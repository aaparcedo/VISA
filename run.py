import torch
import numpy as np
import cv2
import sys
import os
import re
from PIL import Image
import transformers
from transformers import AutoTokenizer, CLIPImageProcessor, LlamaForCausalLM
import torch.nn.functional as F
import torchvision.transforms.functional as TF # New import for XMem

from model.llava import conversation as conversation_lib
from model.VISA import VISAForCausalLM
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IMAGE_TOKEN

from llamavid.model.builder import load_pretrained_model

from XMem.model.network import XMem
from XMem.inference.inference_core import InferenceCore




OUTPUT_DIR = "./video_output" # Directory to save final mask frames
os.makedirs(OUTPUT_DIR, exist_ok=True)
    

# ==============================================================================
# --- 1. Configuration (Set your paths here) ---
# ==============================================================================
IMAGE_SIZE = 1024
OUT_DIM = 256
MODEL_MAX_LENGTH = 2048
DEVICE = "cuda"
DTYPE = torch.bfloat16

# --- VISA Model Paths ---
VISA_MODEL_PATH = "VISA-7B" # Path to your VISA-7B weights
SAM_PRETRAINED_PATH = "sam_vit_h_4b8939.pth" # Path to SAM weights
VISION_TOWER = "openai/clip-vit-large-patch14"

# --- LLaMA-ViD Model Paths (for TFS) ---
# You must download the LLaMA-ViD model weights for this to work
LLAMA_VID_MODEL_PATH = "/home/aparcedo/VISA/llama-vid-13b-full-224-video-fps-1"
LLAMA_VID_MODEL_NAME = "LLaMA-ViD-v1.5-7B" # Model name as used in their repo
TFS_NUM_FRAMES = 100 # Number of frames to sample for LLaMA-ViD query
TFS_K_RESPONSES = 10 # Number of responses to average, as per paper 

XMEM_MODEL_PATH="/home/aparcedo/VISA/XMem/scripts/saves/XMem.pth"

# --- Input Data ---
VIDEO_PATH = "/home/c3-0/datasets/stvg/hcstvg1/v1/video/55_vfjywN5CN0Y.mp4"
CAPTION = "The man in brown clothes pours the contents of the bag into his hand, and then takes out a piece of paper from the bag and opens it"

# ==============================================================================
# --- 2. Load VISA Model (Main Segmentation Model) ---
# ==============================================================================
print(f"Loading VISA model from: {VISA_MODEL_PATH}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    VISA_MODEL_PATH,
    cache_dir=None,
    model_max_length=MODEL_MAX_LENGTH,
    padding_side="right",
    use_fast=False
)
tokenizer.pad_token = tokenizer.unk_token
num_added_tokens = tokenizer.add_tokens("[SEG]")
SEG_TOKEN_IDX = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

print(f'SEG TOKEN INDEX: {SEG_TOKEN_IDX}')

model_args = {
    "train_mask_decoder": False,
    "out_dim": OUT_DIM,
    "seg_token_idx": SEG_TOKEN_IDX,
    "vision_pretrained": SAM_PRETRAINED_PATH,
    "vision_tower": VISION_TOWER,
    "use_im_start_end": False,
}

model = VISAForCausalLM.from_pretrained(
    VISA_MODEL_PATH,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    **model_args
).to(DEVICE)

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=DTYPE, device=DEVICE)

for p in vision_tower.parameters():
    p.requires_grad = False
for p in model.get_model().mm_projector.parameters():
    p.requires_grad = False

conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
model.resize_token_embeddings(len(tokenizer))

print("VISA model loaded.")

# ==============================================================================
# --- 3. Load LLaMA-ViD Model (Text-guided Frame Sampler) ---
# ==============================================================================
print(f"Loading LLaMA-ViD model from: {LLAMA_VID_MODEL_PATH}")

llamavid_tokenizer, llamavid_model, llamavid_processor, _ = load_pretrained_model(
    model_path=LLAMA_VID_MODEL_PATH,
    model_base=None,
    model_name=LLAMA_VID_MODEL_NAME,
    device=DEVICE
)
llamavid_model.to(dtype=DTYPE)

print("LLaMA-ViD model loaded.")

# ==============================================================================
# --- 4. Load Video Frames ---
# ==============================================================================

print(f"Loading video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
frames_np = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frames_np.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

total_frames = len(frames_np)
original_size = frames_np[0].shape[:2]
print(f"Video loaded. Total frames: {total_frames}")

# ==============================================================================
# --- 5. Run Text-guided Frame Sampler (TFS) to find f_tgt ---
# ==============================================================================
print("Running Text-guided Frame Sampler (TFS)...")

# 1. Prepare video tensor for LLaMA-ViD (sample 100 frames)
tfs_indices = np.round(np.linspace(0, total_frames - 1, TFS_NUM_FRAMES)).astype(int)
tfs_frames_pil = [Image.fromarray(frames_np[i]) for i in tfs_indices]
tfs_video_tensor = llamavid_processor(tfs_frames_pil, return_tensors="pt")["pixel_values"].to(DEVICE, dtype=DTYPE)
# 2. Prepare text prompt for LLaMA-ViD
tfs_prompt_template = "<VIDEO> To find {description}, which percentage mark of the video should I check? Please respond with a number between 0% and 100%."
tfs_description = CAPTION.lower().replace("please segment the", "").strip()
tfs_prompt_str = tfs_prompt_template.format(description=tfs_description)

tfs_text_tensor = llamavid_tokenizer(tfs_prompt_str, return_tensors='pt')['input_ids'].to(DEVICE)

# 3. Generate K responses and parse percentages
found_percentages = []
print(f"Querying LLaMA-ViD {TFS_K_RESPONSES} times...")
with torch.inference_mode():
    for _ in range(TFS_K_RESPONSES):
        print('TKS RESPONSE')
        output_ids = llamavid_model.generate(
        input_ids=tfs_text_tensor,
        prompts=[[tfs_prompt_str]],  # Pass the raw string in a list
        images=tfs_video_tensor.unsqueeze(0),
        do_sample=True,
        temperature=0.2,
        max_new_tokens=1024,
        use_cache=True,
    )
        response = llamavid_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        # Parse percentage
        match = re.findall(r"(\d+)\%", response)
        if match:
            found_percentages.append(float(match[0]))

if not found_percentages:
    print("Warning: TFS did not return a valid percentage. Defaulting to middle frame.")
    avg_percentage = 50.0
else:
    avg_percentage = np.mean(found_percentages)

# 4. Calculate target frame index (f_tgt)
target_frame_idx = int(total_frames * (avg_percentage / 100.0))
target_frame_idx = np.clip(target_frame_idx, 0, total_frames - 1)
print(f"TFS complete. Target frame (f_tgt): {target_frame_idx} ({avg_percentage:.1f}%)")

# ==============================================================================
# --- 6. Prepare Frames for VISA (Global-Local Sampling for x_r) ---
# ==============================================================================
print("Applying Global-Local sampling for reference frames (x_r)...")

num_ref_frames = 12
num_global = num_ref_frames // 2
num_local = num_ref_frames - num_global

# Global sampling: uniformly sample through the whole video 
global_indices = np.round(np.linspace(0, total_frames - 1, num_global)).astype(int)

# Local sampling: sample contiguous frames centered by f_tgt 
# We define "contiguous" as a window around the target frame
local_start = max(0, target_frame_idx - num_local * 2) # Heuristic window
local_end = min(total_frames - 1, target_frame_idx + num_local * 2) # Heuristic window
if local_end - local_start < num_local:
    local_start = max(0, total_frames - num_local)
    local_end = total_frames -1
    
local_indices = np.round(np.linspace(local_start, local_end, num_local)).astype(int)

# Combine and get unique indices
ref_indices = np.concatenate([global_indices, local_indices])
all_proc_indices = np.unique(np.append(ref_indices, target_frame_idx))
print(f"Sampling complete. {len(all_proc_indices)} unique frames selected.")


# --- Preprocess frames for VISA ---
image_list_clip = []
clip_image_processor = CLIPImageProcessor.from_pretrained(VISION_TOWER)

# 1. Process all_proc_indices for CLIP (reference frames x_r + f_tgt)
for frm_idx in all_proc_indices:
    image_clip = clip_image_processor.preprocess(
        frames_np[frm_idx], return_tensors="pt"
    )["pixel_values"][0].to(DEVICE, dtype=DTYPE)
    image_list_clip.append(image_clip)

image_clip_tensor = torch.stack(image_list_clip, dim=0)

# 2. Process target_frame_idx for SAM (target frame f_tgt)
def _preprocess_sam(x):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(x.device, x.dtype)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(x.device, x.dtype)
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = IMAGE_SIZE - h
    padw = IMAGE_SIZE - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

transform = ResizeLongestSide(IMAGE_SIZE)
target_frame_sam_resized = transform.apply_image(frames_np[target_frame_idx])
resize_list = [target_frame_sam_resized.shape[:2]]
target_frame_sam = _preprocess_sam(
    torch.from_numpy(target_frame_sam_resized).permute(2, 0, 1).contiguous()
).to(DEVICE, dtype=DTYPE)

# ==============================================================================
# --- 7. Prepare Text Inputs for VISA ---
# ==============================================================================
conv = conversation_lib.conv_templates["llava_v1"].copy()
conv.messages = []

num_frames = len(all_proc_indices) # Count how many frames we are passing

prompt = CAPTION.lower()
prompt = (DEFAULT_IMAGE_TOKEN * num_frames) + "\n" + prompt # Create one token per frame
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], "Sure, [SEG].")
prompt_str = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_str, tokenizer, return_tensors="pt").unsqueeze(0).to(DEVICE)

# ==============================================================================
# --- 8. Run VISA Model Forward Pass ---
# ==============================================================================
print("Running VISA model_forward pass...")

with torch.inference_mode():
    output = model.model_forward(
        images=[target_frame_sam.unsqueeze(0)], # This is f_tgt for SAM
        images_clip=[image_clip_tensor],         # This is x_r + f_tgt for CLIP
        input_ids=input_ids,
        labels=None,
        attention_masks=input_ids.ne(tokenizer.pad_token_id),
        # --- IMPORTANT ---
        # `offset` tracks conversations, not frames.
        # Since we have 1 video and 1 conversation, offset is [0, 1].
        offset=torch.LongTensor([0, 1]).to(DEVICE),
        # -----------------
        masks_list=[], # No GT masks for inference
        label_list=[torch.zeros(original_size)], # Dummy label list
        resize_list=resize_list,
        conversation_list=[prompt_str],
        num_frame_list=[len(all_proc_indices)],
        num_conv_list=[1], # We are processing 1 conversation
        inference=True,
    )

print("Inference complete.")
# output["pred_masks"] contains the segmentation mask for the target frame (f_tgt)
# output["pred_masks"][0] will be a tensor of shape [1, H, W]
print(f"Output mask shape: {output['pred_masks'][0].shape}")

initial_mask = output['pred_masks'][0][0] # Shape [H, W]
print(f"Output mask shape: {initial_mask.shape}")
print(f"Initial mask shape: {initial_mask.shape}, Sum: {initial_mask.sum()}") 


# ==============================================================================
# --- 9. Load XMem Tracker ---
# ==============================================================================
print("Loading XMem model...")
# Set up XMem configuration
xmem_config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_frames': 10,
    'max_long_term_elements': 10000, # <-- ADD THIS LINE
    # 'single_object': True
}

# Load the XMem model from the checkpoint
xmem_model = XMem(xmem_config, XMEM_MODEL_PATH).to(DEVICE).eval()
if DTYPE == torch.bfloat16:
    xmem_model.half() # XMem uses half, not bfloat16

# Initialize the inference core
xmem_tracker = InferenceCore(xmem_model, config=xmem_config)
xmem_tracker.set_all_labels([1])
# Set the target size for XMem (it prefers 480p)
xmem_target_size = (480, 854) # H, W. Adjust as needed.


# ==============================================================================
# --- 10. Run XMem Propagation ---
# ==============================================================================
print(f"Running XMem propagation for {total_frames} frames...")


# Normalize image function for XMem
def xmem_preprocess(image_np):
    image_torch = torch.from_numpy(image_np).permute(2, 0, 1).float() # HWC -> CHW
    image_torch = TF.resize(image_torch, xmem_target_size, interpolation=Image.BILINEAR)
    # Normalize with ImageNet stats
    image_torch = TF.normalize(image_torch / 255.0, 
                             mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    return image_torch.to(DEVICE)

# Loop through all frames
with torch.inference_mode():
    for frame_idx, frame_np in enumerate(frames_np):
        
        # 1. Preprocess the frame for XMem
        frame_torch = xmem_preprocess(frame_np).half()

        # 2. Check if this is the target frame
        if frame_idx == target_frame_idx:
            # This is the frame we have the mask for
            # Binarize the mask from VISA (it's logits)
            mask_torch = (initial_mask > 0).to(torch.uint8) # Shape [H, W]
            
            # Resize mask to XMem's target size
            mask_torch = TF.resize(mask_torch.unsqueeze(0), 
                                   xmem_target_size, 
                                   interpolation=Image.NEAREST)
            
            # Initialize the tracker with this frame and mask
            xmem_tracker.clear_memory() # Clear memory before starting
            predicted_mask = xmem_tracker.step(frame_torch, mask_torch[0].to(DEVICE))

        elif frame_idx < target_frame_idx:
            # This is a frame *before* the target frame
            # We skip them for bi-directional tracking, or run in reverse.
            # For simplicity, we'll just track forward from f_tgt.
            # You can implement backward tracking later if needed.
            if frame_idx == 0:
                # To make the tracker work, we must init on frame 0
                # We'll give it a dummy mask
                dummy_mask = torch.zeros(xmem_target_size, dtype=torch.uint8, device=DEVICE)
                xmem_tracker.step(frame_torch, dummy_mask)
            else:
                 xmem_tracker.step(frame_torch) # No mask provided
            continue # Skip saving output

        else:
            # This is a frame *after* the target frame
            # Track it using memory
            predicted_mask = xmem_tracker.step(frame_torch) # No mask provided
        
        # 3. Save the output mask
        # The predicted_mask is [1, H, W] tensor, values 0 or 1
        print(f"Frame {frame_idx}: Predicted mask is None? {predicted_mask is None}") # <--- ADD THIS
        if predicted_mask is not None:
            print(f"Frame {frame_idx}: Predicted mask shape: {predicted_mask.shape}, Sum: {predicted_mask.sum()}") # <--- ADD THIS
            # Upscale mask to original video size
            final_mask_hw = TF.resize(predicted_mask.unsqueeze(0).to(torch.float32), 
                                      original_size, 
                                      interpolation=Image.NEAREST)[0, 0]
            
            # Convert to numpy array
            final_mask_np = (final_mask_hw.cpu().numpy() * 255).astype(np.uint8)
            
            # Save the mask as an image
            mask_filename = os.path.join(OUTPUT_DIR, f"{frame_idx:05d}.png")
            print(f"Frame {frame_idx}: Saving mask to {mask_filename}") # <--- ADD THIS
            try:
                cv2.imwrite(mask_filename, final_mask_np)
            except Exception as e:
                print(f"Frame {frame_idx}: ERROR saving mask - {e}")

print(f"Propagation complete. All masks saved to {OUTPUT_DIR}")

# --- Bi-directional Tracking (Simplified) ---
# The paper mentions bi-directional tracking. We just did forward-tracking.
# To do backward tracking, you would:
# 1. Re-initialize a new XMem tracker
# 2. Loop from frame_idx = target_frame_idx down to 0
# 3. Feed the initial mask at target_frame_idx
# 4. Call xmem_tracker.step() for all previous frames
# 5. Save those masks
print("Note: Only forward propagation from f_tgt was implemented.")
print("For full bi-directional tracking, a backward pass is also needed.")