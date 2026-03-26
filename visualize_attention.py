import os
import yaml
import torch
import cv2
import numpy as np
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model.ds import DS

parser = argparse.ArgumentParser(description='Visualize Attention')
parser.add_argument('--config_path', type=str, default='./config/test.yaml', help='path to config file')
parser.add_argument('--weights_path', type=str, default='./ckpt_best.pth', help='path to model weights')
parser.add_argument('--image_paths', type=str, nargs='+', required=True, help='paths to 1 or more images to visualize')
parser.add_argument('--save_dir', type=str, default='./figures/attention_map', help='directory to save the output images')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to run the model on (cuda or cpu)')
args = parser.parse_args()

device = torch.device(args.device)

def main():
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Prepare the model
    model = DS(clip_name=config['clip_model_name'],
               adapter_vit_name=config['vit_name'],
               num_quires=config['num_quires'],
               fusion_map=config['fusion_map'],
               mlp_dim=config['mlp_dim'],
               mlp_out_dim=config['mlp_out_dim'],
               head_num=config['head_num'],
               device=args.device)

    # Load weights
    if os.path.exists(args.weights_path):
        ckpt = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded weights from {args.weights_path}")
    else:
        print(f"Weights not found at {args.weights_path}. Visualizing with random weights.")
    
    model.to(device)
    model.eval()
    
    # Setup the target attention block
    target_block = None
    if hasattr(model, 'rec_attn_clip'):
        target_block = model.rec_attn_clip.resblocks[-1]
    else:
        target_block = model.clip_model.visual.transformer.resblocks[-1]
        
    target_block.need_weights = True
    
    # Define the transform pipeline mimicking the test_set loader
    transform = A.Compose([
        A.Resize(config['resolution'], config['resolution']),
        A.Normalize(mean=config['mean'], std=config['std']),
        ToTensorV2(),
    ])
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for img_path in args.image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
            
        print(f"Processing {img_path}...")
        
        # Load image via OpenCV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: Failed to load image {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = transform(image=img_rgb)
        img_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Create a dummy batch for inference
        batch = {
            'image': img_tensor,
            'label': torch.tensor([1]).to(device), # dummy label
            'if_boundary': torch.tensor([0]).to(device) # dummy boundary
        }
        
        # Forward pass
        with torch.no_grad():
            pred_dict = model(batch, inference=True)
            
        # Extract weights
        attn_weights = target_block.last_attn_weights
        if attn_weights is None:
            print("Failed to capture attention weights.")
            continue
            
        attn_map = attn_weights[0] 
        
        num_quires = config.get('num_quires', 128)
        start_idx = num_quires + 1
        
        spatial_attn = attn_map[:start_idx, start_idx:].mean(dim=0)
        spatial_tokens = spatial_attn.shape[0]
        
        grid_size = int(np.sqrt(spatial_tokens))
        spatial_attn = spatial_attn.reshape(grid_size, grid_size).cpu().float().numpy()
        
        # Normalize the heatmap
        spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min() + 1e-8)
        
        # Resize spatial_attn to match original image resolution
        img_h, img_w = img_bgr.shape[:2]
        heatmap = cv2.resize(spatial_attn, (img_w, img_h))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay onto original image
        alpha = 0.5
        overlay = cv2.addWeighted(img_bgr, alpha, heatmap_colored, 1 - alpha, 0)
        
        # Construct save paths
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(args.save_dir, f"{name}{ext}")
        side_by_side_path = os.path.join(args.save_dir, f"{name}_side_by_side{ext}")
        
        # Save results
        cv2.imwrite(save_path, overlay)
        
        # Save side by side (Original | Heatmap | Overlay)
        side_by_side = np.hstack((img_bgr, heatmap_colored, overlay))
        cv2.imwrite(side_by_side_path, side_by_side)
        
        print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    main()
