import torch
import numpy as np
from PIL import Image
import os
from pytorch_i3d import InceptionI3d

# Path to your extracted frames (adjust if needed)
frames_dir = r"C:\Users\karlw\Desktop\tei stuff\thesis\pytorch-i3d-master\testvideos\framevideos\video1"
model_path = r"C:\Users\karlw\Desktop\tei stuff\thesis\pytorch-i3d-master\models\rgb_imagenet.pt"
output_path = r"C:\Users\karlw\Desktop\tei stuff\thesis\pytorch-i3d-master\testvideos\outputfeatures\output_features1.pt"

# === LOAD FRAME FILENAMES ===
print("📂 Loading frames from:", frames_dir)
frame_files = sorted([
    os.path.join(frames_dir, f)
    for f in os.listdir(frames_dir)
    if f.endswith(('.jpg', '.png'))
])
print("🖼️ Files found:", len(frame_files))

# === LOAD MODEL ===
print("🔍 Loading model...")
i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load(model_path, map_location='cpu'))
i3d.eval()
print("✅ Model loaded successfully!")

# === CHUNK PARAMETERS ===
chunk_size = 64  # Try 64, or smaller (e.g. 32) if still too large
all_features = []

# === EXTRACT FEATURES IN CHUNKS ===
with torch.no_grad():
    for i in range(0, len(frame_files), chunk_size):
        chunk_files = frame_files[i:i + chunk_size]
        frames = []
        for f in chunk_files:
            img = Image.open(f).resize((224, 224))
            frames.append(np.array(img).astype(np.float32) / 255.0)
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)

        print(f"⚙️ Processing frames {i+1}–{i+len(chunk_files)}...")
        feats = i3d.extract_features(frames)
        all_features.append(feats.squeeze(0))  # remove batch dim

# === CONCATENATE AND SAVE ===
features = torch.cat(all_features, dim=1)  # concat along time dimension
torch.save(features, output_path)
print("💾 Features saved to:", output_path)
print("🏁 Done! Final feature shape:", features.shape)