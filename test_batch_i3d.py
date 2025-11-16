import torch
import numpy as np
from PIL import Image
import os
from pytorch_i3d import InceptionI3d

# Path to your extracted frames (adjust if needed)
VIDEOS_ROOT = r"C:\Users\karlw\Desktop\tei stuff\thesis\pytorch-i3d-master\testvideos"
OUTPUT_ROOT = r"C:\Users\karlw\Desktop\tei stuff\thesis\pytorch-i3d-master\testvideos\outputfeatures"
MODEL_PATH = r"C:\Users\karlw\Desktop\tei stuff\thesis\pytorch-i3d-master\models\rgb_imagenet.pt"
CHUNK_SIZE = 32  # reduce if memory issues

# === LOAD MODEL ===
print("🔍 Loading I3D model...")
i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
i3d.eval()
print("✅ Model loaded!\n")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def extract_video_features(video_dir, output_file):
    if os.path.exists(output_file):
        print(f"⏩ Skipping (already extracted): {output_file}")
        return

    frame_files = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith(('.jpg', '.png'))
    ])

    if len(frame_files) == 0:
        print(f"⚠️ No frames found in {video_dir}")
        return

    print(f"🎬 Processing: {os.path.basename(video_dir)} ({len(frame_files)} frames)")

    all_feats = []

    with torch.no_grad():
        for i in range(0, len(frame_files), CHUNK_SIZE):
            chunk = frame_files[i:i + CHUNK_SIZE]
            print(f"   🔹 Frames {i+1}–{i+len(chunk)}...")

            frames = []
            for f in chunk:
                img = Image.open(f).resize((224, 224))
                frames.append(np.array(img).astype(np.float32) / 255.0)

            frames = np.stack(frames)
            frames = torch.from_numpy(frames).permute(3,0,1,2).unsqueeze(0)

            feats = i3d.extract_features(frames)
            all_feats.append(feats.squeeze(0))

            del frames, feats  # 🔥 prevent RAM overflow

    features = torch.cat(all_feats, dim=1)
    np.save(output_file, features.cpu().numpy())

    print(f"💾 Saved: {output_file} | Shape: {tuple(features.shape)}\n")
    del all_feats, features


print("📂 Searching for processed videos...\n")
video_folders = [
    os.path.join(VIDEOS_ROOT, f)
    for f in os.listdir(VIDEOS_ROOT)
    if os.path.isdir(os.path.join(VIDEOS_ROOT, f))
]

if not video_folders:
    print("❌ No video frame folders found!")
else:
    print(f"📌 Found {len(video_folders)} video folders\n")

for folder in video_folders:
    base = os.path.basename(folder)
    output_file = os.path.join(OUTPUT_ROOT, f"{base}.npy")
    extract_video_features(folder, output_file)

print("🏁 All done!")