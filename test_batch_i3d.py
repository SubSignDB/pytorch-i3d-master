import torch
import numpy as np
from PIL import Image
import os
from pytorch_i3d import InceptionI3d
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 

# Path to your extracted frames (adjust if needed)
VIDEOS_ROOT = r"C:\Users\karlw\Desktop\tei stuff\thesis\pytorch-i3d-master\testvideos\1016"
OUTPUT_ROOT = r"C:\Users\karlw\Desktop\tei stuff\thesis\pytorch-i3d-master\testvideos\outputfeatures\1016_output"
MODEL_PATH = r"C:\Users\karlw\Desktop\tei stuff\thesis\pytorch-i3d-master\models\rgb_imagenet.pt"

CHUNK_SIZE = 32
MAX_WORKERS = 6  # recommended = number of physical cores

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def extract_video_features(video_dir, output_file, model_path):
    if os.path.exists(output_file):
        print(f"⏩ Skipping (already exists): {output_file}")
        return output_file

    frame_files = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith(('.jpg', '.png'))
    ])

    if not frame_files:
        print(f"⚠️ No frames found in: {video_dir}")
        return None

    video_name = os.path.basename(video_dir)

    # Load model inside worker
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(model_path, map_location='cpu'))
    i3d.eval()

    all_feats = []

    with tqdm(total=len(frame_files), desc=video_name, ncols=100, position=0, leave=True) as pbar:
        with torch.no_grad():
            for i in range(0, len(frame_files), CHUNK_SIZE):
                chunk = frame_files[i:i + CHUNK_SIZE]

                frames = [np.array(Image.open(f).resize((224, 224)), dtype=np.float32)/255.0 for f in chunk]

                # Pad last batch if needed
                if len(chunk) < CHUNK_SIZE:
                    frames.extend([frames[-1]] * (CHUNK_SIZE - len(chunk)))

                frames = np.stack(frames)
                frames = torch.from_numpy(frames).permute(3,0,1,2).unsqueeze(0)

                feats = i3d.extract_features(frames)
                all_feats.append(feats.squeeze(0))

                pbar.update(len(chunk))

                del frames, feats

    features = torch.cat(all_feats, dim=1)
    np.save(output_file, features.cpu().numpy())

    print(f"💾 Saved: {output_file} | Shape: {tuple(features.shape)}")
    return output_file


if __name__ == '__main__':
    # === PARALLEL PROCESSING ===
    print("📂 Searching for video folders...\n")
    video_folders = [
        os.path.join(VIDEOS_ROOT, f)
        for f in os.listdir(VIDEOS_ROOT)
        if os.path.isdir(os.path.join(VIDEOS_ROOT, f))
    ]

    print(f"📌 {len(video_folders)} folders found!\n")

    tasks = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for folder in video_folders:
            base = os.path.basename(folder)
            output_file = os.path.join(OUTPUT_ROOT, f"{base}.npy")
            tasks.append(executor.submit(extract_video_features, folder, output_file, MODEL_PATH))

        for future in as_completed(tasks):
            future.result()

    print("🏁 All videos processed in parallel!")