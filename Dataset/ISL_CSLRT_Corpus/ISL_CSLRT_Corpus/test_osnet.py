from torchreid.reid.utils.feature_extractor import FeatureExtractor
from PIL import Image
import os
import numpy as np

# --------------------------------------------------
# PATH
# --------------------------------------------------

BASE = r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"
WORD_DIR = os.path.join(BASE, "Frames_Word_Level")

# Test only a few signs first
WORDS = {
    "YOU": 5,
    "FREE": 5,
    "TODAY": 5,
    "HAPPY": 5,
    "HOW": 5
}

# --------------------------------------------------
# LOAD OSNET
# --------------------------------------------------

print("\nLoading OSNet...")

extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path=None,
    device="cpu"
)

print("OSNet loaded successfully.\n")

# --------------------------------------------------
# COLLECT IMAGES
# --------------------------------------------------

image_paths = []
labels = []

for word, count in WORDS.items():

    folder = os.path.join(WORD_DIR, word)

    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {word}")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files = sorted(files)[:count]

    for file in files:
        path = os.path.join(folder, file)

        image_paths.append(path)
        labels.append(word)

        print(f"Added: {word} -> {file}")

# --------------------------------------------------
# EXTRACT FEATURES
# --------------------------------------------------

print(f"\nExtracting features from {len(image_paths)} images...")

features = extractor(image_paths)

print("Feature extraction completed.")

print("\nFeature shape:")
print(features.shape)

# --------------------------------------------------
# SAVE FEATURES
# --------------------------------------------------

output_file = "osnet_test_features.npz"

np.savez(
    output_file,
    features=features.cpu().numpy(),
    labels=np.array(labels),
    paths=np.array(image_paths)
)

print(f"\nSaved features to: {output_file}")

# --------------------------------------------------
# BASIC SIMILARITY TEST
# --------------------------------------------------

print("\nPairwise similarity test:")

features_np = features.cpu().numpy()

# Normalize
features_np = features_np / (
    np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8
)

# Compare every image with every other image
for i in range(len(image_paths)):

    best_score = -1
    best_index = -1

    for j in range(len(image_paths)):

        if i == j:
            continue

        score = np.dot(
            features_np[i],
            features_np[j]
        )

        if score > best_score:
            best_score = score
            best_index = j

    print(
        f"{labels[i]:6s} "
        f"{os.path.basename(image_paths[i]):20s}"
        f" -> "
        f"{labels[best_index]:6s} "
        f"{os.path.basename(image_paths[best_index]):20s}"
        f" similarity={best_score:.4f}"
    )