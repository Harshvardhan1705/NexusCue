import cv2
import os
import re
import numpy as np
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

BASE = r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"

WORD_DIR = os.path.join(BASE, "Frames_Word_Level")
MODEL_DIR = os.path.join(BASE, "face_models")

DETECTOR_MODEL = os.path.join(
    MODEL_DIR,
    "face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_MODEL = os.path.join(
    MODEL_DIR,
    "face_recognition_sface_2021dec.onnx"
)

# ============================================================
# SETTINGS
# ============================================================

WORDS = [
    "YOU",
    "FREE",
    "TODAY",
    "HAPPY",
    "HOW"
]

# Similarity threshold used to consider two faces
# likely to belong to the same person.
#
# We start conservatively. We will inspect the results
# before using this in the final application.
SIMILARITY_THRESHOLD = 0.70

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading YuNet...")

detector = cv2.FaceDetectorYN_create(
    DETECTOR_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("Loading SFace...")

recognizer = cv2.FaceRecognizerSF_create(
    RECOGNIZER_MODEL,
    ""
)

print("Models loaded.\n")


# ============================================================
# NATURAL SORT
# ============================================================

def natural_key(filename):

    parts = re.split(
        r"(\d+)",
        filename.lower()
    )

    return [
        int(part) if part.isdigit() else part
        for part in parts
    ]


# ============================================================
# GET WORD IMAGES
# ============================================================

def get_word_images(word):

    folder = os.path.join(
        WORD_DIR,
        word
    )

    if not os.path.exists(folder):
        return []

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    # IMPORTANT:
    # YOUR files are not the YOU sign.
    if word == "YOU":
        files = [
            f for f in files
            if not f.upper().startswith("YOUR")
        ]

    files.sort(key=natural_key)

    return [
        os.path.join(folder, f)
        for f in files
    ]


# ============================================================
# EXTRACT FACE EMBEDDING
# ============================================================

def get_face_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    height, width = image.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    # Select the largest face
    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    try:

        aligned = recognizer.alignCrop(
            image,
            face
        )

        feature = recognizer.feature(
            aligned
        )

        feature = feature.flatten()

        # Normalize
        norm = np.linalg.norm(feature)

        if norm == 0:
            return None

        feature = feature / norm

        return feature

    except Exception:
        return None


# ============================================================
# LOAD ALL FEATURES
# ============================================================

records = []

print("=" * 70)
print("EXTRACTING FACE FEATURES")
print("=" * 70)

for word in WORDS:

    files = get_word_images(word)

    print(
        f"\n{word}: {len(files)} images"
    )

    for index, path in enumerate(files, 1):

        filename = os.path.basename(path)

        feature = get_face_embedding(path)

        if feature is None:

            print(
                f"  ❌ {filename}"
            )

            continue

        records.append({
            "word": word,
            "filename": filename,
            "path": path,
            "feature": feature
        })

        print(
            f"  ✅ {filename}"
        )


print(
    f"\nTotal usable face images: "
    f"{len(records)}"
)


# ============================================================
# PAIRWISE SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print("CROSS-WORD MATCHES")
print("=" * 70)

# Only compare different words.
# This is important because we want to find the same signer
# appearing across different signs.

matches = []

for i in range(len(records)):

    for j in range(i + 1, len(records)):

        a = records[i]
        b = records[j]

        if a["word"] == b["word"]:
            continue

        score = float(
            np.dot(
                a["feature"],
                b["feature"]
            )
        )

        if score >= SIMILARITY_THRESHOLD:

            matches.append({
                "score": score,
                "a": i,
                "b": j
            })


matches.sort(
    key=lambda x: x["score"],
    reverse=True
)


# ============================================================
# PRINT STRONGEST MATCHES
# ============================================================

print(
    f"\nMatches above "
    f"{SIMILARITY_THRESHOLD:.2f}: "
    f"{len(matches)}"
)

print("\nTop 40 cross-word matches:\n")

for match in matches[:40]:

    a = records[match["a"]]
    b = records[match["b"]]

    print(
        f"{match['score']:.4f} | "
        f"{a['word']:5s} | "
        f"{a['filename']:18s} <-> "
        f"{b['word']:5s} | "
        f"{b['filename']}"
    )


# ============================================================
# FIND BEST SIGNER CANDIDATES FOR EACH WORD
# ============================================================

print("\n" + "=" * 70)
print("BEST CROSS-WORD CANDIDATES")
print("=" * 70)

for word in WORDS:

    print(f"\n{word}")

    source_indices = [
        i for i, r in enumerate(records)
        if r["word"] == word
    ]

    candidates = defaultdict(list)

    for source_index in source_indices:

        source = records[source_index]

        for other_index, other in enumerate(records):

            if other["word"] == word:
                continue

            score = float(
                np.dot(
                    source["feature"],
                    other["feature"]
                )
            )

            if score >= SIMILARITY_THRESHOLD:

                candidates[other["word"]].append(
                    (
                        score,
                        source["filename"],
                        other["filename"]
                    )
                )

    for other_word in sorted(candidates):

        values = sorted(
            candidates[other_word],
            reverse=True
        )

        if values:

            score, source_file, other_file = values[0]

            print(
                f"  {other_word:5s} -> "
                f"{score:.4f} | "
                f"{source_file} <-> {other_file}"
            )


# ============================================================
# SAVE FEATURES
# ============================================================

feature_file = "signer_face_features.npz"

np.savez(
    feature_file,

    features=np.array([
        r["feature"]
        for r in records
    ]),

    words=np.array([
        r["word"]
        for r in records
    ]),

    filenames=np.array([
        r["filename"]
        for r in records
    ]),

    paths=np.array([
        r["path"]
        for r in records
    ])
)

print(
    f"\nSaved feature database: "
    f"{feature_file}"
)

print("\nFinished.")