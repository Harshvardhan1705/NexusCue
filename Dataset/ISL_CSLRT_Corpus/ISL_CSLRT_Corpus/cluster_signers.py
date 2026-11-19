import cv2
import os
import re
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

BASE = r"D:\mini project hhai\nexus\dataset\isl_cslrt_corpus\isl_cslrt_corpus"

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

WORDS = [
    "YOU",
    "FREE",
    "TODAY",
    "HAPPY",
    "HOW"
]

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------

# Start with 7 clusters because the dataset documentation
# reports 7 signers.
N_SIGNERS = 7


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading face models...")

detector = cv2.FaceDetectorYN_create(
    DETECTOR_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF_create(
    RECOGNIZER_MODEL,
    ""
)

print("Models loaded.")


# ============================================================
# NATURAL SORT
# ============================================================

def natural_key(filename):

    parts = re.split(
        r"(\d+)",
        filename.lower()
    )

    return [
        int(x) if x.isdigit() else x
        for x in parts
    ]


# ============================================================
# GET IMAGES
# ============================================================

def get_images(word):

    folder = os.path.join(
        WORD_DIR,
        word
    )

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

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
# FACE FEATURE
# ============================================================

def get_feature(path):

    image = cv2.imread(path)

    if image is None:
        return None

    h, w = image.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

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
        ).flatten()

        norm = np.linalg.norm(feature)

        if norm == 0:
            return None

        return feature / norm

    except:
        return None


# ============================================================
# EXTRACT FEATURES
# ============================================================

records = []

print("\nExtracting features...")

for word in WORDS:

    paths = get_images(word)

    print(
        f"{word}: {len(paths)} images"
    )

    for path in paths:

        feature = get_feature(path)

        if feature is None:
            continue

        records.append({
            "word": word,
            "filename": os.path.basename(path),
            "path": path,
            "feature": feature
        })


print(
    "\nUsable images:",
    len(records)
)


# ============================================================
# CLUSTER
# ============================================================

features = np.array([
    r["feature"]
    for r in records
])


print(
    f"\nClustering into {N_SIGNERS} "
    f"signer groups..."
)

clusterer = AgglomerativeClustering(
    n_clusters=N_SIGNERS,
    metric="cosine",
    linkage="average"
)

labels = clusterer.fit_predict(
    features
)


# ============================================================
# ORGANIZE CLUSTERS
# ============================================================

clusters = defaultdict(list)

for record, label in zip(
    records,
    labels
):

    clusters[label].append(
        record
    )


# ============================================================
# PRINT CLUSTER INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SIGNER CLUSTERS")
print("=" * 70)

for cluster_id in sorted(clusters):

    items = clusters[cluster_id]

    word_counts = defaultdict(int)

    for item in items:
        word_counts[item["word"]] += 1

    print(
        f"\nSIGNER GROUP {cluster_id + 1}"
    )

    print(
        "Total images:",
        len(items)
    )

    print(
        "Word coverage:",
        dict(word_counts)
    )

    for word in WORDS:

        matching = [
            x["filename"]
            for x in items
            if x["word"] == word
        ]

        if matching:

            print(
                f"  {word:5s}: "
                f"{', '.join(matching[:8])}"
                + (
                    " ..."
                    if len(matching) > 8
                    else ""
                )
            )


# ============================================================
# COVERAGE TABLE
# ============================================================

print("\n" + "=" * 70)
print("WORD COVERAGE PER SIGNER")
print("=" * 70)

for cluster_id in sorted(clusters):

    words_present = set(
        x["word"]
        for x in clusters[cluster_id]
    )

    print(
        f"Signer {cluster_id + 1}: "
        f"{len(words_present)}/{len(WORDS)} "
        f"words -> "
        f"{', '.join(sorted(words_present))}"
    )


# ============================================================
# SAVE
# ============================================================

np.savez(
    "signer_clusters.npz",
    features=features,
    labels=labels,
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
    "\nSaved: signer_clusters.npz"
)

print("\nFinished.")