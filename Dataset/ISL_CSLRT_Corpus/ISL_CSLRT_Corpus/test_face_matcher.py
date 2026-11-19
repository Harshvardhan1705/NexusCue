import cv2
import os
import numpy as np

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
# INITIALIZE YUNET FACE DETECTOR
# ============================================================

print("\nLoading YuNet face detector...")

detector = cv2.FaceDetectorYN_create(
    DETECTOR_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

# ============================================================
# INITIALIZE SFACE FACE RECOGNIZER
# ============================================================

print("Loading SFace face recognizer...")

recognizer = cv2.FaceRecognizerSF_create(
    RECOGNIZER_MODEL,
    ""
)

print("Face models loaded successfully.\n")


# ============================================================
# DETECT + EXTRACT FACE EMBEDDING
# ============================================================

def get_face_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read:", image_path)
        return None

    height, width = image.shape[:2]

    # Tell detector the actual image size
    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    # Choose the largest detected face
    best_face = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Align face
    aligned_face = recognizer.alignCrop(
        image,
        best_face
    )

    # Extract face feature
    feature = recognizer.feature(
        aligned_face
    )

    # Normalize
    feature = feature / (
        np.linalg.norm(feature) + 1e-8
    )

    return feature.flatten()


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(feature1, feature2):

    return float(
        np.dot(feature1, feature2)
    )


# ============================================================
# FIND FIRST IMAGE IN WORD FOLDER
# ============================================================

def get_images(word, count=5):

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

    # Sort naturally by number
    import re

    def natural_key(filename):

        numbers = re.findall(
            r"\d+",
            filename
        )

        return (
            int(numbers[-1])
            if numbers
            else 0
        )

    files.sort(key=natural_key)

    return [
        os.path.join(folder, f)
        for f in files[:count]
    ]


# ============================================================
# LOAD TEST IMAGES
# ============================================================

WORDS = [
    "YOU",
    "FREE",
    "TODAY",
    "HAPPY",
    "HOW"
]

all_images = {}

print("Detecting faces...\n")

for word in WORDS:

    images = get_images(
        word,
        count=7
    )

    all_images[word] = []

    for path in images:

        feature = get_face_embedding(path)

        filename = os.path.basename(path)

        if feature is None:

            print(
                f"❌ {word:6s} {filename:20s} "
                "No face detected"
            )

        else:

            all_images[word].append(
                (filename, path, feature)
            )

            print(
                f"✅ {word:6s} {filename:20s} "
                "Face detected"
            )


# ============================================================
# CROSS-WORD SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print("CROSS-WORD FACE SIMILARITY")
print("=" * 70)

word_pairs = [
    ("YOU", "FREE"),
    ("YOU", "TODAY"),
    ("YOU", "HAPPY"),
    ("YOU", "HOW"),
    ("FREE", "TODAY"),
    ("FREE", "HAPPY"),
    ("FREE", "HOW"),
    ("TODAY", "HAPPY"),
    ("TODAY", "HOW"),
    ("HAPPY", "HOW")
]

for word1, word2 in word_pairs:

    print(
        f"\n{word1} ↔ {word2}"
    )

    comparisons = []

    for name1, path1, feat1 in all_images[word1]:

        for name2, path2, feat2 in all_images[word2]:

            score = cosine_similarity(
                feat1,
                feat2
            )

            comparisons.append(
                (
                    score,
                    name1,
                    name2
                )
            )

    comparisons.sort(
        reverse=True
    )

    # Show top 3 matches
    for score, name1, name2 in comparisons[:3]:

        print(
            f"  {score:.4f}  "
            f"{name1}  <->  {name2}"
        )


print("\nFinished.")