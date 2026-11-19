import os
import cv2
import numpy as np
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\mini project hhai\nexus\dataset\isl_cslrt_corpus\isl_cslrt_corpus"

CLUSTER_FILE = os.path.join(
    BASE_DIR,
    "signer_clusters.npz"
)

FACE_MODEL = os.path.join(
    BASE_DIR,
    "face_models",
    "face_detection_yunet_2023mar.onnx"
)

TEST_WORDS = ["YOU", "FREE", "TODAY"]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SMART REPRESENTATIVE SIGN IMAGE SELECTOR")
print("=" * 70)

print("\nLoading signer clusters...")

data = np.load(
    CLUSTER_FILE,
    allow_pickle=True
)

labels = data["labels"]
words = data["words"]
filenames = data["filenames"]
paths = data["paths"]

print(f"Loaded {len(paths)} clustered images.")


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

print("\nLoading face detector...")

detector = cv2.FaceDetectorYN.create(
    FACE_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("Face detector loaded.")


# ============================================================
# BUILD WORD -> GROUP -> IMAGES
# ============================================================

word_groups = defaultdict(
    lambda: defaultdict(list)
)

for word, filename, path, label in zip(
    words,
    filenames,
    paths,
    labels
):

    word = str(word).upper()
    filename = str(filename)
    path = str(path)
    label = int(label)

    # Ignore YOUR from YOU
    if word == "YOU" and filename.upper().startswith("YOUR"):
        continue

    word_groups[word][label].append(path)


# ============================================================
# IMAGE QUALITY FUNCTIONS
# ============================================================

def calculate_sharpness(image):
    """
    Variance of Laplacian.
    Higher = sharper image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def detect_face_quality(image):
    """
    Returns:
        face_score
        face_area
        face_detected
    """

    h, w = image.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:

        return 0.0, 0.0, False

    # Select largest detected face
    best_face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    x, y, fw, fh = best_face[:4]

    face_area = max(
        0,
        fw * fh
    )

    image_area = w * h

    face_ratio = face_area / image_area

    # Clamp for normalization
    face_score = min(
        face_ratio / 0.30,
        1.0
    )

    return face_score, face_ratio, True


# ============================================================
# CALCULATE IMAGE SCORE
# ============================================================

def score_image(path):

    image = cv2.imread(path)

    if image is None:

        return {
            "score": -1,
            "sharpness": 0,
            "face_score": 0,
            "face_ratio": 0,
            "face_detected": False
        }

    sharpness = calculate_sharpness(
        image
    )

    face_score, face_ratio, detected = (
        detect_face_quality(image)
    )

    # Normalize sharpness.
    # 1000 is treated as very sharp.
    sharpness_score = min(
        sharpness / 1000.0,
        1.0
    )

    # Weighted quality score
    #
    # Face visibility = 60%
    # Sharpness       = 40%
    #
    final_score = (
        0.60 * face_score +
        0.40 * sharpness_score
    )

    return {
        "score": final_score,
        "sharpness": sharpness,
        "face_score": face_score,
        "face_ratio": face_ratio,
        "face_detected": detected
    }


# ============================================================
# AVAILABLE GROUPS
# ============================================================

print("\n" + "=" * 70)
print("AVAILABLE SIGNER GROUPS")
print("=" * 70)

for word in TEST_WORDS:

    print(f"\n{word}")

    for group, images in sorted(
        word_groups.get(word, {}).items()
    ):

        print(
            f"  Group {group + 1}: "
            f"{len(images)} image(s)"
        )


# ============================================================
# FIND COMMON GROUP
# ============================================================

print("\n" + "=" * 70)
print("FINDING COMMON SIGNER GROUPS")
print("=" * 70)

group_sets = []

for word in TEST_WORDS:

    groups = set(
        word_groups.get(word, {}).keys()
    )

    if groups:
        group_sets.append(groups)


common_groups = set.intersection(
    *group_sets
)

print("\nCommon signer groups:")

for group in sorted(common_groups):

    print(
        f"  Group {group + 1}"
    )


# ============================================================
# SELECT SIGNER GROUP
# ============================================================

group_scores = {}

for group in common_groups:

    count = sum(
        len(word_groups[word][group])
        for word in TEST_WORDS
    )

    group_scores[group] = count


selected_group = max(
    group_scores,
    key=group_scores.get
)

print(
    f"\n🎯 Selected signer group: "
    f"Group {selected_group + 1}"
)

print(
    f"Candidate images: "
    f"{group_scores[selected_group]}"
)


# ============================================================
# SELECT BEST IMAGE FOR EACH WORD
# ============================================================

print("\n" + "=" * 70)
print("QUALITY-BASED REPRESENTATIVE SELECTION")
print("=" * 70)

selected_images = {}


for word in TEST_WORDS:

    candidates = word_groups[word].get(
        selected_group,
        []
    )

    if not candidates:

        print(
            f"\n❌ {word}: "
            f"No image for selected signer."
        )

        continue

    scored = []

    print(f"\nAnalyzing {word}...")

    for path in candidates:

        result = score_image(path)

        scored.append(
            (
                result["score"],
                path,
                result
            )
        )

    # Highest score first
    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_score, best_path, best_result = (
        scored[0]
    )

    selected_images[word] = best_path

    print(
        f"\n  Selected: "
        f"{os.path.basename(best_path)}"
    )

    print(
        f"  Quality score : "
        f"{best_score:.4f}"
    )

    print(
        f"  Sharpness     : "
        f"{best_result['sharpness']:.2f}"
    )

    print(
        f"  Face detected : "
        f"{'YES' if best_result['face_detected'] else 'NO'}"
    )

    print(
        f"  Face ratio    : "
        f"{best_result['face_ratio']:.4f}"
    )

    # Show top 3 candidates
    print("\n  Top candidates:")

    for rank, (
        score,
        path,
        result
    ) in enumerate(
        scored[:3],
        start=1
    ):

        print(
            f"    {rank}. "
            f"{os.path.basename(path)} "
            f"→ {score:.4f}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL REPRESENTATIVE SIGN SEQUENCE")
print("=" * 70)

for word in TEST_WORDS:

    if word in selected_images:

        print(
            f"{word:8} → "
            f"{os.path.basename(selected_images[word])}"
        )

    else:

        print(
            f"{word:8} → ❌ MISSING"
        )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)