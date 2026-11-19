import os
import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\mini project hhai\nexus\dataset\isl_cslrt_corpus\isl_cslrt_corpus"

CLUSTER_FILE = os.path.join(
    BASE_DIR,
    "signer_clusters.npz"
)

HAND_MODEL = os.path.join(
    BASE_DIR,
    "mediapipe_models",
    "hand_landmarker.task"
)

POSE_MODEL = os.path.join(
    BASE_DIR,
    "mediapipe_models",
    "pose_landmarker_lite.task"
)

FACE_MODEL = os.path.join(
    BASE_DIR,
    "face_models",
    "face_detection_yunet_2023mar.onnx"
)

# Test sentence:
# "are you free today"
TEST_WORDS = ["YOU", "FREE", "TODAY"]


# ============================================================
# LOAD CLUSTER DATA
# ============================================================

print("=" * 70)
print("SIGN-AWARE REPRESENTATIVE IMAGE SELECTOR")
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
# BUILD WORD -> SIGNER GROUP -> IMAGES
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

    # YOUR is a separate sign.
    if word == "YOU" and filename.upper().startswith("YOUR"):
        continue

    word_groups[word][label].append(path)


# ============================================================
# MEDIA PIPE TASKS
# ============================================================

print("\nLoading MediaPipe Tasks models...")

BaseOptions = mp.tasks.BaseOptions

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)


# ------------------------------------------------------------
# Hand Landmarker
# ------------------------------------------------------------

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30,
    min_tracking_confidence=0.30
)

hand_detector = HandLandmarker.create_from_options(
    hand_options
)


# ------------------------------------------------------------
# Pose Landmarker
# ------------------------------------------------------------

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.30,
    min_pose_presence_confidence=0.30
)

pose_detector = PoseLandmarker.create_from_options(
    pose_options
)

print("MediaPipe Tasks models loaded.")


# ============================================================
# FACE DETECTOR
# ============================================================

print("\nLoading face detector...")

face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("Face detector loaded.")


# ============================================================
# IMAGE SHARPNESS
# ============================================================

def calculate_sharpness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


# ============================================================
# MEDIAPIPE IMAGE
# ============================================================

def make_mp_image(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


# ============================================================
# HAND SCORE
# ============================================================

def calculate_hand_score(image):

    mp_image = make_mp_image(image)

    result = hand_detector.detect(
        mp_image
    )

    if not result.hand_landmarks:

        return 0.0, 0

    hand_count = len(
        result.hand_landmarks
    )

    # 1 hand = 0.5
    # 2 hands = 1.0

    hand_score = min(
        hand_count / 2.0,
        1.0
    )

    return hand_score, hand_count


# ============================================================
# POSE SCORE
# ============================================================

def calculate_pose_score(image):

    mp_image = make_mp_image(image)

    result = pose_detector.detect(
        mp_image
    )

    if not result.pose_landmarks:

        return 0.0, 0

    landmarks = result.pose_landmarks[0]

    # MediaPipe Pose indices:
    #
    # 0  = nose
    # 11 = left shoulder
    # 12 = right shoulder
    # 13 = left elbow
    # 14 = right elbow
    # 15 = left wrist
    # 16 = right wrist

    important_indices = [
        0,
        11,
        12,
        13,
        14,
        15,
        16
    ]

    visible = 0

    for index in important_indices:

        landmark = landmarks[index]

        if landmark.visibility >= 0.30:
            visible += 1

    pose_score = (
        visible /
        len(important_indices)
    )

    return pose_score, visible


# ============================================================
# FACE SCORE
# ============================================================

def calculate_face_score(image):

    h, w = image.shape[:2]

    face_detector.setInputSize(
        (w, h)
    )

    _, faces = face_detector.detect(
        image
    )

    if faces is None or len(faces) == 0:

        return 0.0

    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    x, y, fw, fh = face[:4]

    ratio = (
        max(0, fw) *
        max(0, fh)
    ) / (w * h)

    # Face is intentionally a secondary signal.
    return min(
        ratio / 0.10,
        1.0
    )


# ============================================================
# COMPLETE IMAGE ANALYSIS
# ============================================================

def analyze_image(path):

    image = cv2.imread(path)

    if image is None:

        return {
            "total": -1,
            "hands": 0,
            "hand_count": 0,
            "pose": 0,
            "pose_points": 0,
            "face": 0,
            "sharpness": 0
        }

    hand_score, hand_count = (
        calculate_hand_score(image)
    )

    pose_score, pose_points = (
        calculate_pose_score(image)
    )

    face_score = calculate_face_score(
        image
    )

    sharpness = calculate_sharpness(
        image
    )

    # Normalize sharpness.
    sharpness_score = min(
        sharpness / 1000.0,
        1.0
    )

    # ========================================================
    # SIGN-AWARE WEIGHTS
    #
    # Hands      = 40%
    # Pose       = 25%
    # Sharpness  = 25%
    # Face       = 10%
    # ========================================================

    total_score = (
        0.40 * hand_score +
        0.25 * pose_score +
        0.25 * sharpness_score +
        0.10 * face_score
    )

    return {
        "total": total_score,
        "hands": hand_score,
        "hand_count": hand_count,
        "pose": pose_score,
        "pose_points": pose_points,
        "face": face_score,
        "sharpness": sharpness
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
# FIND COMMON SIGNER GROUPS
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


if not group_sets:

    print("❌ No signer groups found.")

    hand_detector.close()
    pose_detector.close()

    raise SystemExit


common_groups = set.intersection(
    *group_sets
)

print("\nCommon signer groups:")

for group in sorted(common_groups):

    print(
        f"  Group {group + 1}"
    )


# ============================================================
# SELECT COMMON SIGNER
# ============================================================

group_scores = {}

for group in common_groups:

    candidate_count = sum(
        len(word_groups[word][group])
        for word in TEST_WORDS
    )

    group_scores[group] = candidate_count


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
# SIGN-AWARE IMAGE SELECTION
# ============================================================

print("\n" + "=" * 70)
print("SIGN-AWARE REPRESENTATIVE SELECTION")
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
            f"No candidate image."
        )

        continue

    print(
        f"\nAnalyzing {word} "
        f"({len(candidates)} candidates)..."
    )

    scored = []

    for path in candidates:

        result = analyze_image(
            path
        )

        scored.append(
            (
                result["total"],
                path,
                result
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_score, best_path, best_result = (
        scored[0]
    )

    selected_images[word] = best_path

    print("\n  BEST IMAGE")

    print(
        f"  File        : "
        f"{os.path.basename(best_path)}"
    )

    print(
        f"  Total score : "
        f"{best_score:.4f}"
    )

    print(
        f"  Hands       : "
        f"{best_result['hand_count']} "
        f"→ {best_result['hands']:.3f}"
    )

    print(
        f"  Pose        : "
        f"{best_result['pose_points']}/7 "
        f"→ {best_result['pose']:.3f}"
    )

    print(
        f"  Sharpness   : "
        f"{best_result['sharpness']:.2f}"
    )

    print(
        f"  Face score  : "
        f"{best_result['face']:.3f}"
    )

    print("\n  TOP CANDIDATES:")

    for rank, (
        score,
        path,
        result
    ) in enumerate(
        scored[:5],
        start=1
    ):

        print(
            f"    {rank}. "
            f"{os.path.basename(path):25} "
            f"score={score:.4f} "
            f"hands={result['hand_count']} "
            f"pose={result['pose_points']}/7 "
            f"sharp={result['sharpness']:.0f}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL SIGN-AWARE REPRESENTATIVE SEQUENCE")
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


# ============================================================
# CLEANUP
# ============================================================

hand_detector.close()
pose_detector.close()