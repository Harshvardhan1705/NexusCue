import os
import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

ALPHABET_DIR = r"D:\mini project hhai\nexus\Dataset\ISL_Alphabet"

TRAINING_DIR = os.path.join(
    ALPHABET_DIR,
    "Training"
)

TESTING_DIR = os.path.join(
    ALPHABET_DIR,
    "Testing"
)

VALIDATION_DIR = os.path.join(
    ALPHABET_DIR,
    "Validation"
)


# ============================================================
# MEDIA PIPE TASK MODELS
# ============================================================

HAND_MODEL = (
    r"D:\mini project hhai\nexus\dataset"
    r"\isl_cslrt_corpus\isl_cslrt_corpus"
    r"\mediapipe_models\hand_landmarker.task"
)

POSE_MODEL = (
    r"D:\mini project hhai\nexus\dataset"
    r"\isl_cslrt_corpus\isl_cslrt_corpus"
    r"\mediapipe_models\pose_landmarker_lite.task"
)

FACE_MODEL = (
    r"D:\mini project hhai\nexus\dataset"
    r"\isl_cslrt_corpus\isl_cslrt_corpus"
    r"\face_models\face_detection_yunet_2023mar.onnx"
)


# ============================================================
# INITIALIZE MEDIAPIPE
# ============================================================

print("=" * 70)
print("QUALITY-AWARE ISL ALPHABET FALLBACK")
print("=" * 70)

print("\nLoading MediaPipe models...")

BaseOptions = mp.tasks.BaseOptions

RunningMode = (
    mp.tasks.vision.RunningMode
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)


# ------------------------------------------------------------
# Hand detector
# ------------------------------------------------------------

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30,
    min_tracking_confidence=0.30
)

hand_detector = (
    HandLandmarker.create_from_options(
        hand_options
    )
)


# ------------------------------------------------------------
# Pose detector
# ------------------------------------------------------------

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL
    ),
    running_mode=RunningMode.IMAGE,
    min_pose_detection_confidence=0.30,
    min_pose_presence_confidence=0.30
)

pose_detector = (
    PoseLandmarker.create_from_options(
        pose_options
    )
)


# ------------------------------------------------------------
# Face detector
# ------------------------------------------------------------

face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("Models loaded.")


# ============================================================
# CREATE MEDIAPIPE IMAGE
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
# SHARPNESS
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
# HAND SCORE
# ============================================================

def calculate_hand_score(image):

    mp_image = make_mp_image(
        image
    )

    result = hand_detector.detect(
        mp_image
    )

    if not result.hand_landmarks:
        return 0.0, 0

    hand_count = len(
        result.hand_landmarks
    )

    score = min(
        hand_count / 2.0,
        1.0
    )

    return score, hand_count


# ============================================================
# POSE SCORE
# ============================================================

def calculate_pose_score(image):

    mp_image = make_mp_image(
        image
    )

    result = pose_detector.detect(
        mp_image
    )

    if not result.pose_landmarks:
        return 0.0, 0

    landmarks = result.pose_landmarks[0]

    important_indices = [
        0,   # Nose
        11,  # Left shoulder
        12,  # Right shoulder
        13,  # Left elbow
        14,  # Right elbow
        15,  # Left wrist
        16   # Right wrist
    ]

    visible = 0

    for index in important_indices:

        if landmarks[index].visibility >= 0.30:
            visible += 1

    score = (
        visible /
        len(important_indices)
    )

    return score, visible


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

    return min(
        ratio / 0.10,
        1.0
    )


# ============================================================
# SCORE ALPHABET IMAGE
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
            "sharpness": 0,
            "face": 0
        }

    hand_score, hand_count = (
        calculate_hand_score(image)
    )

    pose_score, pose_points = (
        calculate_pose_score(image)
    )

    sharpness = calculate_sharpness(
        image
    )

    sharpness_score = min(
        sharpness / 1000.0,
        1.0
    )

    face_score = calculate_face_score(
        image
    )

    # ========================================================
    # ALPHABET IMAGE QUALITY
    #
    # Hands      = 50%
    # Pose       = 20%
    # Sharpness  = 25%
    # Face       = 5%
    # ========================================================

    total_score = (
        0.50 * hand_score +
        0.20 * pose_score +
        0.25 * sharpness_score +
        0.05 * face_score
    )

    return {
        "total": total_score,
        "hands": hand_score,
        "hand_count": hand_count,
        "pose": pose_score,
        "pose_points": pose_points,
        "sharpness": sharpness,
        "face": face_score
    }


# ============================================================
# FIND LETTER FOLDER
# ============================================================

def find_letter_folder(letter):

    letter = letter.upper()

    # Prefer Training
    roots = [
        TRAINING_DIR,
        TESTING_DIR,
        VALIDATION_DIR
    ]

    for root in roots:

        folder = os.path.join(
            root,
            letter
        )

        if os.path.isdir(folder):
            return folder

    return None


# ============================================================
# GET LETTER IMAGES
# ============================================================

def get_letter_images(letter):

    folder = find_letter_folder(
        letter
    )

    if folder is None:
        return []

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    images = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            extensions
        ):

            images.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    return sorted(images)


# ============================================================
# SELECT BEST LETTER IMAGE
# ============================================================

def select_letter_image(letter):

    images = get_letter_images(
        letter
    )

    if not images:
        return None

    scored = []

    print(
        f"\nAnalyzing letter {letter} "
        f"({len(images)} images)..."
    )

    for image_path in images:

        result = analyze_image(
            image_path
        )

        scored.append(
            (
                result["total"],
                image_path,
                result
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_score, best_path, best_result = (
        scored[0]
    )

    print(
        f"  Selected: "
        f"{os.path.basename(best_path)}"
    )

    print(
        f"  Score: "
        f"{best_score:.4f}"
    )

    print(
        f"  Hands: "
        f"{best_result['hand_count']}"
    )

    print(
        f"  Pose: "
        f"{best_result['pose_points']}/7"
    )

    print(
        f"  Sharpness: "
        f"{best_result['sharpness']:.2f}"
    )

    return best_path


# ============================================================
# FINGERSPELL WORD
# ============================================================

def fingerspell_word(word):

    word = word.upper().strip()

    sequence = []

    for character in word:

        if not character.isalpha():
            continue

        image = select_letter_image(
            character
        )

        sequence.append({
            "letter": character,
            "image": image
        })

    return sequence


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_word = "VERY"

    print(
        f"\nFingerspelling: {test_word}"
    )

    result = fingerspell_word(
        test_word
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL ALPHABET SEQUENCE"
    )

    print(
        "=" * 70
    )

    found = 0

    for item in result:

        letter = item["letter"]
        image = item["image"]

        if image:

            found += 1

            print(
                f"{letter} → "
                f"{os.path.basename(image)}"
            )

        else:

            print(
                f"{letter} → ❌ MISSING"
            )

    print(
        f"\nLetters found: "
        f"{found}/{len(result)}"
    )

    print(
        "=" * 70
    )


# ============================================================
# CLEANUP
# ============================================================

hand_detector.close()
pose_detector.close()