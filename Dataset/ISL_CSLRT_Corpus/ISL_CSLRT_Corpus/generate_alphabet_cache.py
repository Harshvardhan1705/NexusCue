
import os
import cv2
import json
import mediapipe as mp

# ==========================================
# CONFIGURATION
# ==========================================

ALPHABET_DIR = r"..\..\ISL_Alphabet\Training"

HAND_MODEL = r"mediapipe_models\hand_landmarker.task"
POSE_MODEL = r"mediapipe_models\pose_landmarker_lite.task"

CACHE_FILE = "alphabet_cache.json"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ==========================================
# INITIALIZE MEDIAPIPE TASKS
# ==========================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=VisionRunningMode.IMAGE
)

hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(
    hand_options
)

pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(
    pose_options
)


# ==========================================
# IMAGE QUALITY ANALYSIS
# ==========================================

def analyze_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return 0.0

    height, width = image.shape[:2]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Hand detection
    hand_result = hand_detector.detect(mp_image)

    hand_count = len(hand_result.hand_landmarks)

    hand_score = min(hand_count / 2, 1.0)

    # Pose detection
    pose_result = pose_detector.detect(mp_image)

    pose_score = 0.0

    if pose_result.pose_landmarks:
        visibility_values = [
            landmark.visibility
            for landmark in pose_result.pose_landmarks[0]
        ]

        if visibility_values:
            pose_score = sum(visibility_values) / len(visibility_values)

    # Image sharpness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    sharpness_score = min(sharpness / 500, 1.0)

    # Alphabet image quality score
    final_score = (
        0.50 * hand_score +
        0.20 * pose_score +
        0.30 * sharpness_score
    )

    return round(final_score, 4)


# ==========================================
# FIND BEST IMAGE FOR EACH LETTER
# ==========================================

def generate_cache():

    cache = {}

    print("\nStarting alphabet image analysis...\n")

    for letter in LETTERS:

        letter_folder = os.path.join(
            ALPHABET_DIR,
            letter
        )

        if not os.path.isdir(letter_folder):
            print(f"[MISSING] Folder not found: {letter}")
            continue

        image_files = [
            file for file in os.listdir(letter_folder)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ]

        if not image_files:
            print(f"[EMPTY] No images found for: {letter}")
            continue

        best_score = -1
        best_image = None

        print(
            f"Analyzing {letter}: "
            f"{len(image_files)} images"
        )

        for filename in image_files:

            image_path = os.path.join(
                letter_folder,
                filename
            )

            score = analyze_image(image_path)

            if score > best_score:
                best_score = score
                best_image = filename

        if best_image is not None:

            relative_path = os.path.join(
                letter_folder,
                best_image
            )

            cache[letter] = {
                "image": relative_path,
                "score": best_score
            }

            print(
                f"[SELECTED] {letter} -> {best_image} "
                f"(Score: {best_score})"
            )

    # Save persistent cache
    with open(CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump(
            cache,
            file,
            indent=4
        )

    print("\n================================")
    print("ALPHABET CACHE GENERATION DONE")
    print("================================")
    print(f"Letters cached: {len(cache)}/26")
    print(f"Saved to: {CACHE_FILE}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    try:
        generate_cache()

    finally:
        hand_detector.close()
        pose_detector.close()
