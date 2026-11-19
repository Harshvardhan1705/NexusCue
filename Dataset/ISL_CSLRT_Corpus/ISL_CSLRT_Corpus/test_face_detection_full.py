import os
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = (
    r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus"
    r"\ISL_CSLRT_Corpus\Frames_Word_Level"
)

TEST_SIGNS = [
    "YOU",
    "FREE",
    "TODAY",
    "HAPPY",
    "HOW"
]

MODEL_FILE = "blaze_face_short_range.tflite"


# ============================================================
# GET ALL IMAGES
# ============================================================

def get_images(sign):

    folder = os.path.join(BASE_PATH, sign)

    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        return []

    files = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            files.append(
                os.path.join(folder, filename)
            )

    files.sort()

    return files


# ============================================================
# CREATE FACE DETECTOR
# ============================================================

def create_detector():

    base_options = python.BaseOptions(
        model_asset_path=MODEL_FILE
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )

    detector = vision.FaceDetector.create_from_options(
        options
    )

    return detector


# ============================================================
# DETECT FACE
# ============================================================

def detect_face(detector, image_path):

    image = cv2.imread(image_path)

    if image is None:
        return 0

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    result = detector.detect(mp_image)

    return len(result.detections)


# ============================================================
# PROCESS ONE SIGN
# ============================================================

def process_sign(detector, sign):

    files = get_images(sign)

    if not files:
        return

    total = len(files)
    detected = 0
    missed = 0

    print("\n" + "=" * 60)
    print(f"SIGN: {sign}")
    print("=" * 60)

    for index, image_path in enumerate(files, start=1):

        filename = os.path.basename(image_path)

        face_count = detect_face(
            detector,
            image_path
        )

        if face_count > 0:
            detected += 1
        else:
            missed += 1

        # Progress every 10 images
        if index % 10 == 0 or index == total:

            print(
                f"Processed: {index:>4}/{total:<4} | "
                f"Faces detected: {detected:<4} | "
                f"Missed: {missed:<4}"
            )

    detection_rate = (
        detected / total * 100
        if total > 0
        else 0
    )

    print("\nRESULT")
    print("-" * 60)

    print(f"Total images       : {total}")
    print(f"Faces detected     : {detected}")
    print(f"Faces not detected : {missed}")
    print(f"Detection rate     : {detection_rate:.2f}%")

    return {
        "sign": sign,
        "total": total,
        "detected": detected,
        "missed": missed,
        "rate": detection_rate
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FULL DATASET FACE DETECTION TEST")
    print("=" * 60)

    if not os.path.exists(MODEL_FILE):

        print("\n❌ Face detector model not found:")
        print(MODEL_FILE)

        print(
            "\nMake sure "
            "blaze_face_short_range.tflite "
            "is in the same folder as this script."
        )

        return

    print("\nCreating MediaPipe face detector...")

    detector = create_detector()

    print("✅ MediaPipe face detector initialized.")

    results = []

    for sign in TEST_SIGNS:

        result = process_sign(
            detector,
            sign
        )

        if result:
            results.append(result)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"{'SIGN':<12}"
        f"{'TOTAL':<10}"
        f"{'DETECTED':<12}"
        f"{'MISSED':<10}"
        f"{'RATE':<10}"
    )

    print("-" * 70)

    total_images = 0
    total_detected = 0

    for result in results:

        print(
            f"{result['sign']:<12}"
            f"{result['total']:<10}"
            f"{result['detected']:<12}"
            f"{result['missed']:<10}"
            f"{result['rate']:.2f}%"
        )

        total_images += result["total"]
        total_detected += result["detected"]

    overall_rate = (
        total_detected / total_images * 100
        if total_images > 0
        else 0
    )

    print("-" * 70)

    print(
        f"{'TOTAL':<12}"
        f"{total_images:<10}"
        f"{total_detected:<12}"
        f"{total_images - total_detected:<10}"
        f"{overall_rate:.2f}%"
    )

    print("=" * 70)

    print("\n✅ FULL DATASET TEST COMPLETE")


if __name__ == "__main__":
    main()