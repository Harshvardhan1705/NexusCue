import os
import cv2
import urllib.request
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Word_Level"

OUTPUT_DIR = "face_detection_test"

TEST_SIGNS = [
    "YOU",
    "FREE",
    "TODAY",
    "HAPPY",
    "HOW"
]

SAMPLES_PER_SIGN = 5

MODEL_FILE = "blaze_face_short_range.tflite"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def download_model():

    if os.path.exists(MODEL_FILE):

        print("✅ Face detector model already exists.")

        return

    print("\nDownloading MediaPipe face detector model...")

    try:

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_FILE
        )

        print("✅ Face detector model downloaded.")

    except Exception as e:

        print("\n❌ Could not download model.")
        print("Error:", e)

        raise


# ============================================================
# GET IMAGES
# ============================================================

def get_images(sign):

    folder = os.path.join(
        BASE_PATH,
        sign
    )

    if not os.path.exists(folder):

        print(f"⚠️ Folder not found: {folder}")

        return []

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    files.sort()

    return [
        os.path.join(folder, f)
        for f in files
    ]


# ============================================================
# SELECT SAMPLE IMAGES
# ============================================================

def select_samples(files):

    if len(files) <= SAMPLES_PER_SIGN:

        return files

    step = len(files) // SAMPLES_PER_SIGN

    return [
        files[i * step]
        for i in range(SAMPLES_PER_SIGN)
    ]


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

        print("❌ Could not read:", image_path)

        return None, 0

    rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    result = detector.detect(mp_image)

    face_count = len(result.detections)

    for detection in result.detections:

        bbox = detection.bounding_box

        x = max(0, bbox.origin_x)
        y = max(0, bbox.origin_y)

        w = bbox.width
        h = bbox.height

        x2 = min(
            image.shape[1],
            x + w
        )

        y2 = min(
            image.shape[0],
            y + h
        )

        cv2.rectangle(
            image,
            (x, y),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return image, face_count


# ============================================================
# PROCESS SIGN
# ============================================================

def process_sign(detector, sign):

    files = get_images(sign)

    if not files:

        return

    samples = select_samples(files)

    sign_output_dir = os.path.join(
        OUTPUT_DIR,
        sign
    )

    os.makedirs(
        sign_output_dir,
        exist_ok=True
    )

    print("\n" + "-" * 60)
    print(sign)
    print("-" * 60)

    detected = 0

    for index, image_path in enumerate(samples):

        output_image, face_count = detect_face(
            detector,
            image_path
        )

        filename = os.path.basename(image_path)

        if output_image is None:

            continue

        output_path = os.path.join(
            sign_output_dir,
            f"sample_{index + 1}.jpg"
        )

        cv2.imwrite(
            output_path,
            output_image
        )

        if face_count > 0:

            detected += 1

            print(
                f"{filename:<35} → "
                f"{face_count} face(s) ✅"
            )

        else:

            print(
                f"{filename:<35} → "
                f"0 faces ❌"
            )

    print(
        f"\nDetected faces: "
        f"{detected}/{len(samples)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MEDIAPIPE FACE DETECTION TEST")
    print("=" * 60)

    download_model()

    detector = create_detector()

    print("\n✅ MediaPipe face detector initialized.")

    for sign in TEST_SIGNS:

        process_sign(
            detector,
            sign
        )

    print("\n" + "=" * 60)
    print("✅ FACE DETECTION TEST COMPLETE")
    print("=" * 60)

    print("\nOutput folder:")
    print(
        os.path.abspath(OUTPUT_DIR)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()