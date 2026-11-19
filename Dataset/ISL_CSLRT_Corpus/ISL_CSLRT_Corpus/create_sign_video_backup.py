import os
import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from gloss_engine import get_gloss


# ============================================================
# CONFIGURATION
# ============================================================

WORD_DURATION = 2.0
LETTER_DURATION = 1.2
FPS = 10

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

OUTPUT_FILE = "sign_output.mp4"

WORD_ASSET_DIR = "Frames_Word_Level"
ALPHABET_DIR = r"..\..\ISL_Alphabet\Training"

CLUSTER_FILE = "signer_clusters.npz"

HAND_MODEL = r"mediapipe_models\hand_landmarker.task"
POSE_MODEL = r"mediapipe_models\pose_landmarker_lite.task"

FACE_MODEL = r"face_models\face_detection_yunet_2023mar.onnx"


# ============================================================
# SIGNER CLUSTER DATA
# ============================================================

print("=" * 70)
print("SIGNBRIDGE - SPEECH/TEXT TO SIGN + A-Z FALLBACK")
print("=" * 70)

print("\nLoading signer cluster data...")

cluster_data = np.load(CLUSTER_FILE, allow_pickle=True)

cluster_features = cluster_data["features"]
cluster_labels = cluster_data["labels"]
cluster_words = cluster_data["words"]
cluster_filenames = cluster_data["filenames"]
cluster_paths = cluster_data["paths"]

print(f"Loaded {len(cluster_paths)} clustered images.")


# ============================================================
# MEDIA PIPE
# ============================================================

print("\nLoading MediaPipe models...")

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=VisionRunningMode.IMAGE
)

hand_landmarker = HandLandmarker.create_from_options(hand_options)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)

face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("MediaPipe models loaded.")


# ============================================================
# BUILD WORD → GROUP → IMAGES INDEX
# ============================================================

word_groups = defaultdict(lambda: defaultdict(list))

for path, word, label in zip(
    cluster_paths,
    cluster_words,
    cluster_labels
):
    word_groups[str(word).upper()][int(label)].append(path)


# ============================================================
# ALIASES
# ============================================================

ALIASES = {
    "HI": "HELLO_HI",
    "HELLO": "HELLO_HI",
    "LOVE": "LIKE_LOVE",
    "COLLEGE": "COLLEGE_SCHOOL",
    "SCHOOL": "COLLEGE_SCHOOL",
    "I": "I_ME_MINE_MY",
    "ME": "I_ME_MINE_MY",
    "MY": "I_ME_MINE_MY",
    "AGE": "OLD_AGE",
    "SOMEONE": "SOME ONE",
    "SOMEHOW": "SOME HOW",
    "MEAN": "MEAN IT",
    "TAKE_CARE": "TAKE CARE",
    "ON_THE_WAY": "ON THE WAY",
    "SO_MUCH": "SO MUCH",
}


def normalize_token(token):
    token = token.upper().strip()

    if token in ALIASES:
        return ALIASES[token]

    return token


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(path):

    image = cv2.imread(path)

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # ---------------- HANDS ----------------

    hand_result = hand_landmarker.detect(mp_image)

    hand_count = len(hand_result.hand_landmarks)

    hand_score = min(hand_count / 2.0, 1.0)

    # ---------------- POSE ----------------

    pose_result = pose_landmarker.detect(mp_image)

    pose_score = 0.0

    if pose_result.pose_landmarks:

        landmarks = pose_result.pose_landmarks[0]

        visible = 0

        for landmark in landmarks:

            if landmark.visibility > 0.5:
                visible += 1

        pose_score = min(visible / 7.0, 1.0)

    # ---------------- SHARPNESS ----------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    sharpness_score = min(
        sharpness / 1000.0,
        1.0
    )

    # ---------------- FACE ----------------

    height, width = image.shape[:2]

    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(image)

    face_score = 0.0

    if faces is not None and len(faces) > 0:

        largest = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        face_area = largest[2] * largest[3]

        image_area = width * height

        face_score = min(
            face_area / image_area * 10,
            1.0
        )

    # ========================================================
    # SIGN-AWARE SCORE
    # ========================================================

    score = (
        0.40 * hand_score +
        0.25 * pose_score +
        0.25 * sharpness_score +
        0.10 * face_score
    )

    return {
        "score": score,
        "hands": hand_count,
        "pose": pose_score,
        "sharpness": sharpness
    }


# ============================================================
# SELECT COMMON SIGNER GROUP
# ============================================================

def choose_signer_group(tokens):

    available_groups = []

    for token in tokens:

        normalized = normalize_token(token)

        if normalized not in word_groups:
            continue

        groups = set(
            word_groups[normalized].keys()
        )

        available_groups.append(groups)

    if not available_groups:
        return None

    common = set.intersection(*available_groups)

    if common:

        # Select group with maximum total number of examples
        best_group = max(
            common,
            key=lambda group: sum(
                len(word_groups[normalize_token(token)].get(group, []))
                for token in tokens
                if normalize_token(token) in word_groups
            )
        )

        return best_group

    return None


# ============================================================
# SELECT BEST WORD IMAGE
# ============================================================

def select_word_image(token, signer_group):

    normalized = normalize_token(token)

    if normalized not in word_groups:
        return None

    candidates = []

    if signer_group is not None:

        candidates = word_groups[
            normalized
        ].get(signer_group, [])

    # If no common signer exists, use all available images
    if not candidates:

        for group_images in word_groups[normalized].values():
            candidates.extend(group_images)

    if not candidates:
        return None

    best_path = None
    best_score = -1
    best_info = None

    for path in candidates:

        info = analyze_image(path)

        if info is None:
            continue

        if info["score"] > best_score:

            best_score = info["score"]
            best_path = path
            best_info = info

    if best_path:

        print(f"  Selected: {os.path.basename(best_path)}")
        print(f"  Score: {best_score:.4f}")
        print(f"  Hands: {best_info['hands']}")
        print(
            f"  Pose: "
            f"{best_info['pose']:.0f}/7"
        )
        print(
            f"  Sharpness: "
            f"{best_info['sharpness']:.2f}"
        )

    return best_path


# ============================================================
# ALPHABET DATASET
# ============================================================

alphabet_cache = {}


def get_letter_images(letter):

    letter = letter.upper()

    if letter in alphabet_cache:
        return alphabet_cache[letter]

    folder = os.path.join(
        ALPHABET_DIR,
        letter
    )

    if not os.path.exists(folder):
        alphabet_cache[letter] = []
        return []

    images = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            images.append(
                os.path.join(folder, filename)
            )

    alphabet_cache[letter] = images

    return images


# ============================================================
# SELECT BEST ALPHABET IMAGE
# ============================================================

def select_alphabet_image(letter):

    candidates = get_letter_images(letter)

    if not candidates:
        return None

    best_path = None
    best_score = -1

    for path in candidates:

        info = analyze_image(path)

        if info is None:
            continue

        # Alphabet-specific weighting
        score = (
            0.50 * min(info["hands"] / 2.0, 1.0) +
            0.20 * info["pose"] +
            0.25 * min(
                info["sharpness"] / 1000.0,
                1.0
            ) +
            0.05 * 0
        )

        if score > best_score:

            best_score = score
            best_path = path

    if best_path:

        print(
            f"    {letter} → "
            f"{os.path.basename(best_path)} "
            f"(score {best_score:.4f})"
        )

    return best_path


# ============================================================
# ALPHABET FALLBACK
# ============================================================

def fingerspell_word(word):

    word = word.upper().strip()

    print()
    print(f"🔤 Alphabet fallback: {word}")

    sequence = []

    for letter in word:

        if not letter.isalpha():
            continue

        path = select_alphabet_image(letter)

        if path:

            sequence.append({
                "type": "letter",
                "token": letter,
                "path": path
            })

        else:

            print(
                f"    ⚠️ No alphabet image "
                f"found for {letter}"
            )

    return sequence


# ============================================================
# BUILD SIGN SEQUENCE
# ============================================================

def build_sequence(sentence):

    print()
    print("=" * 70)
    print("BUILDING SIGN SEQUENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Try sentence-level ISL gloss
    # --------------------------------------------------------

    gloss = get_gloss(sentence)

    sequence = []

    if gloss:

        print()
        print(f"ISL Gloss: {gloss}")

        tokens = gloss.split()

        print(
            f"Gloss tokens: {tokens}"
        )

        # ----------------------------------------------------
        # STEP 2: Select common signer
        # ----------------------------------------------------

        signer_group = choose_signer_group(tokens)

        if signer_group is not None:

            print()
            print(
                f"🎯 Selected signer group: "
                f"Group {signer_group}"
            )

        else:

            print()
            print(
                "⚠️ No common signer group "
                "for all signs."
            )

        # ----------------------------------------------------
        # STEP 3: Process each gloss token
        # ----------------------------------------------------

        for token in tokens:

            print()
            print(f"Processing: {token}")

            normalized = normalize_token(token)

            path = select_word_image(
                token,
                signer_group
            )

            if path:

                sequence.append({
                    "type": "word",
                    "token": token,
                    "path": path
                })

            else:

                print(
                    f"  ⚠️ No ISL sign asset "
                    f"for {token}"
                )

                fallback = fingerspell_word(
                    normalized
                )

                sequence.extend(
                    fallback
                )

    else:

        # ----------------------------------------------------
        # SENTENCE NOT FOUND
        #
        # We cannot invent an ISL gloss.
        # Instead, use direct word fallback.
        # ----------------------------------------------------

        print()
        print(
            "⚠️ Sentence not found in "
            "ISL gloss database."
        )

        print(
            "Using word-level fallback."
        )

        words = sentence.upper().split()

        known_words = []
        missing_words = []

        for word in words:

            normalized = normalize_token(word)

            if normalized in word_groups:

                known_words.append(
                    (word, normalized)
                )

            else:

                missing_words.append(word)

        # ----------------------------------------------------
        # Try to maintain same signer for known words
        # ----------------------------------------------------

        signer_tokens = [
            normalized
            for _, normalized in known_words
        ]

        signer_group = None

        if signer_tokens:

            signer_group = choose_signer_group(
                signer_tokens
            )

            if signer_group is not None:

                print()
                print(
                    f"🎯 Selected signer group: "
                    f"Group {signer_group}"
                )

        # ----------------------------------------------------
        # Process original sentence order
        # ----------------------------------------------------

        for word in words:

            normalized = normalize_token(word)

            print()
            print(f"Processing: {word}")

            if normalized in word_groups:

                path = select_word_image(
                    word,
                    signer_group
                )

                if path:

                    sequence.append({
                        "type": "word",
                        "token": word,
                        "path": path
                    })

            else:

                print(
                    f"  ⚠️ No ISL sign asset "
                    f"for {word}"
                )

                fallback = fingerspell_word(
                    word
                )

                sequence.extend(
                    fallback
                )

    return sequence


# ============================================================
# PREPARE FRAME
# ============================================================

def prepare_frame(image):

    return cv2.resize(
        image,
        (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# CREATE VIDEO
# ============================================================

def create_video(sequence):

    print()
    print("=" * 70)
    print("RENDERING SLOW SIGN VIDEO")
    print("=" * 70)

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_FILE,
        fourcc,
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    total_frames = 0

    for item in sequence:

        path = item["path"]

        image = cv2.imread(path)

        if image is None:
            continue

        frame = prepare_frame(image)

        if item["type"] == "word":

            duration = WORD_DURATION

            print()
            print(
                f"SIGN: {item['token']}"
            )

        else:

            duration = LETTER_DURATION

            print()
            print(
                f"LETTER: {item['token']}"
            )

        frame_count = int(
            duration * FPS
        )

        for _ in range(frame_count):

            writer.write(frame)

            total_frames += 1

    writer.release()

    print()
    print("=" * 70)
    print("VIDEO CREATED")
    print("=" * 70)

    print(
        f"Output file: "
        f"{os.path.abspath(OUTPUT_FILE)}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"FPS: {FPS}"
    )

    print(
        f"Approx duration: "
        f"{total_frames / FPS:.2f} seconds"
    )

    if os.path.exists(OUTPUT_FILE):

        print(
            f"File size: "
            f"{os.path.getsize(OUTPUT_FILE)} bytes"
        )


# ============================================================
# MAIN
# ============================================================

try:

    sentence = input(
        "\nEnter sentence: "
    ).strip()

    if not sentence:

        print("No sentence entered.")
        exit()

    sequence = build_sequence(
        sentence
    )

    if not sequence:

        print()
        print(
            "❌ No sign sequence could "
            "be generated."
        )

    else:

        print()
        print("=" * 70)
        print("FINAL SEQUENCE")
        print("=" * 70)

        for item in sequence:

            print(
                f"{item['type'].upper():7} "
                f"{item['token']:8} → "
                f"{os.path.basename(item['path'])}"
            )

        create_video(sequence)

finally:

    hand_landmarker.close()
    pose_landmarker.close()

    print()
    print("Renderer closed.")