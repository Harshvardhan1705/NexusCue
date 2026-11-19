import os
import cv2
import numpy as np
import mediapipe as mp
import json
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from gloss_engine import get_gloss
from sentence_sequence_engine import get_sentence_frames


# ============================================================
# CONFIGURATION
# ============================================================

WORD_DURATION = 2.0
LETTER_DURATION = 1.2
FPS = 10

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

OUTPUT_FILE = "sign_output.mp4"

# True only when the entered sentence has no exact sentence-level
# sequence and no exact sentence-to-gloss mapping in the dataset.
UNSUPPORTED_SENTENCE_MODE = False

WORD_ASSET_DIR = "Frames_Word_Level"
ALPHABET_DIR = r"..\..\ISL_Alphabet\Training"

CLUSTER_FILE = "signer_clusters.npz"

HAND_MODEL = r"mediapipe_models\hand_landmarker.task"
POSE_MODEL = r"mediapipe_models\pose_landmarker_lite.task"

FACE_MODEL = r"face_models\face_detection_yunet_2023mar.onnx"

ALPHABET_CACHE_FILE = "alphabet_cache.json"

# Load precomputed alphabet selections
with open(ALPHABET_CACHE_FILE, "r", encoding="utf-8") as file:
    ALPHABET_CACHE = json.load(file)

print(f"Loaded alphabet cache: {len(ALPHABET_CACHE)}/26 letters")


# ============================================================
# SIGNER CLUSTER DATA
# ============================================================

print("=" * 70)
print("NEXUSCUE - SPEECH TO INDIAN SIGN LANGUAGE")
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

    letter = letter.upper()

    # First, use the precomputed best image
    if letter in ALPHABET_CACHE:

        image_path = ALPHABET_CACHE[letter]["image"]

        if os.path.exists(image_path):
            return image_path

        print(f"[WARNING] Cached image missing for {letter}: {image_path}")

    print(f"[WARNING] No valid cached image for {letter}")

    return None

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

    global UNSUPPORTED_SENTENCE_MODE
    UNSUPPORTED_SENTENCE_MODE = False

    print()
    print("=" * 70)
    print("BUILDING SIGN SEQUENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Prefer an exact recovered sentence-level sequence
    # --------------------------------------------------------

    recovered_frames = get_sentence_frames(sentence)

    if recovered_frames:

        print()
        print("EXACT RECOVERED SENTENCE SEQUENCE FOUND")
        print(f"Sentence frames: {len(recovered_frames)}")
        print("Using original sequential frames at the renderer FPS.")

        return [
            {
                "type": "sentence_frame",
                "token": sentence,
                "path": str(frame_path)
            }
            for frame_path in recovered_frames
        ]

    print("No recovered sentence sequence; checking gloss database.")

    # --------------------------------------------------------
    # STEP 2: Try sentence-level ISL gloss
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
        # UNSUPPORTED SENTENCE
        #
        # Do not guess an ISL gloss or mix in isolated word
        # assets as if they formed a valid sentence translation.
        # Use alphabet images only and label the output clearly.
        # ----------------------------------------------------

        UNSUPPORTED_SENTENCE_MODE = True

        print()
        print("UNSUPPORTED SENTENCE: no exact dataset gloss found.")
        print("Using ENGLISH FINGERSPELLING as a spelling aid only.")
        print("IMPORTANT: This is NOT a verified ISL sentence translation.")

        words = sentence.split()

        for word in words:

            print()
            print(f"Fingerspelling word: {word}")

            fallback = fingerspell_word(word)
            sequence.extend(fallback)

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
# NEXUSCUE VIDEO PRESENTATION
# ============================================================

def create_nexcue_frame(image, token, sign_type, index, total, fallback_mode=False):
    """
    Compose one 640x480 NexusCue presentation frame.
    The original sign image is resized proportionally so it is
    not stretched or cropped.
    """

    # BGR palette: dark navy with restrained blue accents
    background = (58, 39, 27)  # RGB result: dark navy #1B273A
    header_background = (49, 31, 20)  # RGB result: #141F31
    image_panel = (81, 55, 37)  # RGB result: #253751
    white = (255, 255, 255)
    muted_blue = (225, 195, 170)  # BGR for soft blue
    accent_blue = (225, 165, 105)  # BGR for light blue
    border_blue = (148, 108, 76)  # BGR for blue-grey

    canvas = np.full(
        (FRAME_HEIGHT, FRAME_WIDTH, 3),
        background,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # HEADER PANEL
    # --------------------------------------------------------

    cv2.rectangle(
        canvas,
        (0, 0),
        (FRAME_WIDTH, 140),
        header_background,
        -1
    )

    # NQ monogram tile
    cv2.rectangle(
        canvas,
        (24, 19),
        (76, 69),
        (96, 63, 39),  # BGR for navy-blue tile
        -1
    )

    cv2.rectangle(
        canvas,
        (24, 19),
        (76, 69),
        accent_blue,
        1
    )

    cv2.putText(
        canvas,
        "NQ",
        (32, 52),
        cv2.FONT_HERSHEY_DUPLEX,
        0.78,
        white,
        2,
        cv2.LINE_AA
    )

    # Project name
    cv2.putText(
        canvas,
        "NexusCue",
        (90, 43),
        cv2.FONT_HERSHEY_DUPLEX,
        0.88,
        white,
        2,
        cv2.LINE_AA
    )

    # Small project identity
    cv2.putText(
        canvas,
        "INCLUSIVE COMMUNICATION",
        (92, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        muted_blue,
        1,
        cv2.LINE_AA
    )

    # AI-powered pill
    cv2.rectangle(
        canvas,
        (496, 23),
        (615, 53),
        (91, 63, 39),  # BGR for navy-blue badge
        -1
    )

    cv2.rectangle(
        canvas,
        (496, 23),
        (615, 53),
        border_blue,
        1
    )

    cv2.putText(
        canvas,
        "FALLBACK MODE" if fallback_mode else "AI POWERED",
        (507, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 220, 190),  # BGR for pale blue text
        1,
        cv2.LINE_AA
    )

    # Module subtitle
    cv2.putText(
        canvas,
        "English Fingerspelling Fallback" if fallback_mode else "Speech to Indian Sign Language",
        (25, 101),
        cv2.FONT_HERSHEY_DUPLEX,
        0.56,
        white,
        1,
        cv2.LINE_AA
    )

    # Supporting tagline
    cv2.putText(
        canvas,
        "Not a verified ISL sentence translation" if fallback_mode else "Bridging communication through intelligent sign representation",
        (26, 124),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        muted_blue,
        1,
        cv2.LINE_AA
    )

    # Header divider
    cv2.line(
        canvas,
        (24, 140),
        (FRAME_WIDTH - 24, 140),
        accent_blue,
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # SIGN IMAGE PANEL
    # --------------------------------------------------------

    panel_x = 30
    panel_y = 151
    panel_width = FRAME_WIDTH - 60
    panel_height = 238

    cv2.rectangle(
        canvas,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        image_panel,
        -1
    )

    cv2.rectangle(
        canvas,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        border_blue,
        1
    )

    image_height, image_width = image.shape[:2]

    if image_width > 0 and image_height > 0:

        # Fit image inside the panel while preserving aspect ratio
        scale = min(
            (panel_width - 16) / image_width,
            (panel_height - 16) / image_height
        )

        new_width = max(1, int(image_width * scale))
        new_height = max(1, int(image_height * scale))

        resized_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        x_offset = panel_x + (panel_width - new_width) // 2
        y_offset = panel_y + (panel_height - new_height) // 2

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized_image

    # --------------------------------------------------------
    # CURRENT SIGN AND CATEGORY
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "CURRENT SIGN",
        (25, 414),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        muted_blue,
        1,
        cv2.LINE_AA
    )

    token_text = str(token).upper()

    # Fit long words within the available width
    font_scale = 0.88
    max_text_width = FRAME_WIDTH - 50

    text_size = cv2.getTextSize(
        token_text,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        2
    )[0]

    while text_size[0] > max_text_width and font_scale > 0.35:
        font_scale -= 0.05
        text_size = cv2.getTextSize(
            token_text,
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            2
        )[0]

    cv2.putText(
        canvas,
        token_text,
        (25, 449),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        white,
        2,
        cv2.LINE_AA
    )

    if fallback_mode:
        category = "ENGLISH FINGERSPELLING"
    elif sign_type == "word":
        category = "WORD SIGN"
    elif sign_type == "sentence_frame":
        category = "SENTENCE SEQUENCE"
    else:
        category = "FINGERSPELLING"

    cv2.putText(
        canvas,
        category,
        (25, 474),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        muted_blue,
        1,
        cv2.LINE_AA
    )

    # Dynamic sequence progress, based on valid rendered items
    progress_text = f"{index:02d} / {total:02d}"

    progress_size = cv2.getTextSize(
        progress_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        2
    )[0]

    progress_x = FRAME_WIDTH - progress_size[0] - 25

    cv2.putText(
        canvas,
        progress_text,
        (progress_x, 474),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        white,
        2,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# CREATE NEXUSCUE VIDEO WITH SMOOTH TRANSITIONS
# ============================================================

def create_video(sequence):

    print()
    print("=" * 70)
    print("RENDERING NEXUSCUE SIGN VIDEO")
    print("=" * 70)

    if UNSUPPORTED_SENTENCE_MODE:
        print("MODE: ENGLISH FINGERSPELLING FALLBACK")
        print("This output is not a verified ISL sentence translation.")

    # Smooth transitions remain enabled for individual word/letter signs.
    # Sentence-level video frames stay continuous; fades occur only at
    # the beginning and end of the complete sentence sequence.
    fade_duration = 0.3
    pause_duration = 0.1

    valid_items = []

    for item in sequence:
        image = cv2.imread(item["path"])

        if image is None:
            print(f"[SKIPPED] Unable to read: {item['path']}")
            continue

        valid_items.append((item, image))

    total_items = len(valid_items)

    if total_items == 0:
        print("[ERROR] No valid sign images available for rendering.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_FILE,
        fourcc,
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    if not writer.isOpened():
        print("[ERROR] Could not open the video writer.")
        return

    total_frames = 0
    fade_frames = max(1, int(round(fade_duration * FPS)))
    pause_frames = max(0, int(round(pause_duration * FPS)))

    navy_background = np.full(
        (FRAME_HEIGHT, FRAME_WIDTH, 3),
        (58, 39, 27),
        dtype=np.uint8
    )

    def write_fade(source_frame, fade_in=True):
        nonlocal total_frames

        steps = range(1, fade_frames + 1) if fade_in else range(fade_frames - 1, -1, -1)

        for step in steps:
            alpha = step / fade_frames
            transition_frame = cv2.addWeighted(
                navy_background,
                1.0 - alpha,
                source_frame,
                alpha,
                0
            )
            writer.write(transition_frame)
            total_frames += 1

    def write_pause():
        nonlocal total_frames
        for _ in range(pause_frames):
            writer.write(navy_background)
            total_frames += 1

    try:
        for index, (item, image) in enumerate(valid_items, start=1):

            token = item["token"]
            sign_type = item["type"]

            frame = create_nexcue_frame(
                image,
                token,
                sign_type,
                index,
                total_items,
                fallback_mode=UNSUPPORTED_SENTENCE_MODE
            )

            if sign_type == "sentence_frame":
                # Keep every source frame in sequence at 10 FPS.
                # Do not fade between individual frames: that would
                # interrupt the signer's natural movement.
                previous_is_sentence = (
                    index > 1
                    and valid_items[index - 2][0]["type"] == "sentence_frame"
                )
                next_is_sentence = (
                    index < total_items
                    and valid_items[index][0]["type"] == "sentence_frame"
                )

                if not previous_is_sentence:
                    print(
                        f"\nSENTENCE SEQUENCE: {token} "
                        f"({total_items} frames)"
                    )
                    write_fade(frame, fade_in=True)

                writer.write(frame)
                total_frames += 1

                if not next_is_sentence:
                    write_fade(frame, fade_in=False)
                    write_pause()

            else:
                if sign_type == "word":
                    duration = WORD_DURATION
                    print(f"\nWORD SIGN: {token} ({index}/{total_items})")
                else:
                    duration = LETTER_DURATION
                    print(f"\nFINGERSPELLING: {token} ({index}/{total_items})")

                write_fade(frame, fade_in=True)

                frame_count = max(1, int(round(duration * FPS)))
                for _ in range(frame_count):
                    writer.write(frame)
                    total_frames += 1

                write_fade(frame, fade_in=False)
                write_pause()

    finally:
        writer.release()

    print()
    print("=" * 70)
    print("NEXUSCUE VIDEO CREATED")
    print("=" * 70)
    print(f"Output file: {os.path.abspath(OUTPUT_FILE)}")
    print(f"Total sequence items: {total_items}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {FPS}")
    print(f"Approx duration: {total_frames / FPS:.2f} seconds")
    print(f"Fade duration per side: {fade_duration:.2f} seconds")
    print(f"Neutral pause at transition boundaries: {pause_duration:.2f} seconds")

    if os.path.exists(OUTPUT_FILE):
        print(f"File size: {os.path.getsize(OUTPUT_FILE)} bytes")


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