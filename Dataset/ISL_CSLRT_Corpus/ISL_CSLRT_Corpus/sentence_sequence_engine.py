
from pathlib import Path
import csv
import re

from gloss_engine import normalize_text

# ============================================================
# NEXUSCUE - AUTOMATIC SENTENCE SEQUENCE ENGINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = (
    BASE_DIR
    / "bulk_sentence_sequences"
    / "bulk_sequence_manifest.csv"
)

RECOVERED_ROOT = BASE_DIR / "recovered_sentence_frames_10fps"

# Original dataset sentence-frame directory
DATASET_FRAMES_ROOT = BASE_DIR / "Frames_Sentence_Level"

# Prefer the same dataset group across sentences.
# These are dataset group labels, not verified identities.
PREFERRED_GROUP = "1"


# ------------------------------------------------------------
# 1. RECOVERED SENTENCE SEQUENCES
# ------------------------------------------------------------

RECOVERED_SENTENCES = {
    normalize_text("he would be coming today"):
        "23_he_would_be_coming_today",

    normalize_text("i dont agree"):
        "50_i_dont_agree",

    normalize_text("No need to worry dont worry"):
        "64_No_need_to_worry_dont_worry",

    normalize_text("now onwards he will never hurt you"):
        "65_now_onwards_he_will_never_hurt_you"
}


# ------------------------------------------------------------
# 2. ALTERNATE ORIGINAL DATASET FOLDER NAMES
# ------------------------------------------------------------

DATASET_SENTENCE_FOLDERS = {
    normalize_text("which college school are you from"):
        "which collegeschool are you from"
}


# ------------------------------------------------------------
# 3. NATURAL SORTING
# ------------------------------------------------------------

def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_image_frames(folder, recursive=False):
    """Return naturally sorted image paths from a folder."""

    if not folder.exists() or not folder.is_dir():
        return []

    extensions = {".jpg", ".jpeg", ".png"}

    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.iterdir()

    frames = [
        path for path in candidates
        if path.is_file() and path.suffix.lower() in extensions
    ]

    return sorted(frames, key=natural_sort_key)


# ------------------------------------------------------------
# 4. LOAD BULK MANIFEST
# ------------------------------------------------------------

def load_manifest():
    """Load available sentence sequences from the CSV manifest."""

    sequences = {}

    if not MANIFEST_PATH.exists():
        print(f"WARNING: Manifest not found: {MANIFEST_PATH}")
        return sequences

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            sentence = normalize_text(row.get("sentence", ""))
            group = str(row.get("signer_group", "")).strip()
            folder_value = str(row.get("sequence_folder", "")).strip()

            if not sentence or not group or not folder_value:
                continue

            folder = Path(folder_value)

            if not folder.is_absolute():
                folder = BASE_DIR / folder

            if not folder.exists():
                continue

            sequences.setdefault(sentence, {})
            sequences[sentence][group] = folder

    return sequences


BULK_SEQUENCES = load_manifest()


# ------------------------------------------------------------
# 5. SELECT BULK SEQUENCE
# ------------------------------------------------------------

def get_bulk_sequence_folder(sentence):
    """Choose one available group for an exact sentence."""

    normalized = normalize_text(sentence)

    groups = BULK_SEQUENCES.get(normalized)

    if not groups:
        return None

    if PREFERRED_GROUP in groups:
        return groups[PREFERRED_GROUP]

    def group_key(group):
        match = re.search(r"\d+", group)
        return int(match.group()) if match else 999999

    selected_group = sorted(
        groups.keys(),
        key=group_key
    )[0]

    return groups[selected_group]


# ------------------------------------------------------------
# 6. ORIGINAL DATASET FALLBACK
# ------------------------------------------------------------

def get_original_dataset_frames(sentence):
    """
    Find sentence frames in the original dataset using
    explicitly configured alternate folder names.
    """

    normalized = normalize_text(sentence)

    folder_name = DATASET_SENTENCE_FOLDERS.get(normalized)

    if not folder_name:
        return []

    folder = DATASET_FRAMES_ROOT / folder_name

    return get_image_frames(folder, recursive=True)


# ------------------------------------------------------------
# 7. AUTOMATIC SENTENCE FRAME RETRIEVAL
# ------------------------------------------------------------

def get_sentence_frames(sentence):
    """
    Return ordered image paths for a sentence.

    Search priority:
    1. Bulk-generated sentence sequences
    2. Previously recovered sentence sequences
    3. Original dataset alternate-name folders
    """

    if not isinstance(sentence, str) or not sentence.strip():
        return []

    normalized = normalize_text(sentence)

    # --------------------------------------------------------
    # PRIORITY 1: BULK-GENERATED SEQUENCES
    # --------------------------------------------------------

    bulk_folder = get_bulk_sequence_folder(sentence)

    if bulk_folder is not None:

        frames = get_image_frames(bulk_folder)

        if frames:
            return frames

    # --------------------------------------------------------
    # PRIORITY 2: RECOVERED SEQUENCES
    # --------------------------------------------------------

    recovered_folder_name = RECOVERED_SENTENCES.get(normalized)

    if recovered_folder_name:

        recovered_folder = RECOVERED_ROOT / recovered_folder_name

        frames = get_image_frames(recovered_folder)

        if frames:
            return frames

    # --------------------------------------------------------
    # PRIORITY 3: ORIGINAL DATASET ALTERNATE FOLDER
    # --------------------------------------------------------

    original_frames = get_original_dataset_frames(sentence)

    if original_frames:
        return original_frames

    return []


# ------------------------------------------------------------
# 8. CHECK SEQUENCE AVAILABILITY
# ------------------------------------------------------------

def has_sentence_sequence(sentence):
    """Check whether a sentence sequence is available."""

    return len(get_sentence_frames(sentence)) > 0


# ------------------------------------------------------------
# 9. SENTENCE SEQUENCE INFORMATION
# ------------------------------------------------------------

def get_sentence_sequence_info(sentence):
    """Return information about the selected sentence sequence."""

    frames = get_sentence_frames(sentence)

    if not frames:
        return None

    return {
        "sentence": sentence,
        "frame_count": len(frames),
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "frames": frames
    }


# ------------------------------------------------------------
# 10. MAIN TEST
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("NEXUSCUE - AUTOMATIC SENTENCE SEQUENCE ENGINE")
    print("=" * 70)

    print(f"\nBulk manifest loaded: {len(BULK_SEQUENCES)} sentences")
    print(f"Preferred signer group: {PREFERRED_GROUP}")

    test_sentences = [
        "are you free today",
        "how are things",
        "it was nice chatting with you",
        "what is your phone number",
        "he would be coming today",
        "i dont agree",
        "No need to worry dont worry",
        "now onwards he will never hurt you",
        "which college school are you from"
    ]

    print("\nAUTOMATIC SEQUENCE TESTS\n")

    for sentence in test_sentences:

        info = get_sentence_sequence_info(sentence)

        print(f"Sentence: {sentence}")

        if info:
            print("Status: SEQUENCE FOUND")
            print(f"Frames: {info['frame_count']}")
            print(f"First frame: {info['first_frame'].name}")
            print(f"Last frame: {info['last_frame'].name}")
            print(f"Folder: {info['first_frame'].parent}")
        else:
            print("Status: SEQUENCE NOT FOUND")

        print("-" * 70)

    print("\nCUSTOM SENTENCE TEST")

    user_sentence = input(
        "Enter a sentence to check (or press Enter to exit): "
    ).strip()

    if user_sentence:

        info = get_sentence_sequence_info(user_sentence)

        if info:
            print("\nEXACT SENTENCE SEQUENCE FOUND")
            print(f"Total frames: {info['frame_count']}")
            print(f"Frame folder: {info['first_frame'].parent}")
        else:
            print("\nNO SENTENCE SEQUENCE FOUND")
            print("The existing word-sign pipeline can be used as fallback.")


if __name__ == "__main__":
    main()
