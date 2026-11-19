
import csv
import json
import re
from pathlib import Path

from gloss_engine import normalize_text


# =========================================================
# NEXUSCUE - BULK SENTENCE SEQUENCE INDEX BUILDER
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "corpus_csv_files"
    / "ISL Corpus sign glosses.csv"
)

FRAME_ROOT = BASE_DIR / "Frames_Sentence_Level"

OUTPUT_PATH = BASE_DIR / "sentence_sequence_index.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def natural_sort_key(path):
    """Sort frame names naturally: 1, 2, 10 instead of 1, 10, 2."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_sentences():
    """Read the original English sentences from the gloss CSV."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Gloss CSV not found: {CSV_PATH}"
        )

    sentences = []

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {"Sentence"}

        if not reader.fieldnames or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                "CSV must contain a 'Sentence' column."
            )

        for row in reader:
            sentence = (row.get("Sentence") or "").strip()

            if sentence:
                sentences.append(sentence)

    return sentences


def collect_group_frames(sentence_folder):
    """Collect images separately for each signer group."""

    groups = {}

    group_directories = [
        folder
        for folder in sentence_folder.iterdir()
        if folder.is_dir()
    ]

    # Sort group folders numerically where possible.
    group_directories.sort(key=natural_sort_key)

    for group_folder in group_directories:

        frames = sorted(
            [
                image
                for image in group_folder.iterdir()
                if image.is_file()
                and image.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=natural_sort_key
        )

        if frames:
            groups[group_folder.name] = [
                str(frame.relative_to(BASE_DIR))
                for frame in frames
            ]

    # Also support sentence folders containing images directly.
    direct_frames = sorted(
        [
            image
            for image in sentence_folder.iterdir()
            if image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_sort_key
    )

    if direct_frames:
        groups["default"] = [
            str(frame.relative_to(BASE_DIR))
            for frame in direct_frames
        ]

    return groups


def build_index():

    print("=" * 65)
    print("NEXUSCUE - BULK SENTENCE SEQUENCE INDEX BUILDER")
    print("=" * 65)

    if not FRAME_ROOT.exists():
        raise FileNotFoundError(
            f"Sentence frame directory not found: {FRAME_ROOT}"
        )

    sentences = load_sentences()

    print(f"\nGloss CSV sentences: {len(sentences)}")
    print("Scanning sentence frame folders...\n")

    # Build a normalized lookup of available directories.
    folder_lookup = {}

    for folder in FRAME_ROOT.iterdir():

        if folder.is_dir():

            normalized_name = normalize_text(folder.name)

            if normalized_name:
                folder_lookup.setdefault(
                    normalized_name, []
                ).append(folder)

    index = {}

    matched_sentences = 0
    total_groups = 0
    total_frames = 0

    for number, sentence in enumerate(sentences, start=1):

        normalized_sentence = normalize_text(sentence)

        matching_folders = folder_lookup.get(
            normalized_sentence, []
        )

        if not matching_folders:
            print(
                f"{number:03d}. NOT FOUND: {sentence}"
            )
            continue

        # Do not silently choose between ambiguous folder matches.
        if len(matching_folders) > 1:
            print(
                f"{number:03d}. AMBIGUOUS: {sentence}"
            )
            for folder in matching_folders:
                print(f"      {folder.name}")
            continue

        sentence_folder = matching_folders[0]

        groups = collect_group_frames(sentence_folder)

        if not groups:
            print(
                f"{number:03d}. EMPTY: {sentence}"
            )
            continue

        frame_count = sum(
            len(frames) for frames in groups.values()
        )

        index[normalized_sentence] = {
            "sentence": sentence,
            "folder": str(sentence_folder.relative_to(BASE_DIR)),
            "groups": groups,
            "group_count": len(groups),
            "total_frames": frame_count
        }

        matched_sentences += 1
        total_groups += len(groups)
        total_frames += frame_count

        group_summary = ", ".join(
            f"{name}: {len(frames)}"
            for name, frames in groups.items()
        )

        print(
            f"{number:03d}. INDEXED: {sentence}"
        )
        print(f"      Groups: {group_summary}")
        print(f"      Total images: {frame_count}")

    # Save a separate index. Original dataset files remain untouched.
    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            index,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("INDEX BUILD COMPLETED")
    print("=" * 65)

    print(f"CSV sentences: {len(sentences)}")
    print(f"Indexed sentences: {matched_sentences}")
    print(f"Unindexed sentences: {len(sentences) - matched_sentences}")
    print(f"Signer groups indexed: {total_groups}")
    print(f"Total images indexed: {total_frames}")

    print(f"\nIndex saved to:\n{OUTPUT_PATH}")

    print("\nOriginal dataset and gloss CSV were not modified.")


if __name__ == "__main__":

    try:
        build_index()

    except Exception as error:
        print("\nINDEX BUILD FAILED")
        print(error)
