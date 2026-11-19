
import csv
import re
import shutil
from pathlib import Path

from gloss_engine import normalize_text

# ============================================================
# NEXUSCUE - BULK SENTENCE SEQUENCE BUILDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_ROOT = BASE_DIR

FRAME_ROOT = DATASET_ROOT / "Frames_Sentence_Level"

CSV_PATH = (
    DATASET_ROOT
    / "corpus_csv_files"
    / "ISL Corpus sign glosses.csv"
)

OUTPUT_ROOT = BASE_DIR / "bulk_sentence_sequences"

MANIFEST_PATH = OUTPUT_ROOT / "bulk_sequence_manifest.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def natural_sort_key(path):
    """Sort filenames in numerical order."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def make_folder_name(sentence):
    """Create a safe folder name from a sentence."""
    name = re.sub(r"[^a-zA-Z0-9]+", "_", sentence.strip())
    return name.strip("_")


def load_gloss_sentences():
    """Read all sentence and gloss entries from the CSV."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Gloss CSV not found:\n{CSV_PATH}"
        )

    entries = []

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required = {"Sentence", "SIGN GLOSSES"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV must contain Sentence and SIGN GLOSSES columns."
            )

        for row in reader:
            sentence = (row.get("Sentence") or "").strip()
            gloss = (row.get("SIGN GLOSSES") or "").strip()

            if sentence:
                entries.append({
                    "sentence": sentence,
                    "normalized": normalize_text(sentence),
                    "gloss": gloss
                })

    return entries


def find_sentence_folder(sentence):
    """Find the matching sentence frame folder."""

    if not FRAME_ROOT.exists():
        return None

    target = normalize_text(sentence)

    for folder in FRAME_ROOT.iterdir():

        if folder.is_dir():
            if normalize_text(folder.name) == target:
                return folder

    return None


def collect_group_images(group_folder):
    """Collect images from one signer group."""

    images = [
        path
        for path in group_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images, key=natural_sort_key)


def process_sentence(entry):
    """Copy each available signer group into a separate sequence."""

    sentence = entry["sentence"]
    gloss = entry["gloss"]

    source_folder = find_sentence_folder(sentence)

    if source_folder is None:
        return []

    sentence_output = (
        OUTPUT_ROOT / make_folder_name(sentence)
    )

    results = []

    group_folders = sorted(
        [
            folder
            for folder in source_folder.iterdir()
            if folder.is_dir()
        ],
        key=natural_sort_key
    )

    for group_folder in group_folders:

        images = collect_group_images(group_folder)

        if len(images) < 2:
            continue

        group_name = group_folder.name

        destination = (
            sentence_output / f"group_{group_name}"
        )

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        # Remove only previously generated JPG/PNG files
        # inside this output group folder.
        for old_file in destination.iterdir():
            if (
                old_file.is_file()
                and old_file.suffix.lower() in IMAGE_EXTENSIONS
            ):
                old_file.unlink()

        copied_count = 0

        for index, source_image in enumerate(images, start=1):

            output_image = (
                destination
                / f"frame_{index:04d}{source_image.suffix.lower()}"
            )

            shutil.copy2(source_image, output_image)
            copied_count += 1

        results.append({
            "sentence": sentence,
            "gloss": gloss,
            "signer_group": group_name,
            "frame_count": copied_count,
            "first_frame": f"frame_0001{images[0].suffix.lower()}",
            "last_frame": (
                f"frame_{copied_count:04d}"
                f"{images[-1].suffix.lower()}"
            ),
            "sequence_folder": str(destination)
        })

    return results


def main():

    print("=" * 68)
    print("NEXUSCUE - BULK SENTENCE SEQUENCE BUILDER")
    print("=" * 68)

    if not FRAME_ROOT.exists():
        print("\nERROR: Sentence frame directory not found.")
        print(FRAME_ROOT)
        return

    print("\nReading gloss CSV...")

    entries = load_gloss_sentences()

    print(f"CSV sentences: {len(entries)}")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    sentences_found = 0
    sentences_missing = 0

    print("\nProcessing sentence frame folders...\n")

    for index, entry in enumerate(entries, start=1):

        results = process_sentence(entry)

        if results:
            sentences_found += 1
            total_frames = sum(
                result["frame_count"] for result in results
            )

            print(
                f"{index:03d}. {entry['sentence']} | "
                f"Groups: {len(results)} | "
                f"Frames: {total_frames} | READY"
            )

            all_results.extend(results)

        else:
            sentences_missing += 1

            print(
                f"{index:03d}. {entry['sentence']} | "
                f"NO USABLE GROUP FOUND"
            )

    fieldnames = [
        "sentence",
        "gloss",
        "signer_group",
        "frame_count",
        "first_frame",
        "last_frame",
        "sequence_folder"
    ]

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 68)
    print("BULK PROCESSING COMPLETED")
    print("=" * 68)

    print(f"Total CSV sentences: {len(entries)}")
    print(f"Sentences with usable groups: {sentences_found}")
    print(f"Sentences without usable groups: {sentences_missing}")
    print(f"Total signer-group sequences: {len(all_results)}")
    print(
        "Total copied frames: "
        f"{sum(item['frame_count'] for item in all_results)}"
    )

    print(f"\nOutput folder:\n{OUTPUT_ROOT}")
    print(f"\nManifest:\n{MANIFEST_PATH}")

    print("\nOriginal dataset and gloss CSV were not modified.")


if __name__ == "__main__":
    main()
