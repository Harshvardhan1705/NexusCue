
import csv
import re
from pathlib import Path

# ==========================================================
# NEXUSCUE - SENTENCE-LEVEL COVERAGE AUDIT
# Read-only: does not modify the original dataset
# ==========================================================

DATASET_ROOT = Path(__file__).resolve().parent

CSV_PATH = (
    DATASET_ROOT
    / "corpus_csv_files"
    / "ISL Corpus sign glosses.csv"
)

FRAME_ROOT = DATASET_ROOT / "Frames_Sentence_Level"
VIDEO_ROOT = DATASET_ROOT / "Videos_Sentence_Level"

OUTPUT_CSV = DATASET_ROOT / "sentence_coverage_inventory.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

ALREADY_INTEGRATED = {
    "he would be coming today",
    "i dont agree",
    "no need to worry dont worry",
    "now onwards he will never hurt you",
}


def normalize_name(value):
    """Normalize labels for case- and punctuation-insensitive matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def collect_directories(root):
    if not root.exists():
        return []

    return [
        path for path in root.rglob("*")
        if path.is_dir()
    ]


def collect_files(root, extensions):
    if not root.exists():
        return []

    return [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]


def main():

    print("=" * 68)
    print("NEXUSCUE - SENTENCE-LEVEL COVERAGE AUDIT")
    print("=" * 68)

    if not CSV_PATH.exists():
        print(f"\nERROR: Gloss CSV not found:\n{CSV_PATH}")
        return

    if not FRAME_ROOT.exists():
        print(f"\nERROR: Frame directory not found:\n{FRAME_ROOT}")
        return

    if not VIDEO_ROOT.exists():
        print(f"\nERROR: Video directory not found:\n{VIDEO_ROOT}")
        return

    print("\nReading gloss CSV...")

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {"Sentence", "SIGN GLOSSES"}

        if not required.issubset(reader.fieldnames or []):
            print("ERROR: Required CSV columns are missing.")
            print("Found columns:", reader.fieldnames)
            return

        sentences = [
            row for row in reader
            if row.get("Sentence", "").strip()
        ]

    print(f"CSV sentences: {len(sentences)}")

    print("\nScanning sentence frame folders...")
    frame_directories = collect_directories(FRAME_ROOT)

    frame_index = {}

    for folder in frame_directories:
        key = normalize_name(folder.name)

        if key:
            frame_index.setdefault(key, []).append(folder)

    print(f"Frame directories scanned: {len(frame_directories)}")

    print("\nScanning sentence video folders...")
    video_directories = collect_directories(VIDEO_ROOT)
    video_files_all = collect_files(VIDEO_ROOT, VIDEO_EXTENSIONS)

    video_directory_index = {}
    video_file_index = {}

    for folder in video_directories:
        key = normalize_name(folder.name)

        if key:
            video_directory_index.setdefault(key, []).append(folder)

    for video in video_files_all:
        key = normalize_name(video.stem)

        if key:
            video_file_index.setdefault(key, []).append(video)

    print(f"Video directories scanned: {len(video_directories)}")
    print(f"Video files scanned: {len(video_files_all)}")

    results = []

    exact_frame_matches = 0
    frame_candidates = 0
    video_matches = 0
    integrated_matches = 0

    print("\nMatching sentence entries...")

    for index, row in enumerate(sentences, start=1):

        sentence = row["Sentence"].strip()
        gloss = row["SIGN GLOSSES"].strip()

        key = normalize_name(sentence)

        matched_frame_folders = frame_index.get(key, [])
        matched_video_folders = video_directory_index.get(key, [])
        matched_video_files = video_file_index.get(key, [])

        # Count unique image files across matching folders.
        image_paths = set()

        for folder in matched_frame_folders:
            for image_path in folder.rglob("*"):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    image_paths.add(image_path)

        # Collect videos inside matching sentence folders.
        video_paths = set(matched_video_files)

        for folder in matched_video_folders:
            for video_path in folder.rglob("*"):
                if (
                    video_path.is_file()
                    and video_path.suffix.lower() in VIDEO_EXTENSIONS
                ):
                    video_paths.add(video_path)

        frame_count = len(image_paths)
        video_count = len(video_paths)

        if matched_frame_folders:
            exact_frame_matches += 1

        if frame_count > 1:
            frame_candidates += 1

        if video_count > 0:
            video_matches += 1

        if normalize_name(sentence) in {
            normalize_name(item) for item in ALREADY_INTEGRATED
        }:
            status = "ALREADY_INTEGRATED"
            integrated_matches += 1
        elif frame_count > 1:
            status = "FRAME_SEQUENCE_CANDIDATE"
        elif video_count > 0:
            status = "VIDEO_ONLY_REVIEW"
        else:
            status = "NO_MATCH_FOUND"

        results.append({
            "row": index,
            "sentence": sentence,
            "gloss": gloss,
            "frame_folder_matches": " ; ".join(
                str(path) for path in matched_frame_folders
            ),
            "frame_count": frame_count,
            "video_folder_matches": " ; ".join(
                str(path) for path in matched_video_folders
            ),
            "video_count": video_count,
            "video_files": " ; ".join(
                str(path) for path in sorted(video_paths)
            ),
            "status": status,
        })

        print(
            f"{index:03d}. {sentence} | "
            f"Frames: {frame_count} | "
            f"Videos: {video_count} | "
            f"{status}"
        )

    columns = [
        "row",
        "sentence",
        "gloss",
        "frame_folder_matches",
        "frame_count",
        "video_folder_matches",
        "video_count",
        "video_files",
        "status",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 68)
    print("SENTENCE COVERAGE AUDIT COMPLETED")
    print("=" * 68)

    print(f"Total CSV sentences: {len(sentences)}")
    print(f"Exact frame-folder matches: {exact_frame_matches}")
    print(f"Sequences with more than one frame: {frame_candidates}")
    print(f"Sentences with matching videos: {video_matches}")
    print(f"Already integrated sequences: {integrated_matches}")

    print("\nReport saved to:")
    print(OUTPUT_CSV)

    print("\nOriginal dataset and gloss CSV were not modified.")


if __name__ == "__main__":
    main()
