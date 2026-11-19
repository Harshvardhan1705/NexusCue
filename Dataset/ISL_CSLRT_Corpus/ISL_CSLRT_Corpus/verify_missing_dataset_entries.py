
import os
import re
import csv
from pathlib import Path

# =====================================================
# NEXUSCUE - MISSING DATASET ENTRY VERIFICATION
# Read-only inspection: original dataset is not modified.
# =====================================================

DATASET_ROOT = Path(
    r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"
)

VIDEO_DIR = DATASET_ROOT / "Videos_Sentence_Level"
FRAME_DIR = DATASET_ROOT / "Frames_Sentence_Level"

OUTPUT_FILE = Path(__file__).resolve().parent / "missing_entries_report.csv"

TARGETS = [
    {
        "row": 23,
        "sentence": "he would be coming today",
        "alternatives": []
    },
    {
        "row": 50,
        "sentence": "i dont agree",
        "alternatives": ["i do not agree"]
    },
    {
        "row": 64,
        "sentence": "no need to worry dont worry",
        "alternatives": ["do not worry"]
    },
    {
        "row": 65,
        "sentence": "now onwards he will never hurt you",
        "alternatives": []
    }
]

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def normalize(value):
    """Normalize text for filename/folder comparison."""
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def inspect_directory(directory, target, extensions):
    """
    Find exact matching folder names and file names.
    Count media files inside matching folders.
    """
    result = {
        "exists": directory.exists(),
        "matching_folders": [],
        "matching_files": [],
        "media_count": 0,
        "sample_files": [],
        "error": None
    }

    if not directory.exists():
        return result

    target_normalized = normalize(target)

    try:
        for root, dirs, files in os.walk(directory):
            # Ignore inaccessible folders without stopping the scan.
            dirs.sort()
            files.sort()

            current = Path(root)

            if normalize(current.name) == target_normalized:
                result["matching_folders"].append(str(current))

                for child_root, _, child_files in os.walk(current):
                    for filename in child_files:
                        file_path = Path(child_root) / filename

                        if file_path.suffix.lower() in extensions:
                            result["media_count"] += 1

                            if len(result["sample_files"]) < 10:
                                result["sample_files"].append(
                                    str(file_path)
                                )

                # Avoid counting the same nested folder twice.
                dirs[:] = []

            for filename in files:
                file_path = current / filename

                if (
                    normalize(file_path.stem) == target_normalized
                    and file_path.suffix.lower() in extensions
                ):
                    result["matching_files"].append(str(file_path))

    except PermissionError as error:
        result["error"] = str(error)

    return result


def print_section(title, result):
    print(f"\n{title}")
    print("-" * 55)

    if not result["exists"]:
        print("DIRECTORY NOT FOUND")
        return

    print("Matching folders:", len(result["matching_folders"]))

    for folder in result["matching_folders"]:
        print("  Folder:", folder)

    print("Exact matching files:", len(result["matching_files"]))

    for file_path in result["matching_files"][:10]:
        print("  File:", file_path)

    print("Media files inside matching folders:", result["media_count"])

    for sample in result["sample_files"]:
        print("  Sample:", sample)

    if result["error"]:
        print("Permission warning:", result["error"])


def main():
    print("=" * 65)
    print("NEXUSCUE - DATASET VERIFICATION")
    print("=" * 65)

    print("\nDataset root:", DATASET_ROOT)
    print("Video directory:", VIDEO_DIR)
    print("Frame directory:", FRAME_DIR)

    if not DATASET_ROOT.exists():
        print("\nERROR: Dataset root does not exist.")
        return

    report_rows = []

    for item in TARGETS:
        row = item["row"]
        sentence = item["sentence"]

        print("\n\n" + "=" * 65)
        print(f"ROW {row}: {sentence}")
        print("=" * 65)

        video_result = inspect_directory(
            VIDEO_DIR, sentence, VIDEO_EXTENSIONS
        )

        frame_result = inspect_directory(
            FRAME_DIR, sentence, IMAGE_EXTENSIONS
        )

        print_section("VIDEO DATASET RESULTS", video_result)
        print_section("FRAME DATASET RESULTS", frame_result)

        for alternative in item["alternatives"]:
            print("\nChecking alternative label:", alternative)

            alt_video = inspect_directory(
                VIDEO_DIR, alternative, VIDEO_EXTENSIONS
            )

            alt_frames = inspect_directory(
                FRAME_DIR, alternative, IMAGE_EXTENSIONS
            )

            print_section("ALTERNATIVE VIDEO RESULTS", alt_video)
            print_section("ALTERNATIVE FRAME RESULTS", alt_frames)

        report_rows.append({
            "CSV Row": row,
            "Sentence": sentence,
            "Video Folder Matches": len(video_result["matching_folders"]),
            "Exact Video Files": len(video_result["matching_files"]),
            "Video Files in Matching Folders": video_result["media_count"],
            "Frame Folder Matches": len(frame_result["matching_folders"]),
            "Exact Frame Files": len(frame_result["matching_files"]),
            "Frames in Matching Folders": frame_result["media_count"]
        })

    # Save a small report outside the dataset.
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=report_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n\n" + "=" * 65)
    print("VERIFICATION COMPLETED")
    print("=" * 65)
    print("Report saved to:", OUTPUT_FILE)
    print("Original dataset files were not modified.")


if __name__ == "__main__":
    main()
