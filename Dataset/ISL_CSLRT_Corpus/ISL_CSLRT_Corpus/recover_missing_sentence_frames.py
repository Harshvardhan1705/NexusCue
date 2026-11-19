
import cv2
import csv
from pathlib import Path

# ============================================================
# NEXUSCUE - SENTENCE LEVEL FRAME RECOVERY
# Non-destructive: original dataset files are never modified.
# ============================================================

DATASET_ROOT = Path(
    r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"
)

VIDEO_ROOT = DATASET_ROOT / "Videos_Sentence_Level"

# Output is created beside this Python script.
OUTPUT_ROOT = Path(__file__).resolve().parent / "recovered_sentence_frames"

FRAME_INTERVAL_SECONDS = 0.5
JPEG_QUALITY = 95

TARGETS = [
    {
        "row": 23,
        "sentence": "he would be coming today",
        "preferred_video": "he would be coming today.mp4"
    },
    {
        "row": 50,
        "sentence": "i dont agree",
        "preferred_video": "MVI_6494.MP4"
    },
    {
        "row": 64,
        "sentence": "No need to worry dont worry",
        "preferred_video": "MVI_6496.MP4"
    },
    {
        "row": 65,
        "sentence": "now onwards he will never hurt you",
        "preferred_video": "now onwards he will never hurt you.mp4"
    }
]


def find_video(target):
    """Locate the selected original video."""

    folder = VIDEO_ROOT / target["sentence"]

    if not folder.exists():
        return None

    preferred = folder / target["preferred_video"]

    if preferred.exists():
        return preferred

    # Case-insensitive fallback.
    preferred_name = target["preferred_video"].lower()

    for file_path in folder.iterdir():
        if (
            file_path.is_file()
            and file_path.name.lower() == preferred_name
        ):
            return file_path

    return None


def extract_frames(video_path, output_folder, row):
    """Extract frames at fixed time intervals."""

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video:", video_path)
        return [], 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("ERROR: Invalid video FPS.")
        cap.release()
        return [], 0, 0

    duration = total_frames / fps if total_frames > 0 else 0

    print("\nVideo:", video_path.name)
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print("Total source frames:", total_frames)

    manifest_rows = []
    saved_count = 0
    timestamp = 0.0

    while True:
        # Seek to the required timestamp.
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

        success, frame = cap.read()

        if not success:
            break

        filename = (
            f"frame_{saved_count + 1:04d}_"
            f"{timestamp:.2f}s.jpg"
        )

        output_path = output_folder / filename

        saved = cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if saved:
            saved_count += 1

            manifest_rows.append({
                "CSV Row": row,
                "Source Video": str(video_path),
                "Frame Filename": filename,
                "Timestamp Seconds": round(timestamp, 2),
                "Frame Path": str(output_path),
                "Width": frame.shape[1],
                "Height": frame.shape[0]
            })

            print("Saved:", filename)

        timestamp += FRAME_INTERVAL_SECONDS

    cap.release()

    print("Frames recovered:", saved_count)

    return manifest_rows, duration, fps


def main():
    print("=" * 65)
    print("NEXUSCUE - MISSING SENTENCE FRAME RECOVERY")
    print("=" * 65)

    if not DATASET_ROOT.exists():
        print("\nERROR: Dataset root not found.")
        return

    if not VIDEO_ROOT.exists():
        print("\nERROR: Videos_Sentence_Level folder not found.")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_manifest_rows = []
    summary_rows = []

    for target in TARGETS:
        row = target["row"]
        sentence = target["sentence"]

        print("\n" + "=" * 65)
        print(f"ROW {row}: {sentence}")
        print("=" * 65)

        video_path = find_video(target)

        if video_path is None:
            print("ERROR: Source video not found.")
            summary_rows.append({
                "CSV Row": row,
                "Sentence": sentence,
                "Source Video": "NOT FOUND",
                "Recovered Frames": 0,
                "Status": "VIDEO NOT FOUND"
            })
            continue

        folder_name = f"{row}_{sentence.replace(' ', '_')}"
        output_folder = OUTPUT_ROOT / folder_name

        manifest, duration, fps = extract_frames(
            video_path,
            output_folder,
            row
        )

        all_manifest_rows.extend(manifest)

        status = "SUCCESS" if manifest else "NO FRAMES EXTRACTED"

        summary_rows.append({
            "CSV Row": row,
            "Sentence": sentence,
            "Source Video": str(video_path),
            "Recovered Frames": len(manifest),
            "Status": status
        })

    # Save detailed frame manifest.
    manifest_file = OUTPUT_ROOT / "recovered_frames_manifest.csv"

    with open(
        manifest_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fields = [
            "CSV Row",
            "Source Video",
            "Frame Filename",
            "Timestamp Seconds",
            "Frame Path",
            "Width",
            "Height"
        ]

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_manifest_rows)

    # Save summary report.
    summary_file = OUTPUT_ROOT / "recovery_summary.csv"

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fields = [
            "CSV Row",
            "Sentence",
            "Source Video",
            "Recovered Frames",
            "Status"
        ]

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n\n" + "=" * 65)
    print("FRAME RECOVERY COMPLETED")
    print("=" * 65)

    total = sum(
        item["Recovered Frames"] for item in summary_rows
    )

    for item in summary_rows:
        print(
            f"Row {item['CSV Row']}: "
            f"{item['Recovered Frames']} frames - "
            f"{item['Status']}"
        )

    print("\nTotal recovered frames:", total)
    print("\nOutput folder:", OUTPUT_ROOT)
    print("Manifest:", manifest_file)
    print("Summary:", summary_file)

    print("\nOriginal dataset files were not modified.")


if __name__ == "__main__":
    main()
