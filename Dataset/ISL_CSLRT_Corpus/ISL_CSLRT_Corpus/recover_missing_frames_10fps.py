
import cv2
import csv
from pathlib import Path

# =====================================================
# NEXUSCUE - 10 FPS FRAME RECOVERY
# =====================================================

DATASET_ROOT = Path(
    r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"
)

VIDEO_ROOT = DATASET_ROOT / "Videos_Sentence_Level"

# Separate output folder: existing recovered frames stay untouched
OUTPUT_ROOT = DATASET_ROOT / "recovered_sentence_frames_10fps"

TARGET_FPS = 10

TARGETS = [
    {
        "row": 23,
        "sentence": "he would be coming today",
        "video": "he would be coming today.mp4"
    },
    {
        "row": 50,
        "sentence": "i dont agree",
        "video": "MVI_6494.MP4"
    },
    {
        "row": 64,
        "sentence": "No need to worry dont worry",
        "video": "MVI_6496.MP4"
    },
    {
        "row": 65,
        "sentence": "now onwards he will never hurt you",
        "video": "now onwards he will never hurt you.mp4"
    }
]


def find_video(filename):
    """Find the exact video filename, ignoring case."""

    matches = [
        path for path in VIDEO_ROOT.rglob("*")
        if path.is_file()
        and path.name.lower() == filename.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        print(f"ERROR: Video not found: {filename}")
    else:
        print(f"ERROR: Multiple matching videos found: {filename}")
        for match in matches:
            print(match)

    return None


def extract_frames(target):

    row = target["row"]
    sentence = target["sentence"]
    video_name = target["video"]

    print("\n" + "=" * 65)
    print(f"ROW {row}: {sentence}")
    print("=" * 65)

    video_path = find_video(video_name)

    if video_path is None:
        return {
            "row": row,
            "sentence": sentence,
            "frames": 0,
            "status": "VIDEO NOT FOUND"
        }

    output_folder = OUTPUT_ROOT / (
        f"{row}_{sentence.replace(' ', '_')}"
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")

        return {
            "row": row,
            "sentence": sentence,
            "frames": 0,
            "status": "VIDEO OPEN FAILED"
        }

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0 or total_frames <= 0:
        print("ERROR: Invalid video metadata.")
        cap.release()

        return {
            "row": row,
            "sentence": sentence,
            "frames": 0,
            "status": "INVALID VIDEO METADATA"
        }

    duration = total_frames / source_fps

    print(f"Video: {video_path.name}")
    print(f"Source FPS: {source_fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total source frames: {total_frames}")
    print(f"Target extraction FPS: {TARGET_FPS}")

    # Calculate source frame indices closest to each
    # desired 10 FPS timestamp.
    target_indices = []
    sample_number = 0

    while True:
        source_index = int(
            sample_number * source_fps / TARGET_FPS + 0.5
        )

        if source_index >= total_frames:
            break

        if not target_indices or source_index > target_indices[-1]:
            target_indices.append(source_index)

        sample_number += 1

    frame_index = 0
    saved_count = 0

    manifest_path = OUTPUT_ROOT / "recovered_frames_manifest.csv"

    write_header = not manifest_path.exists()

    with open(
        manifest_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as manifest_file:

        writer = csv.writer(manifest_file)

        if write_header:
            writer.writerow([
                "Dataset Row",
                "Sentence",
                "Source Video",
                "Source Frame Index",
                "Timestamp Seconds",
                "Extracted Frame Path"
            ])

        target_position = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            if (
                target_position < len(target_indices)
                and frame_index == target_indices[target_position]
            ):

                saved_count += 1

                timestamp = frame_index / source_fps

                filename = (
                    f"frame_{saved_count:04d}_"
                    f"{timestamp:.2f}s.jpg"
                )

                output_path = output_folder / filename

                saved = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )

                if saved:
                    print(f"Saved: {filename}")

                    writer.writerow([
                        row,
                        sentence,
                        video_path.name,
                        frame_index,
                        round(timestamp, 3),
                        str(output_path)
                    ])

                else:
                    print(f"Could not save: {filename}")

                target_position += 1

            frame_index += 1

    cap.release()

    status = (
        "SUCCESS"
        if saved_count == len(target_indices)
        else "PARTIAL"
    )

    print(f"Frames recovered: {saved_count}")
    print(f"Status: {status}")

    return {
        "row": row,
        "sentence": sentence,
        "frames": saved_count,
        "status": status
    }


def main():

    print("=" * 65)
    print("NEXUSCUE - 10 FPS MISSING FRAME RECOVERY")
    print("=" * 65)

    if not VIDEO_ROOT.exists():
        print(f"ERROR: Video directory not found: {VIDEO_ROOT}")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Start a fresh manifest for this extraction run
    manifest_path = OUTPUT_ROOT / "recovered_frames_manifest.csv"

    if manifest_path.exists():
        manifest_path.unlink()

    results = []

    for target in TARGETS:
        result = extract_frames(target)
        results.append(result)

    summary_path = OUTPUT_ROOT / "recovery_summary.csv"

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as summary_file:

        writer = csv.writer(summary_file)

        writer.writerow([
            "Dataset Row",
            "Sentence",
            "Frames Recovered",
            "Status"
        ])

        for result in results:
            writer.writerow([
                result["row"],
                result["sentence"],
                result["frames"],
                result["status"]
            ])

    print("\n" + "=" * 65)
    print("10 FPS FRAME RECOVERY COMPLETED")
    print("=" * 65)

    total = 0

    for result in results:
        print(
            f"Row {result['row']}: "
            f"{result['frames']} frames - {result['status']}"
        )

        total += result["frames"]

    print("-" * 65)
    print(f"TOTAL FRAMES RECOVERED: {total}")
    print(f"Output folder: {OUTPUT_ROOT}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    print("\nOriginal dataset files were not modified.")


if __name__ == "__main__":
    main()
