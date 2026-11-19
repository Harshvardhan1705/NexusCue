
from pathlib import Path
import cv2
import re
import csv

# --------------------------------------------------
# NEXUSCUE - VALIDATE CANDIDATE 4, GROUP 3
# Sentence: what is your phone number
# --------------------------------------------------

DATASET_ROOT = Path(
    r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"
)

FRAME_DIR = (
    DATASET_ROOT
    / "Frames_Sentence_Level"
    / "what is your phone number"
    / "3"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "candidate_4_group3_validation"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "candidate4_group3_preview.mp4"
REPORT_FILE = OUTPUT_DIR / "group3_validation_report.csv"

FPS = 10
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def main():

    print("=" * 65)
    print("NEXUSCUE - GROUP 3 VALIDATION")
    print("=" * 65)

    if not FRAME_DIR.exists():
        print("ERROR: Group 3 folder not found:")
        print(FRAME_DIR)
        return

    images = sorted(
        [
            p for p in FRAME_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ],
        key=natural_key
    )

    print(f"\nFrames discovered: {len(images)}")

    if not images:
        print("ERROR: No images found.")
        return

    valid_frames = []
    report = []

    for index, image_path in enumerate(images, start=1):

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"UNREADABLE: {image_path.name}")
            report.append([index, image_path.name, "UNREADABLE", "", ""])
            continue

        height, width = frame.shape[:2]

        valid_frames.append((image_path, frame))

        report.append([
            index,
            image_path.name,
            "READABLE",
            width,
            height
        ])

    with open(REPORT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Sequence Number",
            "Filename",
            "Status",
            "Width",
            "Height"
        ])
        writer.writerows(report)

    print(f"Readable frames: {len(valid_frames)}")
    print(f"Unreadable frames: {len(images) - len(valid_frames)}")

    if not valid_frames:
        print("ERROR: No readable frames available.")
        return

    # Use the first valid frame's dimensions for the preview.
    first_frame = valid_frames[0][1]
    height, width = first_frame.shape[:2]

    # MP4 commonly requires even dimensions.
    width = width - (width % 2)
    height = height - (height % 2)

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Could not create preview video.")
        return

    for image_path, frame in valid_frames:

        frame = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

        writer.write(frame)

    writer.release()

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETED")
    print("=" * 65)

    print(f"Total source images: {len(images)}")
    print(f"Frames included in preview: {len(valid_frames)}")
    print(f"Preview FPS: {FPS}")
    print(f"Preview duration: {len(valid_frames) / FPS:.2f} seconds")
    print(f"Video: {OUTPUT_VIDEO}")
    print(f"Report: {REPORT_FILE}")

    print("\nOriginal dataset and renderer were not modified.")


if __name__ == "__main__":
    main()
