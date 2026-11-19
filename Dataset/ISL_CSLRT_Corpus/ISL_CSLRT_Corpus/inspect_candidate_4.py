
from pathlib import Path
import cv2
import numpy as np
import csv
import re

# --------------------------------------------------
# NEXUSCUE - CANDIDATE 4 FRAME INSPECTION
# --------------------------------------------------

DATASET_ROOT = Path(
    r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"
)

FRAME_ROOT = DATASET_ROOT / "Frames_Sentence_Level" / "what is your phone number"

OUTPUT_DIR = Path(__file__).resolve().parent / "candidate_4_inspection"
OUTPUT_DIR.mkdir(exist_ok=True)

CONTACT_SHEET = OUTPUT_DIR / "candidate_4_contact_sheet.jpg"
REPORT_FILE = OUTPUT_DIR / "candidate_4_frame_report.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SAMPLES_PER_GROUP = 12

TILE_WIDTH = 300
IMAGE_HEIGHT = 180
LABEL_HEIGHT = 38
COLS = 4
ROWS = 3

PANEL_WIDTH = COLS * TILE_WIDTH
PANEL_HEIGHT = ROWS * (IMAGE_HEIGHT + LABEL_HEIGHT) + 65


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_images(folder):
    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_key
    )


def select_samples(images):
    if len(images) <= SAMPLES_PER_GROUP:
        return images

    indices = np.linspace(
        0, len(images) - 1,
        SAMPLES_PER_GROUP,
        dtype=int
    )

    return [images[i] for i in indices]


def create_contact_sheet():
    if not FRAME_ROOT.exists():
        print("ERROR: Frame directory not found:")
        print(FRAME_ROOT)
        return

    groups = sorted(
        [p for p in FRAME_ROOT.iterdir() if p.is_dir()],
        key=natural_key
    )

    if not groups:
        print("ERROR: No frame groups found.")
        return

    panel_rows = (len(groups) + 1) // 2

    sheet = np.full(
        (panel_rows * PANEL_HEIGHT, PANEL_WIDTH * 2, 3),
        245,
        dtype=np.uint8
    )

    report_rows = []

    print("=" * 65)
    print("NEXUSCUE - CANDIDATE 4 FRAME INSPECTION")
    print("=" * 65)

    for group_index, group in enumerate(groups):

        images = get_images(group)
        samples = select_samples(images)

        print(f"\nGroup {group.name}: {len(images)} images")

        report_rows.append({
            "Group": group.name,
            "Total Images": len(images),
            "First Image": images[0].name if images else "",
            "Last Image": images[-1].name if images else "",
            "Sample Images": "; ".join(p.name for p in samples)
        })

        panel_col = group_index % 2
        panel_row = group_index // 2

        x_offset = panel_col * PANEL_WIDTH
        y_offset = panel_row * PANEL_HEIGHT

        cv2.putText(
            sheet,
            f"GROUP {group.name} - {len(images)} FRAMES",
            (x_offset + 15, y_offset + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (25, 35, 50),
            2
        )

        for sample_index, image_path in enumerate(samples):

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Could not read: {image_path.name}")
                continue

            image = cv2.resize(
                image,
                (TILE_WIDTH, IMAGE_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            row = sample_index // COLS
            col = sample_index % COLS

            x = x_offset + col * TILE_WIDTH
            y = y_offset + 50 + row * (IMAGE_HEIGHT + LABEL_HEIGHT)

            sheet[
                y:y + IMAGE_HEIGHT,
                x:x + TILE_WIDTH
            ] = image

            label = image_path.name[:38]

            cv2.putText(
                sheet,
                label,
                (x + 5, y + IMAGE_HEIGHT + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (20, 20, 20),
                1
            )

    cv2.imwrite(
        str(CONTACT_SHEET),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    with open(REPORT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Group",
                "Total Images",
                "First Image",
                "Last Image",
                "Sample Images"
            ]
        )

        writer.writeheader()
        writer.writerows(report_rows)

    print("\n" + "=" * 65)
    print("INSPECTION COMPLETED")
    print("=" * 65)
    print(f"Groups inspected: {len(groups)}")
    print(f"Contact sheet: {CONTACT_SHEET}")
    print(f"CSV report: {REPORT_FILE}")
    print("\nOriginal dataset was not modified.")


if __name__ == "__main__":
    create_contact_sheet()
