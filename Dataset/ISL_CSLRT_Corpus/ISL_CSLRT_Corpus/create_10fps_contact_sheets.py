
import cv2
import math
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "recovered_sentence_frames_10fps"
OUTPUT_DIR = INPUT_DIR / "contact_sheets"

OUTPUT_DIR.mkdir(exist_ok=True)

THUMB_WIDTH = 320
THUMB_HEIGHT = 240
LABEL_HEIGHT = 35
CARD_HEIGHT = THUMB_HEIGHT + LABEL_HEIGHT

COLUMNS = 4
GAP = 12

folders = sorted(
    folder for folder in INPUT_DIR.iterdir()
    if folder.is_dir()
)

for folder in folders:

    images = sorted(folder.glob("*.jpg"))

    if not images:
        continue

    cards = []

    for image_path in images:

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image = cv2.resize(
            image,
            (THUMB_WIDTH, THUMB_HEIGHT)
        )

        card = np.full(
            (CARD_HEIGHT, THUMB_WIDTH, 3),
            245,
            dtype=np.uint8
        )

        card[:THUMB_HEIGHT] = image

        cv2.putText(
            card,
            image_path.name,
            (8, THUMB_HEIGHT + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA
        )

        cards.append(card)

    if not cards:
        continue

    rows = math.ceil(len(cards) / COLUMNS)

    sheet_width = COLUMNS * THUMB_WIDTH + (COLUMNS + 1) * GAP
    sheet_height = rows * CARD_HEIGHT + (rows + 1) * GAP

    sheet = np.full(
        (sheet_height, sheet_width, 3),
        245,
        dtype=np.uint8
    )

    for index, card in enumerate(cards):

        row = index // COLUMNS
        col = index % COLUMNS

        x = GAP + col * (THUMB_WIDTH + GAP)
        y = GAP + row * (CARD_HEIGHT + GAP)

        sheet[
            y:y + CARD_HEIGHT,
            x:x + THUMB_WIDTH
        ] = card

    output_file = OUTPUT_DIR / (
        folder.name + "_contact_sheet.jpg"
    )

    cv2.imwrite(
        str(output_file),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    print(f"Created: {output_file.name}")
    print(f"Frames: {len(cards)}")

print("\nAll 10 FPS contact sheets created.")
