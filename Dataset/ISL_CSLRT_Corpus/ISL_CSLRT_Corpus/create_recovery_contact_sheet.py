
import cv2
import math
import numpy as np
from pathlib import Path

# NexusCue recovered frames directory
BASE_DIR = Path(__file__).resolve().parent
RECOVERY_DIR = BASE_DIR / "recovered_sentence_frames"

OUTPUT_FILE = RECOVERY_DIR / "recovered_frames_contact_sheet.jpg"

# Contact sheet settings
THUMB_WIDTH = 320
THUMB_HEIGHT = 240
LABEL_HEIGHT = 55
CARD_HEIGHT = THUMB_HEIGHT + LABEL_HEIGHT

COLUMNS = 4
GAP = 12

cards = []

# Find recovered frame folders
frame_folders = sorted(
    folder for folder in RECOVERY_DIR.iterdir()
    if folder.is_dir()
)

for folder in frame_folders:

    image_files = sorted(folder.glob("*.jpg"))

    for image_path in image_files:

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        image = cv2.resize(
            image,
            (THUMB_WIDTH, THUMB_HEIGHT)
        )

        # Create a complete card including label area
        card = np.full(
            (CARD_HEIGHT, THUMB_WIDTH, 3),
            245,
            dtype=np.uint8
        )

        # Place image at the top
        card[0:THUMB_HEIGHT, :] = image

        # Add folder name
        cv2.putText(
            card,
            folder.name[:42],
            (8, THUMB_HEIGHT + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (25, 25, 25),
            1,
            cv2.LINE_AA
        )

        # Add filename
        cv2.putText(
            card,
            image_path.name,
            (8, THUMB_HEIGHT + 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (25, 25, 25),
            1,
            cv2.LINE_AA
        )

        cards.append(card)

if not cards:
    print("No recovered images found.")
    raise SystemExit

# Calculate sheet dimensions
rows = math.ceil(len(cards) / COLUMNS)

sheet_width = COLUMNS * THUMB_WIDTH + (COLUMNS + 1) * GAP
sheet_height = rows * CARD_HEIGHT + (rows + 1) * GAP

# Create contact sheet
sheet = np.full(
    (sheet_height, sheet_width, 3),
    245,
    dtype=np.uint8
)

# Place cards
for index, card in enumerate(cards):

    row = index // COLUMNS
    col = index % COLUMNS

    x = GAP + col * (THUMB_WIDTH + GAP)
    y = GAP + row * (CARD_HEIGHT + GAP)

    sheet[
        y:y + CARD_HEIGHT,
        x:x + THUMB_WIDTH
    ] = card

# Save contact sheet
success = cv2.imwrite(
    str(OUTPUT_FILE),
    sheet,
    [cv2.IMWRITE_JPEG_QUALITY, 95]
)

print("=" * 55)
print("NEXUSCUE - RECOVERY CONTACT SHEET")
print("=" * 55)
print(f"Total frames displayed: {len(cards)}")
print(f"Contact sheet dimensions: {sheet_width} x {sheet_height}")
print(f"Output: {OUTPUT_FILE}")

if success:
    print("Contact sheet created successfully!")
else:
    print("ERROR: Could not save contact sheet.")
