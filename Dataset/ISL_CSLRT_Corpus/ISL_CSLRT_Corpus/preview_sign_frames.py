import cv2
import os
import math


BASE_PATH = r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Word_Level"

OUTPUT_PATH = "sign_previews"


SIGNS = [
    "YOU",
    "FREE",
    "TODAY",
    "HAPPY",
    "HOW"
]


IMAGE_SIZE = 180
LABEL_HEIGHT = 35
COLUMNS = 5


def natural_sort_key(filename):

    name = filename.lower()

    number = ""

    for char in name:

        if char.isdigit():
            number += char

    if number:
        return int(number)

    return 0


def create_preview(sign):

    folder = os.path.join(BASE_PATH, sign)

    if not os.path.exists(folder):

        print(f"❌ Folder not found: {folder}")
        return

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files.sort(key=natural_sort_key)

    if not files:

        print(f"❌ No images found for {sign}")
        return

    rows = math.ceil(len(files) / COLUMNS)

    sheet_width = COLUMNS * IMAGE_SIZE

    sheet_height = rows * (IMAGE_SIZE + LABEL_HEIGHT)

    sheet = 255 * __import__("numpy").ones(
        (sheet_height, sheet_width, 3),
        dtype="uint8"
    )

    for index, filename in enumerate(files):

        path = os.path.join(folder, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        row = index // COLUMNS
        column = index % COLUMNS

        x = column * IMAGE_SIZE
        y = row * (IMAGE_SIZE + LABEL_HEIGHT)

        sheet[
            y:y + IMAGE_SIZE,
            x:x + IMAGE_SIZE
        ] = image

        label = f"{index + 1}"

        cv2.putText(
            sheet,
            label,
            (x + 5, y + IMAGE_SIZE + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    os.makedirs(
        OUTPUT_PATH,
        exist_ok=True
    )

    output_file = os.path.join(
        OUTPUT_PATH,
        f"{sign}_preview.jpg"
    )

    cv2.imwrite(
        output_file,
        sheet
    )

    print(f"✅ Created: {output_file}")


def main():

    print("=" * 60)
    print("SIGN FRAME PREVIEW GENERATOR")
    print("=" * 60)

    for sign in SIGNS:

        print(f"\nProcessing: {sign}")

        create_preview(sign)

    print("\n" + "=" * 60)
    print("✅ PREVIEWS CREATED")
    print("=" * 60)

    print(f"Location: {OUTPUT_PATH}")


if __name__ == "__main__":

    main()