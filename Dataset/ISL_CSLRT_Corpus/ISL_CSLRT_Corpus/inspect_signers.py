import os
import cv2
import math


BASE_PATH = (
    r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus"
    r"\ISL_CSLRT_Corpus\Frames_Word_Level"
)

OUTPUT_FILE = "signer_inspection.jpg"

SIGNS = [
    "YOU",
    "FREE",
    "TODAY",
    "HAPPY",
    "HOW"
]

IMAGES_PER_SIGN = 10

THUMB_WIDTH = 180
THUMB_HEIGHT = 180

LABEL_HEIGHT = 35

COLUMNS = 5


def get_images(sign):

    folder = os.path.join(BASE_PATH, sign)

    if not os.path.exists(folder):
        return []

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    files.sort()

    return [
        os.path.join(folder, f)
        for f in files
    ]


def select_images(files):

    if len(files) <= IMAGES_PER_SIGN:
        return files

    step = len(files) / IMAGES_PER_SIGN

    selected = []

    for i in range(IMAGES_PER_SIGN):

        index = int(i * step)

        if index >= len(files):
            index = len(files) - 1

        selected.append(files[index])

    return selected


def create_thumbnail(image_path, sign, index):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(
        image,
        (THUMB_WIDTH, THUMB_HEIGHT)
    )

    canvas = cv2.copyMakeBorder(
        image,
        0,
        LABEL_HEIGHT,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30)
    )

    filename = os.path.basename(image_path)

    label = f"{sign} | {index + 1}"

    cv2.putText(
        canvas,
        label,
        (5, THUMB_HEIGHT + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return canvas


def main():

    print("=" * 60)
    print("SIGNER INSPECTION CONTACT SHEET")
    print("=" * 60)

    all_images = []

    for sign in SIGNS:

        files = get_images(sign)

        selected = select_images(files)

        print(
            f"{sign:<10} → "
            f"{len(files)} total, "
            f"{len(selected)} selected"
        )

        for index, image_path in enumerate(selected):

            thumb = create_thumbnail(
                image_path,
                sign,
                index
            )

            if thumb is not None:
                all_images.append(thumb)

    if not all_images:

        print("\n❌ No images found.")
        return

    rows = math.ceil(
        len(all_images) / COLUMNS
    )

    sheet_width = COLUMNS * THUMB_WIDTH

    sheet_height = rows * (
        THUMB_HEIGHT + LABEL_HEIGHT
    )

    sheet = 255 * (
        __import__("numpy").ones(
            (
                sheet_height,
                sheet_width,
                3
            ),
            dtype="uint8"
        )
    )

    for i, image in enumerate(all_images):

        row = i // COLUMNS
        col = i % COLUMNS

        y = row * (
            THUMB_HEIGHT + LABEL_HEIGHT
        )

        x = col * THUMB_WIDTH

        sheet[
            y:y + image.shape[0],
            x:x + image.shape[1]
        ] = image

    cv2.imwrite(
        OUTPUT_FILE,
        sheet
    )

    print("\n✅ Contact sheet created:")
    print(os.path.abspath(OUTPUT_FILE))


if __name__ == "__main__":
    main()