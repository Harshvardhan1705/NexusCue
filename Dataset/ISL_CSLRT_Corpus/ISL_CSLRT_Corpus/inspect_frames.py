import os

BASE_PATH = r".\Frames_Word_Level"


def get_frame_files(folder_path):
    """Return image files inside a sign asset folder."""

    valid_extensions = {".jpg", ".jpeg", ".png"}

    files = []

    for filename in os.listdir(folder_path):
        extension = os.path.splitext(filename)[1].lower()

        if extension in valid_extensions:
            files.append(filename)

    return sorted(files)


assets_to_check = [
    "YOU",
    "FREE",
    "TODAY",
    "HELLO_HI",
    "HOW",
    "HAPPY",
]


print("=" * 60)
print("SIGN FRAME INSPECTION")
print("=" * 60)

for asset in assets_to_check:

    folder = os.path.join(BASE_PATH, asset)

    print(f"\n{asset}")

    if not os.path.isdir(folder):
        print("  Folder NOT FOUND")
        continue

    frames = get_frame_files(folder)

    print(f"  Frames: {len(frames)}")

    for frame in frames[:5]:
        print(f"    {frame}")

    if len(frames) > 5:
        print(f"    ... and {len(frames) - 5} more")


print("\n" + "=" * 60)