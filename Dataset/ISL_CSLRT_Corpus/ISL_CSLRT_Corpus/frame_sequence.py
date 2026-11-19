import os
import re


BASE_PATH = r".\Frames_Word_Level"


def natural_sort_key(filename):
    """
    Sort filenames naturally:
    frame 1, frame 2, frame 3 ... frame 10
    instead of:
    frame 1, frame 10, frame 11 ... frame 2
    """

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", filename)
    ]


def get_frames(asset_name):
    """
    Get all image frames belonging to a sign asset.
    """

    folder = os.path.join(BASE_PATH, asset_name)

    if not os.path.isdir(folder):
        return []

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    frames = []

    for filename in os.listdir(folder):

        extension = os.path.splitext(filename)[1].lower()

        if extension in valid_extensions:
            frames.append(
                os.path.join(folder, filename)
            )

    return sorted(frames, key=natural_sort_key)


def build_sign_sequence(assets):
    """
    Combine frame sequences of multiple signs.
    """

    sequence = []

    for asset in assets:

        frames = get_frames(asset)

        if not frames:
            print(f"WARNING: No frames found for {asset}")
            continue

        print(
            f"{asset:<20} → {len(frames)} frames"
        )

        sequence.extend(frames)

    return sequence


if __name__ == "__main__":

    assets = [
        "YOU",
        "FREE",
        "TODAY"
    ]

    print("=" * 60)
    print("SIGN FRAME SEQUENCE")
    print("=" * 60)

    sequence = build_sign_sequence(assets)

    print("\n" + "-" * 60)

    print(
        f"Total animation frames: {len(sequence)}"
    )

    print("\nFirst 10 frames:")

    for frame in sequence[:10]:
        print(frame)

    print("\nLast 5 frames:")

    for frame in sequence[-5:]:
        print(frame)

    print("=" * 60)