
from pathlib import Path
import cv2

# ==========================================================
# NEXUSCUE - CANDIDATE SENTENCE INSPECTOR
# Read-only inspection of folders and source videos
# ==========================================================

ROOT = Path(__file__).resolve().parent

FRAME_ROOT = ROOT / "Frames_Sentence_Level"
VIDEO_ROOT = ROOT / "Videos_Sentence_Level"

CANDIDATES = [
    "are you free today",
    "how are things",
    "it was nice chatting with you",
    "what is your phone number",
    "could you please talk slower",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def inspect_frame_folder(sentence):

    folder = FRAME_ROOT / sentence

    print("\nFRAME DIRECTORY STRUCTURE")

    if not folder.exists():
        print("Frame folder not found.")
        return

    print(f"Root: {folder}")

    subfolders = sorted(
        p for p in folder.iterdir()
        if p.is_dir()
    )

    if not subfolders:
        images = [
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"Images directly/recursively found: {len(images)}")
        return

    for subfolder in subfolders:

        images = [
            p for p in subfolder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(
            f"{subfolder.name}: "
            f"{len(images)} images"
        )


def inspect_videos(sentence):

    folder = VIDEO_ROOT / sentence

    print("\nSOURCE VIDEO DETAILS")

    if not folder.exists():
        print("Video folder not found.")
        return

    videos = sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    for video in videos:

        cap = cv2.VideoCapture(str(video))

        if not cap.isOpened():
            print(f"\n{video.name}: UNABLE TO OPEN")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = total / fps if fps > 0 else 0

        print(f"\nFile: {video.name}")
        print(f"Size: {video.stat().st_size / 1024:.1f} KB")
        print(f"Resolution: {width} x {height}")
        print(f"FPS: {fps:.2f}")
        print(f"Frames: {total}")
        print(f"Duration: {duration:.2f} seconds")

        cap.release()


def main():

    print("=" * 65)
    print("NEXUSCUE - SENTENCE CANDIDATE INSPECTION")
    print("=" * 65)

    for number, sentence in enumerate(CANDIDATES, start=1):

        print("\n" + "=" * 65)
        print(f"CANDIDATE {number}: {sentence}")
        print("=" * 65)

        inspect_frame_folder(sentence)
        inspect_videos(sentence)

    print("\n" + "=" * 65)
    print("INSPECTION COMPLETED")
    print("No original dataset files were modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
