
from pathlib import Path
import csv
import cv2

from sentence_sequence_engine import get_sentence_frames

# ============================================================
# NEXUSCUE - COMPLETE 101 SENTENCE VALIDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GLOSS_CSV = (
    BASE_DIR
    / "corpus_csv_files"
    / "ISL Corpus sign glosses.csv"
)

REPORT_PATH = BASE_DIR / "nexcue_all_sentence_validation.csv"


def main():

    print("=" * 75)
    print("NEXUSCUE - COMPLETE SENTENCE SEQUENCE VALIDATION")
    print("=" * 75)

    if not GLOSS_CSV.exists():
        print(f"\nERROR: Gloss CSV not found: {GLOSS_CSV}")
        return

    results = []

    total = 0
    found = 0
    missing = 0
    unreadable = 0
    total_frames = 0

    with open(
        GLOSS_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            sentence = (row.get("Sentence") or "").strip()
            gloss = (row.get("SIGN GLOSSES") or "").strip()

            if not sentence:
                continue

            total += 1

            frames = get_sentence_frames(sentence)

            bad_frames = []

            if not frames:
                status = "MISSING"
                missing += 1

            else:
                total_frames += len(frames)

                for frame in frames:
                    image = cv2.imread(str(frame))

                    if image is None:
                        bad_frames.append(frame.name)

                if bad_frames:
                    status = "UNREADABLE FRAMES"
                    unreadable += 1
                else:
                    status = "READY"
                    found += 1

            folder = str(frames[0].parent) if frames else ""

            results.append({
                "sentence": sentence,
                "gloss": gloss,
                "status": status,
                "frame_count": len(frames),
                "unreadable_count": len(bad_frames),
                "unreadable_files": "; ".join(bad_frames),
                "frame_folder": folder,
                "first_frame": frames[0].name if frames else "",
                "last_frame": frames[-1].name if frames else ""
            })

            print(f"{total:03d}. {sentence}")
            print(f"     Status: {status}")
            print(f"     Frames: {len(frames)}")

            if bad_frames:
                print(f"     Unreadable: {len(bad_frames)}")

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    fieldnames = [
        "sentence",
        "gloss",
        "status",
        "frame_count",
        "unreadable_count",
        "unreadable_files",
        "frame_folder",
        "first_frame",
        "last_frame"
    ]

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 75)

    print(f"Total sentences:          {total}")
    print(f"Sequences READY:          {found}")
    print(f"Missing sequences:        {missing}")
    print(f"Sequences with unreadable frames: {unreadable}")
    print(f"Total frames inspected:   {total_frames}")

    print(f"\nValidation report saved:")
    print(REPORT_PATH)

    print("\nOriginal dataset and gloss CSV were not modified.")

    if found == total and unreadable == 0:
        print("\nALL SENTENCES PASSED FRAME VALIDATION.")

    print("=" * 75)


if __name__ == "__main__":
    main()
