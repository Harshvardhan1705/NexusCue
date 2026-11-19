from pathlib import Path
import csv
import re
from collections import defaultdict

import cv2

from gloss_engine import normalize_text

# ============================================================
# NEXUSCUE - BATCH SENTENCE SEQUENCE VALIDATOR
# Validates all sentence entries against the bulk manifest.
# Does NOT modify the dataset, manifest, or gloss CSV.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GLOSS_CSV = (
    BASE_DIR
    / "corpus_csv_files"
    / "ISL Corpus sign glosses.csv"
)

MANIFEST_CSV = (
    BASE_DIR
    / "bulk_sentence_sequences"
    / "bulk_sequence_manifest.csv"
)

PREFERRED_GROUP = "1"

REPORT_CSV = BASE_DIR / "sentence_sequence_validation_report.csv"
GROUP_REPORT_CSV = BASE_DIR / "sentence_sequence_group_report.csv"


def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def resolve_manifest_folder(raw_folder):
    """Resolve absolute or manifest-relative sequence folder paths."""
    if not raw_folder:
        return None

    folder = Path(raw_folder.strip())

    if folder.is_absolute():
        return folder

    # First try relative to the project directory.
    candidate = (BASE_DIR / folder).resolve()
    if candidate.exists():
        return candidate

    # Then try relative to the manifest directory.
    candidate = (MANIFEST_CSV.parent / folder).resolve()
    if candidate.exists():
        return candidate

    return candidate


def load_gloss_sentences():
    if not GLOSS_CSV.exists():
        raise FileNotFoundError(
            f"Gloss CSV not found:\n{GLOSS_CSV}"
        )

    sentences = []
    seen = set()

    with open(
        GLOSS_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {"Sentence", "SIGN GLOSSES"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Gloss CSV must contain columns: Sentence and SIGN GLOSSES"
            )

        for row in reader:
            sentence = (row.get("Sentence") or "").strip()
            gloss = (row.get("SIGN GLOSSES") or "").strip()
            key = normalize_text(sentence)

            if not sentence or not key or key in seen:
                continue

            seen.add(key)
            sentences.append({
                "sentence": sentence,
                "normalized": key,
                "gloss": gloss
            })

    return sentences


def load_manifest():
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"Bulk manifest not found:\n{MANIFEST_CSV}"
        )

    sequences = defaultdict(dict)

    with open(
        MANIFEST_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {"sentence", "signer_group", "sequence_folder"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Manifest must contain columns: "
                "sentence, signer_group, sequence_folder"
            )

        for row in reader:
            sentence = normalize_text(row.get("sentence") or "")
            group = str(row.get("signer_group") or "").strip()
            folder = resolve_manifest_folder(
                row.get("sequence_folder") or ""
            )

            if not sentence or not group:
                continue

            sequences[sentence][group] = folder

    return sequences


def inspect_group_folder(folder):
    """Return image count and unreadable image details for one group."""
    if folder is None:
        return {
            "folder_exists": False,
            "frame_count": 0,
            "unreadable_count": 0,
            "unreadable_files": [],
            "first_frame": "",
            "last_frame": ""
        }

    if not folder.exists() or not folder.is_dir():
        return {
            "folder_exists": False,
            "frame_count": 0,
            "unreadable_count": 0,
            "unreadable_files": [],
            "first_frame": "",
            "last_frame": ""
        }

    frames = sorted(
        [
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ],
        key=natural_sort_key
    )

    unreadable = []
    for frame_path in frames:
        image = cv2.imread(str(frame_path))
        if image is None or image.size == 0:
            unreadable.append(frame_path.name)

    return {
        "folder_exists": True,
        "frame_count": len(frames),
        "unreadable_count": len(unreadable),
        "unreadable_files": unreadable,
        "first_frame": frames[0].name if frames else "",
        "last_frame": frames[-1].name if frames else ""
    }


def choose_group(groups):
    if PREFERRED_GROUP in groups:
        return PREFERRED_GROUP

    def group_key(group):
        match = re.search(r"\d+", group)
        return int(match.group()) if match else 999999

    return sorted(groups.keys(), key=group_key)[0] if groups else ""


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 72)
    print("NEXUSCUE - BATCH SENTENCE SEQUENCE VALIDATOR")
    print("=" * 72)

    try:
        sentences = load_gloss_sentences()
        manifest = load_manifest()
    except Exception as error:
        print(f"\nVALIDATION COULD NOT START: {error}")
        return

    print(f"\nGloss CSV sentences: {len(sentences)}")
    print(f"Manifest sentence entries: {len(manifest)}")
    print(f"Preferred signer group: {PREFERRED_GROUP}")
    print("\nValidating folders and image files...\n")

    sentence_rows = []
    group_rows = []

    ready_count = 0
    missing_count = 0
    unreadable_sentence_count = 0
    total_valid_frames = 0
    total_groups = 0

    for index, item in enumerate(sentences, start=1):
        sentence = item["sentence"]
        normalized = item["normalized"]
        groups = manifest.get(normalized, {})

        group_results = []
        for group, folder in sorted(
            groups.items(),
            key=lambda pair: (
                int(re.search(r"\d+", pair[0]).group())
                if re.search(r"\d+", pair[0]) else 999999
            )
        ):
            result = inspect_group_folder(folder)
            total_groups += 1
            total_valid_frames += (
                result["frame_count"] - result["unreadable_count"]
            )

            group_rows.append({
                "sentence": sentence,
                "signer_group": group,
                "folder": str(folder) if folder else "",
                "folder_exists": result["folder_exists"],
                "frame_count": result["frame_count"],
                "unreadable_count": result["unreadable_count"],
                "unreadable_files": "; ".join(result["unreadable_files"]),
                "first_frame": result["first_frame"],
                "last_frame": result["last_frame"]
            })

            group_results.append((group, folder, result))

        selected_group = choose_group(groups)
        selected = None

        if selected_group:
            for group, folder, result in group_results:
                if group == selected_group:
                    selected = (folder, result)
                    break

        if not groups:
            status = "NO MANIFEST ENTRY"
            selected_frame_count = 0
            selected_unreadable = 0
            selected_folder = ""
            missing_count += 1
        elif selected is None or not selected[1]["folder_exists"]:
            status = "MISSING SELECTED FOLDER"
            selected_frame_count = 0
            selected_unreadable = 0
            selected_folder = str(selected[0]) if selected else ""
            missing_count += 1
        else:
            selected_folder, selected_info = selected
            selected_frame_count = selected_info["frame_count"]
            selected_unreadable = selected_info["unreadable_count"]

            if selected_frame_count == 0:
                status = "EMPTY SEQUENCE"
                missing_count += 1
            elif selected_unreadable > 0:
                status = "UNREADABLE FRAMES"
                unreadable_sentence_count += 1
            else:
                status = "READY"
                ready_count += 1

        sentence_rows.append({
            "sentence_number": index,
            "sentence": sentence,
            "gloss": item["gloss"],
            "available_groups": len(groups),
            "selected_group": selected_group,
            "selected_frame_count": selected_frame_count,
            "unreadable_frames": selected_unreadable,
            "selected_folder": str(selected_folder) if selected_folder else "",
            "status": status
        })

        print(
            f"{index:03d}. {sentence} | "
            f"Groups: {len(groups)} | "
            f"Selected frames: {selected_frame_count} | {status}"
        )

    sentence_fields = [
        "sentence_number", "sentence", "gloss", "available_groups",
        "selected_group", "selected_frame_count", "unreadable_frames",
        "selected_folder", "status"
    ]

    group_fields = [
        "sentence", "signer_group", "folder", "folder_exists",
        "frame_count", "unreadable_count", "unreadable_files",
        "first_frame", "last_frame"
    ]

    write_csv(REPORT_CSV, sentence_rows, sentence_fields)
    write_csv(GROUP_REPORT_CSV, group_rows, group_fields)

    print("\n" + "=" * 72)
    print("BATCH VALIDATION SUMMARY")
    print("=" * 72)
    print(f"Total unique CSV sentences: {len(sentences)}")
    print(f"Sentences READY: {ready_count}")
    print(f"Sentences with missing/empty sequences: {missing_count}")
    print(f"Sentences with unreadable selected frames: {unreadable_sentence_count}")
    print(f"Total manifest signer-group entries inspected: {total_groups}")
    print(f"Total readable frames across all groups: {total_valid_frames}")
    print(f"\nSentence report: {REPORT_CSV}")
    print(f"Group report: {GROUP_REPORT_CSV}")
    print("\nOriginal dataset, manifest, and gloss CSV were not modified.")
    print("=" * 72)


if __name__ == "__main__":
    main()
