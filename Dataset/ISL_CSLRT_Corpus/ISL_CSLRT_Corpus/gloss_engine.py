
import csv
import sys
import re
from pathlib import Path
from functools import lru_cache


# ============================================================
# NEXUSCUE - ENGLISH TO ISL GLOSS ENGINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "corpus_csv_files"
    / "ISL Corpus sign glosses.csv"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize English input for reliable dataset lookup.

    Handles:
    - Uppercase and lowercase differences
    - Extra spaces
    - Common punctuation
    - Curly apostrophes
    - Hyphenated expressions
    """

    if not text:
        return ""

    text = str(text).strip().lower()

    # Standardize curly apostrophes
    text = text.replace("’", "'")
    text = text.replace("‘", "'")

    # Replace punctuation with spaces, preserving
    # apostrophes inside words such as don't
    text = re.sub(r"[^\w\s']", " ", text)

    # Remove standalone apostrophes
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOAD DATASET GLOSS DICTIONARY
# ============================================================

@lru_cache(maxsize=1)
def load_gloss_dictionary():
    """
    Load English sentence -> ISL gloss mappings
    from the ISL-CSLRT dataset CSV.

    The dictionary is cached in memory so the CSV
    does not need to be loaded repeatedly.
    """

    gloss_dictionary = {}

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"ISL gloss dataset not found:\n{CSV_PATH}"
        )

    with open(
        CSV_PATH,
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = csv.DictReader(file)

        required_columns = {
            "Sentence",
            "SIGN GLOSSES"
        }

        available_columns = set(rows.fieldnames or [])

        missing_columns = required_columns - available_columns

        if missing_columns:

            raise ValueError(
                "Missing required CSV columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in rows:

            sentence = normalize_text(
                row.get("Sentence", "")
            )

            gloss = (
                row.get("SIGN GLOSSES") or ""
            ).strip()

            # Ignore incomplete rows
            if not sentence or not gloss:
                continue

            gloss_dictionary[sentence] = gloss

    return gloss_dictionary


# ============================================================
# GET ISL GLOSS
# ============================================================

def get_gloss(sentence):
    """
    Convert an English sentence into its corresponding
    dataset-provided ISL gloss.

    Uses exact normalized matching.

    Returns:
        str  -> Gloss found
        None -> No exact dataset mapping found

    No artificial or guessed gloss is generated.
    """

    if not sentence or not str(sentence).strip():
        return None

    dictionary = load_gloss_dictionary()

    normalized_sentence = normalize_text(sentence)

    return dictionary.get(normalized_sentence)


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_info():
    """
    Return basic information about the loaded
    sentence-to-gloss dictionary.
    """

    dictionary = load_gloss_dictionary()

    return {
        "dataset_file": CSV_PATH.name,
        "total_mappings": len(dictionary)
    }


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    if len(sys.argv) < 2:

        print("\nUsage:")
        print('py gloss_engine.py "are you free today"')

        return

    sentence = " ".join(sys.argv[1:])

    try:

        gloss = get_gloss(sentence)

        dataset_info = get_dataset_info()

        print("\n" + "=" * 60)
        print("NEXUSCUE - ENGLISH TO ISL GLOSS ENGINE")
        print("=" * 60)

        print("\nEnglish Input:")
        print(sentence)

        print("\nDataset Information:")
        print(
            f"Available mappings: "
            f"{dataset_info['total_mappings']}"
        )

        print("\nISL Gloss:")

        if gloss:

            print(gloss)

            print("\nStatus: EXACT DATASET MATCH")

        else:

            print("NOT FOUND")

            print(
                "\nThis sentence is not available as an exact "
                "sentence in the current ISL-CSLRT dataset."
            )

            print(
                "The system will use the existing "
                "word-level fallback."
            )

        print("\n" + "=" * 60)

    except (FileNotFoundError, ValueError) as error:

        print(f"\n[DATASET ERROR] {error}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
