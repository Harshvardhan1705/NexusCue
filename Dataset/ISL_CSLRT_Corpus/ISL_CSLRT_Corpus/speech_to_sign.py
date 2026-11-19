import csv
import os
import re
import sys


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = r".\corpus_csv_files\ISL Corpus sign glosses.csv"
ASSET_PATH = r".\Frames_Word_Level"


# Known differences between gloss tokens and asset names
ALIASES = {
    "HI": "HELLO_HI",
    "HELLO": "HELLO_HI",

    "LOVE": "LIKE_LOVE",

    "COLLEGE": "COLLEGE_SCHOOL",
    "SCHOOL": "COLLEGE_SCHOOL",

    "I": "I_ME_MINE_MY",
    "ME": "I_ME_MINE_MY",
    "MY": "I_ME_MINE_MY",

    "AGE": "OLD_AGE",

    "SOMEONE": "SOME ONE",
    "SOMEHOW": "SOME HOW",

    "MEAN": "MEAN IT",

    "TAKE_CARE": "TAKE CARE",
    "ON_THE_WAY": "ON THE WAY",

    "SO_MUCH": "SO MUCH",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Normalize English sentence for dataset lookup."""

    text = text.lower().strip()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# LOAD GLOSS DICTIONARY
# ============================================================

def load_gloss_dictionary():

    dictionary = {}

    with open(CSV_PATH, encoding="utf-8-sig") as file:

        rows = csv.DictReader(file)

        for row in rows:

            sentence = normalize_text(row["Sentence"])
            gloss = row["SIGN GLOSSES"].strip()

            dictionary[sentence] = gloss

    return dictionary


# ============================================================
# LOAD SIGN ASSETS
# ============================================================

def load_assets():

    assets = set()

    for folder in os.listdir(ASSET_PATH):

        full_path = os.path.join(ASSET_PATH, folder)

        if os.path.isdir(full_path):
            assets.add(folder.upper())

    return assets


# ============================================================
# FIND ASSET
# ============================================================

def find_asset(token, assets):

    token = token.strip().upper()

    # Direct match
    if token in assets:
        return token

    # Alias match
    if token in ALIASES:

        alias = ALIASES[token].upper()

        if alias in assets:
            return alias

    return None


# ============================================================
# GLOSS → ASSET SEQUENCE
# ============================================================

def map_gloss_to_assets(gloss, assets):

    tokens = gloss.split()

    sequence = []

    for token in tokens:

        asset = find_asset(token, assets)

        sequence.append({
            "gloss": token,
            "asset": asset
        })

    return sequence


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_sentence(sentence):

    print("\n" + "=" * 60)

    print("SPEECH-TO-SIGN PIPELINE")

    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: English input
    # --------------------------------------------------------

    print("\nEnglish Input:")
    print(sentence)

    # --------------------------------------------------------
    # STEP 2: Gloss lookup
    # --------------------------------------------------------

    dictionary = load_gloss_dictionary()

    normalized_sentence = normalize_text(sentence)

    gloss = dictionary.get(normalized_sentence)

    if not gloss:

        print("\nISL Gloss:")
        print("NOT FOUND")

        print("\nThis sentence is not available in the")
        print("current ISL-CSLRT sentence mappings.")

        return

    print("\nISL Gloss:")
    print(gloss)

    # --------------------------------------------------------
    # STEP 3: Load assets
    # --------------------------------------------------------

    assets = load_assets()

    # --------------------------------------------------------
    # STEP 4: Map gloss → assets
    # --------------------------------------------------------

    sequence = map_gloss_to_assets(gloss, assets)

    print("\nSign Sequence:")
    print("-" * 60)

    found = 0
    missing = 0

    for index, item in enumerate(sequence, start=1):

        token = item["gloss"]
        asset = item["asset"]

        if asset:

            print(
                f"{index:02}. {token:<20} → {asset}"
            )

            found += 1

        else:

            print(
                f"{index:02}. {token:<20} → NOT FOUND"
            )

            missing += 1

    # --------------------------------------------------------
    # STEP 5: Summary
    # --------------------------------------------------------

    print("\n" + "-" * 60)

    print(f"Total gloss tokens : {len(sequence)}")
    print(f"Assets found       : {found}")
    print(f"Assets missing     : {missing}")

    print("=" * 60)

# ============================================================
# FUNCTION FOR VIDEO GENERATION
# ============================================================

def get_sign_assets(sentence):

    dictionary = load_gloss_dictionary()

    normalized_sentence = normalize_text(sentence)

    gloss = dictionary.get(normalized_sentence)

    if not gloss:
        return {
            "gloss": None,
            "sequence": []
        }

    assets = load_assets()

    sequence = map_gloss_to_assets(gloss, assets)

    return {
        "gloss": gloss,
        "sequence": sequence
    }


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            'Usage: py speech_to_sign.py '
            '"are you free today"'
        )

        sys.exit()

    sentence = " ".join(sys.argv[1:])

    process_sentence(sentence)