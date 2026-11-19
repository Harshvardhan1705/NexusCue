import os

# Location of word-level sign assets
ASSET_PATH = r".\Frames_Word_Level"


# Known aliases between gloss tokens and dataset assets.
# These are based on the actual asset names in this dataset.
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
    "MEAN_IT": "MEAN IT",

    "TAKE_CARE": "TAKE CARE",
    "ON_THE_WAY": "ON THE WAY",

    "SO_MUCH": "SO MUCH",
}


def load_assets():
    """Load all available sign asset folder names."""

    assets = set()

    for folder in os.listdir(ASSET_PATH):
        full_path = os.path.join(ASSET_PATH, folder)

        if os.path.isdir(full_path):
            assets.add(folder.upper())

    return assets


def find_asset(gloss_token, assets):
    """
    Find the dataset asset corresponding to a gloss token.
    """

    token = gloss_token.strip().upper()

    # 1. Direct match
    if token in assets:
        return token

    # 2. Alias match
    if token in ALIASES:
        alias = ALIASES[token].upper()

        if alias in assets:
            return alias

    return None


def map_gloss_to_assets(gloss):
    """
    Convert an ISL gloss sequence into available sign assets.
    """

    assets = load_assets()

    tokens = gloss.split()

    result = []

    for token in tokens:

        asset = find_asset(token, assets)

        result.append({
            "gloss": token,
            "asset": asset
        })

    return result


def print_mapping(gloss):
    print("\nISL Gloss:")
    print(gloss)

    print("\nAsset Mapping:")
    print("-" * 50)

    results = map_gloss_to_assets(gloss)

    for item in results:

        if item["asset"]:
            print(
                f'✓ {item["gloss"]:<20} → {item["asset"]}'
            )
        else:
            print(
                f'✗ {item["gloss"]:<20} → NOT FOUND'
            )


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print(
            'Usage: py asset_mapper.py "YOU FREE TODAY"'
        )
        sys.exit()

    gloss = " ".join(sys.argv[1:])

    print_mapping(gloss)