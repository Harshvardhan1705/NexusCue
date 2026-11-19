# ============================================================
# FALLBACK HANDLER
# ============================================================

def fingerspell_word(word):
    """
    Temporary fingerspelling fallback.

    Currently this only returns the letters.
    Actual A-Z sign assets will be connected later.
    """

    word = word.upper().strip()

    letters = list(word)

    print(f"\n🔤 Fingerspelling fallback: {word}")

    print("Letters:")
    print(" → ".join(letters))

    return letters


def handle_missing_token(token):

    print(f"\n⚠️ No sign asset found for: {token}")

    letters = fingerspell_word(token)

    return {
        "token": token,
        "method": "FINGERSPELLING",
        "letters": letters
    }