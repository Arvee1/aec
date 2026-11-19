import unicodedata


_ALLOWED_NAME_PUNCTUATION = {"'", "’", "-", "‐"}


def validate_name(name: str) -> str:
    """Normalize and validate a user's display name.

    Names may contain Unicode letters, spaces, apostrophes, and hyphens. The
    normalized value is returned so callers store and display the same name.
    """
    normalized = " ".join(unicodedata.normalize("NFC", name).split())

    if not normalized:
        raise ValueError("Please enter your name to continue.")
    if len(normalized) < 2:
        raise ValueError("Name must be at least 2 characters long.")
    if len(normalized) > 100:
        raise ValueError("Name must be 100 characters or fewer.")
    if any(
        not (
            unicodedata.category(character)[0] in {"L", "M"}
            or character == " "
            or character in _ALLOWED_NAME_PUNCTUATION
        )
        for character in normalized
    ):
        raise ValueError("Use letters, spaces, apostrophes, and hyphens only.")

    has_letter = any(
        unicodedata.category(character).startswith("L") for character in normalized
    )
    if not has_letter:
        raise ValueError("Name must contain at least one letter.")

    return normalized
