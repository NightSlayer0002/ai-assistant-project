"""Input/output safety filters using keyword-based pattern matching."""

# Substring patterns checked case-insensitively against user input
BLOCKED_PATTERNS = [
    "how to make a bomb",
    "how to hack",
    "how to make explosives",
    "synthesize drugs",
    "create a weapon",
    "bypass security",
    "steal identity",
    "make poison",
]


def check_input(text: str) -> tuple[bool, str]:
    """Checks if user input is safe to send to the model."""
    text_lower = text.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern in text_lower:
            return (False, "Input blocked: potentially harmful content detected.")

    return (True, "")


def check_output(text: str) -> tuple[bool, str]:
    """Checks if the model's output is safe to display."""
    if len(text.strip()) < 10:
        return (False, "Output suspiciously short — possible error or truncation.")

    concerning_output_patterns = [
        "here are the steps to hack",
        "here's how to make",
        "i'll help you hack",
        "step 1: obtain",
    ]

    text_lower = text.lower()
    for pattern in concerning_output_patterns:
        if pattern in text_lower:
            return (False, "Output flagged: may contain harmful instructions.")

    return (True, "")
