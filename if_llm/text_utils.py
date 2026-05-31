"""Text processing and cleaning utilities."""
import re
import logging

logger = logging.getLogger(__name__)


def clean_text(generated_text, remove_weights=True, remove_author=True):
    """Clean text while preserving intentional line breaks."""
    # Split into lines first to preserve breaks
    lines = generated_text.split('\n')
    cleaned_lines = []

    for line in lines:
        if line.strip():  # Only process non-empty lines
            # Remove author attribution if requested
            if remove_author:
                line = re.sub(r"\bby:.*", "", line)

            # Remove weights if requested
            if remove_weights:
                line = re.sub(r"\(([^)]*):[\d\.]*\)", r"\1", line)
                line = re.sub(r"(\w+):[\d\.]*(?=[ ,]|$)", r"\1", line)

            # Remove markup tags
            line = re.sub(r"<[^>]*>", "", line)

            # Remove lonely symbols and formatting
            line = re.sub(r"(?<=\s):(?=\s)", "", line)
            line = re.sub(r"(?<=\s);(?=\s)", "", line)
            line = re.sub(r"(?<=\s),(?=\s)", "", line)
            line = re.sub(r"(?<=\s)#(?=\s)", "", line)

            # Clean up extra spaces while preserving line structure
            line = re.sub(r"\s{2,}", " ", line)
            line = re.sub(r"\.,", ",", line)
            line = re.sub(r",,", ",", line)

            # Remove audio tags from the line
            if "<audio" in line:
                logger.debug("iF_prompt_MKR: Audio has been generated.")
                line = re.sub(r"<audio.*?>.*?</audio>", "", line)

            cleaned_lines.append(line.strip())

    # Join with newlines to preserve line structure
    return "\n".join(cleaned_lines)
