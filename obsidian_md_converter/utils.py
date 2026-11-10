import datetime
import re
import unicodedata
from pathlib import Path


def parse_date(date_str: str) -> datetime.date | None:
    """
    Parse a date string in various formats into a datetime.date object.

    Supported formats:
    - YYYY-MM-DD (2024-12-20)
    - DD.MM.YYYY (20.12.2024)
    - DD/MM/YYYY (20/12/2024)
    - MM/DD/YYYY (12/20/2024)
    - Month DD, YYYY (December 20, 2024)
    - DD Month YYYY (20 December 2024)
    - YYYYMMDD (20241220)

    Returns:
        datetime.date object if parsing succeeds, None otherwise
    """
    if not date_str:
        return None
    date_str = date_str.strip()

    # Try common formats in order
    formats = [
        '%Y-%m-%d',    # 2024-12-20
        '%d.%m.%Y',    # 20.12.2024
        '%d/%m/%Y',    # 20/12/2024
        '%m/%d/%Y',    # 12/20/2024
        '%B %d, %Y',   # December 20, 2024
        '%d %B %Y',    # 20 December 2024
        '%Y%m%d',      # 20241220
    ]

    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None

def slugify_text(text: str) -> str:
    """
    Transform text into a URL-safe slug: lowercase ASCII letters and digits joined by single hyphens.

    Example:
        "ABA problém (2)" -> "aba-problem-2"
    """
    # Decomposing splits an accented letter into the letter and its accent, so dropping
    # everything non-ASCII keeps the letter ("é" -> "e").
    text = unicodedata.normalize('NFKD', text.strip())
    text = ''.join(char for char in text if char.isascii())

    text = re.sub(r'[^\w\s-]', '', text)  # Remove remaining non-word chars
    text = re.sub(r'[\s_-]+', '-', text)  # Convert spaces/underscores to hyphens

    return text.strip('-').lower()

def enforce_dir(path: Path):
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)


def nested_get(d: dict, *keys):
    """Safely traverse nested dict keys, return None if any key is missing.

    Example:
        nested_get(config, 'paths', 'obsidian_root')  # config['paths']['obsidian_root'] or None
    """
    value = d
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value