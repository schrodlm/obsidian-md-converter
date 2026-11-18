import datetime
from pathlib import Path

from obsidian_md_converter.processors.base import Processor
from obsidian_md_converter.utils import parse_date


class JekyllProcessor(Processor):
    """How a run's pages are expressed for a Jekyll site."""

    def page_path(self, neutral_path: Path, front_matter: dict) -> Path:
        """
        The neutral path, except that a post gets the dated name Jekyll's '_posts' collection requires.

        Example: 'turing-machine.md' with 'layout: post' and 'date: 2024-12-20' -> '2024-12-20-turing-machine.md'
        """
        if front_matter.get("layout") != "post":
            return neutral_path

        date_value = front_matter.get("date")
        if not date_value:
            raise ValueError("Missing 'date' field for post layout")

        # YAML reads an ISO date as a date object; other formats arrive as text.
        if isinstance(date_value, datetime.date):
            parsed_date = date_value
        else:
            parsed_date = parse_date(str(date_value))
        if parsed_date is None:
            raise ValueError("Invalid date format in front matter for post layout")

        date_prefix = f"{parsed_date.year:04d}-{parsed_date.month:02d}-{parsed_date.day:02d}"
        return neutral_path.with_name(f"{date_prefix}-{neutral_path.name}")
