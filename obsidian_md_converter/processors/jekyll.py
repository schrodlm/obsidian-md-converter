import datetime

from obsidian_md_converter.utils import parse_date


class JekyllProcessor:
    """How a run's pages are expressed for a Jekyll site."""

    @classmethod
    def apply_slug_transform(cls, slug: str, metadata: dict) -> str:
        """Apply Jekyll post naming convention.

        Jekyll's '_posts' collection requires filenames like
        YYYY-MM-DD-title.md. When layout is 'post', this prepends
        the date from front matter to the slug.
        """
        if metadata.get("layout") != "post":
            return slug

        date_value = metadata.get("date")
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
        return f"{date_prefix}-{slug}"
