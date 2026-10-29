import datetime

from .base import FrontMatter
from obsidian_md_converter.utils import parse_date


class JekyllFrontMatter(FrontMatter):
    """Front matter handler with Jekyll-specific defaults.

    Jekyll uses 'layout' to select an HTML template for rendering a page.
    When creating front matter from scratch, this class adds a default
    layout field alongside the title.
    """

    def __init__(self, content: str, config: dict = None):
        jekyll_config = (config or {}).get('jekyll', {})
        self._default_layout = jekyll_config.get('default_layout', 'note')

        # Merge auto_add from jekyll section into the base config
        merged_config = dict(config or {})
        merged_config['auto_add'] = jekyll_config.get('auto_add', False)

        super().__init__(content, merged_config)

    def _create_front_matter(self, title: str):
        """Create front matter with layout and title.

        Overrides base to include a Jekyll layout field.
        """
        self._has_front_matter = True
        self._raw_front_matter = f'layout: {self._default_layout}\ntitle: "{title}"'
        self._metadata = {'layout': self._default_layout, 'title': title}
        if not self._body.startswith('\n'):
            self._body = '\n' + self._body

    def ensure(self, title: str):
        """Ensure front matter has a title and, when auto_add is enabled, a layout.

        A page whose front matter names no layout renders without the site's template.
        """
        super().ensure(title)
        if self._has_front_matter and 'layout' not in self._metadata and self._config.get('auto_add', False):
            self._raw_front_matter += f'\nlayout: {self._default_layout}'
            self._metadata['layout'] = self._default_layout

    def apply_slug_transform(self, slug: str) -> str:
        """Apply Jekyll post naming convention.

        Jekyll's '_posts' collection requires filenames like
        YYYY-MM-DD-title.md. When layout is 'post', this prepends
        the date from front matter to the slug.
        """
        if self._metadata.get("layout") != "post":
            return slug

        date_value = self._metadata.get("date")
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
