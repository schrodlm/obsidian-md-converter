from .base import FrontMatter
from utils import parse_date


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

    def apply_slug_transform(self, slug: str) -> str:
        """Apply Jekyll post naming convention.

        Jekyll's '_posts' collection requires filenames like
        YYYY-MM-DD-title.md. When layout is 'post', this prepends
        the date from front matter to the slug.
        """
        if self._metadata.get("layout") != "post":
            return slug

        date_str = self._metadata.get("date")
        if not date_str:
            raise ValueError("Missing 'date' field for post layout")

        parsed_date = parse_date(date_str)
        if parsed_date is None:
            raise ValueError("Invalid date format in front matter for post layout")

        date_prefix = f"{parsed_date.year:04d}-{parsed_date.month:02d}-{parsed_date.day:02d}"
        return f"{date_prefix}-{slug}"
