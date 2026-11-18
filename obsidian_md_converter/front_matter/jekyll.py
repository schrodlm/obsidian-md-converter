from .base import FrontMatter, yaml_fields


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
        self._raw_front_matter = yaml_fields({'title': title, 'layout': self._default_layout})
        self._metadata = {'title': title, 'layout': self._default_layout}
        if not self._body.startswith('\n'):
            self._body = '\n' + self._body

    def ensure(self, title: str):
        """Ensure front matter has a title and, when auto_add is enabled, a layout.

        A page whose front matter names no layout renders without the site's template.
        """
        super().ensure(title)
        if self._has_front_matter and 'layout' not in self._metadata and self._config.get('auto_add', False):
            self._raw_front_matter += '\n' + yaml_fields({'layout': self._default_layout})
            self._metadata['layout'] = self._default_layout
