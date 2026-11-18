from .base import FrontMatter, yaml_fields


class JekyllFrontMatter(FrontMatter):
    """Front matter handler with Jekyll-specific defaults.

    Jekyll uses 'layout' to select an HTML template for rendering a page.
    When auto_add is on, this class adds a default layout to front matter that names none.
    """

    def __init__(self, content: str, config: dict = None):
        jekyll_config = (config or {}).get('jekyll', {})
        self._default_layout = jekyll_config.get('default_layout', 'note')

        # Merge auto_add from jekyll section into the base config
        merged_config = dict(config or {})
        merged_config['auto_add'] = jekyll_config.get('auto_add', False)

        super().__init__(content, merged_config)

    def ensure(self, title: str):
        """Ensure front matter has a title and, when auto_add is enabled, a layout.

        A page whose front matter names no layout renders without the site's template.
        """
        super().ensure(title)
        if 'layout' not in self._metadata and self._config.get('auto_add', False):
            self._raw_front_matter += '\n' + yaml_fields({'layout': self._default_layout})
            self._metadata['layout'] = self._default_layout
