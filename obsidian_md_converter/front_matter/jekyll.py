from .base import FrontMatter


class JekyllFrontMatter(FrontMatter):
    """Front matter handler with Jekyll-specific defaults.

    Jekyll uses 'layout' to select an HTML template for rendering a page.
    When auto_add is on, a page whose front matter names no layout gets the default one.
    """

    def __init__(self, content: str, config: dict = None):
        jekyll_config = (config or {}).get('jekyll', {})
        self._default_layout = jekyll_config.get('default_layout', 'note')
        self._auto_add = jekyll_config.get('auto_add', False)
        super().__init__(content)

    def defaults(self) -> dict[str, str]:
        """The default layout when auto_add is on: a page whose front matter names no layout renders without the
        site's template."""
        return {'layout': self._default_layout} if self._auto_add else {}
