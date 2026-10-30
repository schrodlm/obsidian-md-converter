import re
import yaml


class FrontMatter:
    """Parses and manipulates YAML front matter from markdown content.

    Parses front matter exactly once from a content string. All operations
    work on the in-memory representation. Call serialize() to reconstruct
    the full file content.
    """

    def __init__(self, content: str, config: dict = None):
        """
        Args:
            content: Full markdown file content.
            config: The front_matter section of the configuration dictionary.
        """
        self._config = config or {}
        self._raw_front_matter: str = ''
        self._body: str = content
        self._has_front_matter: bool = False
        self._metadata: dict = {}
        self._yaml_error: str | None = None
        self._parse(content)

    def _parse(self, content: str):
        if not content.startswith('---'):
            return

        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
        if not match:
            return

        self._has_front_matter = True
        self._raw_front_matter = match.group(1)
        self._body = match.group(2)
        self._parse_metadata()

    def _parse_metadata(self):
        self._metadata = {}
        self._yaml_error = None
        try:
            parsed = yaml.safe_load(self._raw_front_matter)
        except yaml.YAMLError as e:
            self._yaml_error = self._describe_yaml_error(e)
            return
        if parsed is None:
            return
        if not isinstance(parsed, dict):
            self._yaml_error = "front matter is not a list of 'key: value' fields"
            return
        self._metadata = parsed

    @staticmethod
    def _describe_yaml_error(error: yaml.YAMLError) -> str:
        problem = getattr(error, 'problem', None) or str(error)
        mark = getattr(error, 'problem_mark', None)
        if mark is None:
            return problem
        # The mark counts from the first front matter line; the file has the opening '---' above it.
        return f"{problem} (line {mark.line + 2})"

    def apply_slug_transform(self, slug: str) -> str:
        """Apply the site generator's file-naming convention to a neutral output name; none by default.

        Raises:
            ValueError: if the front matter lacks what the convention needs.
        """
        return slug
    
    @property
    def has_front_matter(self) -> bool:
        return self._has_front_matter

    @property
    def yaml_error(self) -> str | None:
        """Why the front matter is not valid YAML, or None when it is (or there is none)."""
        return self._yaml_error

    @property
    def metadata(self) -> dict:
        """Returns a copy of the parsed metadata key-value pairs."""
        return dict(self._metadata)

    @property
    def body(self) -> str:
        return self._body

    def get(self, key: str, default=None):
        """Get a metadata value by key."""
        return self._metadata.get(key, default)

    def fix_tabs(self) -> bool:
        """Replace tab indentation with spaces in front matter.

        Tabs are not valid for YAML indentation per the YAML spec.

        Returns:
            True if tabs were found and fixed, False otherwise.
        """
        if not self._has_front_matter:
            return False
        if '\t' not in self._raw_front_matter:
            return False
        self._raw_front_matter = self._raw_front_matter.replace('\t', '  ')
        self._parse_metadata()
        return True

    def _ensure_title(self, title: str):
        """Add a title field if front matter exists but has no title.

        Preserves the original filename as title since filenames get
        slugified during conversion (e.g. "ABA problém" -> "aba-problem.md").
        """
        if 'title' in self._metadata:
            return
        self._raw_front_matter += f'\ntitle: "{title}"'
        self._metadata['title'] = title

    def _create_front_matter(self, title: str):
        """Create minimal front matter with just a title.

        Subclasses can override to add additional fields (e.g. layout).
        """
        self._has_front_matter = True
        self._raw_front_matter = f'title: "{title}"'
        self._metadata = {'title': title}
        if not self._body.startswith('\n'):
            self._body = '\n' + self._body

    def ensure(self, title: str):
        """Ensure front matter is properly configured.

        If front matter exists, ensures title is present.
        If front matter is missing and auto_add is enabled in config, creates it.

        Args:
            title: Title to use (typically the original filename before slugification).
        """
        if self._has_front_matter:
            self._ensure_title(title)
        elif self._config.get('auto_add', False):
            self._create_front_matter(title)

    def serialize(self) -> str:
        """Reconstruct the full file content from front matter and body."""
        if not self._has_front_matter:
            return self._body
        return f"---\n{self._raw_front_matter}\n---\n{self._body}"
