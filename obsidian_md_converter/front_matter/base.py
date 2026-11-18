import json

import yaml

from obsidian_md_converter import scanning


def yaml_fields(fields: dict[str, str]) -> str:
    """Fields the converter adds, as front matter lines. Each value is written as a double-quoted YAML string and
    escaped, so any text stays valid YAML. Example: {'title': 'Say "hi"'} -> 'title: "Say \\"hi\\""'"""
    # A JSON string is a valid double-quoted YAML string.
    return '\n'.join(f'{key}: {json.dumps(value, ensure_ascii=False)}' for key, value in fields.items())


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
        raw, body = scanning.split_front_matter(content)
        if raw is None:
            return

        self._has_front_matter = True
        self._raw_front_matter = raw
        self._body = body
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
        self._raw_front_matter += '\n' + yaml_fields({'title': title})
        self._metadata['title'] = title

    def _create_front_matter(self, title: str):
        """Create minimal front matter with just a title."""
        self._has_front_matter = True
        self._raw_front_matter = yaml_fields({'title': title})
        self._metadata = {'title': title}
        if not self._body.startswith('\n'):
            self._body = '\n' + self._body

    def ensure(self, title: str):
        """Ensure the front matter has a title; a note without front matter gets one holding just the title.

        Args:
            title: Title to use (typically the original filename before slugification).
        """
        if self._has_front_matter:
            self._ensure_title(title)
        else:
            self._create_front_matter(title)

    def serialize(self) -> str:
        """Reconstruct the full file content from front matter and body."""
        if not self._has_front_matter:
            return self._body
        return f"---\n{self._raw_front_matter}\n---\n{self._body}"
