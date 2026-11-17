from .base import FrontMatter
from .jekyll import JekyllFrontMatter


def front_matter_class(fm_config: dict) -> type[FrontMatter]:
    """The FrontMatter class (base or subclass) that the front_matter section's `type` names."""
    if fm_config.get('type', 'base') == 'jekyll':
        return JekyllFrontMatter
    return FrontMatter


def create_front_matter(content: str, config: dict) -> FrontMatter:
    #TODO: Move this whole mechanism into Config
    # Meaning we should introduce classes like FrontMatterConfig, JekkylFrontMatterConfig.
    # These classes will be able to create their own FrontMatter manipulator, because they know to which front matter they belong to
    """Factory method to create the appropriate FrontMatter instance based on config.

    Args:
        content: Full markdown file content.
        config: The full configuration dictionary (from config.yaml).

    Returns:
        A FrontMatter instance (base or subclass) based on config['front_matter']['type'].
    """
    fm_config = config.get('front_matter', {})
    return front_matter_class(fm_config)(content, fm_config)
