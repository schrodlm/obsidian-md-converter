from .base import FrontMatter
from .jekyll import JekyllFrontMatter


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
    fm_type = fm_config.get('type', 'base')

    if fm_type == 'jekyll':
        return JekyllFrontMatter(content, fm_config)

    return FrontMatter(content, fm_config)
