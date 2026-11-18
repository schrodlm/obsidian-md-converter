from .base import FrontMatter
from .jekyll import JekyllFrontMatter


def create_front_matter(content: str, config: dict) -> FrontMatter:
    #TODO: Move this whole mechanism into Config
    # Meaning we should introduce classes like FrontMatterConfig, JekkylFrontMatterConfig.
    # These classes will be able to create their own FrontMatter manipulator, because they know to which front matter they belong to
    """Factory method to create the FrontMatter instance a run writes, configured by the front_matter section.

    Args:
        content: Full markdown file content.
        config: The full configuration dictionary (from config.yaml).
    """
    fm_config = config.get('front_matter', {})
    return JekyllFrontMatter(content, fm_config)
