from obsidian_md_converter.config import Config

from .base import Processor
from .jekyll import JekyllProcessor


def processor_for(config: Config) -> Processor:
    """The processor a run writes its pages with, set up from the front_matter section's 'jekyll' settings."""
    jekyll = config.front_matter_config.get('jekyll', {})
    # Only with auto_add on does a page get the default layout, which is 'note' unless set.
    default_layout = jekyll.get('default_layout', 'note') if jekyll.get('auto_add', False) else None
    return JekyllProcessor(url_format=config.url_format, default_layout=default_layout)
