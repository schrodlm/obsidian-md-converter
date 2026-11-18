from obsidian_md_converter.config import Config

from .base import Processor
from .jekyll import JekyllProcessor


def processor_for(config: Config) -> Processor:
    """The processor a run writes its pages with."""
    return JekyllProcessor(url_format=config.url_format)
