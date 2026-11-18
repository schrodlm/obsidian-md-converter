from obsidian_md_converter.config import Config
from obsidian_md_converter.errors import ConfigError

from .base import Processor
from .jekyll import JekyllProcessor


def processor_for(config: Config) -> Processor:
    """
    The processor a run writes its pages with, chosen and set up by the config's 'processor' section.

    Example: {'type': 'jekyll', 'default_layout': 'note'} -> JekyllProcessor(url_format, default_layout='note')
    """
    options = dict(config.processor_config)
    kind = options.pop('type', 'jekyll')
    if kind != 'jekyll':
        raise ConfigError(f"Unknown processor type '{kind}'; the only type is 'jekyll'.")
    unknown = sorted(set(options) - {'default_layout'})
    if unknown:
        raise ConfigError(f"Unknown option for the jekyll processor: {', '.join(unknown)}; the only one is 'default_layout'.")
    return JekyllProcessor(url_format=config.url_format, **options)
