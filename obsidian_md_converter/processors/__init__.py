from obsidian_md_converter.config import Config
from obsidian_md_converter.errors import ConfigError

from .base import Processor
from .jekyll import JekyllProcessor
from .markdown import MarkdownProcessor


def processor_for(config: Config) -> Processor:
    """
    The processor a run writes its pages with, chosen and set up by the config's 'processor' section.

    Examples:
        {'type': 'jekyll', 'default_layout': 'note'} -> JekyllProcessor(default_layout='note')
        {'type': 'markdown'}                         -> MarkdownProcessor()
    """
    options = dict(config.processor_config)
    kind = options.pop('type', 'jekyll')
    if kind == 'markdown':
        if options:
            raise ConfigError(f"Unknown option for the markdown processor: {', '.join(sorted(options))}; it has none.")
        return MarkdownProcessor()
    if kind != 'jekyll':
        raise ConfigError(f"Unknown processor type '{kind}'; the types are 'jekyll' and 'markdown'.")
    unknown = sorted(set(options) - {'default_layout', 'permalink'})
    if unknown:
        raise ConfigError(f"Unknown option for the jekyll processor: {', '.join(unknown)}; "
                          "the options are 'default_layout' and 'permalink'.")
    if options.get('permalink', 'pretty') != 'pretty':
        raise ConfigError(f"Unknown jekyll permalink '{options['permalink']}'; the only one is 'pretty'.")
    return JekyllProcessor(**options)
