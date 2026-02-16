from typing import Union, TypeVar, Type
from pathlib import Path


T = TypeVar('T', bound='BaseValidatedPath')


class BaseValidatedPath:
    """Base class for validated path wrappers that implements os.PathLike protocol."""
    _root: Path  # Must be set in child classes or via configure_root()

    def __init__(self, path: Union[str, Path]):
        self._path = Path(path).expanduser().absolute()
        self._validate()

    def _validate(self):
        """Ensure path is within the configured root."""
        try:
            if not self._path.is_relative_to(self.__class__._root):
                raise ValueError(f"Path {self._path} is outside {self.__class__.__name__} root")
        except (FileNotFoundError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {e}")

    # Overrding root - for testing purposes
    @classmethod
    def configure_root(cls: Type[T], root: Union[str, Path]) -> None:
        """Configure the root directory for validation."""
        cls._root = Path(root).resolve()

    @property
    def path(self) -> Path:
        """Access the raw Path object when needed."""
        return self._path

    def __fspath__(self):
        """Return the file system path representation (os.PathLike protocol)."""
        return str(self._path)

    def __truediv__(self, other):
        """Support the / operator for path concatenation."""
        return self.__class__(self._path / other)

    # Delegate all other attributes to the wrapped Path
    def __getattr__(self, name):
        return getattr(self._path, name)


class ObsidianPath(BaseValidatedPath):
    """Path guaranteed to be within the Obsidian vault."""
    _root = None  # Will be set from config in main()


class OutputPath(BaseValidatedPath):
    """Path guaranteed to be within the output site."""
    _root = None  # Will be set from config in main()
