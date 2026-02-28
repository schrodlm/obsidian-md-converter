class ConversionError(Exception):
    """Exception raised when a file cannot be converted."""

    def __init__(self, filepath: str, reason: str):
        """
        Args:
            filepath: Path to the file that failed conversion
            reason: Explanation of why the conversion failed
        """
        self.filepath = filepath
        self.reason = reason
        message = f"Failed to convert '{filepath}': {reason}"
        super().__init__(message)

class ConfigError(Exception):
    """Exception raised when needed config value is not provided or is invalid."""
    def __init__(self, message: str):
        super().__init__(message)