from loguru import logger


class StreamToLogger:
    """Taken from https://loguru.readthedocs.io/en/stable/resources/recipes.html#capturing-standard-stdout-stderr-and-warnings"""

    def __init__(self, level: str = "INFO"):
        """Constructor

        Args:
            level: Level of logging to capture. Defaults to "INFO".
        """
        self._level = level

    def write(self, buffer: str) -> None:
        """Writes std stream to logger.

        Args:
            buffer: Incoming stream.
        """
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self) -> None:
        """Flush stream."""
        pass
