"""Version number."""

from importlib.metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version(__package__)
except PackageNotFoundError:
    __version__ = "included"
