"""Package containing tools and napari-widgets for manual correction and analysis of RGB-segmentation masks as used in the analysis of CLEM experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyclem")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Andreas M Arnold"
__email__ = "andreas.m.arnold@gmail.com"
