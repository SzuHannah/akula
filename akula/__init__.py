__all__ = (
    "DATA_DIR",
    "LCAModel",
    "TEAModel",
    "GraphiteLCAModel",
    "GraphiteTEAModel",
    # "generate_markets_datapackage",
)

from pathlib import Path

DATA_DIR = Path(__file__).parent.resolve() / "data"

from .version import version as __version__
# from .markets import generate_markets_datapackage
from .model_framework import LCAModel, TEAModel
from .graphite_models import GraphiteLCAModel, GraphiteTEAModel
