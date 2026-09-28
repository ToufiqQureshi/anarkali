"""Anarkali: experimental structured decisions.

Importing this package keeps neural and serving dependencies unloaded.
"""

from .schema import Candidate, ChoiceRequest
from .policy import DecisionPolicy

__version__ = "0.3.0"
__all__ = ["Candidate", "ChoiceRequest", "DecisionPolicy", "Engine"]


def __getattr__(name):
    if name == "Engine":
        from .engine import Engine
        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
