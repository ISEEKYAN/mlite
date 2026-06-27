"""Native GLM-5 lite implementation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.model.glm5.lite.model import Glm5Model

__all__ = ["Glm5Model"]


def __getattr__(name: str):
    """Load the TE-dependent model only when callers request the model class."""
    if name == "Glm5Model":
        from megatron.lite.model.glm5.lite.model import Glm5Model

        return Glm5Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
