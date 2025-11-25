"""
Operators and neural operator wrappers.
"""

from neuralop.models import FNO as _BaseFNO


class FNOWithoutMeta(_BaseFNO):
    """
    Subclass of neuralop FNO that strips ``_metadata`` from checkpoints for
    compatibility across PyTorch versions.
    """

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state.pop("_metadata", None)
        return state

    def load_state_dict(self, state, **kwargs):
        state = dict(state)
        state.pop("_metadata", None)
        return super().load_state_dict(state, **kwargs)


class FNO(FNOWithoutMeta):
    """Default FNO export with metadata removal applied."""


__all__ = ["FNO", "FNOWithoutMeta"]
