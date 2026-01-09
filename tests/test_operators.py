import torch
import torch.nn as nn

from neuromancer.modules.operators import _StripMetadataMixin


class _DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.received_state_keys = None

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_metadata"] = {"dummy": True}
        return state

    def load_state_dict(self, state, **kwargs):
        self.received_state_keys = set(state.keys())
        return super().load_state_dict(state, **kwargs)


class _DummyStripModel(_StripMetadataMixin, _DummyBase):
    pass


def test_strip_state():
    """Ensure _metadata is stripped from the state dict."""
    model = _DummyStripModel()
    state = model.state_dict()
    assert "_metadata" not in state
    assert "linear.weight" in state


def test_strip_load():
    """Ensure _metadata is ignored on load_state_dict and input is unchanged."""
    model = _DummyStripModel()
    base = _DummyBase()
    state = base.state_dict()
    state_keys = set(state.keys())

    model.load_state_dict(state, strict=False)

    assert "_metadata" in state_keys
    assert set(state.keys()) == state_keys
    assert "_metadata" not in model.received_state_keys
