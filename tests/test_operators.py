import pytest
import torch
import torch.nn as nn

from neuromancer.modules.operators import (
    DeepONetCartesianProd,
    DeepXDEWrapper,
    _StripMetadataMixin,
)


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


class _DummyDeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_inputs = None

    def forward(self, inputs):
        self.last_inputs = inputs
        branch, trunk = inputs
        return branch.mean() + trunk.mean()


class _DummyMIONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_inputs = None

    def forward(self, inputs):
        self.last_inputs = inputs
        branch1, branch2, trunk = inputs
        return branch1.mean() + branch2.mean() + trunk.mean()


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


def test_deepxde_cartesian():
    """Check cartesian trunk (B, m, dim_x) uses shared grid."""
    model = _DummyDeepONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=True)
    branch = torch.randn(2, 3)
    trunk = torch.randn(2, 4, 5)

    _ = wrapper(branch, trunk)

    assert torch.allclose(model.last_inputs[1], trunk[0])


def test_deepxde_cartesian_grid():
    """Check cartesian trunk (m, dim_x) passes through unchanged."""
    model = _DummyDeepONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=True)
    branch = torch.randn(2, 3)
    trunk = torch.randn(4, 5)

    _ = wrapper(branch, trunk)

    assert torch.allclose(model.last_inputs[1], trunk)


def test_deepxde_pointwise():
    """Check pointwise trunk (B, dim_x) passes through unchanged."""
    model = _DummyDeepONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=False)
    branch = torch.randn(2, 3)
    trunk = torch.randn(2, 5)

    _ = wrapper(branch, trunk)

    assert torch.allclose(model.last_inputs[1], trunk)


def test_deepxde_dict():
    """Check dict input returns preds and targets."""
    model = _DummyDeepONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=False)
    batch = {
        "branch_inputs": torch.randn(2, 3),
        "trunk_inputs": torch.randn(2, 5),
        "outputs": torch.randn(2, 1),
    }

    preds, targets = wrapper(batch)

    assert torch.is_tensor(preds)
    assert torch.allclose(targets, batch["outputs"])


def test_deepxde_bad_trunk():
    """Check invalid trunk shape raises ValueError."""
    model = _DummyDeepONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=False)
    branch = torch.randn(2, 3)
    trunk = torch.randn(2, 4, 5)

    with pytest.raises(ValueError):
        _ = wrapper(branch, trunk)


def test_deepxde_two_branch_positional():
    """Check two-branch positional inputs are routed correctly."""
    model = _DummyMIONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=True, branch_keys=["b1", "b2"])
    branch1 = torch.randn(2, 3)
    branch2 = torch.randn(2, 4)
    trunk = torch.randn(2, 5, 6)

    _ = wrapper(branch1, branch2, trunk)

    assert torch.allclose(model.last_inputs[0], branch1)
    assert torch.allclose(model.last_inputs[1], branch2)
    assert torch.allclose(model.last_inputs[2], trunk[0])


def test_deepxde_branch_list_single():
    """Check list input with single branch is accepted."""
    model = _DummyDeepONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=False, branch_keys=["b1"])
    branch = torch.randn(2, 3)
    trunk = torch.randn(2, 5)

    _ = wrapper([branch], trunk)

    assert torch.allclose(model.last_inputs[0], branch)
    assert torch.allclose(model.last_inputs[1], trunk)


def test_deepxde_branch_tuple_two():
    """Check tuple input with two branches is accepted."""
    model = _DummyMIONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=False, branch_keys=("b1", "b2"))
    branch1 = torch.randn(2, 3)
    branch2 = torch.randn(2, 4)
    trunk = torch.randn(2, 5)

    _ = wrapper((branch1, branch2), trunk)

    assert torch.allclose(model.last_inputs[0], branch1)
    assert torch.allclose(model.last_inputs[1], branch2)
    assert torch.allclose(model.last_inputs[2], trunk)


def test_deepxde_branch_count_mismatch():
    """Check mismatch between expected and received branches."""
    model = _DummyDeepONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=False, branch_keys=["b1", "b2"])
    branch = torch.randn(2, 3)
    trunk = torch.randn(2, 5)

    with pytest.raises(ValueError, match=r"Expected 2 branch inputs, received 1"):
        _ = wrapper(branch, trunk)


def test_deepxde_bad_branch_keys_type():
    """Check invalid branch_keys types raise TypeError."""
    model = _DummyDeepONet()
    with pytest.raises(TypeError, match="branch_keys must be a string or list/tuple"):
        _ = DeepXDEWrapper(model, is_cartesian=False, branch_keys=1)


def test_cartprod_shared_grid():
    """Check cartesian prod uses the first trunk grid."""
    branch_net = nn.Identity()
    trunk_net = nn.Identity()
    model = DeepONetCartesianProd(branch_net, trunk_net, bias=False)
    branch = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, -1.0]])
    grid0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    grid1 = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    trunk = torch.stack([grid0, grid1], dim=0)

    out = model(branch, trunk)

    expected = torch.einsum("bp,mp->bm", branch, grid0)
    assert torch.allclose(out, expected)


def test_cartprod_transpose_bias():
    """Check transpose output and bias addition."""
    branch_net = nn.Identity()
    trunk_net = nn.Identity()
    model = DeepONetCartesianProd(branch_net, trunk_net, bias=True, return_transposed=True)
    model.bias.data.fill_(0.5)
    branch = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, -1.0]])
    grid = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    trunk = torch.stack([grid, grid], dim=0)

    out = model(branch, trunk)

    expected = torch.einsum("bp,mp->bm", branch, grid) + 0.5
    assert out.shape == (grid.shape[0], branch.shape[0])
    assert torch.allclose(out, expected.transpose(0, 1))
