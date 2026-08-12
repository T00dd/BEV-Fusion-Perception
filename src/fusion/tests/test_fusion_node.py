import pytest
import torch
import torch.nn as nn

from fusion.fusion_node import FusionConfig, FusionNode

H = W = 32


def make_inputs(cfg=FusionConfig(), b=2):
    return (
        torch.randn(b, cfg.lidar_channels, H, W),
        torch.randn(b, cfg.camera_channels, H, W),
        torch.rand(b, cfg.lidar_priors, H, W),
        torch.rand(b, cfg.camera_priors, H, W),
    )


def test_output_is_bit_identical_to_lidar_at_init():

    node = FusionNode().eval()
    f_l, f_c, p_l, p_c = make_inputs()
    with torch.no_grad():
        out, _ = node(f_l, f_c, p_l, p_c)
    assert torch.equal(out, f_l)


def test_identity_holds_in_train_mode_too():
    node = FusionNode().train()
    f_l, f_c, p_l, p_c = make_inputs()
    out, _ = node(f_l, f_c, p_l, p_c)
    assert torch.equal(out, f_l)


def test_zero_conv_learns_from_the_first_step():

    node = FusionNode().train()
    f_l, f_c, p_l, p_c = make_inputs()
    out, _ = node(f_l, f_c, p_l, p_c)
    out.sum().backward()

    grad = node.zero_conv.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_upstream_gets_no_gradient_at_step_zero():

    node = FusionNode().train()
    f_l, f_c, p_l, p_c = make_inputs()
    out, _ = node(f_l, f_c, p_l, p_c)
    out.sum().backward()

    assert node.context[0].weight.grad.abs().sum() == 0


def test_delta_becomes_nonzero_after_a_step():
    node = FusionNode().train()
    opt = torch.optim.SGD(node.parameters(), lr=0.1)
    f_l, f_c, p_l, p_c = make_inputs()

    out, _ = node(f_l, f_c, p_l, p_c)
    out.pow(2).mean().backward()
    opt.step()

    out2, aux = node(f_l, f_c, p_l, p_c, return_aux=True)
    assert not torch.equal(out2, f_l)
    assert aux["delta_norm"] > 0


def test_delta_can_be_negative():

    node = FusionNode().train()
    with torch.no_grad():
        node.zero_conv.weight.normal_(0, 0.1)
    f_l, f_c, p_l, p_c = make_inputs()
    _, aux = node(f_l, f_c, p_l, p_c, return_aux=True)
    assert aux["delta"].min() < 0


def test_assert_zero_init_catches_overwrite():
    node = FusionNode()
    node.assert_zero_init()
    node.apply(lambda m: nn.init.normal_(m.weight) if isinstance(m, nn.Conv2d) else None)
    with pytest.raises(AssertionError):
        node.assert_zero_init()

def test_gate_is_exactly_uniform_at_init():
    node = FusionNode().eval()
    f_l, f_c, p_l, p_c = make_inputs()
    with torch.no_grad():
        _, aux = node(f_l, f_c, p_l, p_c, return_aux=True)
    g = aux["gate"]

    assert g.shape == (2, 1, H, W)
    assert torch.allclose(g, torch.full_like(g, torch.sigmoid(torch.tensor(2.0)).item()))
    assert g.std() < 1e-6, "gate is not uniform at init"


def test_gate_is_single_channel():

    node = FusionNode()
    assert node.gate[-1].out_channels == 1


def test_gate_capacity_stays_small():
    node = FusionNode()
    n = sum(p.numel() for p in node.gate.parameters())
    assert n < 10_000, f"gate has {n} params; it will start detecting"


def test_group_order_is_lidar_then_camera():

    node = FusionNode()
    h = node.cfg.hidden_channels
    assert node.lidar_slice == slice(0, h)
    assert node.camera_slice == slice(h, 2 * h)

    norms = node.modality_weight_norms()
    assert abs(norms["w_lidar"] - norms["w_camera"]) < 0.5, \
        "groups start at different scales; the ratio would be unreadable"


def test_projections_are_1x1():
    node = FusionNode()
    assert node.proj_lidar[0].kernel_size == (1, 1)
    assert node.proj_camera[0].kernel_size == (1, 1)


def test_context_is_3x3():

    node = FusionNode()
    assert node.context[0].kernel_size == (3, 3)


def test_no_normalisation_after_zero_conv():

    node = FusionNode()
    assert isinstance(node.zero_conv, nn.Conv2d)


def test_conv_bias_disabled_before_batchnorm():
    node = FusionNode()
    assert node.proj_lidar[0].bias is None
    assert node.context[0].bias is None


def test_shape_mismatch_is_loud():
    node = FusionNode()
    f_l, f_c, p_l, p_c = make_inputs()
    with pytest.raises(ValueError, match="align_"):
        node(f_l, torch.randn(2, 128, 20, 20), p_l, p_c)
    with pytest.raises(ValueError, match="channels"):
        node(f_l, torch.randn(2, 64, H, W), p_l, p_c)


def test_modality_dropout_blanks_camera_everywhere():
    cfg = FusionConfig(modality_dropout=1.0)
    node = FusionNode(cfg).train()
    with torch.no_grad():
        node.zero_conv.weight.normal_(0, 0.1)
    f_l, f_c, p_l, p_c = make_inputs(cfg)

    _, dropped = node(f_l, f_c, p_l, p_c, return_aux=True)
    _, blanked = node(f_l, torch.zeros_like(f_c), p_l, torch.zeros_like(p_c),
                      return_aux=True)
    assert torch.allclose(dropped["delta"], blanked["delta"], atol=1e-5)


def test_no_decay_group_covers_the_zero_conv():
    node = FusionNode()
    ids = {id(p) for p in node.no_decay_parameters()}
    assert id(node.zero_conv.weight) in ids
    assert id(node.zero_conv.bias) in ids


def test_parameter_budget():

    node = FusionNode()
    total = sum(p.numel() for p in node.parameters())
    context = sum(p.numel() for p in node.context.parameters())
    assert total == 387_521, f"unexpected budget: {total}"
    assert context / total > 0.7, "context conv should dominate the budget"