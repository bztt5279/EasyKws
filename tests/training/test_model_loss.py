import math

import pytest
import torch

from easykws.training.loss import frame_max_score, max_pooling_loss, padding_mask
from easykws.training.model import build_model, count_parameters, model_config


def cmvn():
    return {"mean": [1.0] * 40, "istd": [2.0] * 40}


def small_model(**changes):
    return build_model(cmvn(), {"hidden_dim": 8, "num_layers": 2, "kernel_size": 3, **changes})


def test_default_architecture_and_persistent_cmvn():
    assert model_config() == {"input_dim": 40, "hidden_dim": 256, "num_layers": 4,
                              "kernel_size": 8, "dropout": 0.1, "norm_var": True}
    model = build_model(cmvn())
    assert count_parameters(model) == 287233
    assert model.backbone.padding == 105
    assert {"cmvn.mean", "cmvn.istd"} <= set(model.state_dict())
    assert "backbone.network.0.cnn.0.weight" in model.state_dict()
    assert torch.equal(model.cmvn(torch.full((1, 2, 40), 3.0)), torch.full((1, 2, 40), 4.0))
    restored = build_model({"mean": [0.0] * 40, "istd": [1.0] * 40})
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored.cmvn.mean, model.cmvn.mean)


@pytest.mark.parametrize("config", [{"hidden_dim": 0}, {"num_layers": True}, {"kernel_size": 0},
                                    {"input_dim": 39}, {"dropout": float("nan")},
                                    {"dropout": 1}, {"norm_var": 1}, {"unknown": 1}])
def test_invalid_config_rejected(config):
    with pytest.raises(ValueError):
        model_config(config)


@pytest.mark.parametrize("bad", [{}, {"mean": [0.0] * 39, "istd": [1.0] * 39},
                                 {"mean": [float("inf")] * 40, "istd": [1.0] * 40},
                                 {"mean": [0.0] * 40, "istd": [0.0] * 40}])
def test_invalid_cmvn_rejected(bad):
    with pytest.raises(ValueError):
        build_model(bad)


@pytest.mark.parametrize("kernel", [1, 3])
def test_eval_streaming_matches_whole_sequence(kernel):
    torch.manual_seed(91)
    model = small_model(kernel_size=kernel).eval()
    inputs = torch.randn(2, 13, 40)
    with torch.no_grad():
        whole, whole_cache = model(inputs)
        cache = None
        pieces = []
        for begin, end in [(0, 1), (1, 5), (5, 13)]:
            output, cache = model(inputs[:, begin:end], cache)
            pieces.append(output)
    torch.testing.assert_close(torch.cat(pieces, dim=1), whole, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(cache, whole_cache, rtol=1e-6, atol=1e-7)
    assert cache.shape == (2, 8, (kernel - 1) * 3)


def test_batch_one_has_finite_gradients_and_changes_parameters():
    torch.manual_seed(3)
    model = small_model()
    before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    output, _ = model(torch.randn(1, 9, 40))
    loss, accuracy = max_pooling_loss(output, torch.tensor([0]), torch.tensor([7]))
    assert math.isfinite(loss.item()) and accuracy in (0.0, 1.0)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)


def test_pooling_excludes_extra_padding_and_its_gradient():
    probabilities = torch.tensor([[[0.2], [0.8], [0.99], [0.99]],
                                  [[0.1], [0.3], [0.99], [0.99]]], requires_grad=True)
    target = torch.tensor([0, -1])
    lengths = torch.tensor([2, 2])
    loss, accuracy = max_pooling_loss(probabilities, target, lengths)
    torch.testing.assert_close(loss, torch.tensor((-math.log(0.8) - math.log(0.7)) / 2))
    assert accuracy == 1.0
    torch.testing.assert_close(frame_max_score(probabilities, lengths), torch.tensor([0.8, 0.3]))
    loss.backward()
    assert torch.equal(probabilities.grad[:, 2:], torch.zeros(2, 2, 1))


def test_prefix_is_only_excluded_from_positive_loss():
    probabilities = torch.tensor([[[0.9], [0.2]], [[0.9], [0.2]]])
    loss, accuracy = max_pooling_loss(probabilities, torch.tensor([0, -1]), torch.tensor([2, 2]), 1)
    torch.testing.assert_close(loss, torch.tensor((-math.log(0.2) - math.log(0.1)) / 2))
    assert accuracy == 0.5
    with pytest.raises(ValueError, match="every valid frame"):
        max_pooling_loss(probabilities[:1], torch.tensor([0]), torch.tensor([1]), 1)


def test_strict_threshold_and_zero_one_probabilities_remain_finite():
    probabilities = torch.tensor([[[0.5]], [[0.5]], [[0.0]], [[1.0]]], requires_grad=True)
    loss, accuracy = max_pooling_loss(probabilities, torch.tensor([0, -1, 0, -1]), torch.ones(4, dtype=torch.long))
    assert accuracy == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(probabilities.grad).all()


@pytest.mark.parametrize("target", [torch.tensor([0.1]), torch.tensor([0.0]),
                                    torch.tensor([True]), torch.tensor([1]), torch.tensor([-2]),
                                    torch.tensor([float("nan")]), torch.tensor([[0]])])
def test_bad_labels_rejected(target):
    with pytest.raises(ValueError):
        max_pooling_loss(torch.full((1, 3, 1), 0.5), target, torch.tensor([3]))


@pytest.mark.parametrize("lengths", [torch.tensor([0]), torch.tensor([-1]), torch.tensor([4]),
                                     torch.tensor([2.5]), torch.tensor([float("inf")]),
                                     torch.tensor([True]), torch.tensor([[2]]), torch.tensor([2, 2])])
def test_bad_lengths_rejected_for_loss_and_score(lengths):
    for operation in (lambda p: max_pooling_loss(p, torch.tensor([0]), lengths),
                      lambda p: frame_max_score(p, lengths)):
        with pytest.raises(ValueError):
            operation(torch.full((1, 3, 1), 0.5))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_bad_probabilities_rejected_even_in_padding(value):
    probabilities = torch.tensor([[[0.2], [value]]])
    with pytest.raises(ValueError):
        frame_max_score(probabilities, torch.tensor([1]))
    with pytest.raises(ValueError):
        max_pooling_loss(probabilities, torch.tensor([0]), torch.tensor([1]))


def test_mask_width_and_model_input_validation():
    assert padding_mask(torch.tensor([1, 3]), 4).tolist() == [[False, True, True, True], [False, False, False, True]]
    with pytest.raises(ValueError):
        padding_mask(torch.tensor([3]), 2)
    model = small_model().eval()
    with pytest.raises(ValueError):
        model(torch.empty(1, 0, 40))
    with pytest.raises(ValueError):
        model(torch.full((1, 2, 40), float("nan")))
    with pytest.raises(ValueError):
        model(torch.randn(1, 2, 40), torch.zeros(1, 8, 1))
    with pytest.raises(ValueError):
        model(torch.randn(1, 2, 40), torch.tensor(0.0))
