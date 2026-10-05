import pytest

torch = pytest.importorskip("torch")

from uratori.train.losses import (  # noqa: E402
    combine_by_type,
    hard_label_loss,
    ordinal_loss,
    soft_label_loss,
)

NEG = torch.finfo(torch.float32).min


def test_hard_label_loss_ignores_masked_options():
    logits = torch.tensor([[2.0, 0.0, NEG], [2.0, 0.0, 0.0]])
    loss = hard_label_loss(logits, torch.tensor([0, 0]))
    two_way = -torch.log_softmax(torch.tensor([2.0, 0.0]), dim=0)[0]
    assert loss[0].item() == pytest.approx(two_way.item())
    assert loss[1] > loss[0]


def test_soft_label_loss_is_zero_when_prediction_equals_teacher():
    q = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.5, 0.0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    logits = torch.log(q.clamp_min(1e-30)).masked_fill(~mask, NEG)
    assert soft_label_loss(logits, mask, q).tolist() == pytest.approx([0.0, 0.0], abs=1e-5)
    off = soft_label_loss(torch.zeros(2, 3).masked_fill(~mask, NEG), mask, q)
    assert (off[0] > 0) and off[1].item() == pytest.approx(0.0, abs=1e-5)


def test_ordinal_loss_grows_with_distance_from_target():
    mask = torch.ones(1, 5, dtype=torch.bool)
    target = torch.tensor([0])

    def at(level: int) -> float:
        logits = torch.full((1, 5), -20.0)
        logits[0, level] = 20.0
        return ordinal_loss(logits, mask, target).item()

    assert at(0) == pytest.approx(0.0, abs=1e-6)
    assert at(1) == pytest.approx(1.0, abs=1e-4)
    assert at(4) == pytest.approx(4.0, abs=1e-4)
    assert at(1) < at(2) < at(4)


def test_ordinal_loss_handles_fewer_levels_than_batch_width():
    logits = torch.tensor([[20.0, -20.0, -20.0, NEG, NEG]])
    mask = torch.tensor([[True, True, True, False, False]])
    assert ordinal_loss(logits, mask, torch.tensor([2])).item() == pytest.approx(2.0, abs=1e-4)


def test_combine_by_type_weights_types_equally():
    per_example = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0])
    type_ids = torch.tensor([0, 0, 0, 0, 1])
    assert combine_by_type(per_example, type_ids).item() == pytest.approx(3.0)
    assert per_example.mean().item() == pytest.approx(1.8)
