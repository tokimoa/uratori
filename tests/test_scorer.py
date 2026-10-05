import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from uratori.models.scorer import OptionScorer, gather_option_scores  # noqa: E402

MARKER = 5


def test_gather_keeps_marker_order_and_masks_missing_options():
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]])
    is_marker = torch.tensor(
        [[False, True, False, True, False, True], [True, False, False, False, True, False]]
    )
    logits, mask = gather_option_scores(scores, is_marker)
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    assert logits[0].tolist() == [1.0, 3.0, 5.0]
    assert logits[1, :2].tolist() == [10.0, 14.0]
    probs = logits.softmax(dim=-1)
    assert probs[1, 2].item() == 0.0
    assert probs.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])


def test_gather_rejects_inputs_with_fewer_than_two_markers():
    with pytest.raises(ValueError, match="2 個未満"):
        gather_option_scores(torch.zeros(1, 4), torch.tensor([[False, True, False, False]]))


def test_scorer_runs_on_a_tiny_backbone_and_gets_gradients():
    config = transformers.BertConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = OptionScorer(transformers.BertModel(config), 16, MARKER)
    input_ids = torch.tensor([[7, 8, MARKER, 9, MARKER, MARKER], [7, MARKER, 8, MARKER, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    logits, mask = model(input_ids, attention_mask)
    assert logits.shape == (2, 3) and mask.sum().item() == 5
    torch.nn.functional.cross_entropy(logits, torch.tensor([2, 1])).backward()
    assert model.head[0].weight.grad.abs().sum() > 0
