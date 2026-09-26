from __future__ import annotations

import pytest
import torch
from torch import nn

from marl.modules.critic import centralized_critic_input
from marl.target_updates import frozen_target
from marl.training.checkpoint import (
    build_trainer_checkpoint_state,
    capture_torch_rng_state,
    load_checkpoint_state,
    restore_torch_rng_state,
    save_checkpoint_state,
    validate_trainer_checkpoint_state,
)
from marl.training.gradients import isolate_agent_action


def test_frozen_target_is_equal_independent_and_has_no_trainable_parameters() -> None:
    online = nn.Linear(3, 2)
    target = frozen_target(online)

    assert target is not online
    assert all(
        torch.equal(target_value, online_value)
        for target_value, online_value in zip(
            target.state_dict().values(), online.state_dict().values(), strict=True
        )
    )
    assert not any(parameter.requires_grad for parameter in target.parameters())

    with torch.no_grad():
        online.weight.add_(1.0)
    assert not torch.equal(target.weight, online.weight)


def test_centralized_critic_input_preserves_leading_dimensions() -> None:
    observations = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    actions = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

    result = centralized_critic_input(observations, actions)

    assert result.shape == (2, 10)
    assert torch.equal(result[:, :6], observations.flatten(-2))
    assert torch.equal(result[:, 6:], actions.flatten(-2))


def test_centralized_critic_input_rejects_mismatched_layout() -> None:
    with pytest.raises(ValueError, match="批次维"):
        centralized_critic_input(torch.zeros(2, 2, 3), torch.zeros(3, 2, 1))
    with pytest.raises(ValueError, match="智能体数量"):
        centralized_critic_input(torch.zeros(2, 2, 3), torch.zeros(2, 3, 1))


def test_isolate_agent_action_only_propagates_selected_agent_gradient() -> None:
    actions = torch.randn(3, 2, 4, requires_grad=True)

    isolate_agent_action(actions, 1).sum().backward()

    assert actions.grad is not None
    assert torch.equal(actions.grad[:, 0], torch.zeros_like(actions.grad[:, 0]))
    assert torch.equal(actions.grad[:, 1], torch.ones_like(actions.grad[:, 1]))
    with pytest.raises(IndexError, match="agent_index"):
        isolate_agent_action(actions, 2)


def test_checkpoint_helpers_restore_rng_and_validate_top_level(tmp_path) -> None:
    torch.manual_seed(17)
    rng_state = capture_torch_rng_state()
    expected = torch.rand(4)
    torch.rand(4)

    restore_torch_rng_state(rng_state)
    assert torch.equal(torch.rand(4), expected)

    checkpoint = tmp_path / "nested" / "trainer.pt"
    save_checkpoint_state({"value": torch.tensor(3)}, checkpoint)
    loaded = load_checkpoint_state(checkpoint)
    assert torch.equal(loaded["value"], torch.tensor(3))

    invalid = tmp_path / "invalid.pt"
    torch.save([1, 2, 3], invalid)
    with pytest.raises(TypeError, match="mapping"):
        load_checkpoint_state(invalid)


def test_trainer_checkpoint_helpers_reject_schema_and_config_mismatch() -> None:
    state = build_trainer_checkpoint_state(
        {"weight": torch.tensor(1)},
        {"algorithm": "test"},
        {"optimizers": {}},
    )
    validate_trainer_checkpoint_state(state, {"algorithm": "test"})

    with pytest.raises(ValueError, match="旧 schema"):
        validate_trainer_checkpoint_state({**state, "schema_version": 1}, state["config"])
    with pytest.raises(ValueError, match="config"):
        validate_trainer_checkpoint_state(state, {"algorithm": "different"})
