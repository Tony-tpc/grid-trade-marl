"""组件装配示例：.venv\\Scripts\\python.exe examples\\minimal_usage.py"""

import torch

from marl.models import MLPBackbone
from marl.modules import Actor, DiscreteActionHead, QMixer


def main() -> None:
    num_agents, observation_dim, action_dim = 3, 12, 5

    # 同一个 Actor 可被 MAAC/MAPPO 等策略梯度算法持有；替换 Backbone 不影响动作头。
    actor = Actor(
        MLPBackbone(observation_dim, hidden_dims=(128, 128), output_dim=128),
        DiscreteActionHead(feature_dim=128, action_dim=action_dim),
    )
    observations = torch.randn(32, num_agents, observation_dim)
    policy_output = actor(observations)

    # QMIX 的 agent Q 网络应产生 [batch, agent, action]；这里只演示 mixer 的输入输出。
    chosen_agent_q = torch.randn(32, num_agents)
    global_state = torch.randn(32, 24)
    q_total = QMixer(num_agents=num_agents, state_dim=24)(chosen_agent_q, global_state)

    print("actions:", tuple(policy_output.actions.shape))
    print("joint Q:", tuple(q_total.shape))


if __name__ == "__main__":
    main()
