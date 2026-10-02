# 全库函数索引

[手册首页](../README.md) · [目录职责](../architecture.md)

覆盖全部 marl 源文件、四个原有示例和手册教学脚本；包含私有/嵌套函数，
overload 在同一锚点展示最终实现。测试按专题见 [验证导航](../training.md#verification)，
基准实验入口与内部组织见 [基准说明](../../benchmarks/README.md)。

| 源文件 | 类/函数条目数 |
|---|---:|
| [doc/examples/cue_environment.py](doc-examples-cue_environment.md) | 6 |
| [doc/examples/train_walkthrough.py](doc-examples-train_walkthrough.md) | 5 |
| [examples/custom_component.py](examples-custom_component.md) | 5 |
| [examples/minimal_usage.py](examples-minimal_usage.md) | 1 |
| [examples/plot_demand_response.py](examples-plot_demand_response.md) | 2 |
| [examples/prepare_demand_response.py](examples-prepare_demand_response.md) | 1 |
| [examples/train_demand_response.py](examples-train_demand_response.md) | 4 |
| [examples/train_energy_maac.py](examples-train_energy_maac.md) | 1 |
| [examples/train_mappo.py](examples-train_mappo.md) | 1 |
| [examples/train_sn_mappo.py](examples-train_sn_mappo.md) | 3 |
| [marl/__init__.py](marl-__init__.md) | 0 |
| [marl/algorithms/__init__.py](marl-algorithms-__init__.md) | 0 |
| [marl/algorithms/base.py](marl-algorithms-base.md) | 6 |
| [marl/algorithms/maac.py](marl-algorithms-maac.md) | 14 |
| [marl/algorithms/maddpg.py](marl-algorithms-maddpg.md) | 14 |
| [marl/algorithms/mappo.py](marl-algorithms-mappo.md) | 16 |
| [marl/algorithms/masac.py](marl-algorithms-masac.md) | 16 |
| [marl/algorithms/qmix.py](marl-algorithms-qmix.md) | 11 |
| [marl/algorithms/sn_mappo.py](marl-algorithms-sn_mappo.md) | 10 |
| [marl/config.py](marl-config.md) | 10 |
| [marl/core/__init__.py](marl-core-__init__.md) | 0 |
| [marl/core/batch.py](marl-core-batch.md) | 8 |
| [marl/core/layout.py](marl-core-layout.md) | 7 |
| [marl/core/output.py](marl-core-output.md) | 1 |
| [marl/core/recurrent.py](marl-core-recurrent.md) | 4 |
| [marl/envs/__init__.py](marl-envs-__init__.md) | 0 |
| [marl/envs/base.py](marl-envs-base.md) | 14 |
| [marl/envs/config.py](marl-envs-config.md) | 9 |
| [marl/envs/demand_response.py](marl-envs-demand_response.md) | 37 |
| [marl/envs/demand_response_data.py](marl-envs-demand_response_data.md) | 5 |
| [marl/envs/energy_trading.py](marl-envs-energy_trading.md) | 17 |
| [marl/envs/energy_trading_adapter.py](marl-envs-energy_trading_adapter.md) | 7 |
| [marl/envs/memory_cue.py](marl-envs-memory_cue.md) | 6 |
| [marl/envs/mpe2_simple_adversary_adapter.py](marl-envs-mpe2_simple_adversary_adapter.md) | 20 |
| [marl/envs/sequential.py](marl-envs-sequential.md) | 9 |
| [marl/experiment.py](marl-experiment.md) | 4 |
| [marl/extensions.py](marl-extensions.md) | 12 |
| [marl/models/__init__.py](marl-models-__init__.md) | 0 |
| [marl/models/base.py](marl-models-base.md) | 6 |
| [marl/models/encoder.py](marl-models-encoder.md) | 15 |
| [marl/models/gnn.py](marl-models-gnn.md) | 6 |
| [marl/models/gru.py](marl-models-gru.md) | 7 |
| [marl/models/lstm.py](marl-models-lstm.md) | 7 |
| [marl/models/mlp.py](marl-models-mlp.md) | 6 |
| [marl/models/transformer.py](marl-models-transformer.md) | 6 |
| [marl/modules/__init__.py](marl-modules-__init__.md) | 0 |
| [marl/modules/action_head.py](marl-modules-action_head.md) | 16 |
| [marl/modules/actor.py](marl-modules-actor.md) | 7 |
| [marl/modules/critic.py](marl-modules-critic.md) | 41 |
| [marl/modules/encoding.py](marl-modules-encoding.md) | 10 |
| [marl/modules/mixer.py](marl-modules-mixer.md) | 11 |
| [marl/modules/policy.py](marl-modules-policy.md) | 49 |
| [marl/objectives.py](marl-objectives.md) | 30 |
| [marl/returns.py](marl-returns.md) | 16 |
| [marl/runtime.py](marl-runtime.md) | 16 |
| [marl/target_updates.py](marl-target_updates.md) | 16 |
| [marl/training/__init__.py](marl-training-__init__.md) | 0 |
| [marl/training/checkpoint.py](marl-training-checkpoint.md) | 6 |
| [marl/training/gradients.py](marl-training-gradients.md) | 2 |
| [marl/training/implicit.py](marl-training-implicit.md) | 12 |
| [marl/training/off_policy.py](marl-training-off_policy.md) | 34 |
| [marl/training/on_policy.py](marl-training-on_policy.md) | 48 |
| [marl/training/optimization.py](marl-training-optimization.md) | 15 |
| [marl/training/sequential.py](marl-training-sequential.md) | 18 |
| [marl/value_scaling.py](marl-value_scaling.md) | 7 |
