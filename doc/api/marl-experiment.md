# marl/experiment.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/experiment.py)

唯一装配入口；算法选择只发生在这里，trainer 不识别算法名称。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [Experiment](#experiment)
- [build_experiment](#build_experiment)
- [_off_policy](#_off_policy)

<a id="experiment"></a>

## Experiment

[源码位置](../../marl/experiment.py#L31) · [页内目录](#符号目录)

`class Experiment(Generic[A, T])`

同一次装配得到的 config/algorithm/trainer；三者不再各自重复创建模型。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
config: AlgorithmConfig
algorithm: A
trainer: T
```

<a id="build_experiment"></a>

## build_experiment

[源码位置](../../marl/experiment.py#L71) · [页内目录](#符号目录)

```python
def build_experiment(environment: Environment, config: AlgorithmConfig, *, device: str | torch.device='cpu', seed: int=42) -> Experiment[BaseMARLAlgorithm, OnPolicyTrainer] | Experiment[BaseMARLAlgorithm, OffPolicyTrainer]
```

先绑定 spec、构造网络并移动设备，再创建 optimizer。

overload 仅用于静态类型推导，不是多套运行入口。内置类型的显式分支保留
各 optimizer 的参数归属；不再抽象成 registry、工厂链或 UpdatePlan。

<a id="_off_policy"></a>

## _off_policy

[源码位置](../../marl/experiment.py#L123) · [页内目录](#符号目录)

```python
def _off_policy(spec: EnvironmentSpec, config: OffConfig, device: str | torch.device, seed: int, snapshot: dict) -> Experiment[BaseMARLAlgorithm, OffPolicyTrainer]
```

按离策略配置构造网络、独立命名 optimizer、target 更新器和 replay trainer，返回 Experiment。
