# doc/examples/train_walkthrough.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../doc/examples/train_walkthrough.py)

从新 adapter 训练、保存、重建和续训；在仓库根目录运行，输出到忽略目录。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [TanhValueConfig](#tanhvalueconfig)
- [TanhValueConfig.__post_init__](#tanhvalueconfig-__post_init__)
- [TanhValueConfig.build](#tanhvalueconfig-build)
- [main](#main)
- [main.train_round](#main-train_round)

<a id="tanhvalueconfig"></a>

## TanhValueConfig

[源码位置](../../doc/examples/train_walkthrough.py#L28) · [页内目录](#符号目录)

`class TanhValueConfig()`

教学用外部 critic 配置；输入 state [...,S]，输出 value [...,N]。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
hidden_dim: int = 16
kind: str = 'tutorial_tanh_value'
```

<a id="tanhvalueconfig-__post_init__"></a>

## TanhValueConfig.__post_init__

[源码位置](../../doc/examples/train_walkthrough.py#L34) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

无显式输入；校验配置宽度，非法值抛 ValueError。

<a id="tanhvalueconfig-build"></a>

## TanhValueConfig.build

[源码位置](../../doc/examples/train_walkthrough.py#L39) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> nn.Sequential
```

输入环境 spec；返回注册了全部参数、输出逐智能体 value 的网络。

<a id="main"></a>

## main

[源码位置](../../doc/examples/train_walkthrough.py#L48) · [页内目录](#符号目录)

```python
def main() -> None
```

输入 CLI 参数；输出每轮 JSON 指标和 checkpoint，并实际恢复后再更新一轮。

<a id="main-train_round"></a>

## main.train_round

[源码位置](../../doc/examples/train_walkthrough.py#L103) · [页内目录](#符号目录)

```python
def train_round(trainer: OnPolicyTrainer | OffPolicyTrainer, model: nn.Module, round_index: int) -> dict[str, float]
```

输入 trainer/模型/轮次；输出有限指标，采集 seed 从基础 seed 和轮次确定。
