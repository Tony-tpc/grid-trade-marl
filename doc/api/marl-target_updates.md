# marl/target_updates.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/target_updates.py)

与算法解耦的 target network 更新策略。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [frozen_target](#frozen_target)
- [SoftTargetUpdate](#softtargetupdate)
- [SoftTargetUpdate.__post_init__](#softtargetupdate-__post_init__)
- [SoftTargetUpdate.step](#softtargetupdate-step)
- [HardTargetUpdate](#hardtargetupdate)
- [HardTargetUpdate.__post_init__](#hardtargetupdate-__post_init__)
- [HardTargetUpdate.should_step](#hardtargetupdate-should_step)
- [HardTargetUpdate.step](#hardtargetupdate-step)
- [TargetUpdate](#targetupdate)
- [TargetUpdate.step](#targetupdate-step)
- [SoftTargetConfig](#softtargetconfig)
- [SoftTargetConfig.__post_init__](#softtargetconfig-__post_init__)
- [SoftTargetConfig.build](#softtargetconfig-build)
- [HardTargetConfig](#hardtargetconfig)
- [HardTargetConfig.__post_init__](#hardtargetconfig-__post_init__)
- [HardTargetConfig.build](#hardtargetconfig-build)

<a id="frozen_target"></a>

## frozen_target

[源码位置](../../marl/target_updates.py#L16) · [页内目录](#符号目录)

```python
def frozen_target(module: ModuleT) -> ModuleT
```

复制 online 网络并冻结参数，作为注册到算法上的 target 网络。

<a id="softtargetupdate"></a>

## SoftTargetUpdate

[源码位置](../../marl/target_updates.py#L27) · [页内目录](#符号目录)

`class SoftTargetUpdate()`

参数执行 Polyak 平均，非参数 buffer（如 BN 统计）完整复制。

buffer 可以是整数计数器，不能统一执行 lerp。目标网络是否处于 eval 模式由
调用者决定；requires_grad=False 本身不会关闭 dropout 或 BN 统计更新。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
tau: float = 0.005
```

<a id="softtargetupdate-__post_init__"></a>

## SoftTargetUpdate.__post_init__

[源码位置](../../marl/target_updates.py#L36) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 SoftTargetUpdate 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="softtargetupdate-step"></a>

## SoftTargetUpdate.step

[源码位置](../../marl/target_updates.py#L41) · [页内目录](#符号目录)

```python
def step(self, pairs: Iterable[tuple[nn.Module, nn.Module]]) -> None
```

输入 (target,online) 对；no_grad 下参数按 tau 做 Polyak 平均，buffers 完整复制，返回 None。

<a id="hardtargetupdate"></a>

## HardTargetUpdate

[源码位置](../../marl/target_updates.py#L54) · [页内目录](#符号目录)

`class HardTargetUpdate()`

按固定更新次数完整复制 online 参数。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
interval: int = 1
```

<a id="hardtargetupdate-__post_init__"></a>

## HardTargetUpdate.__post_init__

[源码位置](../../marl/target_updates.py#L59) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 HardTargetUpdate 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="hardtargetupdate-should_step"></a>

## HardTargetUpdate.should_step

[源码位置](../../marl/target_updates.py#L63) · [页内目录](#符号目录)

```python
def should_step(self, update_count: int) -> bool
```

输入算法 update_count，返回是否被 interval 整除的 bool。

<a id="hardtargetupdate-step"></a>

## HardTargetUpdate.step

[源码位置](../../marl/target_updates.py#L67) · [页内目录](#符号目录)

```python
def step(self, pairs: Iterable[tuple[nn.Module, nn.Module]]) -> None
```

输入 (target,online) 对；完整复制 state_dict，返回 None。间隔判定由调用方完成。

<a id="targetupdate"></a>

## TargetUpdate

[源码位置](../../marl/target_updates.py#L71) · [页内目录](#符号目录)

`class TargetUpdate(Protocol)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="targetupdate-step"></a>

## TargetUpdate.step

[源码位置](../../marl/target_updates.py#L72) · [页内目录](#符号目录)

```python
def step(self, pairs: Iterable[tuple[nn.Module, nn.Module]]) -> None
```

声明目标网络原地同步接口，输入 target/online 对，返回 None。

<a id="softtargetconfig"></a>

## SoftTargetConfig

[源码位置](../../marl/target_updates.py#L76) · [页内目录](#符号目录)

`class SoftTargetConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['soft'] = 'soft'
tau: float = 0.005
```

<a id="softtargetconfig-__post_init__"></a>

## SoftTargetConfig.__post_init__

[源码位置](../../marl/target_updates.py#L80) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 SoftTargetConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="softtargetconfig-build"></a>

## SoftTargetConfig.build

[源码位置](../../marl/target_updates.py#L84) · [页内目录](#符号目录)

```python
def build(self) -> SoftTargetUpdate
```

从 SoftTargetConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。

<a id="hardtargetconfig"></a>

## HardTargetConfig

[源码位置](../../marl/target_updates.py#L89) · [页内目录](#符号目录)

`class HardTargetConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['hard'] = 'hard'
interval: int = 1
```

<a id="hardtargetconfig-__post_init__"></a>

## HardTargetConfig.__post_init__

[源码位置](../../marl/target_updates.py#L93) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 HardTargetConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="hardtargetconfig-build"></a>

## HardTargetConfig.build

[源码位置](../../marl/target_updates.py#L97) · [页内目录](#符号目录)

```python
def build(self) -> HardTargetUpdate
```

从 HardTargetConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。
