# marl/envs/config.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/envs/config.py)

安全、不可变的环境 YAML config。

算法 config 只描述训练数学；环境 config 只描述环境动力学和动作编码。环境尺寸由
构建后的 ``EnvironmentAdapter.spec`` 推导，绝不能复制到算法 YAML 中。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [_freeze](#_freeze)
- [EnvironmentConfig](#environmentconfig)
- [EnvironmentConfig.__post_init__](#environmentconfig-__post_init__)
- [environment_config_from_dict](#environment_config_from_dict)
- [load_environment_config](#load_environment_config)
- [energy_trading_config_from_environment](#energy_trading_config_from_environment)
- [build_energy_trading_adapter](#build_energy_trading_adapter)
- [mpe2_simple_adversary_config_from_environment](#mpe2_simple_adversary_config_from_environment)
- [build_mpe2_simple_adversary_adapter](#build_mpe2_simple_adversary_adapter)

<a id="_freeze"></a>

## _freeze

[源码位置](../../marl/envs/config.py#L34) · [页内目录](#符号目录)

```python
def _freeze(value: object) -> EnvironmentValue
```

将嵌套环境配置递归转为只读 mapping/tuple，返回不可变数据快照。

<a id="environmentconfig"></a>

## EnvironmentConfig

[源码位置](../../marl/envs/config.py#L47) · [页内目录](#符号目录)

`class EnvironmentConfig()`

一份环境 YAML 的不可变表示。

``environment`` 选择环境实现，``action_kind`` 选择适配器向算法公开的动作编码，
``options`` 只包含环境物理、市场和设备参数。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
schema_version: int
environment: str
action_kind: ActionKind
options: Mapping[str, EnvironmentValue] = field(default_factory=dict)
```

<a id="environmentconfig-__post_init__"></a>

## EnvironmentConfig.__post_init__

[源码位置](../../marl/envs/config.py#L59) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 EnvironmentConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="environment_config_from_dict"></a>

## environment_config_from_dict

[源码位置](../../marl/envs/config.py#L81) · [页内目录](#符号目录)

```python
def environment_config_from_dict(data: Mapping[str, object]) -> EnvironmentConfig
```

严格解析环境 config；未知或缺失的顶层字段立即报错。

<a id="load_environment_config"></a>

## load_environment_config

[源码位置](../../marl/envs/config.py#L101) · [页内目录](#符号目录)

```python
def load_environment_config(path: str | Path) -> EnvironmentConfig
```

使用 ``yaml.safe_load`` 读取环境配置，不执行任何动态 Python 导入。

<a id="energy_trading_config_from_environment"></a>

## energy_trading_config_from_environment

[源码位置](../../marl/envs/config.py#L119) · [页内目录](#符号目录)

```python
def energy_trading_config_from_environment(config: EnvironmentConfig) -> EnergyTradingConfig
```

把通用环境 config 严格绑定为 ``EnergyTradingConfig``。

<a id="build_energy_trading_adapter"></a>

## build_energy_trading_adapter

[源码位置](../../marl/envs/config.py#L141) · [页内目录](#符号目录)

```python
def build_energy_trading_adapter(config: EnvironmentConfig, *, profiles: EnergyProfiles | None=None) -> EnergyTradingAdapter
```

根据独立环境 YAML 创建一个全新的环境和适配器实例。

<a id="mpe2_simple_adversary_config_from_environment"></a>

## mpe2_simple_adversary_config_from_environment

[源码位置](../../marl/envs/config.py#L160) · [页内目录](#符号目录)

```python
def mpe2_simple_adversary_config_from_environment(config: EnvironmentConfig) -> MPE2SimpleAdversaryConfig
```

严格绑定 Farama MPE2 simple_adversary 的场景参数。

<a id="build_mpe2_simple_adversary_adapter"></a>

## build_mpe2_simple_adversary_adapter

[源码位置](../../marl/envs/config.py#L175) · [页内目录](#符号目录)

```python
def build_mpe2_simple_adversary_adapter(config: EnvironmentConfig) -> MPE2SimpleAdversaryAdapter
```

根据独立环境 YAML 创建 MPE2 对抗场景适配器。
