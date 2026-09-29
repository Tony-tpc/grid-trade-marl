# marl/extensions.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/extensions.py)

可选外部组件目录。内置组件直接 build，不经过此目录。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [Buildable](#buildable)
- [Buildable.kind](#buildable-kind)
- [Buildable.build](#buildable-build)
- [ExternalConfig](#externalconfig)
- [ExternalConfig.build](#externalconfig-build)
- [_Entry](#_entry)
- [ExtensionCatalog](#extensioncatalog)
- [ExtensionCatalog.__init__](#extensioncatalog-__init__)
- [ExtensionCatalog.register](#extensioncatalog-register)
- [ExtensionCatalog.register.configure](#extensioncatalog-register-configure)
- [ExtensionCatalog.configure](#extensioncatalog-configure)
- [ExtensionCatalog.configure.build](#extensioncatalog-configure-build)

<a id="buildable"></a>

## Buildable

[源码位置](../../marl/extensions.py#L19) · [页内目录](#符号目录)

`class Buildable(Protocol[T_co])`

配置只需暴露 build；内置与外部实现共享这个很小的接口。

<a id="buildable-kind"></a>

## Buildable.kind

[源码位置](../../marl/extensions.py#L23) · [页内目录](#符号目录)

```python
def kind(self) -> str
```

返回组件选择名称，供配置序列化或 YAML 选择使用。

<a id="buildable-build"></a>

## Buildable.build

[源码位置](../../marl/extensions.py#L25) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> T_co
```

输入 EnvironmentSpec，返回该配置定义的组件；结构化能力协议不提供实现。

<a id="externalconfig"></a>

## ExternalConfig

[源码位置](../../marl/extensions.py#L29) · [页内目录](#符号目录)

`class ExternalConfig(Generic[T])`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: str
settings: Mapping[str, object]
create: Callable[[EnvironmentSpec], T] = field(repr=False, compare=False)
```

<a id="externalconfig-build"></a>

## ExternalConfig.build

[源码位置](../../marl/extensions.py#L34) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> T
```

输入 spec，调用加载配置时绑定的 create，返回已校验兼容性的外部组件。

<a id="_entry"></a>

## _Entry

[源码位置](../../marl/extensions.py#L39) · [页内目录](#符号目录)

`class _Entry()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
configure: Callable[[Mapping[str, object]], Callable[[EnvironmentSpec], nn.Module]]
action_kinds: frozenset[ActionKind]
reward_structures: frozenset[RewardStructure]
```

<a id="extensioncatalog"></a>

## ExtensionCatalog

[源码位置](../../marl/extensions.py#L45) · [页内目录](#符号目录)

`class ExtensionCatalog()`

仅注册外部 policy/critic/mixer；调用者显式传给 YAML loader。

<a id="extensioncatalog-__init__"></a>

## ExtensionCatalog.__init__

[源码位置](../../marl/extensions.py#L48) · [页内目录](#符号目录)

```python
def __init__(self) -> None
```

构造 ExtensionCatalog，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="extensioncatalog-register"></a>

## ExtensionCatalog.register

[源码位置](../../marl/extensions.py#L51) · [页内目录](#符号目录)

```python
def register(self, category: Category, name: str, parse: Callable[[Mapping[str, object]], C], build: Callable[[C, EnvironmentSpec], nn.Module], *, action_kinds: frozenset[ActionKind], reward_structures: frozenset[RewardStructure]=frozenset(RewardStructure)) -> None
```

登记 category/name 对应的 parse/build 和动作/奖励兼容集合；重复名称报错，返回 None。

<a id="extensioncatalog-register-configure"></a>

## ExtensionCatalog.register.configure

[源码位置](../../marl/extensions.py#L66) · [页内目录](#符号目录)

```python
def configure(data: Mapping[str, object]) -> Callable[[EnvironmentSpec], nn.Module]
```

用注册的 parse 解析 settings，返回绑定解析结果的网络构造闭包。

<a id="extensioncatalog-configure"></a>

## ExtensionCatalog.configure

[源码位置](../../marl/extensions.py#L72) · [页内目录](#符号目录)

```python
def configure(self, category: Category, name: str, settings: Mapping[str, object], capability: type[T]) -> ExternalConfig[T]
```

解析外部 settings 并绑定构造闭包，返回 ExternalConfig；网络稍后由 build(spec) 创建。

<a id="extensioncatalog-configure-build"></a>

## ExtensionCatalog.configure.build

[源码位置](../../marl/extensions.py#L83) · [页内目录](#符号目录)

```python
def build(spec: EnvironmentSpec) -> T
```

校验 spec 的动作/奖励兼容性后实例化 nn.Module，验证能力接口并返回组件。
