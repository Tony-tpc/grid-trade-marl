# marl/config.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/config.py)

安全 YAML 边界：显式选择算法配置，所有默认值定义在配置 dataclass。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [_mapping](#_mapping)
- [_flat](#_flat)
- [_backbone](#_backbone)
- [_component](#_component)
- [algorithm_config_from_dict](#algorithm_config_from_dict)
- [load_algorithm_config](#load_algorithm_config)
- [config_to_dict](#config_to_dict)
- [config_to_dict.encode](#config_to_dict-encode)
- [config_to_dict.deep_settings](#config_to_dict-deep_settings)

<a id="_mapping"></a>

## _mapping

[源码位置](../../marl/config.py#L56) · [页内目录](#符号目录)

```python
def _mapping(data: object, location: str) -> Mapping[str, Any]
```

检查输入为字符串键 Mapping，返回原映射；location 用于错误定位。

<a id="_flat"></a>

## _flat

[源码位置](../../marl/config.py#L62) · [页内目录](#符号目录)

```python
def _flat(cls: type[T], data: object, location: str) -> T
```

只解析叶子配置；算法组合在下方显式列出。Any 仅限 YAML 边界。

<a id="_backbone"></a>

## _backbone

[源码位置](../../marl/config.py#L104) · [页内目录](#符号目录)

```python
def _backbone(data: object, location: str) -> BackboneConfig
```

读取 kind 并严格解析 MLP/GRU/LSTM 配置；未知 kind 报错，返回对应 BackboneConfig。

<a id="_component"></a>

## _component

[源码位置](../../marl/config.py#L116) · [页内目录](#符号目录)

```python
def _component(data: object, builtin: type[Buildable[T]], capability: type, category: Category, catalog: ExtensionCatalog | None) -> Buildable[T]
```

选择内置组件配置或显式 catalog 的外部组件；返回 Buildable，尚不实例化网络。

<a id="algorithm_config_from_dict"></a>

## algorithm_config_from_dict

[源码位置](../../marl/config.py#L145) · [页内目录](#符号目录)

```python
def algorithm_config_from_dict(data: Mapping[str, object], *, catalog: ExtensionCatalog | None=None) -> AlgorithmConfig
```

输入算法字典和可选 catalog，校验版本/字段/类型后返回五种算法 Config 之一。

<a id="load_algorithm_config"></a>

## load_algorithm_config

[源码位置](../../marl/config.py#L231) · [页内目录](#符号目录)

```python
def load_algorithm_config(path: str | Path, *, catalog: ExtensionCatalog | None=None) -> AlgorithmConfig
```

从路径安全加载 YAML，然后返回经过校验的算法 Config；catalog 仅用于外部组件。

<a id="config_to_dict"></a>

## config_to_dict

[源码位置](../../marl/config.py#L241) · [页内目录](#符号目录)

```python
def config_to_dict(config: AlgorithmConfig) -> dict[str, Any]
```

Checkpoint/YAML 数据快照；外部组件只保存名称和配置，不序列化构造器。

<a id="config_to_dict-encode"></a>

## config_to_dict.encode

[源码位置](../../marl/config.py#L244) · [页内目录](#符号目录)

```python
def encode(value: Any) -> Any
```

递归把 dataclass、Enum、dtype、外部配置等转成可序列化值，返回规范化数据。

<a id="config_to_dict-deep_settings"></a>

## config_to_dict.deep_settings

[源码位置](../../marl/config.py#L266) · [页内目录](#符号目录)

```python
def deep_settings(settings: Mapping[str, object]) -> dict[str, Any]
```

递归编码外部组件 settings，返回独立的可序列化字典。
