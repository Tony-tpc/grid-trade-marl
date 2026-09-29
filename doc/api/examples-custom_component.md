# examples/custom_component.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../examples/custom_component.py)

通过可选 ExtensionCatalog 接入库外 critic，并完成一次 MAPPO 更新。

内置组件不需要此目录。Python-only 用法可直接 MAPPOConfig(critic=TanhValueConfig())。
此例从字典模拟 YAML 数据；文件加载时把同一个 catalog 传给 load_algorithm_config。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [TanhValueConfig](#tanhvalueconfig)
- [TanhValueConfig.__post_init__](#tanhvalueconfig-__post_init__)
- [TanhValueConfig.build](#tanhvalueconfig-build)
- [read_settings](#read_settings)
- [main](#main)

<a id="tanhvalueconfig"></a>

## TanhValueConfig

[源码位置](../../examples/custom_component.py#L26) · [页内目录](#符号目录)

`class TanhValueConfig()`

输入全局 state [*B,S]，输出各 agent value [*B,N]。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
hidden_dim: int = 32
kind: str = 'tanh_value'
```

<a id="tanhvalueconfig-__post_init__"></a>

## TanhValueConfig.__post_init__

[源码位置](../../examples/custom_component.py#L32) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 TanhValueConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="tanhvalueconfig-build"></a>

## TanhValueConfig.build

[源码位置](../../examples/custom_component.py#L36) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> nn.Sequential
```

从 TanhValueConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。

<a id="read_settings"></a>

## read_settings

[源码位置](../../examples/custom_component.py#L46) · [页内目录](#符号目录)

```python
def read_settings(data: Mapping[str, object]) -> TanhValueConfig
```

外部组件负责自己的字段校验；不编写通用 schema 框架。

<a id="main"></a>

## main

[源码位置](../../examples/custom_component.py#L57) · [页内目录](#符号目录)

```python
def main() -> None
```

读取命令行参数或示例配置，执行该脚本描述的演示并打印结果；返回 None。
