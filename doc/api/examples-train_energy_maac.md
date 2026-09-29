# examples/train_energy_maac.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../examples/train_energy_maac.py)

个体收益 P2P 博弈：环境 -> 适配器 -> MARLBatch -> MAAC。

运行：.venv\Scripts\python.exe examples\train_energy_maac.py --episodes 5
使用合成曲线与 54 个量化动作，仅演示训练接口，不代表论文数值复现。
每个 round 同时采集 num-envs 条 episode，--episodes 表示 round 数。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [main](#main)

<a id="main"></a>

## main

[源码位置](../../examples/train_energy_maac.py#L27) · [页内目录](#符号目录)

```python
def main() -> None
```

读取命令行参数或示例配置，执行该脚本描述的演示并打印结果；返回 None。
