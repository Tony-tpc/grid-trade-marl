# reproduction/sn_mappo/plot_demand_response.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../reproduction/sn_mappo/plot_demand_response.py)

绘制真实诊断和独立评估；散点评估，滚动均值仅用于训练诊断趋势。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [plot_run](#plot_run)
- [main](#main)

<a id="plot_run"></a>

## plot_run

[源码位置](../../reproduction/sn_mappo/plot_demand_response.py#L11) · [页内目录](#符号目录)

```python
def plot_run(root: Path) -> None
```

读取一个实验目录的 training/evaluation
JSONL，按实际物理步绘图并保存
diagnostics.png；评估仅散点，训练趋势为尾随均值。

<a id="main"></a>

## main

[源码位置](../../reproduction/sn_mappo/plot_demand_response.py#L61) · [页内目录](#符号目录)

```python
def main() -> None
```

解析脚本命令行并执行文档所述入口；输出写入显式指定的本地目录。
