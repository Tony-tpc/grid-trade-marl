# reproduction/sn_mappo/train_sn_mappo.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../reproduction/sn_mappo/train_sn_mappo.py)

顺序 1+3 实验：默认仅一轮数值验收；求解不达标时记录并停止长训练。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [evaluate](#evaluate)
- [run](#run)
- [main](#main)

<a id="evaluate"></a>

## evaluate

[源码位置](../../reproduction/sn_mappo/train_sn_mappo.py#L21) · [页内目录](#符号目录)

```python
def evaluate(algorithm: SNMAPPO, environment: SequentialEnvironment, seed: int) -> dict
```

确定性完整事件评估；保持三个角色 GRU 状态，恢复所有模块 train/eval 标志。

<a id="run"></a>

## run

[源码位置](../../reproduction/sn_mappo/train_sn_mappo.py#L65) · [页内目录](#符号目录)

```python
def run(args: argparse.Namespace, seed: int) -> None
```

以指定 seed 构造实验、执行预算内训练并保存逐次诊断、独立评估、配置和
checkpoint；不覆盖已有实验。

<a id="main"></a>

## main

[源码位置](../../reproduction/sn_mappo/train_sn_mappo.py#L118) · [页内目录](#符号目录)

```python
def main() -> None
```

解析脚本命令行并执行文档所述入口；输出写入显式指定的本地目录。
