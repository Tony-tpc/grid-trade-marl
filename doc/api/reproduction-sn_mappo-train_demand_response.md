# reproduction/sn_mappo/train_demand_response.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../reproduction/sn_mappo/train_demand_response.py)

真实 OPSD 日实验：固定 UC 或同步连续 MAPPO，原始诊断与评估分开保存。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [evaluate](#evaluate)
- [run](#run)
- [run.profiles](#run-profiles)
- [main](#main)

<a id="evaluate"></a>

## evaluate

[源码位置](../../reproduction/sn_mappo/train_demand_response.py#L20) · [页内目录](#符号目录)

```python
def evaluate(algorithm: MAPPO, environment: DailyDemandAdapter, seed: int) -> dict
```

独立确定性日评估；循环状态持续传递，并恢复模型 train/eval 标志。

<a id="run"></a>

## run

[源码位置](../../reproduction/sn_mappo/train_demand_response.py#L56) · [页内目录](#符号目录)

```python
def run(args: argparse.Namespace, seed: int) -> dict
```

以指定 seed 构造实验、执行预算内训练并保存逐次诊断、独立评估、配置和
checkpoint；不覆盖已有实验。

<a id="run-profiles"></a>

## run.profiles

[源码位置](../../reproduction/sn_mappo/train_demand_response.py#L61) · [页内目录](#符号目录)

```python
def profiles(split: str) -> DemandProfiles
```

读取指定 train/validation/test 分割，选择消费者列并保留 DER 插补标记和训练尺度。

<a id="main"></a>

## main

[源码位置](../../reproduction/sn_mappo/train_demand_response.py#L117) · [页内目录](#符号目录)

```python
def main() -> None
```

解析脚本命令行并执行文档所述入口；输出写入显式指定的本地目录。
