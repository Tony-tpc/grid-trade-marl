# reproduction/sn_mappo/demand_response_data.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../reproduction/sn_mappo/demand_response_data.py)

OPSD 累计表读数转区间功率；数据处理依赖只在准备入口加载。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [DemandProfiles](#demandprofiles)
- [DemandProfiles.__post_init__](#demandprofiles-__post_init__)
- [DemandProfiles.window](#demandprofiles-window)
- [DemandProfiles.load](#demandprofiles-load)
- [prepare_opsd](#prepare_opsd)

<a id="demandprofiles"></a>

## DemandProfiles

[源码位置](../../reproduction/sn_mappo/demand_response_data.py#L23) · [页内目录](#符号目录)

`class DemandProfiles()`

完整连续区间：负荷 [T,N] kW、DER [T] kW、时间及插补标记。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
load_kw: np.ndarray
der_kw: np.ndarray
local_slot: np.ndarray
utc: np.ndarray
imputed: np.ndarray
load_scale: np.ndarray
der_scale: float
source_sha256: str
```

<a id="demandprofiles-__post_init__"></a>

## DemandProfiles.__post_init__

[源码位置](../../reproduction/sn_mappo/demand_response_data.py#L34) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

验证数值范围、形状和时间间隔；非法物理或数据配置尽早抛 ValueError。

<a id="demandprofiles-window"></a>

## DemandProfiles.window

[源码位置](../../reproduction/sn_mappo/demand_response_data.py#L56) · [页内目录](#符号目录)

```python
def window(self, start: int, horizon: int) -> DemandProfiles
```

返回 [start,start+horizon)；不重新拟合归一化尺度。

<a id="demandprofiles-load"></a>

## DemandProfiles.load

[源码位置](../../reproduction/sn_mappo/demand_response_data.py#L68) · [页内目录](#符号目录)

```python
def load(cls, path: str | Path) -> DemandProfiles
```

读取 allow_pickle=False 的 NPZ 并构造经验证的
DemandProfiles；拒绝缺测、负值或不连续区间。

<a id="prepare_opsd"></a>

## prepare_opsd

[源码位置](../../reproduction/sn_mappo/demand_response_data.py#L81) · [页内目录](#符号目录)

```python
def prepare_opsd(destination: str | Path) -> dict[str, object]
```

下载版本固定的公开数据并生成 train/validation/test NPZ 与审计 manifest。

原始表点作为区间结束时间；功率属于前 15 分钟，按区间开始的本地月份分割。
插补标记合并差分的两个端点。失败保留原文件，不写伪造的零负荷。
