"""OPSD 累计表读数转区间功率；数据处理依赖只在准备入口加载。"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPSD_URL = 'https://data.open-power-system-data.org/household_data/2020-04-15/'
OPSD_FILE = 'household_data_15min_singleindex.csv'
OPSD_COLUMNS = (
    'DE_KN_residential2_grid_import',
    'DE_KN_residential5_grid_import',
    'DE_KN_industrial3_area_offices',
    'DE_KN_residential3_pv',
)


@dataclass(frozen=True, slots=True)
class DemandProfiles:
    """完整连续区间：负荷 [T,N] kW、DER [T] kW、时间及插补标记。"""
    load_kw: np.ndarray
    der_kw: np.ndarray
    local_slot: np.ndarray
    utc: np.ndarray
    imputed: np.ndarray
    load_scale: np.ndarray
    der_scale: float
    source_sha256: str

    def __post_init__(self) -> None:
        """验证数值范围、形状和时间间隔；非法物理或数据配置尽早抛 ValueError。"""
        t = len(self.load_kw)
        if self.load_kw.ndim != 2 or t == 0:
            raise ValueError('load_kw 必须是非空 [T,N]')
        if any(x.shape != (t,) for x in (self.der_kw, self.local_slot, self.utc)):
            raise ValueError('DER/时间必须是 [T]')
        if self.imputed.shape != (t, self.load_kw.shape[1] + 1):
            raise ValueError('imputed 必须是 [T,N+1]')
        if self.load_scale.shape != (self.load_kw.shape[1],):
            raise ValueError('load_scale 必须是 [N]')
        for value in (self.load_kw, self.der_kw, self.load_scale, np.array([self.der_scale])):
            if not np.isfinite(value).all() or (value < 0).any():
                raise ValueError('负荷/DER/尺度必须有限且非负，不允许隐式补零')
        if (self.load_scale <= 0).any() or self.der_scale <= 0:
            raise ValueError('训练尺度必须为正')
        if ((self.local_slot < 0) | (self.local_slot > 95)).any():
            raise ValueError('local_slot 必须在 0..95')
        spacing = np.diff(self.utc.astype('datetime64[s]'))
        if t > 1 and not np.all(spacing == np.timedelta64(900, 's')):
            raise ValueError('必须是连续 15 分钟区间')

    def window(self, start: int, horizon: int) -> DemandProfiles:
        """返回 [start,start+horizon)；不重新拟合归一化尺度。"""
        if start < 0 or horizon < 1 or start + horizon > len(self.load_kw):
            raise ValueError('窗口越界')
        end = start + horizon
        return DemandProfiles(
            self.load_kw[start:end], self.der_kw[start:end], self.local_slot[start:end],
            self.utc[start:end], self.imputed[start:end], self.load_scale, self.der_scale,
            self.source_sha256,
        )

    @classmethod
    def load(cls, path: str | Path) -> DemandProfiles:
        """
        读取 allow_pickle=False 的 NPZ 并构造经验证的
        DemandProfiles；拒绝缺测、负值或不连续区间。
        """
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data['load_kw'], data['der_kw'], data['local_slot'], data['utc'],
                data['imputed'], data['load_scale'], float(data['der_scale']),
                str(data['source_sha256']),
            )


def prepare_opsd(destination: str | Path) -> dict[str, object]:
    """下载版本固定的公开数据并生成 train/validation/test NPZ 与审计 manifest。

    原始表点作为区间结束时间；功率属于前 15 分钟，按区间开始的本地月份分割。
    插补标记合并差分的两个端点。失败保留原文件，不写伪造的零负荷。
    """
    import pandas as pd  # type: ignore[import-untyped]

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    raw = root / OPSD_FILE
    if not raw.exists():
        temporary = raw.with_suffix('.download')
        with urllib.request.urlopen(OPSD_URL + OPSD_FILE, timeout=90) as response:
            with temporary.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        temporary.replace(raw)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    columns = ['utc_timestamp', 'interpolated', *OPSD_COLUMNS]
    frame = pd.read_csv(raw, usecols=columns)
    end = pd.to_datetime(frame['utc_timestamp'], utc=True)
    start = end - pd.Timedelta(minutes=15)
    local = start.dt.tz_convert('Europe/Berlin')
    power = frame[list(OPSD_COLUMNS)].diff() / .25
    gaps = end.diff().dt.total_seconds().ne(900)
    power.loc[gaps] = np.nan
    flags = frame['interpolated'].fillna('')
    imputed = np.column_stack([
        (flags.str.contains(col, regex=False) | flags.shift(1, fill_value='').str.contains(
            col, regex=False)).to_numpy() for col in OPSD_COLUMNS
    ])
    intervals = {
        'train': ('2016-06-01', '2016-08-01'),
        'validation': ('2016-08-01', '2016-09-01'),
        'test': ('2016-09-01', '2016-10-01'),
    }
    masks = {
        name: ((local >= pd.Timestamp(left, tz='Europe/Berlin'))
               & (local < pd.Timestamp(right, tz='Europe/Berlin'))).to_numpy()
        for name, (left, right) in intervals.items()
    }
    train = power.to_numpy()[masks['train']]
    scales = np.quantile(train, .95, axis=0)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('训练窗口缺测或尺度无效')
    summary: dict[str, object] = {}
    for name, mask in masks.items():
        values = power.to_numpy()[mask]
        if not np.isfinite(values).all() or (values < -1e-8).any():
            raise ValueError(f'{name} 存在缺测或负差分；不能零填/截断来掩盖')
        values = np.maximum(values, 0)  # 仅处理差分舍入误差 <= 1e-8 kW。
        utc = start[mask].dt.tz_localize(None).to_numpy(dtype='datetime64[s]')
        slots = (local[mask].dt.hour * 4 + local[mask].dt.minute // 15).to_numpy()
        profiles = DemandProfiles(
            values[:, :3], values[:, 3], slots, utc, imputed[mask], scales[:3],
            float(scales[3]), digest,
        )
        np.savez_compressed(
            root / f'{name}.npz', load_kw=profiles.load_kw, der_kw=profiles.der_kw,
            local_slot=slots, utc=utc, imputed=profiles.imputed,
            load_scale=scales[:3], der_scale=scales[3], source_sha256=digest,
        )
        summary[name] = {
            'local_window': intervals[name], 'steps': len(values),
            'imputed_intervals_per_column': profiles.imputed.sum(0).tolist(),
            'mean_kw': values.mean(0).tolist(), 'maximum_kw': values.max(0).tolist(),
            'missing': 0, 'negative_beyond_tolerance': 0,
        }
    manifest: dict[str, object] = {
        'url': OPSD_URL + OPSD_FILE, 'license': 'CC-BY-4.0', 'sha256': digest,
        'columns': list(OPSD_COLUMNS), 'time_zone': 'Europe/Berlin',
        'interval_label': 'start', 'raw_unit': 'cumulative kWh', 'output_unit': 'kW',
        'delta_hours': .25, 'scale': 'training-only 95th percentile',
        'scale_kw': scales.tolist(), 'splits': summary,
        'limitations': 'Proxy public meters; not the paper authors original experiment data.',
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
