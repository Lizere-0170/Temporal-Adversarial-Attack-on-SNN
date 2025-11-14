# src/utils.py
"""
SHD (Spiking Heidelberg Digits) 专用工具函数

功能：
- 自动用 tonic 下载 / 读取 SHD（若 tonic 可用）
- 将单样本规范化为 {'times', 'x', 'y', 'polarity', 'label'} 格式（numpy 1D arrays）
- 时间单位规范化为秒（user 可指定原始时间单位缩放）
- 对事件按时间升序排序
- 支持保存 / 加载标准化 .npz 文件以便离线实验

依赖：
- numpy
- tonic (推荐，但如果没有，load_shd_dataset 会抛出友好提示)
- matplotlib (仅用于可选的 raster 可视化)

安装示例：
pip install numpy tonic matplotlib

使用示例：
from src.utils import download_shd_if_needed, load_shd_sample, save_events_npz, plot_raster

download_shd_if_needed(root='./data')
evt = load_shd_sample(root='./data', split='train', index=0)
plot_raster(evt)
save_events_npz(evt, './data/shd_sample_0.npz')
"""

import os
import numpy as np

# Optional imports
try:
    import tonic
    from tonic.datasets import SHD  # tonic's HSD loader includes SHD
    _HAS_TONIC = True
except Exception:
    _HAS_TONIC = False

try:
    import matplotlib.pyplot as _plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

# -------------------------
# Helper / core functions
# -------------------------
def download_shd_if_needed(root='./data'):
    """
    确保 SHD 数据已下载（使用 tonic）。若 tonic 未安装，抛出异常提示安装。
    root: 存放数据的根目录
    返回: tonic dataset 对象（HSD）
    """
    if not _HAS_TONIC:
        raise RuntimeError("未检测到 'tonic'。请先安装：pip install tonic")
    os.makedirs(root, exist_ok=True)
    # HSD 即 tonic 中的 Spiking Heidelberg dataset 接口（有时命名为 HSD）
    ds = SHD(save_to=root)
    return ds

def _normalize_events_dict(raw_events, label=None, time_scale=1.0):
    """
    把原始 sample 中的事件结构规范成：
      {'times': np.ndarray(float, N), 'x': np.ndarray(int, N),
       'y': np.ndarray(int, N), 'polarity': np.ndarray(int, N), 'label': int}
    说明：
      - raw_events 可以是 tonic sample（dict like）或 numpy structured array，或 dict。
      - time_scale: 用于把原始时间单位换算到秒，例如若原始单位为 microsecond，time_scale=1e-6。
    """
    # raw_events may be dict-like from tonic: {'events': ndarray, 'label': int}
    # common shapes: structured array with fields ('t','ch') or ('times','channels')
    if isinstance(raw_events, dict) and 'events' in raw_events:
        data = raw_events['events']
    else:
        data = raw_events

    # structured numpy array with named fields
    times = None
    xs = None
    ys = None
    pols = None

    # Try many common field names
    def _pick_field(obj, candidates):
        # obj may be dict or structured array
        if obj is None:
            return None
        if isinstance(obj, dict):
            keys = list(obj.keys())
            lower_map = {k.lower(): k for k in keys}
            for c in candidates:
                if c in lower_map:
                    return np.asarray(obj[lower_map[c]])
            return None
        # structured numpy array
        if hasattr(obj, 'dtype') and getattr(obj.dtype, 'names', None):
            for c in candidates:
                for f in obj.dtype.names:
                    if f.lower() == c:
                        return np.asarray(obj[f])
        # plain ndarray fallback handled later
        return None

    times = _pick_field(data, ['t', 'time', 'times', 'ts'])
    xs = _pick_field(data, ['x', 'col', 'u', 'channel', 'ch', 'channels'])
    ys = _pick_field(data, ['y', 'row', 'v'])
    pols = _pick_field(data, ['p', 'pol', 'polarity', 'sign'])

    # If structured array but only (time, channel) in plain ndarray form
    if times is None and isinstance(data, np.ndarray) and data.ndim == 2:
        if data.shape[1] >= 2:
            times = data[:, 0]
            xs = data[:, 1].astype(int)
            if data.shape[1] >= 3:
                pols = data[:, 2].astype(int)
            else:
                pols = np.ones_like(times, dtype=int)
        else:
            raise ValueError("无法解析 ndarray 事件形式，期待至少 (time, channel).")

    # If SHD often stores channel only (0..699), map channel->x, and y=0
    if xs is None:
        ch = _pick_field(data, ['channel', 'ch', 'channels'])
        if ch is not None:
            xs = np.asarray(ch).astype(int)
            ys = np.zeros_like(xs, dtype=int)

    # default polarity
    if pols is None:
        pols = np.ones_like(times, dtype=int)

    # coerce
    times = np.asarray(times, dtype=float).ravel()
    xs = np.asarray(xs, dtype=int).ravel()
    ys = np.asarray(ys, dtype=int).ravel() if ys is not None else np.zeros_like(xs)
    pols = np.asarray(pols, dtype=int).ravel()

    N = times.shape[0]
    # Ensure lengths match
    if not (xs.shape[0] == N and ys.shape[0] == N and pols.shape[0] == N):
        raise ValueError(f"事件字段长度不匹配：times {N}, x {xs.shape[0]}, y {ys.shape[0]}, pol {pols.shape[0]}")

    # apply time scale
    times = times.astype(float) * float(time_scale)

    # sort by time
    order = np.argsort(times)
    times = times[order]
    xs = xs[order]
    ys = ys[order]
    pols = pols[order]

    out = {'times': times, 'x': xs, 'y': ys, 'polarity': pols}
    if label is not None:
        out['label'] = int(label)
    return out

def load_shd_sample(root='./data', split='train', index=0, time_unit='s'):
    """
    加载 SHD 指定分割（train/test）下的单样本并返回标准事件 dict。

    参数：
      - root: 存放 tonic 下载数据的目录（与 download_shd_if_needed 中一致）
      - split: 'train' 或 'test'
      - index: 样本索引（整数）
      - time_unit: 指定返回的时间单位，支持 's'（秒）、'ms'（毫秒）、'us'（微秒）
    返回:
      {'times','x','y','polarity','label'}

    说明：
      tonic 下载的 HSD dataset 可通过 HSD(save_to=root).train / .test 访问
    """
    if not _HAS_TONIC:
        raise RuntimeError("需要 'tonic' 来加载 SHD。请安装：pip install tonic")
    ds = SHD(save_to=root, train=(split == 'train'))
    if split not in ['train', 'test']:
        raise ValueError("split 必须是 'train' 或 'test'")
    # subset = ds.train if split == 'train' else ds.test
    # sample = subset[index]  # sample is usually dict-like {'events':..., 'label':...}
    # sample = ds[index]
    # # infer time scale: tonic usually provides times in seconds already; allow override by user if needed
    # # we'll assume tonic returns times in seconds. If not, user can multiply externally.
    # unit_scale = {'s': 1.0, 'ms': 1e3, 'us': 1e6}
    # if time_unit not in unit_scale:
    #     raise ValueError("time_unit must be one of 's','ms','us'")
    # # tonic likely returns times in seconds; to produce requested time unit, multiply by factor
    # # But our internal representation will be in seconds; we return times in seconds (consistent).
    # # If the user requested 'ms' or 'us', we'll scale before returning.
    # normalized = _normalize_events_dict(sample, label=sample.get('label', None), time_scale=1.0)
    sample = ds[index]
    # tonic returns samples as (events, label) tuple; handle tuple/list and dict-like cases
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        events_obj, label_obj = sample[0], sample[1]
    elif isinstance(sample, dict):
        events_obj = sample.get('events', sample)
        label_obj = sample.get('label', None)
    else:
        events_obj = sample
        label_obj = None

    unit_scale = {'s': 1.0, 'ms': 1e3, 'us': 1e6}
    if time_unit not in unit_scale:
        raise ValueError("time_unit must be one of 's','ms','us'")

    normalized = _normalize_events_dict(events_obj, label=label_obj, time_scale=1.0)
    if time_unit == 's':
        return normalized
    elif time_unit == 'ms':
        normalized['times'] = normalized['times'] * 1e3
        return normalized
    else:  # 'us'
        normalized['times'] = normalized['times'] * 1e6
        return normalized

def save_events_npz(events_dict, out_path):
    """
    把标准化事件 dict 保存为 .npz
    events_dict keys: 'times','x','y','polarity' (可选 'label')
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    keys = ['times', 'x', 'y', 'polarity', 'label']
    save_dict = {k: events_dict[k] for k in events_dict if k in keys}
    np.savez_compressed(out_path, **save_dict)
    return out_path

def load_events_npz(path):
    """
    读取由 save_events_npz 保存的 .npz，返回标准事件 dict
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    d = np.load(path, allow_pickle=True)
    out = {'times': d['times'], 'x': d['x'], 'y': d['y'], 'polarity': d['polarity']}
    if 'label' in d:
        out['label'] = int(d['label'].tolist())
    return out

def plot_raster(events, title='raster', figsize=(8,4), max_points=200000):
    """
    简单 raster 绘图：x 轴为时间（秒），y 轴为 channel（或 x）
    events: 标准事件 dict
    """
    if not _HAS_MATPLOTLIB:
        raise RuntimeError("plot_raster 需要 matplotlib。请安装：pip install matplotlib")
    times = np.asarray(events['times'])
    xs = np.asarray(events['x'])
    N = times.shape[0]
    if N > max_points:
        idx = np.linspace(0, N-1, max_points).astype(int)
        times = times[idx]
        xs = xs[idx]
    _plt.figure(figsize=figsize)
    _plt.scatter(times, xs, s=1)
    _plt.xlabel('time (s)')
    _plt.ylabel('channel / x')
    _plt.title(title)
    _plt.show()

# -------------------------
# Batch helpers
# -------------------------
def export_shd_subset_to_npz(root='./data', split='train', out_dir='./data/shd_npz', max_samples=None):
    """
    把 SHD 指定子集（train/test）导出成单独的 .npz 文件，便于离线使用。
    out files 格式: out_dir/{split}_{index}.npz
    """
    if not _HAS_TONIC:
        raise RuntimeError("需要 'tonic' 来导出 SHD。请安装：pip install tonic")
    os.makedirs(out_dir, exist_ok=True)
    # ds = SHD(save_to=root)
    # subset = ds.train if split == 'train' else ds.test
    # count = len(subset)
    ds = SHD(save_to=root, train=(split == 'train'))
    count = len(ds)
    if max_samples is not None:
        count = min(count, max_samples)
    for i in range(count):
        sample = ds[i]
        evt = _normalize_events_dict(sample, label=sample.get('label', None), time_scale=1.0)
        out_path = os.path.join(out_dir, f"{split}_{i}.npz")
        save_events_npz(evt, out_path)
    return out_dir

# -------------------------
# quick test / debug runner
# -------------------------
if __name__ == '__main__':
    # quick smoke test (only runs when invoked directly)
    print("SHD utils smoke test (requires tonic).")
    if not _HAS_TONIC:
        print("请先 pip install tonic 再运行本测试。")
    else:
        ds_train = SHD(save_to='./data', train=True)
        ds_test = SHD(save_to='./data', train=False)
        print("dataset ready. train size:", len(ds_train), "test size:", len(ds_test))
        sample = load_shd_sample('./data', split='train', index=0)
        print("sample keys:", list(sample.keys()))
        print("first 5 times (s):", sample['times'][:5])
        try:
            plot_raster(sample, title='SHD sample #0')
        except Exception as e:
            print("plot failed:", e)
