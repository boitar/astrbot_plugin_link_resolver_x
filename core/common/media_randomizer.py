"""X 图片随机预处理: 在发送前对已下载图片做轻微的随机化改动.

接入位置为图片下载完成之后、创建 Image 组件之前, 后续构造 Node、上传和
转发都复用处理后的文件。设计要点:

- 默认开启, 可通过 media_randomizer 配置组关闭;
- 静态图支持六种策略: reencode / border / crop / canvas / edge_noise / pixel_noise,
  按 "先过滤、再加权随机" 选择, 无可用策略时回退原图;
- 动图 (GIF / 动画 WebP / APNG) 仅支持 skip, 直接使用原文件;
- JPEG / PNG / 静态 WebP 保留原格式、原分辨率编码参数与透明通道,
  输出扩展名按实际格式确定 (Pillow 按内容识别格式, 保存时需显式指定);
- 解码、处理、编码和校验都在后台线程执行; 任务取消时等待线程结束并清理
  半成品文件后再向上传播 (与 file_lifecycle.save_image_file 相同的模式);
- 事务内按文件 SHA-256 去重, 相同内容只随机处理一次, 跳过与失败结果同样缓存;
  不同事务各自创建 MediaPrepareContext, 重新随机选择, 不共享处理结果;
- 临时副本写入数据目录 temp/media_randomizer/ 下的 UUID 文件名, 发送结束后
  默认立即清理, 另有启动时与周期性的兜底清理 (跳过活跃事务文件)。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover - 本地测试环境没有 AstrBot
    import logging

    logger = logging.getLogger("astrbot_plugin_link_resolver")

try:
    from PIL import Image
except ImportError:  # pragma: no cover - 缺少 Pillow 时直接回退原图
    Image = None

from .paths import get_media_randomizer_path

# region 常量与默认配置

SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")
FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}

STATIC_STRATEGIES = ("reencode", "border", "crop", "canvas", "edge_noise", "pixel_noise")
ANIMATION_STRATEGIES = ("skip",)

DEFAULT_WEIGHTS = {
    "reencode": 30,
    "border": 20,
    "crop": 15,
    "canvas": 15,
    "edge_noise": 10,
    "pixel_noise": 10,
}
DEFAULT_ANIMATION_WEIGHTS = {"skip": 100}

# 配置值 (黑/白颜色选项) → 实际颜色集合
_COLOR_CHOICES = {
    "随机黑白": ("black", "white"),
    "黑": ("black",),
    "白": ("white",),
}

# 周期清理任务的默认参数
DEFAULT_CLEANUP_INTERVAL_MINUTES = 60.0
DEFAULT_CLEANUP_RETENTION_HOURS = 24.0

# endregion


# region 配置快照
@dataclass(frozen=True)
class MediaRandomizerConfig:
    """一次事务内使用的配置快照, 由 from_plugin 校验并固化."""

    enabled: bool = True
    cleanup_after_send: bool = True
    weights: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    animation_weights: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_ANIMATION_WEIGHTS)
    )
    jpeg_quality: int = 95
    webp_quality: int = 95
    png_compress_level: int = 6
    max_modified_ratio: float = 0.001
    border_px: tuple[int, int] = (1, 3)
    crop_px: tuple[int, int] = (1, 2)
    canvas_px: tuple[int, int] = (1, 6)
    noise_region_px: int = 2
    edge_noise_count: tuple[int, int] = (2, 20)
    pixel_noise_count: tuple[int, int] = (5, 30)
    pixel_noise_delta: tuple[int, int] = (1, 3)
    border_colors: tuple[str, ...] = ("black", "white")
    canvas_colors: tuple[str, ...] = ("black", "white")
    noise_colors: tuple[str, ...] = ("black", "white")

    @classmethod
    def from_plugin(cls, plugin: Any) -> "MediaRandomizerConfig":
        """从插件实例读取 media_randomizer_* 属性并校验.

        缺失的属性使用默认值, 便于测试桩复用默认行为。
        """

        def attr(name: str, default: Any) -> Any:
            return getattr(plugin, f"media_randomizer_{name}", default)

        weights = _sanitize_weights(attr("weights", None), DEFAULT_WEIGHTS, "weights")
        animation_weights = _sanitize_weights(
            attr("animation_weights", None),
            DEFAULT_ANIMATION_WEIGHTS,
            "animation_weights",
        )

        jpeg_quality = _coerce_bounded(attr("jpeg_quality", 95), 95, 1, 100, "jpeg_quality")
        webp_quality = _coerce_bounded(attr("webp_quality", 95), 95, 1, 100, "webp_quality")
        png_compress_level = _coerce_bounded(
            attr("png_compress_level", 6), 6, 0, 9, "png_compress_level"
        )

        raw_ratio = attr("max_modified_ratio", 0.001)
        try:
            max_modified_ratio = float(raw_ratio)
        except (TypeError, ValueError):
            max_modified_ratio = 0.001
        noise_eligible = True
        if not (0.0 < max_modified_ratio <= 1.0):
            _warn_once(
                "max_modified_ratio",
                f"max_modified_ratio={raw_ratio!r} 应在 (0, 1] 内, 噪声策略已停用",
            )
            noise_eligible = False
            max_modified_ratio = 0.001

        border_px = _coerce_range(attr("border_px", None), (1, 3))
        crop_px = _coerce_range(attr("crop_px", None), (1, 2))
        canvas_px = _coerce_range(attr("canvas_px", None), (1, 6))
        noise_region_px = _coerce_bounded(
            attr("noise_region_px", 2), 2, 1, 64, "noise_region_px"
        )
        edge_noise_count = _coerce_range(attr("edge_noise_count", None), (2, 20))
        pixel_noise_count = _coerce_range(attr("pixel_noise_count", None), (5, 30))
        pixel_noise_delta = _coerce_range(attr("pixel_noise_delta", None), (1, 3))

        for name, value_range in (
            ("border", border_px),
            ("crop", crop_px),
            ("canvas", canvas_px),
        ):
            if value_range[0] < 1 or value_range[0] > value_range[1]:
                _warn_once(
                    f"param.{name}",
                    f"{name} 参数 {value_range} 无效, 已停用该策略",
                )
                weights[name] = 0
        if noise_eligible and edge_noise_count[0] < 1:
            _warn_once("param.edge_noise", f"edge_noise 参数 {edge_noise_count} 无效, 已停用该策略")
            weights["edge_noise"] = 0
        if noise_eligible and (
            pixel_noise_count[0] < 1
            or pixel_noise_delta[0] < 1
            or pixel_noise_count[0] > pixel_noise_count[1]
            or pixel_noise_delta[0] > pixel_noise_delta[1]
        ):
            _warn_once(
                "param.pixel_noise",
                f"pixel_noise 参数 {pixel_noise_count}/{pixel_noise_delta} 无效, 已停用该策略",
            )
            weights["pixel_noise"] = 0
        if not noise_eligible:
            weights["edge_noise"] = 0
            weights["pixel_noise"] = 0

        return cls(
            enabled=bool(attr("enabled", True)),
            cleanup_after_send=bool(attr("cleanup_after_send", True)),
            weights=weights,
            animation_weights=animation_weights,
            jpeg_quality=jpeg_quality,
            webp_quality=webp_quality,
            png_compress_level=png_compress_level,
            max_modified_ratio=max_modified_ratio,
            border_px=border_px,
            crop_px=crop_px,
            canvas_px=canvas_px,
            noise_region_px=noise_region_px,
            edge_noise_count=edge_noise_count,
            pixel_noise_count=pixel_noise_count,
            pixel_noise_delta=pixel_noise_delta,
            border_colors=_resolve_colors(attr("border_color", "随机黑白"), "border_color"),
            canvas_colors=_resolve_colors(
                attr("canvas_background", "随机黑白"), "canvas_background"
            ),
            noise_colors=_resolve_colors(attr("noise_color", "随机黑白"), "noise_color"),
        )


_WARNED_CONFIG_KEYS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """非法配置只在首次出现时记录 Warning, 避免每条消息重复刷屏."""
    if key in _WARNED_CONFIG_KEYS:
        return
    _WARNED_CONFIG_KEYS.add(key)
    logger.warning("⚠️ X 图片随机预处理配置无效: %s", message)


def _sanitize_weights(
    raw: Any, default_weights: dict[str, int], label: str
) -> dict[str, int]:
    """权重采用非负相对权重, 不要求合计为 100; 非法权重按 0 处理."""
    source = raw if isinstance(raw, dict) else {}
    result: dict[str, int] = {}
    for name, default in default_weights.items():
        value = source.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            result[name] = 0
            continue
        coerced = int(value)
        if coerced < 0:
            coerced = 0
        result[name] = coerced
    return result


def _coerce_bounded(value: Any, default: int, lo: int, hi: int, label: str) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        _warn_once(f"param.{label}", f"{label}={value!r} 不是有效整数, 使用默认值 {default}")
        return default
    if coerced < lo or coerced > hi:
        clamped = min(max(coerced, lo), hi)
        _warn_once(f"param.{label}", f"{label}={coerced} 超出 [{lo}, {hi}], 已调整为 {clamped}")
        return clamped
    return coerced


def _coerce_range(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return default
    return default


def _resolve_colors(value: Any, label: str) -> tuple[str, ...]:
    colors = _COLOR_CHOICES.get(str(value).strip())
    if not colors:
        _warn_once(f"color.{label}", f"{label}={value!r} 无效, 使用默认的随机黑白")
        return _COLOR_CHOICES["随机黑白"]
    return colors


# endregion


# region 事务上下文
_MISSING = object()


class MediaPrepareContext:
    """一次消息处理事务内的图片预处理上下文.

    事务内以文件 SHA-256 去重: 同一内容重复引用只随机处理一次,
    跳过与失败结果同样缓存。不同事务各自创建实例, 重新随机选择。
    """

    def __init__(
        self,
        config: MediaRandomizerConfig | None = None,
        rng: random.Random | None = None,
        work_dir: Path | None = None,
    ) -> None:
        self.config = config if config is not None else MediaRandomizerConfig()
        self.rng = rng if rng is not None else random.Random()
        self._hash_by_path: dict[str, str] = {}
        self._result_by_hash: dict[str, Path | None] = {}
        self.generated_files: list[Path] = []
        self.processed_count = 0
        self.elapsed_seconds = 0.0
        self._work_dir = work_dir

    @classmethod
    def from_plugin(cls, plugin: Any) -> "MediaPrepareContext":
        return cls(config=MediaRandomizerConfig.from_plugin(plugin))

    def ensure_work_dir(self) -> Path:
        if self._work_dir is None:
            self._work_dir = get_media_randomizer_path()
        return self._work_dir
    async def digest_for(self, path: Path) -> str | None:
        """计算 (或复用) 文件的 SHA-256; 读取失败返回 None 表示直接回退."""
        key = str(path)
        cached = self._hash_by_path.get(key)
        if cached is not None:
            return cached
        try:
            digest = await asyncio.to_thread(_sha256_file, path)
        except OSError as exc:
            logger.debug("🖼️ X图片随机预处理: 读取文件失败, 回退原图: %s (%s)", path, exc)
            return None
        self._hash_by_path[key] = digest
        return digest

    def cached_result(self, digest: str) -> Path | None | object:
        """返回缓存的预处理结果; 无缓存时返回 _MISSING 哨兵."""
        return self._result_by_hash.get(digest, _MISSING)

    def remember_prepared(self, digest: str, path: Path, elapsed: float) -> None:
        self._result_by_hash[digest] = path
        self.generated_files.append(path)
        self.processed_count += 1
        self.elapsed_seconds += elapsed
        _register_active_file(path)

    def remember_fallback(self, digest: str) -> None:
        self._result_by_hash[digest] = None

    async def cleanup(self) -> None:
        """事务结束 (发送返回或异常) 后清理本事务生成的副本.

        cleanup_after_send=false 时只解除活跃标记, 文件由周期清理兜底。
        """
        paths = self.generated_files
        self.generated_files = []
        self._result_by_hash.clear()
        self._hash_by_path.clear()
        if not paths:
            return
        if self.config.cleanup_after_send:
            await asyncio.to_thread(_remove_files, tuple(paths))
        else:
            for path in paths:
                _unregister_active_file(path)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _remove_files(paths: tuple[Path, ...]) -> None:
    for path in paths:
        _unregister_active_file(path)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("🧹 X图片随机预处理副本清理失败: %s (%s)", path, exc)


# endregion


# region 活跃文件登记与目录清理
_ACTIVE_FILES: set[str] = set()
_ACTIVE_LOCK = threading.Lock()


def _register_active_file(path: Path) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_FILES.add(str(path))


def _unregister_active_file(path: Path) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_FILES.discard(str(path))


def _active_snapshot() -> set[str]:
    with _ACTIVE_LOCK:
        return set(_ACTIVE_FILES)


def cleanup_media_randomizer_directory(
    work_dir: Path, retention_hours: float, active: set[str] | None = None
) -> tuple[int, int]:
    """删除专用目录内超过保留期的副本, 返回 (已删除, 已检查).

    仅作用于 temp/media_randomizer/ 目录, 且跳过活跃事务登记的文件。
    active 缺省时自动取当前活跃文件快照。
    """
    try:
        if not work_dir.is_dir():
            return 0, 0
        entries = list(work_dir.iterdir())
    except OSError:
        return 0, 0
    active = _active_snapshot() if active is None else active
    now = time.time()
    retention_seconds = max(float(retention_hours), 0.0) * 3600.0
    deleted = 0
    examined = 0
    for entry in entries:
        try:
            if not entry.is_file():
                continue
            examined += 1
            if str(entry) in active:
                continue
            if now - entry.stat().st_mtime <= retention_seconds:
                continue
            entry.unlink(missing_ok=True)
            deleted += 1
        except OSError:
            continue
    return deleted, examined


_CLEANER_TASK: asyncio.Task | None = None
_CLEANER_STOP: asyncio.Event | None = None
_CLEANER_LOCK = threading.Lock()


async def _cleaner_loop(
    work_dir: Path, interval_seconds: float, retention_hours: float
) -> None:
    stop_event = _CLEANER_STOP
    while True:
        try:
            deleted, examined = await asyncio.to_thread(
                cleanup_media_randomizer_directory,
                work_dir,
                retention_hours,
                _active_snapshot(),
            )
            if deleted:
                logger.debug(
                    "🧹 X图片随机预处理目录清理: 删除=%d, 检查=%d", deleted, examined
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("🧹 X图片随机预处理目录清理失败: %s", exc)
        try:
            if stop_event is None:
                return
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            return
        except asyncio.TimeoutError:
            continue


def start_media_randomizer_cleaner(
    interval_minutes: float = DEFAULT_CLEANUP_INTERVAL_MINUTES,
    retention_hours: float = DEFAULT_CLEANUP_RETENTION_HOURS,
) -> bool:
    """启动周期清理任务 (启动后先立即清理一次). 返回是否新启动了任务."""
    global _CLEANER_TASK, _CLEANER_STOP
    with _CLEANER_LOCK:
        if _CLEANER_TASK is not None and not _CLEANER_TASK.done():
            return False
        interval_seconds = max(float(interval_minutes), 0.1) * 60.0
        retention = max(float(retention_hours), 0.0)
        _CLEANER_STOP = asyncio.Event()
        _CLEANER_TASK = asyncio.create_task(
            _cleaner_loop(get_media_randomizer_path(), interval_seconds, retention)
        )
        return True


async def stop_media_randomizer_cleaner() -> None:
    """停止周期清理任务 (插件卸载时调用)."""
    global _CLEANER_TASK, _CLEANER_STOP
    with _CLEANER_LOCK:
        task = _CLEANER_TASK
        stop_event = _CLEANER_STOP
        _CLEANER_TASK = None
        _CLEANER_STOP = None
    if stop_event is not None:
        stop_event.set()
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# endregion


# region 图片状态检查
@dataclass(frozen=True)
class _ImageState:
    format: str
    mode: str  # 处理用模式: RGB / RGBA
    width: int
    height: int
    has_alpha: bool
    exif: bytes | None
    icc_profile: bytes | None
    webp_lossless: bool


def _static_skip_reason(raw: "Image.Image") -> str | None:
    fmt = (getattr(raw, "format", "") or "").upper()
    if fmt not in SUPPORTED_FORMATS:
        return f"格式不支持({fmt or '未知'}), 使用原文件"
    n_frames = getattr(raw, "n_frames", 1)
    if bool(getattr(raw, "is_animated", False)) or n_frames > 1:
        return "动图, 按动画策略跳过"
    if raw.mode not in ("RGB", "RGBA", "P"):
        return f"色彩模式不支持({raw.mode}), 使用原文件"
    return None


def _detect_webp_lossless(path: Path) -> bool:
    """解析 RIFF 块判断 WebP 是 VP8L (无损) 还是 VP8 (有损); 无法判定按有损处理."""
    try:
        with open(path, "rb") as file:
            data = file.read(65536)
    except OSError:
        return False
    if len(data) < 16 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return False
    offset = 12
    total = len(data)
    while offset + 8 <= total:
        fourcc = data[offset : offset + 4]
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        if fourcc == b"VP8L":
            return True
        if fourcc == b"VP8 ":
            return False
        offset += 8 + size + (size & 1)
    return False


def _build_state(raw: "Image.Image", source: Path) -> _ImageState:
    fmt = (raw.format or "").upper()
    mode = raw.mode
    if mode == "P":
        mode = "RGBA" if "transparency" in raw.info else "RGB"

    def _bytes(value: Any) -> bytes | None:
        if isinstance(value, (bytes, bytearray)) and value:
            return bytes(value)
        return None

    return _ImageState(
        format=fmt,
        mode=mode,
        width=raw.width,
        height=raw.height,
        has_alpha=(mode == "RGBA"),
        exif=_bytes(raw.info.get("exif")),
        icc_profile=_bytes(raw.info.get("icc_profile")),
        webp_lossless=_detect_webp_lossless(source) if fmt == "WEBP" else False,
    )


# endregion


# region 策略实现
def _mono_value(color: str) -> int:
    return 0 if color == "black" else 255


def _pick_color(rng: random.Random, colors: tuple[str, ...]) -> str:
    if len(colors) == 1:
        return colors[0]
    return rng.choice(colors)


def _background_fill(mode: str, color: str) -> tuple[int, ...]:
    value = _mono_value(color)
    if mode == "RGBA":
        return (value, value, value, 255)
    return (value, value, value)


def _apply_reencode(
    img: "Image.Image", state: _ImageState, config: MediaRandomizerConfig, rng: random.Random
) -> tuple["Image.Image", tuple[int, int], dict[str, Any]]:
    return img, img.size, {}


def _apply_border(
    img: "Image.Image", state: _ImageState, config: MediaRandomizerConfig, rng: random.Random
) -> tuple["Image.Image", tuple[int, int], dict[str, Any]]:
    thickness = rng.randint(config.border_px[0], config.border_px[1])
    color = _pick_color(rng, config.border_colors)
    width, height = img.size
    background = Image.new(
        img.mode,
        (width + thickness * 2, height + thickness * 2),
        _background_fill(img.mode, color),
    )
    # 直接复制原图像素, 保留原内容的 Alpha; 新增边框区域为不透明背景
    background.paste(img, (thickness, thickness))
    return background, background.size, {"thickness": thickness, "color": color}


def _apply_canvas(
    img: "Image.Image", state: _ImageState, config: MediaRandomizerConfig, rng: random.Random
) -> tuple["Image.Image", tuple[int, int], dict[str, Any]]:
    pad_w = rng.randint(config.canvas_px[0], config.canvas_px[1])
    pad_h = rng.randint(config.canvas_px[0], config.canvas_px[1])
    offset_x = rng.randint(0, pad_w)
    offset_y = rng.randint(0, pad_h)
    color = _pick_color(rng, config.canvas_colors)
    width, height = img.size
    background = Image.new(
        img.mode,
        (width + pad_w, height + pad_h),
        _background_fill(img.mode, color),
    )
    background.paste(img, (offset_x, offset_y))
    return (
        background,
        background.size,
        {"pad": (pad_w, pad_h), "offset": (offset_x, offset_y), "color": color},
    )


def _crop_options(state: _ImageState, config: MediaRandomizerConfig) -> dict[str, tuple[int, int]]:
    lo, hi = config.crop_px
    horizontal = (lo, min(hi, state.width - 1))
    vertical = (lo, min(hi, state.height - 1))
    options: dict[str, tuple[int, int]] = {}
    if horizontal[0] <= horizontal[1]:
        options["left"] = horizontal
        options["right"] = horizontal
    if vertical[0] <= vertical[1]:
        options["top"] = vertical
        options["bottom"] = vertical
    return options


def _apply_crop(
    img: "Image.Image", state: _ImageState, config: MediaRandomizerConfig, rng: random.Random
) -> tuple["Image.Image", tuple[int, int], dict[str, Any]]:
    options = _crop_options(state, config)
    edge = rng.choice(sorted(options))
    amount = rng.randint(options[edge][0], options[edge][1])
    width, height = img.size
    boxes = {
        "left": (amount, 0, width, height),
        "right": (0, 0, width - amount, height),
        "top": (0, amount, width, height),
        "bottom": (0, 0, width, height - amount),
    }
    cropped = img.crop(boxes[edge])
    return cropped, cropped.size, {"edge": edge, "amount": amount}


def _noise_region_pixels(state: _ImageState, config: MediaRandomizerConfig) -> list[tuple[int, int]]:
    region = min(config.noise_region_px, state.width // 2, state.height // 2)
    if region < 1:
        return []
    width, height = state.width, state.height
    return [
        (x, y)
        for y in range(height)
        for x in range(width)
        if x < region or x >= width - region or y < region or y >= height - region
    ]


def _noise_plan(
    state: _ImageState,
    config: MediaRandomizerConfig,
    count_range: tuple[int, int],
) -> tuple[list[tuple[int, int]], int, int] | None:
    """噪声像素数 = 随机范围 ∩ 比例上限 ∩ 区域实际像素数; 不可行时返回 None."""
    coords = _noise_region_pixels(state, config)
    if not coords:
        return None
    cap = math.floor(state.width * state.height * config.max_modified_ratio)
    hi = min(count_range[1], cap)
    if hi < 1:
        return None
    lo = min(count_range[0], hi)
    if lo < 1:
        return None
    return coords, lo, hi


def _apply_edge_noise(
    img: "Image.Image", state: _ImageState, config: MediaRandomizerConfig, rng: random.Random
) -> tuple["Image.Image", tuple[int, int], dict[str, Any]]:
    plan = _noise_plan(state, config, config.edge_noise_count)
    assert plan is not None
    coords, lo, hi = plan
    count = min(rng.randint(lo, hi), len(coords))
    chosen = rng.sample(coords, count)
    pixels = img.load()
    for x, y in chosen:
        value = _mono_value(_pick_color(rng, config.noise_colors))
        current = pixels[x, y]
        if state.has_alpha:
            # 保持 Alpha 不变, 仅覆盖 RGB
            pixels[x, y] = (value, value, value, current[3])
        else:
            pixels[x, y] = (value, value, value)
    return img, img.size, {"count": count, "region": config.noise_region_px}


def _apply_pixel_noise(
    img: "Image.Image", state: _ImageState, config: MediaRandomizerConfig, rng: random.Random
) -> tuple["Image.Image", tuple[int, int], dict[str, Any]]:
    plan = _noise_plan(state, config, config.pixel_noise_count)
    assert plan is not None
    coords, lo, hi = plan
    count = min(rng.randint(lo, hi), len(coords))
    chosen = rng.sample(coords, count)
    delta_lo, delta_hi = config.pixel_noise_delta
    pixels = img.load()
    for x, y in chosen:
        current = pixels[x, y]
        shifted = []
        for channel in range(3):
            delta = rng.randint(delta_lo, delta_hi) * rng.choice((-1, 1))
            shifted.append(min(255, max(0, current[channel] + delta)))
        if state.has_alpha:
            shifted.append(current[3])
        pixels[x, y] = tuple(shifted)
    return img, img.size, {"count": count, "region": config.noise_region_px}


def _check_reencode(config: MediaRandomizerConfig, state: _ImageState) -> bool:
    return True


def _check_border(config: MediaRandomizerConfig, state: _ImageState) -> bool:
    return True


def _check_canvas(config: MediaRandomizerConfig, state: _ImageState) -> bool:
    return True


def _check_crop(config: MediaRandomizerConfig, state: _ImageState) -> bool:
    return bool(_crop_options(state, config))


def _check_edge_noise(config: MediaRandomizerConfig, state: _ImageState) -> bool:
    return _noise_plan(state, config, config.edge_noise_count) is not None


def _check_pixel_noise(config: MediaRandomizerConfig, state: _ImageState) -> bool:
    return _noise_plan(state, config, config.pixel_noise_count) is not None


_STRATEGY_APPLY: dict[str, Callable[..., tuple["Image.Image", tuple[int, int], dict[str, Any]]]] = {
    "reencode": _apply_reencode,
    "border": _apply_border,
    "crop": _apply_crop,
    "canvas": _apply_canvas,
    "edge_noise": _apply_edge_noise,
    "pixel_noise": _apply_pixel_noise,
}

_STRATEGY_CHECKS: dict[str, Callable[[MediaRandomizerConfig, _ImageState], bool]] = {
    "reencode": _check_reencode,
    "border": _check_border,
    "crop": _check_crop,
    "canvas": _check_canvas,
    "edge_noise": _check_edge_noise,
    "pixel_noise": _check_pixel_noise,
}


def _choose_static_strategy(
    config: MediaRandomizerConfig, state: _ImageState, rng: random.Random
) -> str | None:
    """先过滤 (零权重 / 参数无效 / 尺寸不允许) 再加权随机."""
    eligible_names: list[str] = []
    eligible_weights: list[int] = []
    for name in STATIC_STRATEGIES:
        weight = config.weights.get(name, 0)
        if weight <= 0:
            continue
        if not _STRATEGY_CHECKS[name](config, state):
            continue
        eligible_names.append(name)
        eligible_weights.append(weight)
    if not eligible_names:
        return None
    return rng.choices(eligible_names, weights=eligible_weights, k=1)[0]


# endregion


# region 保存与校验
def _save_image(
    img: "Image.Image", part_path: Path, state: _ImageState, config: MediaRandomizerConfig
) -> None:
    kwargs: dict[str, Any] = {}
    if state.format == "JPEG":
        kwargs["quality"] = config.jpeg_quality
        kwargs["subsampling"] = 0
    elif state.format == "PNG":
        kwargs["compress_level"] = config.png_compress_level
    elif state.format == "WEBP":
        if state.webp_lossless:
            kwargs["lossless"] = True
        else:
            kwargs["quality"] = config.webp_quality
        # exact=True 保留透明位置的 RGB
        kwargs["exact"] = True
    if state.exif:
        kwargs["exif"] = state.exif
    if state.icc_profile:
        kwargs["icc_profile"] = state.icc_profile
    img.save(part_path, format=state.format, **kwargs)


def _verify_output(
    path: Path, expected_format: str, expected_size: tuple[int, int]
) -> None:
    """重新打开 .part 文件确认格式、尺寸与单帧状态, 失败则不发布."""
    with Image.open(path) as output:
        output.load()
        actual_format = (output.format or "").upper()
        if actual_format != expected_format:
            raise ValueError(f"输出格式校验失败: 期望 {expected_format}, 实际 {actual_format}")
        if tuple(output.size) != tuple(expected_size):
            raise ValueError(f"输出尺寸校验失败: 期望 {expected_size}, 实际 {output.size}")
        if getattr(output, "is_animated", False) or getattr(output, "n_frames", 1) > 1:
            raise ValueError("输出校验失败: 出现多帧")


# endregion


# region 后台线程处理
@dataclass
class _PrepareOutcome:
    status: str  # "prepared" | "skipped" | "failed"
    path: Path | None = None
    created: tuple[Path, ...] = ()
    detail: str = ""
    elapsed: float = 0.0

    def cleanup_files(self) -> None:
        for path in self.created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _prepare_in_thread(
    source: Path,
    config: MediaRandomizerConfig,
    work_dir: Path,
    rng_seed: int,
) -> _PrepareOutcome:
    started = time.perf_counter()
    rng = random.Random(rng_seed)
    created: list[Path] = []
    try:
        with Image.open(source) as raw:
            raw.load()
            skip_reason = _static_skip_reason(raw)
            if skip_reason is not None:
                return _PrepareOutcome(
                    "skipped", detail=skip_reason, elapsed=time.perf_counter() - started
                )
            state = _build_state(raw, source)
            strategy = _choose_static_strategy(config, state, rng)
            if strategy is None:
                return _PrepareOutcome(
                    "skipped", detail="无可用策略", elapsed=time.perf_counter() - started
                )
            img = raw if raw.mode == state.mode else raw.convert(state.mode)
            working, expected_size, params = _STRATEGY_APPLY[strategy](img, state, config, rng)

            extension = FORMAT_EXTENSIONS[state.format]
            work_dir.mkdir(parents=True, exist_ok=True)
            final_path = work_dir / f"{uuid.uuid4().hex}{extension}"
            part_path = final_path.with_name(final_path.name + ".part")
            created.append(part_path)
            _save_image(working, part_path, state, config)
            _verify_output(part_path, state.format, expected_size)
            part_path.replace(final_path)
            created.append(final_path)

            detail = f"策略={strategy}"
            if params:
                detail += ", " + ", ".join(f"{key}={value}" for key, value in params.items())
            return _PrepareOutcome(
                "prepared",
                path=final_path,
                created=(final_path,),
                detail=detail,
                elapsed=time.perf_counter() - started,
            )
    except Exception as exc:
        for path in created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return _PrepareOutcome(
            "failed",
            detail=f"{type(exc).__name__}: {exc}",
            elapsed=time.perf_counter() - started,
        )


# endregion


# region 对外接口
async def prepare_media(source: Path, context: MediaPrepareContext | None) -> Path:
    """对已下载的图片执行随机预处理, 返回处理后的文件路径.

    功能关闭、动图、格式不支持或处理失败时返回原路径;
    事务内同一内容 (SHA-256) 只处理一次, 结果复用。
    """
    if context is None or not context.config.enabled:
        return source
    if Image is None:
        return source
    src = Path(source)
    digest = await context.digest_for(src)
    if digest is None:
        return src
    cached = context.cached_result(digest)
    if cached is not _MISSING:
        if isinstance(cached, Path):
            return cached
        return src

    work_dir = context.ensure_work_dir()
    rng_seed = context.rng.getrandbits(64)
    task = asyncio.create_task(
        asyncio.to_thread(_prepare_in_thread, src, context.config, work_dir, rng_seed)
    )
    outcome: _PrepareOutcome | None
    cancelled = False
    try:
        outcome = await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled = True
        # 等待后台线程结束, 防止清理后线程继续写出文件
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            outcome = task.result()
        except (Exception, asyncio.CancelledError):
            outcome = None
    except Exception:
        outcome = None

    if cancelled:
        if outcome is not None:
            outcome.cleanup_files()
        raise asyncio.CancelledError

    if outcome is not None and outcome.status == "prepared" and outcome.path is not None:
        context.remember_prepared(digest, outcome.path, outcome.elapsed)
        logger.debug(
            "🖼️ X图片随机预处理: 源=%s, %s, 耗时=%.3fs, 输出=%s",
            src.name,
            outcome.detail,
            outcome.elapsed,
            outcome.path.name,
        )
        return outcome.path

    fallback_reason = outcome.detail if outcome is not None else "处理线程异常"
    fallback_elapsed = outcome.elapsed if outcome is not None else 0.0
    context.remember_fallback(digest)
    logger.debug(
        "🖼️ X图片随机预处理回退原图: 源=%s, 原因=%s, 耗时=%.3fs",
        src.name,
        fallback_reason,
        fallback_elapsed,
    )
    return src


# endregion
