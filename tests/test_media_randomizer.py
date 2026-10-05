# ruff: noqa: E402
"""X 图片随机预处理单元测试.

本地运行:
    PYTHONPATH=<包含 astrbot_plugin_link_resolver 目录的父目录> \\
        python -m pytest tests/test_media_randomizer.py -v

AstrBot 容器内运行:
    cd /AstrBot
    python data/plugins/astrbot_plugin_link_resolver/tests/test_media_randomizer.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from astrbot_plugin_link_resolver.core.common.media_randomizer import (
    MediaPrepareContext,
    MediaRandomizerConfig,
    STATIC_STRATEGIES,
    _PrepareOutcome,
    cleanup_media_randomizer_directory,
    prepare_media,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _only_strategy(name: str, **overrides) -> MediaRandomizerConfig:
    weights = {s: (1 if s == name else 0) for s in STATIC_STRATEGIES}
    return MediaRandomizerConfig(weights=weights, **overrides)


def _make_source(tmp_path: Path, fmt: str, mode: str = "RGB", size=(64, 48)) -> Path:
    ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]
    source = tmp_path / f"src{ext}"
    color = (200, 100, 50, 128) if mode == "RGBA" else (200, 100, 50)
    Image.new(mode, size, color).save(source, format=fmt)
    return source


def _assert_strategy_size(strategy: str, before: tuple, after: tuple) -> None:
    dw = after[0] - before[0]
    dh = after[1] - before[1]
    if strategy in ("reencode", "edge_noise", "pixel_noise"):
        assert (dw, dh) == (0, 0)
    elif strategy == "border":
        assert 2 <= dw <= 6 and 2 <= dh <= 6
        assert dw == dh  # 四边使用同一厚度
        assert dw % 2 == 0 and dh % 2 == 0
    elif strategy == "crop":
        assert (dw < 0 and dh == 0) or (dw == 0 and dh < 0)
        assert -2 <= min(dw, dh) <= -1
    elif strategy == "canvas":
        assert 1 <= dw <= 6 and 1 <= dh <= 6


# region 六种策略 × 三种静态格式
@pytest.mark.parametrize("strategy", STATIC_STRATEGIES)
@pytest.mark.parametrize(
    "fmt,mode", [("JPEG", "RGB"), ("PNG", "RGBA"), ("WEBP", "RGBA")]
)
@pytest.mark.asyncio
async def test_strategy_x_format_matrix(tmp_path, strategy, fmt, mode):
    source = _make_source(tmp_path, fmt, mode)
    before_hash = _sha256(source)
    config = _only_strategy(strategy)
    context = MediaPrepareContext(
        config=config, rng=random.Random(7), work_dir=tmp_path / "work"
    )

    result = await prepare_media(source, context)

    assert result != source
    assert result.exists()
    assert result.parent == context.ensure_work_dir()
    assert result.suffix == {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]
    with Image.open(source) as original, Image.open(result) as output:
        assert output.format == fmt
        _assert_strategy_size(strategy, original.size, output.size)
        if fmt != "JPEG" and mode == "RGBA":
            # 透明通道保留
            assert output.mode == "RGBA"
    # 原图只读
    assert _sha256(source) == before_hash
    assert context.processed_count == 1
    assert context.generated_files == [result]

    await context.cleanup()
    assert not result.exists()


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
@pytest.mark.asyncio
async def test_same_seed_same_strategy_produces_identical_bytes(tmp_path, fmt):
    source = _make_source(tmp_path, fmt)
    results = []
    for index in range(2):
        context = MediaPrepareContext(
            config=_only_strategy("pixel_noise"),
            rng=random.Random(99),
            work_dir=tmp_path / f"work-{index}",
        )
        results.append(await prepare_media(source, context))

    assert results[0].read_bytes() == results[1].read_bytes()


# endregion


# region PNG 像素级校验
@pytest.mark.asyncio
async def test_border_preserves_content_and_alpha(tmp_path):
    source = _make_source(tmp_path, "PNG", "RGBA", size=(32, 32))
    result = await prepare_media(
        source,
        MediaPrepareContext(
            config=_only_strategy("border"),
            rng=random.Random(3),
            work_dir=tmp_path / "work",
        ),
    )
    with Image.open(source) as original, Image.open(result) as output:
        op, np_pixels = original.load(), output.load()
        w0, h0 = original.size
        w1, h1 = output.size
        thickness = (w1 - w0) // 2
        assert 1 <= thickness <= 3
        assert w1 == w0 + 2 * thickness and h1 == h0 + 2 * thickness
        # 四角为不透明黑/白
        for corner in ((0, 0), (w1 - 1, 0), (0, h1 - 1), (w1 - 1, h1 - 1)):
            assert np_pixels[corner] in ((0, 0, 0, 255), (255, 255, 255, 255))
        # 原内容与 Alpha 完整保留
        assert np_pixels[thickness, thickness] == op[0, 0]
        assert np_pixels[w1 - 1 - thickness, h1 - 1 - thickness] == op[w0 - 1, h0 - 1]


@pytest.mark.asyncio
async def test_crop_output_equals_one_valid_edge_crop(tmp_path):
    source = _make_source(tmp_path, "PNG", "RGB", size=(32, 32))
    result = await prepare_media(
        source,
        MediaPrepareContext(
            config=_only_strategy("crop"),
            rng=random.Random(11),
            work_dir=tmp_path / "work",
        ),
    )
    with Image.open(source) as original, Image.open(result) as output:
        w, h = original.size
        candidates = []
        for amount in (1, 2):
            candidates.extend(
                [
                    original.crop((amount, 0, w, h)),
                    original.crop((0, 0, w - amount, h)),
                    original.crop((0, amount, w, h)),
                    original.crop((0, 0, w, h - amount)),
                ]
            )
        assert any(output.tobytes() == candidate.tobytes() for candidate in candidates)


@pytest.mark.asyncio
async def test_canvas_places_original_with_opaque_background(tmp_path):
    source = _make_source(tmp_path, "PNG", "RGBA", size=(32, 32))
    result = await prepare_media(
        source,
        MediaPrepareContext(
            config=_only_strategy("canvas"),
            rng=random.Random(5),
            work_dir=tmp_path / "work",
        ),
    )
    with Image.open(source) as original, Image.open(result) as output:
        w0, h0 = original.size
        w1, h1 = output.size
        pad_w, pad_h = w1 - w0, h1 - h0
        assert 1 <= pad_w <= 6 and 1 <= pad_h <= 6
        found_offset = False
        for offset_x in range(pad_w + 1):
            for offset_y in range(pad_h + 1):
                if (
                    output.crop(
                        (offset_x, offset_y, offset_x + w0, offset_y + h0)
                    ).tobytes()
                    == original.tobytes()
                ):
                    found_offset = True
        assert found_offset
        # 角落背景不透明黑/白
        assert output.load()[0, 0] in ((0, 0, 0, 255), (255, 255, 255, 255))


def _outer_region(size: tuple, region: int = 2):
    width, height = size
    return {
        (x, y)
        for y in range(height)
        for x in range(width)
        if x < region or x >= width - region or y < region or y >= height - region
    }


@pytest.mark.asyncio
async def test_edge_noise_only_touches_outer_region(tmp_path):
    source = _make_source(tmp_path, "PNG", "RGBA", size=(128, 128))
    result = await prepare_media(
        source,
        MediaPrepareContext(
            config=_only_strategy("edge_noise"),
            rng=random.Random(23),
            work_dir=tmp_path / "work",
        ),
    )
    with Image.open(source) as original, Image.open(result) as output:
        op, np_pixels = original.load(), output.load()
        changed = {
            (x, y)
            for y in range(128)
            for x in range(128)
            if np_pixels[x, y] != op[x, y]
        }
        # 比例上限: floor(128*128*0.001) = 16
        assert 2 <= len(changed) <= 20
        assert changed <= _outer_region((128, 128))
        for x, y in changed:
            expected_rgb = {(0, 0, 0), (255, 255, 255)}
            assert np_pixels[x, y][:3] in expected_rgb
            assert np_pixels[x, y][3] == op[x, y][3]  # Alpha 不变


@pytest.mark.asyncio
async def test_pixel_noise_shifts_limited_channels(tmp_path):
    source = _make_source(tmp_path, "PNG", "RGBA", size=(128, 128))
    result = await prepare_media(
        source,
        MediaPrepareContext(
            config=_only_strategy("pixel_noise"),
            rng=random.Random(31),
            work_dir=tmp_path / "work",
        ),
    )
    with Image.open(source) as original, Image.open(result) as output:
        op, np_pixels = original.load(), output.load()
        changed = {
            (x, y)
            for y in range(128)
            for x in range(128)
            if np_pixels[x, y] != op[x, y]
        }
        # 比例上限: floor(128*128*0.001) = 16
        assert 5 <= len(changed) <= 30
        assert changed <= _outer_region((128, 128))
        for x, y in changed:
            deltas = [np_pixels[x, y][c] - op[x, y][c] for c in range(3)]
            assert all(abs(delta) <= 3 for delta in deltas)
            assert any(deltas)
            assert np_pixels[x, y][3] == op[x, y][3]


# endregion


# region 事务去重
@pytest.mark.asyncio
async def test_duplicate_paths_processed_once(tmp_path):
    source = _make_source(tmp_path, "PNG")
    context = MediaPrepareContext(
        config=_only_strategy("border"), rng=random.Random(1), work_dir=tmp_path / "work"
    )
    first = await prepare_media(source, context)
    second = await prepare_media(source, context)
    assert first == second
    assert context.processed_count == 1


@pytest.mark.asyncio
async def test_same_content_different_paths_processed_once(tmp_path):
    a = _make_source(tmp_path, "PNG")
    b = tmp_path / "copy.png"
    b.write_bytes(a.read_bytes())
    context = MediaPrepareContext(
        config=_only_strategy("border"), rng=random.Random(2), work_dir=tmp_path / "work"
    )
    first = await prepare_media(a, context)
    second = await prepare_media(b, context)
    assert first == second
    assert context.processed_count == 1
    assert context.generated_files == [first]


@pytest.mark.asyncio
async def test_new_context_rerandomizes(tmp_path):
    source = _make_source(tmp_path, "PNG")
    first_ctx = MediaPrepareContext(
        config=_only_strategy("border"), rng=random.Random(3), work_dir=tmp_path / "work1"
    )
    second_ctx = MediaPrepareContext(
        config=_only_strategy("border"), rng=random.Random(3), work_dir=tmp_path / "work2"
    )
    first = await prepare_media(source, first_ctx)
    second = await prepare_media(source, second_ctx)
    assert first != second
    assert first_ctx.processed_count == 1
    assert second_ctx.processed_count == 1


@pytest.mark.asyncio
async def test_different_images_processed_individually(tmp_path):
    a = _make_source(tmp_path, "PNG")
    b = tmp_path / "other.png"
    Image.new("RGB", (40, 30), (1, 2, 3)).save(b, format="PNG")
    context = MediaPrepareContext(
        config=_only_strategy("border"), rng=random.Random(4), work_dir=tmp_path / "work"
    )
    first = await prepare_media(a, context)
    second = await prepare_media(b, context)
    assert first != second
    assert context.processed_count == 2


# endregion


# region 动画保护
@pytest.mark.parametrize("fmt", ["GIF", "PNG", "WEBP"])
@pytest.mark.asyncio
async def test_animated_images_use_original(tmp_path, fmt):
    ext = {"GIF": ".gif", "PNG": ".png", "WEBP": ".webp"}[fmt]
    source = tmp_path / f"anim{ext}"
    base = Image.new("RGB", (16, 16), (255, 0, 0))
    base.save(
        source,
        format=fmt,
        save_all=True,
        append_images=[Image.new("RGB", (16, 16), (0, 255, 0))],
        duration=100,
        loop=0,
    )
    before = _sha256(source)
    context = MediaPrepareContext(
        config=MediaRandomizerConfig(), rng=random.Random(1), work_dir=tmp_path / "work"
    )
    result = await prepare_media(source, context)
    assert result == source
    assert _sha256(source) == before
    assert context.processed_count == 0
    assert context.generated_files == []


# endregion


# region 失败与回退
@pytest.mark.asyncio
async def test_corrupt_file_falls_back_and_is_cached(tmp_path, monkeypatch):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image at all")
    calls: list[Path] = []

    def counting_worker(source, config, work_dir, rng_seed):
        calls.append(source)
        return _PrepareOutcome("failed", detail="cannot identify image")

    monkeypatch.setattr(
        "astrbot_plugin_link_resolver.core.common.media_randomizer._prepare_in_thread",
        counting_worker,
    )
    context = MediaPrepareContext(work_dir=tmp_path / "work")
    first = await prepare_media(bad, context)
    second = await prepare_media(bad, context)
    assert first == second == bad
    assert context.processed_count == 0
    assert context.generated_files == []
    assert len(calls) == 1  # 失败结果在事务内缓存, 不重复尝试


@pytest.mark.asyncio
async def test_real_corrupt_file_falls_back_without_copy(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image at all")
    context = MediaPrepareContext(work_dir=tmp_path / "work")
    result = await prepare_media(bad, context)
    assert result == bad
    assert context.processed_count == 0
    assert context.generated_files == []
    assert not (tmp_path / "work").exists() or not list((tmp_path / "work").glob("*"))


@pytest.mark.asyncio
async def test_disabled_context_returns_source_without_processing(tmp_path, monkeypatch):
    source = _make_source(tmp_path, "PNG")
    calls: list = []

    def fail_worker(*args, **kwargs):
        calls.append(args)
        raise AssertionError("禁用后不应进入处理线程")

    monkeypatch.setattr(
        "astrbot_plugin_link_resolver.core.common.media_randomizer._prepare_in_thread",
        fail_worker,
    )
    context = MediaPrepareContext(
        config=MediaRandomizerConfig(enabled=False), work_dir=tmp_path / "work"
    )
    assert await prepare_media(source, context) == source
    assert calls == []


@pytest.mark.asyncio
async def test_zero_weights_fall_back_to_original(tmp_path):
    source = _make_source(tmp_path, "PNG")
    context = MediaPrepareContext(
        config=MediaRandomizerConfig(weights={s: 0 for s in STATIC_STRATEGIES}),
        work_dir=tmp_path / "work",
    )
    result = await prepare_media(source, context)
    assert result == source
    assert context.processed_count == 0


@pytest.mark.asyncio
async def test_tiny_image_with_noise_only_has_no_eligible_strategy(tmp_path):
    source = _make_source(tmp_path, "PNG", "RGB", size=(4, 4))
    context = MediaPrepareContext(
        config=_only_strategy("edge_noise"), rng=random.Random(1), work_dir=tmp_path / "work"
    )
    result = await prepare_media(source, context)
    # 比例上限 floor(16 * 0.001) = 0, 噪声不可行
    assert result == source
    assert context.processed_count == 0


@pytest.mark.asyncio
async def test_zero_ratio_disables_noise_but_keeps_other_strategies(tmp_path):
    source = _make_source(tmp_path, "PNG")
    context = MediaPrepareContext(
        config=MediaRandomizerConfig(
            weights={"reencode": 0, "border": 0, "crop": 0, "canvas": 0,
                     "edge_noise": 1, "pixel_noise": 1},
            max_modified_ratio=0.0,
        ),
        work_dir=tmp_path / "work",
    )
    result = await prepare_media(source, context)
    assert result == source
    assert context.processed_count == 0


@pytest.mark.asyncio
async def test_cleanup_after_send_false_keeps_copy(tmp_path):
    source = _make_source(tmp_path, "PNG")
    context = MediaPrepareContext(
        config=_only_strategy("border", cleanup_after_send=False),
        rng=random.Random(1),
        work_dir=tmp_path / "work",
    )
    result = await prepare_media(source, context)
    await context.cleanup()
    assert result.exists()  # 由周期清理兜底


# endregion


# region 配置快照校验
def test_from_plugin_defaults():
    config = MediaRandomizerConfig.from_plugin(SimpleNamespace())
    assert config.enabled is True
    assert config.cleanup_after_send is True
    assert config.weights == {
        "reencode": 30,
        "border": 20,
        "crop": 15,
        "canvas": 15,
        "edge_noise": 10,
        "pixel_noise": 10,
    }
    assert config.animation_weights == {"skip": 100}
    assert config.border_colors == ("black", "white")


def test_from_plugin_disables_strategy_with_invalid_params():
    plugin = SimpleNamespace(
        media_randomizer_border_px=(3, 1),
        media_randomizer_max_modified_ratio=5.0,
        media_randomizer_enabled=True,
    )
    config = MediaRandomizerConfig.from_plugin(plugin)
    assert config.weights["border"] == 0
    assert config.weights["edge_noise"] == 0
    assert config.weights["pixel_noise"] == 0
    assert config.weights["reencode"] == 30  # 其他策略不受影响


def test_from_plugin_sanitizes_weights_and_colors():
    plugin = SimpleNamespace(
        media_randomizer_weights={"reencode": -5, "border": "x", "crop": 7},
        media_randomizer_border_color="红色",
        media_randomizer_noise_color="黑",
    )
    config = MediaRandomizerConfig.from_plugin(plugin)
    assert config.weights["reencode"] == 0  # 非法权重按 0 处理
    assert config.weights["border"] == 0
    assert config.weights["crop"] == 7
    assert config.weights["canvas"] == 15  # 未配置的键保持默认
    assert config.border_colors == ("black", "white")  # 非法颜色回退默认
    assert config.noise_colors == ("black",)


def test_from_plugin_respects_enabled_flag():
    config = MediaRandomizerConfig.from_plugin(
        SimpleNamespace(media_randomizer_enabled=False)
    )
    assert config.enabled is False


# endregion


# region 目录清理
def test_directory_cleanup_respects_retention_and_active_files(tmp_path):
    import time as time_module

    now = time_module.time()
    old = tmp_path / "old.png"
    old.write_bytes(b"old")
    os.utime(old, (now - 25 * 3600, now - 25 * 3600))
    old_part = tmp_path / "old.png.part"
    old_part.write_bytes(b"part")
    os.utime(old_part, (now - 25 * 3600, now - 25 * 3600))
    fresh = tmp_path / "fresh.png"
    fresh.write_bytes(b"fresh")
    active = tmp_path / "active.png"
    active.write_bytes(b"active")
    os.utime(active, (now - 25 * 3600, now - 25 * 3600))

    from astrbot_plugin_link_resolver.core.common.media_randomizer import (
        _register_active_file,
        _unregister_active_file,
    )

    _register_active_file(active)
    try:
        deleted, examined = cleanup_media_randomizer_directory(tmp_path, 24.0)
        assert deleted == 2
        assert examined == 4
        assert not old.exists()
        assert not old_part.exists()
        assert fresh.exists()
        assert active.exists()  # 活跃事务文件不清理
    finally:
        _unregister_active_file(active)


def test_directory_cleanup_missing_dir_is_noop(tmp_path):
    deleted, examined = cleanup_media_randomizer_directory(tmp_path / "nope", 24.0)
    assert (deleted, examined) == (0, 0)


@pytest.mark.asyncio
async def test_cleaner_start_stop_roundtrip(tmp_path):
    from astrbot_plugin_link_resolver.core.common.media_randomizer import (
        start_media_randomizer_cleaner,
        stop_media_randomizer_cleaner,
    )

    with patch(
        "astrbot_plugin_link_resolver.core.common.media_randomizer.get_media_randomizer_path",
        return_value=tmp_path,
    ):
        assert start_media_randomizer_cleaner(interval_minutes=0.1, retention_hours=24)
        # 已启动时不重复启动
        assert not start_media_randomizer_cleaner(interval_minutes=0.1)
        await stop_media_randomizer_cleaner()
        # 停止后可重新启动
        assert start_media_randomizer_cleaner(interval_minutes=0.1)
        await stop_media_randomizer_cleaner()


# endregion


# region 取消与线程收敛
@pytest.mark.asyncio
async def test_cancellation_waits_for_thread_and_removes_copy(tmp_path, monkeypatch):
    started = threading.Event()
    allow_finish = threading.Event()
    late_file = tmp_path / "late.png"

    def slow_worker(source, config, work_dir, rng_seed):
        started.set()
        allow_finish.wait(timeout=2)
        late_file.write_bytes(b"late copy")
        return _PrepareOutcome("prepared", path=late_file, created=(late_file,))

    monkeypatch.setattr(
        "astrbot_plugin_link_resolver.core.common.media_randomizer._prepare_in_thread",
        slow_worker,
    )
    source = _make_source(tmp_path, "PNG")
    context = MediaPrepareContext(work_dir=tmp_path / "work")

    task = asyncio.create_task(prepare_media(source, context))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    allow_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    # 线程结束后才清理, 半成品不残留
    assert not late_file.exists()
    assert context.processed_count == 0


@pytest.mark.asyncio
async def test_repeated_cancellation_still_cleans(tmp_path, monkeypatch):
    started = threading.Event()
    allow_finish = threading.Event()
    late_file = tmp_path / "late.png"

    def slow_worker(source, config, work_dir, rng_seed):
        started.set()
        allow_finish.wait(timeout=2)
        late_file.write_bytes(b"late copy")
        return _PrepareOutcome("prepared", path=late_file, created=(late_file,))

    monkeypatch.setattr(
        "astrbot_plugin_link_resolver.core.common.media_randomizer._prepare_in_thread",
        slow_worker,
    )
    source = _make_source(tmp_path, "PNG")
    context = MediaPrepareContext(work_dir=tmp_path / "work")

    task = asyncio.create_task(prepare_media(source, context))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    allow_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not late_file.exists()


@pytest.mark.asyncio
async def test_worker_exception_falls_back_to_original(tmp_path, monkeypatch):
    def broken_worker(source, config, work_dir, rng_seed):
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(
        "astrbot_plugin_link_resolver.core.common.media_randomizer._prepare_in_thread",
        broken_worker,
    )
    source = _make_source(tmp_path, "PNG")
    context = MediaPrepareContext(work_dir=tmp_path / "work")
    result = await prepare_media(source, context)
    assert result == source
    assert context.processed_count == 0


# endregion


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
