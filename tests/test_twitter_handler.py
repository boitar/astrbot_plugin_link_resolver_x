# ruff: noqa: E402
"""Integration-style tests for Twitter/X handler entrypoints.

Run inside AstrBot container:
    cd /AstrBot
    python /AstrBot/data/plugins/astrbot_plugin_link_resolver/tests/test_twitter_handler.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

for candidate in Path(__file__).resolve().parents:
    if (candidate / "data" / "plugins").exists():
        root_path = str(candidate)
        if root_path not in sys.path:
            sys.path.insert(0, root_path)
        break

from astrbot.api.message_components import Image, Plain, Video
from PIL import Image as PILImage

from data.plugins.astrbot_plugin_link_resolver.core.twitter import TwitterResult
from data.plugins.astrbot_plugin_link_resolver.core.twitter.handler import TwitterMixin
from data.plugins.astrbot_plugin_link_resolver.main import LinkResolver

MEDIA_RANDOMIZER_MODULE = (
    "data.plugins.astrbot_plugin_link_resolver.core.common.media_randomizer"
)


class DummyEvent:
    def __init__(self, message_str: str = "", components: list | None = None):
        self.message_str = message_str
        self.message_obj = SimpleNamespace(message=components or [], raw_message=None)
        self.bot = None
        self.sent = []
        self._llm = False

    def get_sender_id(self):
        return "10001"

    def get_self_id(self):
        return "20002"

    def get_group_id(self):
        return "30003"

    def should_call_llm(self, value: bool):
        self._llm = value

    async def send(self, chain):
        self.sent.append(chain)


class TestTwitterHandler(unittest.IsolatedAsyncioTestCase):
    def test_conf_schema_exposes_x_platform_and_twitter_settings(self):
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertIn("X", schema["enable_platforms"]["options"])
        self.assertIn("X", schema["enable_platforms"]["default"])
        self.assertEqual(
            schema["twitter_settings"]["items"]["max_media"]["default"], 99
        )
        self.assertFalse(schema["twitter_settings"]["items"]["merge_send"]["default"])

    async def test_process_twitter_image_post_merges_with_plain_summary(self):
        event = DummyEvent()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_a = Path(tmpdir) / "a.jpg"
            image_b = Path(tmpdir) / "b.jpg"
            image_a.write_bytes(b"a")
            image_b.write_bytes(b"b")

            plugin = SimpleNamespace(
                twitter_enabled=True,
                twitter_merge_send=False,
                twitter_max_media=99,
                retry_count=0,
                max_video_size_mb=200,
                twitter_extractor=SimpleNamespace(
                    parse=AsyncMock(
                        return_value=TwitterResult(
                            text="图片正文",
                            author="Alice(@alice)",
                            created_at="2025-04-01",
                            image_urls=[
                                "https://pbs.twimg.com/media/a.jpg",
                                "https://pbs.twimg.com/media/b.jpg",
                            ],
                            video_urls=[],
                            source_url="https://x.com/alice/status/1234567890123456789",
                            tweet_id="1234567890123456789",
                        )
                    )
                ),
                _refresh_config=lambda: None,
                _send_reaction_emoji=AsyncMock(),
                _download_twitter_image=AsyncMock(side_effect=[image_a, image_b]),
                _download_twitter_video=AsyncMock(),
                _prepare_component_for_merge_send=AsyncMock(
                    side_effect=lambda component: component
                ),
                _get_merge_sender_uin=lambda event: "10001",
                cleanup_files=AsyncMock(),
            )
            plugin._build_twitter_summary = TwitterMixin._build_twitter_summary.__get__(
                plugin, TwitterMixin
            )

            await TwitterMixin._process_twitter(
                plugin, event, "https://x.com/alice/status/1234567890123456789"
            )

        self.assertEqual(len(event.sent), 1)
        nodes = event.sent[0].chain[0]
        self.assertGreaterEqual(len(nodes.nodes), 3)
        first_component = nodes.nodes[0].content[0]
        self.assertIsInstance(first_component, Plain)
        self.assertIn("X @Alice(@alice)", first_component.text)
        self.assertIn("图片正文", first_component.text)

    async def test_process_twitter_single_video_non_merged_sends_only_video(self):
        event = DummyEvent()
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "demo.mp4"
            video_path.write_bytes(b"video")

            plugin = SimpleNamespace(
                twitter_enabled=True,
                twitter_merge_send=False,
                twitter_max_media=99,
                retry_count=0,
                max_video_size_mb=200,
                twitter_extractor=SimpleNamespace(
                    parse=AsyncMock(
                        return_value=TwitterResult(
                            text="视频正文",
                            author="Bob(@bob)",
                            created_at="2025-05-02",
                            image_urls=[],
                            video_urls=[
                                "https://video.twimg.com/ext_tw_video/demo.mp4"
                            ],
                            source_url="https://x.com/bob/status/1234567890123456789",
                            tweet_id="1234567890123456789",
                        )
                    )
                ),
                _refresh_config=lambda: None,
                _send_reaction_emoji=AsyncMock(),
                _download_twitter_video=AsyncMock(return_value=video_path),
                _download_twitter_image=AsyncMock(),
                _prepare_component_for_merge_send=AsyncMock(),
                _get_merge_sender_uin=lambda event: "10001",
                cleanup_files=AsyncMock(),
            )
            plugin._build_twitter_summary = TwitterMixin._build_twitter_summary.__get__(
                plugin, TwitterMixin
            )

            await TwitterMixin._process_twitter(
                plugin, event, "https://x.com/bob/status/1234567890123456789"
            )

        plugin._prepare_component_for_merge_send.assert_not_awaited()
        self.assertEqual(len(event.sent), 1)
        chain = event.sent[0].chain
        self.assertEqual(len(chain), 1)
        self.assertIsInstance(chain[0], Video)

    async def test_process_twitter_cleans_download_when_send_fails(self):
        event = DummyEvent()
        event.send = AsyncMock(side_effect=RuntimeError("send failed"))
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "demo.mp4"
            video_path.write_bytes(b"video")
            plugin = SimpleNamespace(
                twitter_enabled=True,
                twitter_merge_send=False,
                twitter_max_media=99,
                retry_count=0,
                max_video_size_mb=200,
                twitter_extractor=SimpleNamespace(
                    parse=AsyncMock(
                        return_value=TwitterResult(
                            text="视频正文",
                            author="Bob(@bob)",
                            created_at="2025-05-02",
                            image_urls=[],
                            video_urls=["https://video.twimg.com/demo.mp4"],
                            source_url="https://x.com/bob/status/1234567890123456789",
                            tweet_id="1234567890123456789",
                        )
                    )
                ),
                _refresh_config=lambda: None,
                _send_reaction_emoji=AsyncMock(),
                _download_twitter_video=AsyncMock(return_value=video_path),
                _download_twitter_image=AsyncMock(),
                _prepare_component_for_merge_send=AsyncMock(),
                _get_merge_sender_uin=lambda event: "10001",
                cleanup_files=AsyncMock(),
            )
            plugin._build_twitter_summary = TwitterMixin._build_twitter_summary.__get__(
                plugin, TwitterMixin
            )

            with self.assertRaisesRegex(RuntimeError, "send failed"):
                await TwitterMixin._process_twitter(
                    plugin, event, "https://x.com/bob/status/1234567890123456789"
                )

        plugin.cleanup_files.assert_awaited_once_with([video_path], [])

    async def test_process_twitter_single_video_merged_sends_summary_plus_video(self):
        event = DummyEvent()
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "demo.mp4"
            video_path.write_bytes(b"video")

            plugin = SimpleNamespace(
                twitter_enabled=True,
                twitter_merge_send=True,
                twitter_max_media=99,
                retry_count=0,
                max_video_size_mb=200,
                twitter_extractor=SimpleNamespace(
                    parse=AsyncMock(
                        return_value=TwitterResult(
                            text="视频正文",
                            author="Bob(@bob)",
                            created_at="2025-05-02",
                            image_urls=[],
                            video_urls=[
                                "https://video.twimg.com/ext_tw_video/demo.mp4"
                            ],
                            source_url="https://x.com/bob/status/1234567890123456789",
                            tweet_id="1234567890123456789",
                        )
                    )
                ),
                _refresh_config=lambda: None,
                _send_reaction_emoji=AsyncMock(),
                _download_twitter_video=AsyncMock(return_value=video_path),
                _download_twitter_image=AsyncMock(),
                _prepare_component_for_merge_send=AsyncMock(
                    side_effect=lambda component: component
                ),
                _get_merge_sender_uin=lambda event: "10001",
                cleanup_files=AsyncMock(),
            )
            plugin._build_twitter_summary = TwitterMixin._build_twitter_summary.__get__(
                plugin, TwitterMixin
            )

            await TwitterMixin._process_twitter(
                plugin, event, "https://x.com/bob/status/1234567890123456789"
            )

        self.assertEqual(len(event.sent), 1)
        nodes = event.sent[0].chain[0]
        self.assertEqual(len(nodes.nodes), 2)
        self.assertIsInstance(nodes.nodes[0].content[0], Plain)
        plugin._prepare_component_for_merge_send.assert_awaited_once()

    async def test_process_twitter_mixed_media_forces_merge_even_when_disabled(self):
        event = DummyEvent()
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "demo.mp4"
            image_path = Path(tmpdir) / "demo.jpg"
            video_path.write_bytes(b"video")
            image_path.write_bytes(b"image")

            plugin = SimpleNamespace(
                twitter_enabled=True,
                twitter_merge_send=False,
                twitter_max_media=99,
                retry_count=0,
                max_video_size_mb=200,
                twitter_extractor=SimpleNamespace(
                    parse=AsyncMock(
                        return_value=TwitterResult(
                            text="混合正文",
                            author="Carol(@carol)",
                            created_at="2025-06-07",
                            image_urls=["https://pbs.twimg.com/media/a.jpg"],
                            video_urls=["https://video.twimg.com/ext_tw_video/a.mp4"],
                            source_url="https://x.com/carol/status/1234567890123456789",
                            tweet_id="1234567890123456789",
                        )
                    )
                ),
                _refresh_config=lambda: None,
                _send_reaction_emoji=AsyncMock(),
                _download_twitter_video=AsyncMock(return_value=video_path),
                _download_twitter_image=AsyncMock(return_value=image_path),
                _prepare_component_for_merge_send=AsyncMock(
                    side_effect=lambda component: component
                ),
                _get_merge_sender_uin=lambda event: "10001",
                cleanup_files=AsyncMock(),
            )
            plugin._build_twitter_summary = TwitterMixin._build_twitter_summary.__get__(
                plugin, TwitterMixin
            )

            await TwitterMixin._process_twitter(
                plugin, event, "https://x.com/carol/status/1234567890123456789"
            )

        self.assertEqual(len(event.sent), 1)
        nodes = event.sent[0].chain[0]
        self.assertEqual(len(nodes.nodes), 3)
        self.assertIsInstance(nodes.nodes[0].content[0], Plain)
        self.assertEqual(plugin._prepare_component_for_merge_send.await_count, 2)

    async def test_process_twitter_randomizes_images_and_cleans_copies(self):
        event = DummyEvent()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            work_dir = tmp / "randomizer"
            image_a = tmp / "a.jpg"
            image_b = tmp / "b.jpg"
            PILImage.new("RGB", (32, 24), (10, 20, 30)).save(image_a, format="JPEG")
            PILImage.new("RGB", (40, 30), (50, 60, 70)).save(image_b, format="JPEG")
            original_hashes = {image_a.read_bytes(), image_b.read_bytes()}

            plugin = SimpleNamespace(
                twitter_enabled=True,
                twitter_merge_send=False,
                twitter_max_media=99,
                retry_count=0,
                max_video_size_mb=200,
                media_randomizer_enabled=True,
                media_randomizer_weights={
                    "reencode": 1,
                    "border": 0,
                    "crop": 0,
                    "canvas": 0,
                    "edge_noise": 0,
                    "pixel_noise": 0,
                },
                twitter_extractor=SimpleNamespace(
                    parse=AsyncMock(
                        return_value=TwitterResult(
                            text="图片正文",
                            author="Alice(@alice)",
                            created_at="2025-04-01",
                            image_urls=[
                                "https://pbs.twimg.com/media/a.jpg",
                                "https://pbs.twimg.com/media/b.jpg",
                            ],
                            video_urls=[],
                            source_url="https://x.com/alice/status/1234567890123456789",
                            tweet_id="1234567890123456789",
                        )
                    )
                ),
                _refresh_config=lambda: None,
                _send_reaction_emoji=AsyncMock(),
                _download_twitter_image=AsyncMock(side_effect=[image_a, image_b]),
                _download_twitter_video=AsyncMock(),
                _prepare_component_for_merge_send=AsyncMock(
                    side_effect=lambda component: component
                ),
                _get_merge_sender_uin=lambda event: "10001",
                cleanup_files=AsyncMock(),
            )
            plugin._build_twitter_summary = TwitterMixin._build_twitter_summary.__get__(
                plugin, TwitterMixin
            )

            sent_images: list[dict] = []

            async def send_hook(chain):
                for node in chain.chain[0].nodes:
                    for component in node.content:
                        if not isinstance(component, Image):
                            continue
                        path = Path(component.path)
                        sent_images.append(
                            {
                                "path": path,
                                "existed": path.exists(),
                                "bytes": path.read_bytes() if path.exists() else None,
                            }
                        )

            event.send = AsyncMock(side_effect=send_hook)

            with patch(
                f"{MEDIA_RANDOMIZER_MODULE}.get_media_randomizer_path",
                return_value=work_dir,
            ):
                await TwitterMixin._process_twitter(
                    plugin, event, "https://x.com/alice/status/1234567890123456789"
                )

        # 发送的是预处理副本: 位于专用目录、与原图字节不同、发送时真实存在
        self.assertEqual(len(sent_images), 2)
        for sent in sent_images:
            self.assertEqual(sent["path"].parent, work_dir.resolve())
            self.assertEqual(sent["path"].suffix, ".jpg")
            self.assertTrue(sent["existed"])
            self.assertIsNotNone(sent["bytes"])
            self.assertNotIn(sent["bytes"], original_hashes)
        # 发送结束后副本被清理
        for sent in sent_images:
            self.assertFalse(sent["path"].exists())
        # 原图仍走现有下载文件清理机制
        plugin.cleanup_files.assert_awaited_once_with([image_a, image_b], [])

    async def test_process_twitter_randomizer_disabled_sends_original(self):
        event = DummyEvent()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            work_dir = tmp / "randomizer"
            image_a = tmp / "a.jpg"
            PILImage.new("RGB", (32, 24), (10, 20, 30)).save(image_a, format="JPEG")

            plugin = SimpleNamespace(
                twitter_enabled=True,
                twitter_merge_send=False,
                twitter_max_media=99,
                retry_count=0,
                max_video_size_mb=200,
                media_randomizer_enabled=False,
                twitter_extractor=SimpleNamespace(
                    parse=AsyncMock(
                        return_value=TwitterResult(
                            text="图片正文",
                            author="Alice(@alice)",
                            created_at="2025-04-01",
                            image_urls=["https://pbs.twimg.com/media/a.jpg"],
                            video_urls=[],
                            source_url="https://x.com/alice/status/1234567890123456789",
                            tweet_id="1234567890123456789",
                        )
                    )
                ),
                _refresh_config=lambda: None,
                _send_reaction_emoji=AsyncMock(),
                _download_twitter_image=AsyncMock(return_value=image_a),
                _download_twitter_video=AsyncMock(),
                _prepare_component_for_merge_send=AsyncMock(
                    side_effect=lambda component: component
                ),
                _get_merge_sender_uin=lambda event: "10001",
                cleanup_files=AsyncMock(),
            )
            plugin._build_twitter_summary = TwitterMixin._build_twitter_summary.__get__(
                plugin, TwitterMixin
            )

            sent_image_paths: list[Path] = []

            async def send_hook(chain):
                for node in chain.chain[0].nodes:
                    for component in node.content:
                        if isinstance(component, Image):
                            sent_image_paths.append(Path(component.path))

            event.send = AsyncMock(side_effect=send_hook)

            with patch(
                f"{MEDIA_RANDOMIZER_MODULE}.get_media_randomizer_path",
                return_value=work_dir,
            ):
                await TwitterMixin._process_twitter(
                    plugin, event, "https://x.com/alice/status/1234567890123456789"
                )

        self.assertEqual(len(sent_image_paths), 1)
        self.assertEqual(sent_image_paths[0], image_a.resolve())
        self.assertFalse(work_dir.exists())  # 关闭时不创建任何副本目录
        plugin.cleanup_files.assert_awaited_once_with([image_a], [])

    async def test_handle_json_card_dispatches_twitter_link(self):
        event = DummyEvent(
            components=[
                {
                    "type": "json",
                    "data": {
                        "meta": {
                            "detail_1": {
                                "url": "https://x.com/demo/status/1234567890123456789"
                            }
                        }
                    },
                }
            ]
        )
        plugin = LinkResolver.__new__(LinkResolver)
        plugin.bili_enabled = False
        plugin.douyin_enabled = False
        plugin.xhs_enabled = False
        plugin.weibo_enabled = False
        plugin.twitter_enabled = True
        plugin.group_filter_mode = "黑名单"
        plugin.group_filter_list = []
        plugin._register_parse_task = lambda *args, **kwargs: None
        plugin._is_bot_muted = AsyncMock(return_value=False)
        plugin._process_twitter = AsyncMock()
        plugin.extract_links_from_json = LinkResolver.extract_links_from_json.__get__(
            plugin, LinkResolver
        )
        plugin._coerce_json_payload = LinkResolver._coerce_json_payload.__get__(
            plugin, LinkResolver
        )

        async for _ in LinkResolver.handle_json_card(plugin, event):
            pass

        plugin._process_twitter.assert_awaited_once_with(
            event, "https://x.com/demo/status/1234567890123456789", is_from_card=True
        )
        self.assertTrue(event._llm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
