"""Image markdown extraction for terminal display."""

from esdc.chat.media import extract_image_urls


def test_extracts_single_image():
    md = "Here is the plot:\n\n![forecast](http://ot.local/files/f_20260714.png)"
    assert extract_image_urls(md) == ["http://ot.local/files/f_20260714.png"]


def test_extracts_multiple_in_order_deduplicated():
    md = (
        "![a](http://x/1.png) text ![b](http://x/2.png) "
        "again ![a2](http://x/1.png)"
    )
    assert extract_image_urls(md) == ["http://x/1.png", "http://x/2.png"]


def test_no_images_returns_empty():
    assert extract_image_urls("plain **markdown**, [link](http://x) only") == []


def test_ignores_empty_target():
    assert extract_image_urls("![broken]()") == []


class TestCompleteBranchImageSurfacing:
    def test_complete_surfaces_images_and_stores_last(self, monkeypatch):
        from esdc.chat.app import ESDCChatApp

        app = ESDCChatApp()
        shown = []
        monkeypatch.setattr(
            app, "display_message", lambda role, c: shown.append((role, c))
        )
        app._accumulated_content = "done ![plot](http://x/plot.png)"
        app._handle_stream_chunk({"type": "complete", "success": True})
        assert app._last_image_url == "http://x/plot.png"
        assert any("http://x/plot.png" in c for _, c in shown)
