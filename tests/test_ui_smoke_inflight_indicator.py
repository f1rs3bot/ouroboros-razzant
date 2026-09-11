"""Browser smoke test for the in-flight direct/ephemeral turn indicator.

Lives in its own module (not test_ui_smoke_playwright.py) so the giant smoke
module stays under the size-ratchet byte gate. Reuses its server fixture.
"""

from __future__ import annotations

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import


@pytest.mark.ui_browser
def test_ui_smoke_chat_inflight_indicator_lifecycle(direct_server_with_data):  # noqa: F811
    """Direct and ephemeral turns display in-flight Thinking... indicator and transitions."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    # NB: the server runs as a subprocess; its DirectActivityRegistry is not
    # reachable from this process. The lifecycle below is driven client-side
    # through the window.__ouroWs debug hook, which is exactly the surface the
    # browser exercises for real WS frames.
    url = direct_server_with_data["url"]

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                status_badge = page.locator("#chat-status")
                status_badge.wait_for(state="attached", timeout=30_000)
                # Idle state
                page.wait_for_function(
                    """() => {
                        const el = document.querySelector('#chat-status');
                        return el && (el.textContent.trim() === 'Online' || el.textContent.trim() === 'Ready');
                    }""",
                    timeout=10_000,
                )

                # Send typing event directly through the window.__ouroWs test hook
                page.evaluate("""() => {
                    if (window.__ouroWs) {
                        window.__ouroWs.emit('typing', {
                            type: 'typing',
                            chat_id: 1,
                            activity_id: 'act-smoke-1',
                            client_message_id: 'msg-smoke-1',
                            phase: 'thinking',
                            kind: 'direct_chat'
                        });
                    }
                }""")

                # In-flight state
                page.wait_for_function(
                    """() => {
                        const el = document.querySelector('#chat-status');
                        return el && el.textContent.trim() === 'Thinking...';
                    }""",
                    timeout=5_000,
                )

                # Dots indicator visible (main chat instance id is #typing-indicator)
                typing_el = page.locator("#typing-indicator")
                typing_el.wait_for(state="visible", timeout=5_000)

                # Dispatch chat assistant message to clear activity
                page.evaluate("""() => {
                    if (window.__ouroWs) {
                        window.__ouroWs.emit('chat', {
                            type: 'chat',
                            role: 'assistant',
                            content: 'Done!',
                            task_id: 'act-smoke-1',
                            chat_id: 1
                        });
                    }
                }""")

                page.wait_for_function(
                    """() => {
                        const el = document.querySelector('#chat-status');
                        return el && el.textContent.trim() === 'Online';
                    }""",
                    timeout=5_000,
                )
                typing_el.wait_for(state="hidden", timeout=5_000)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
