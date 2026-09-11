"""Real Skills geometry and local receipt clearing against an isolated server."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ouroboros.marketplace.provenance import read_publication_record, write_publication_record, merge_state_record
from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state, save_enabled
from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data

pytestmark = pytest.mark.ui_browser
direct_server_with_data = _direct_server_with_data


def _skill(data: Path, name: str, *, reviewed=True, grant=False, version='2.0.0', bucket='external') -> Path:
    payload = data / 'skills' / bucket / name
    (payload / 'scripts').mkdir(parents=True)
    (payload / 'SKILL.md').write_text(
        f'---\nname: {name}\ndescription: Publication layout fixture\nversion: "{version}"\n'
        'type: script\nruntime: python3\nscripts:\n  - name: check.py\n    description: Fixture\n'
        + ('env_from_settings: [OPENROUTER_API_KEY]\n' if grant else '') + '---\n# Fixture\n')
    (payload / 'scripts/check.py').write_text("print('fixture')\n")
    if reviewed:
        save_review_state(data, name, SkillReviewState(status='pass', content_hash=compute_content_hash(payload)))
    save_enabled(data, name, False)
    return payload


def _screenshot(page, name):
    directory = os.environ.get('OUROBOROS_PUBLICATION_SCREENSHOTS')
    if directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output / name))


def test_skills_card_geometry_tracks_card_width_and_keeps_grant_on_card(direct_server_with_data):
    from playwright.sync_api import sync_playwright, expect

    data, url = direct_server_with_data['data_dir'], direct_server_with_data['url']
    names = ['publication_pending_with_a_long_but_valid_skill_name',
             'publication_grant_access_with_a_very_long_skill_name',
             'publication_ready_with_a_long_but_valid_skill_name']
    _skill(data, names[0], reviewed=False)
    _skill(data, names[1], grant=True)
    _skill(data, names[2])
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 719, 'height': 900})
            page.route('**/api/marketplace/ouroboroshub/catalog*', lambda route: route.fulfill(json={'results': []}))
            page.goto(url, wait_until='domcontentloaded')
            page.click('[data-nav-page="skills"]')
            page.wait_for_selector(f'.skills-card[data-skill="{names[1]}"]')
            measurements = []
            for viewport, sidebar in [(719, 280), (1100, 560), (1440, 280)]:
                page.set_viewport_size({'width': viewport, 'height': 900})
                page.evaluate("width => document.getElementById('app').style.setProperty('--sidebar-width', `${width}px`)", sidebar)
                for name in names:
                    card = page.locator(f'.skills-card[data-skill="{name}"]')
                    card.scroll_into_view_if_needed()
                    facts = card.evaluate('''card => {
                        const rect = card.getBoundingClientRect();
                        const title = card.querySelector('.skills-card-title').getBoundingClientRect();
                        const actions = [...card.querySelectorAll('.skills-card-toggle > button, .skills-card-menu-trigger')]
                            .map(button => ({x:button.getBoundingClientRect().x, right:button.getBoundingClientRect().right}));
                        return {card:rect.width, title:title.width, x:rect.x, right:rect.right, actions,
                            direction:getComputedStyle(card.querySelector('.skills-card-head')).flexDirection,
                            sidebar:document.getElementById('primary-sidebar').getBoundingClientRect().width};
                    }''')
                    assert abs(facts['sidebar'] - sidebar) <= 2, facts
                    assert facts['title'] >= 180, facts
                    assert all(action['x'] >= facts['x'] and action['right'] <= facts['right'] for action in facts['actions']), facts
                    assert facts['direction'] == ('column' if viewport != 1440 else 'row'), facts
                    measurements.append({'viewport': viewport, 'name': name, **facts})
                grant = page.locator(f'.skills-card[data-skill="{names[1]}"]')
                grant.scroll_into_view_if_needed()
                expect(grant.get_by_role('button', name='Grant access', exact=True)).to_be_visible()
                _screenshot(page, f'skills-width-{viewport}-sidebar-{sidebar}.png')
            grant.get_by_role('button', name='Grant access', exact=True).click()
            expect(page.locator('.confirm-dialog')).to_contain_text(f'Grant access to {names[1]}')
            _screenshot(page, 'grant-access-dialog.png')
            page.locator('.confirm-dialog [data-confirm-cancel]').last.click()
            if directory := os.environ.get('OUROBOROS_PUBLICATION_SCREENSHOTS'):
                (Path(directory) / 'geometry.json').write_text(json.dumps(measurements, indent=2))
        finally:
            browser.close()


@pytest.mark.parametrize('bucket', ['external', 'ouroboroshub'])
def test_clear_submission_changes_real_receipt_and_refreshes_both_skill_views(direct_server_with_data, bucket):
    from playwright.sync_api import sync_playwright, expect

    data, url = direct_server_with_data['data_dir'], direct_server_with_data['url']
    name = 'publication_waiting_fixture'
    payload = _skill(data, name, bucket=bucket)
    original_hash = compute_content_hash(payload)

    def receipt(number):
        return {'slug': name, 'version': '2.0.0', 'content_hash': original_hash,
                'repository': 'hub/project', 'pr_number': number,
                'pr_url': f'https://github.com/hub/project/pull/{number}', 'published_at': '2026-09-06T00:00:00Z'}

    write_publication_record(data, name, receipt(7))
    merge_state_record(data, name, 'ouroboroshub.json', {'future': {'keep': True}})
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1100, 'height': 900})
            external = []
            page.route('https://github.com/**', lambda route: (external.append(route.request.url), route.abort()))
            page.route('**/api/marketplace/ouroboroshub/catalog*', lambda route: route.fulfill(json={'results': [{
                'slug': name, 'sanitized_name': name, 'display_name': name, 'latest_version': '1.0.0',
                'summary': 'Controlled catalog entry', 'identity_conflict': False}]}))
            page.goto(url, wait_until='domcontentloaded')
            page.click('[data-nav-page="skills"]')
            own = page.locator(f'.skills-card[data-skill="{name}"]')
            expect(own).to_contain_text('Submitted PR #7')
            page.click('.skills-tab[data-tab="ouroboroshub"]')
            clear = page.locator(f'[data-oh-clear-publication="{name}"]')
            expect(clear).to_be_visible()
            hint = page.locator('#oh-results .marketplace-secondary-actions > .muted')
            assert hint.evaluate("node => getComputedStyle(node).fontSize === getComputedStyle(document.documentElement).getPropertyValue('--type-meta').trim()")
            badge = page.locator('#oh-results .skills-badge').filter(has_text='Submitted PR #7')
            assert badge.evaluate("node => { const range = document.createRange(); range.selectNodeContents(node); return range.getClientRects().length === 1; }")
            _screenshot(page, f'submission-waiting-{bucket}.png')
            # The visible old button cannot clear a concurrent newer publication.
            write_publication_record(data, name, receipt(8))
            clear.click()
            expect(page.locator('#oh-status')).to_contain_text('publication_changed')
            assert read_publication_record(data, name)[0] == receipt(8)
            page.click('[data-oh-search]')
            expect(page.locator('#oh-results')).to_contain_text('Submitted PR #8')
            clear.click()
            expect(page.locator('#oh-status')).to_contain_text('local submission record cleared')
            assert read_publication_record(data, name) == (None, None)
            assert json.loads((data / f'state/skills/{name}/ouroboroshub.json').read_text())['future'] == {'keep': True}
            next_action = 'adopt' if bucket == 'external' else 'update'
            expect(page.locator(f'#oh-results [data-oh-action="{next_action}"]')).to_be_visible()
            expect(page.locator('.confirm-dialog')).to_have_count(0)
            _screenshot(page, f'submission-cleared-{bucket}.png')
            page.click('.skills-tab[data-tab="installed"]')
            expect(own).not_to_contain_text('Submitted PR')
            assert compute_content_hash(payload) == original_hash
            # Edited-since-submission keeps the same explicit local clear affordance.
            write_publication_record(data, name, receipt(9))
            (payload / 'scripts/check.py').write_text("print('edited payload stays')\n")
            edited_hash = compute_content_hash(payload)
            page.click('.skills-tab[data-tab="ouroboroshub"]')
            if bucket == 'external':
                expect(page.locator('#oh-results')).to_contain_text('edited since submission')
            expect(clear).to_be_visible()
            clear.click()
            expect(page.locator('#oh-status')).to_contain_text('local submission record cleared')
            assert read_publication_record(data, name) == (None, None)
            assert compute_content_hash(payload) == edited_hash
            assert not external
        finally:
            browser.close()
