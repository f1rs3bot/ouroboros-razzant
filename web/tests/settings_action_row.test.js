import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const settingsCss = await readFile(new URL('../settings.css', import.meta.url), 'utf8');
const onboardingCss = await readFile(new URL('../onboarding.css', import.meta.url), 'utf8');
const styleCss = await readFile(new URL('../style.css', import.meta.url), 'utf8');
const settingsJs = await readFile(new URL('../modules/settings.js', import.meta.url), 'utf8');
const catalogJs = await readFile(new URL('../modules/settings_catalog.js', import.meta.url), 'utf8');
const uiHelpersJs = await readFile(new URL('../modules/ui_helpers.js', import.meta.url), 'utf8');

test('neutral controls have one shared button role in both UI shells', () => {
    assert.match(styleCss, /\.btn-default\s*\{/);
    assert.match(styleCss, /\.btn:focus-visible\s*\{/);
    assert.match(onboardingCss, /\.btn-default\s*\{/);
    assert.match(onboardingCss, /--button-min-height:\s*34px/);
    assert.doesNotMatch(settingsCss, /settings-ghost-btn/);
    assert.doesNotMatch(onboardingCss, /settings-ghost-btn/);
});

test('single-action settings rows reserve a flexible status and responsive action edge', () => {
    assert.match(settingsCss, /\.settings-action-row\s*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\) auto/);
    assert.match(settingsCss, /\.settings-action-row\s*>\s*\.btn-default\s*\{[^}]*justify-self:\s*end/);
    assert.match(settingsCss, /@media \(max-width: 760px\)[\s\S]*?\.settings-action-row\s*>\s*\.settings-inline-status:empty\s*\{[^}]*display:\s*none/);
    assert.match(settingsCss, /@media \(max-width: 760px\)[\s\S]*?\.settings-action-row\s*\{[^}]*grid-template-columns:\s*1fr/);
});

test('async actions expose the same busy and status semantics', () => {
    assert.match(settingsJs, /function setButtonBusy\(button, busy\)/);
    assert.match(settingsJs, /setAttribute\('aria-busy', 'true'\)/);
    assert.match(settingsJs, /setInlineStatus\(status, 'Testing…', 'muted'\)/);
    assert.match(settingsJs, /providerTestStatusText\(data\), data\?\.ok \? 'ok' : 'danger'/);
    assert.match(settingsJs, /refreshModelCatalog\(\{ button: byId\('btn-refresh-model-catalog'\) \}\)/);
    assert.match(settingsJs, /setInlineStatus\(el, '', 'muted'\)/);
    assert.match(uiHelpersJs, /if \(el\.textContent !== next\) el\.textContent = next/);
    assert.match(catalogJs, /import \{ setInlineStatus \} from '\.\/ui_helpers\.js'/);
    assert.match(catalogJs, /setInlineStatus\(statusEl, text, tone\)/);
    assert.match(catalogJs, /refreshModelCatalog\(\{ button \} = \{\}\)/);
    assert.match(catalogJs, /refreshSeq === catalogRefreshSeq/);
});
