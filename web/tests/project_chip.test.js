import assert from 'node:assert/strict';
import test from 'node:test';

import { PAGE_ICONS } from '../modules/page_icons.js';
import { renderProjectChip } from '../modules/ui_helpers.js';

// Minimal element stub: the chip is built with createElement/append/textContent
// only, so a flat stub proves the DOM contract without a browser.
class NodeStub {
    constructor(tag) {
        this.tagName = tag.toUpperCase();
        this.children = [];
        this.attributes = {};
        this.listeners = new Map();
        this.className = '';
        this.innerHTML = '';
        this._text = '';
    }
    set textContent(value) { this._text = String(value ?? ''); }
    get textContent() { return this._text; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    append(...nodes) { this.children.push(...nodes); }
    addEventListener(type, fn) { this.listeners.set(type, fn); }
}

function withDocument(fn) {
    const prior = globalThis.document;
    globalThis.document = { createElement: (tag) => new NodeStub(tag) };
    try { return fn(); } finally { globalThis.document = prior; }
}

test('the project chip is one DOM contract for the bound footer and the converted card', () => {
    withDocument(() => {
        let clicks = 0;
        const footer = renderProjectChip({
            name: 'OpenClaw 2.0 <b>x</b>', status: 'in project ↗',
            className: 'chat-live-bound-pointer', onClick: () => { clicks += 1; },
        });
        assert.equal(footer.tagName, 'BUTTON');
        assert.equal(footer.type, 'button');
        assert.equal(footer.className, 'chat-live-project-card-btn chat-live-bound-pointer');
        const [icon, name, status] = footer.children;
        // Vector icon from the shared Projects glyph, decorative, never an emoji.
        assert.equal(icon.className, 'chat-live-project-icon');
        assert.equal(icon.attributes['aria-hidden'], 'true');
        assert.equal(icon.innerHTML, PAGE_ICONS.projects);
        assert.match(icon.innerHTML, /^<svg /);
        // The name is text: a project name can never inject markup.
        assert.equal(name.className, 'chat-live-project-name');
        assert.equal(name.textContent, 'OpenClaw 2.0 <b>x</b>');
        assert.equal(status.className, 'chat-live-project-status');
        assert.equal(status.textContent, 'in project ↗');
        footer.listeners.get('click')();
        assert.equal(clicks, 1);

        const converted = renderProjectChip({ name: 'P', status: 'running in background ↗' });
        assert.equal(converted.className, 'chat-live-project-card-btn');
        assert.equal(converted.children[2].textContent, 'running in background ↗');
        assert.equal(converted.listeners.size, 0, 'no handler is attached without onClick');
    });
});
