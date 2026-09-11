// The shared Models editor for Settings and onboarding. Model strings remain
// the routing authority; account and context choices belong to the exact role.
// Catalog arrival only enriches choices. It never authors an assignment.
import { fetchJson } from './api_client.js';
import { MODEL_CATALOG_TIMEOUT_MS } from './settings_catalog.js';
import { bindStatusSurface, claudexorStatus } from './claudexor_status_store.js';
import { parseModelSource, composeModelSource, indexProfilesByHarness, profileOptionsFor, selectHtml, mintStableId } from './route_editor_primitives.js';
import { revealNewRow } from './ui_helpers.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const MODEL_ACCOUNTS_KEY = 'OUROBOROS_MODEL_ACCOUNTS';
export const MODEL_CONTEXT_KEY = 'OUROBOROS_MODEL_CONTEXT_WINDOWS';

export { parseModelSource, composeModelSource } from './route_editor_primitives.js';

export function modelRoleMap(value) {
    if (typeof value === 'string') {
        try { value = JSON.parse(value); } catch (_) { return {}; }
    }
    return value && typeof value === 'object' && !Array.isArray(value) ? { ...value } : {};
}

export function modelSourceGroups({ sources = [], providers = {}, current = '' } = {}) {
    const subscriptions = sources.map((source) => ({ value: `subscription:${source.id}`,
        label: source.label || source.id }));
    const api = Object.entries(providers).filter(([id]) => !['local', 'direct-multi'].includes(id))
        .map(([id, provider]) => ({ value: id, label: provider.label || id }));
    if (!api.some((row) => row.value === 'openrouter')) api.unshift({ value: 'openrouter', label: 'OpenRouter' });
    if (current && current !== 'inherit' && ![...subscriptions, ...api].some((row) => row.value === current)) {
        (current.startsWith('subscription:') ? subscriptions : api).push({
            value: current, label: `${current.replace('subscription:', '')} (not checked)`,
        });
    }
    return [
        ...(current === 'inherit' ? [{ options: [{ value: 'inherit', label: 'Uses Main' }] }] : []),
        { label: 'Claudexor · subscriptions', options: subscriptions.length ? subscriptions : [
            { value: '', label: 'Connect a model source in Accounts', disabled: true },
        ] },
        { label: 'API', options: api },
    ];
}

export function modelContextNote(item, override = 0) {
    if (Number(override) > 0) return `${Number(override).toLocaleString('en-US')} tokens · set by you; the provider may accept less.`;
    const window = Number(item?.max_context_window || item?.context_window || 0);
    return window > 0 ? `Auto: ${window.toLocaleString('en-US')} tokens · advertised for this route.`
        : 'Auto: context limit not known for this route.';
}

export function modelRolesHost(id) {
    return `<div id="${escapeHtml(id)}" class="model-role-editor"></div>`;
}

/** In-memory form controller; refreshes have no settings or login side effects. */
export function createModelRolesEditor({ hostId, store = claudexorStatus,
    doc = () => document, onChange = () => {}, catalogFetch = fetchJson, showContext = true } = {}) {
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    let slots = [];
    let rows = [];
    let settings = {};
    let providers = {};
    let sources = [];
    let apiItems = [];
    let loaded = false;
    let destroyed = false;
    let disposeStatus = null;
    const catalogs = new Map();
    const requests = new Map();
    const host = () => getDoc()?.getElementById(hostId);

    function rowState(slot, value, index = -1) {
        const account = modelRoleMap(settings[MODEL_ACCOUNTS_KEY])[slot.slot];
        const context = modelRoleMap(settings[MODEL_CONTEXT_KEY])[slot.slot];
        const parsed = parseModelSource(value);
        if (!parsed.model && !['main', 'fallback'].includes(slot.slot)) parsed.source = 'inherit';
        return { id: index < 0 ? slot.slot : mintStableId('fallback', rows.map((row) => row.id)), slot,
            ...parsed, account: String(index < 0 ? account || '' : account?.[index] || ''),
            context: index < 0 ? context || 0 : context?.[index] || 0,
            local: settings[`USE_LOCAL_${slot.slot.toUpperCase()}`] === true
                || settings[`USE_LOCAL_${slot.slot.toUpperCase()}`] === 'true' };
    }

    function collect() {
        if (!loaded) return {};
        const result = {};
        const accounts = modelRoleMap(settings[MODEL_ACCOUNTS_KEY]);
        const windows = modelRoleMap(settings[MODEL_CONTEXT_KEY]);
        for (const slot of slots) {
            const matching = rows.filter((row) => row.slot.slot === slot.slot);
            const fallback = slot.slot === 'fallback';
            const values = matching.map((row) => composeModelSource(row.source, row.model));
            result[slot.settingKey] = fallback ? values.join(', ') : values[0] || '';
            const pins = matching.map((row) => sourceId(row) ? row.account : '');
            const contexts = matching.map((row) => Number(row.context || 0));
            if (pins.some(Boolean) || slot.slot in accounts) accounts[slot.slot] = fallback ? pins : pins[0] || '';
            if (contexts.some(Boolean) || slot.slot in windows) windows[slot.slot] = fallback ? contexts : contexts[0] || 0;
            if (slot.settingsToggleId) result[`USE_LOCAL_${slot.slot.toUpperCase()}`] = matching[0]?.local || false;
        }
        if (Object.keys(accounts).length) result[MODEL_ACCOUNTS_KEY] = accounts;
        if (Object.keys(windows).length) result[MODEL_CONTEXT_KEY] = windows;
        return result;
    }

    function changed() { onChange(collect()); }
    function effectiveSource(row) { return row.source === 'inherit' ? rows.find((entry) => entry.slot.slot === 'main')?.source || 'openrouter' : row.source; }
    function sourceId(row) { const source = effectiveSource(row); return source.startsWith('subscription:') ? source.slice(13) : ''; }
    function catalogKey(row) { return JSON.stringify([sourceId(row), row.account]); }
    function itemsFor(row) {
        return sourceId(row) ? (catalogs.get(catalogKey(row))?.items || [])
            : apiItems.filter((item) => parseModelSource(item.value || item.id).source === row.source);
    }
    function currentItem(row) {
        const model = row.source === 'inherit' ? rows.find((entry) => entry.slot.slot === 'main')?.model : row.model;
        return itemsFor(row).find((item) => parseModelSource(item.value || item.id).model === model
            || item.id === model);
    }

    function inputIdFor(row) {
        return row.slot.slot === 'fallback' && rows.find((entry) => entry.slot === row.slot) !== row
            ? `${hostId}-${row.id}` : row.slot.inputId;
    }

    function detailsHtml(row) {
        if (!showContext) return '';
        return `<details class="model-role-details"><summary>Context</summary>
            <div class="model-role-context">
                <label>Window <input data-model-role-context type="number" min="0" step="1"
                    placeholder="Auto" value="${escapeHtml(row.context || '')}" aria-label="${escapeHtml(row.slot.label)} context window"></label>
                <span data-model-context-note>${escapeHtml(modelContextNote(currentItem(row), row.context))}</span>
            </div></details>`;
    }

    function rowHtml(row, index, total) {
        const isFallback = row.slot.slot === 'fallback';
        const inputId = inputIdFor(row);
        return `<div class="model-role-row" data-model-role="${escapeHtml(row.id)}">
            <div class="model-role-controls">
                ${selectHtml('data-model-role-source aria-label="Source"', modelSourceGroups({ sources, providers, current: row.source }), row.source)}
                <input id="${escapeHtml(inputId)}" data-model-role-model aria-label="${escapeHtml(row.slot.label)}" value="${escapeHtml(row.model)}" list="${escapeHtml(hostId)}-${escapeHtml(row.id)}-models" placeholder="${row.slot.slot === 'main' ? 'Choose a model' : 'Empty uses Main'}" autocomplete="off" spellcheck="false">
                <select data-model-role-account aria-label="Account" ${sourceId(row) ? '' : 'hidden'}></select>
                ${isFallback ? `<span class="model-role-order"><button type="button" class="btn btn-default" data-model-up aria-label="Move fallback up" ${index === 0 ? 'disabled' : ''}>↑</button><button type="button" class="btn btn-default" data-model-down aria-label="Move fallback down" ${index === total - 1 ? 'disabled' : ''}>↓</button><button type="button" class="btn btn-default" data-model-remove aria-label="Remove fallback">Remove</button></span>` : ''}
            </div>
            <datalist id="${escapeHtml(hostId)}-${escapeHtml(row.id)}-models"></datalist>
            <div class="model-role-notes"><span class="model-role-meta" data-model-role-status></span>${detailsHtml(row)}</div>
        </div>`;
    }

    function render() {
        const element = host();
        if (!element || destroyed || !loaded) return;
        element.innerHTML = slots.map((slot) => {
            const matching = rows.filter((row) => row.slot.slot === slot.slot);
            const local = matching[0]?.local || false;
            return `<section class="model-role-group" data-model-role-group="${escapeHtml(slot.slot)}">
                <div class="model-role-head"><h4 title="${escapeHtml(slot.note || '')}">${escapeHtml(slot.label.replace(/ Model$/, ''))}</h4>
                    ${slot.settingsToggleId ? `<label class="local-toggle"><input id="${escapeHtml(slot.settingsToggleId)}" type="checkbox" data-model-local ${local ? 'checked' : ''}> Local</label>` : ''}
                    ${slot.slot === 'fallback' ? '<button type="button" class="btn btn-default" data-model-add>Add fallback</button>' : ''}
                </div>
                ${slot.slot === 'fallback' ? '<p class="model-role-copy">Tried in this order. Subscription quota waits for your choice before using API.</p>' : ''}
                ${matching.map((row, index) => rowHtml(row, index, matching.length)).join('')}
            </section>`;
        }).join('');
        bindRows(element);
        updateCatalogViews();
        for (const row of rows) void refreshRow(row);
    }

    function updateCatalogViews() {
        const element = host();
        if (!element || destroyed) return;
        const profiles = indexProfilesByHarness(store.snapshot);
        for (const row of rows) {
            const node = element.querySelector(`[data-model-role="${row.id}"]`);
            if (!node) continue;
            const source = node.querySelector('[data-model-role-source]');
            const sourceHtml = selectHtml('', modelSourceGroups({ sources, providers, current: row.source }), row.source);
            const options = sourceHtml.slice(sourceHtml.indexOf('>') + 1, sourceHtml.lastIndexOf('</select>'));
            if (source.innerHTML !== options) source.innerHTML = options;
            const account = node.querySelector('[data-model-role-account]');
            account.hidden = !sourceId(row);
            const credentialHarness = sources.find((entry) => entry.id === sourceId(row))?.credentialHarness || '';
            const accountHtml = selectHtml('', [{ options: profileOptionsFor(profiles[credentialHarness], row.account, { accountsKnown: store.accountsKnown && Boolean(credentialHarness) }) }], row.account);
            const accountOptions = accountHtml.slice(accountHtml.indexOf('>') + 1, accountHtml.lastIndexOf('</select>'));
            if (account.innerHTML !== accountOptions) account.innerHTML = accountOptions;
            const list = node.querySelector('datalist');
            list.innerHTML = itemsFor(row).map((item) => `<option value="${escapeHtml(parseModelSource(item.value || item.id).model)}">${escapeHtml(item.name || item.label || item.id)}</option>`).join('');
            if (showContext) node.querySelector('[data-model-context-note]').textContent = modelContextNote(currentItem(row), row.context);
            const status = node.querySelector('[data-model-role-status]');
            const catalog = catalogs.get(catalogKey(row));
            status.textContent = row.local ? 'Uses the local runtime.'
                : !row.model ? (row.slot.slot === 'main' ? 'Choose a model to continue.' : 'Uses Main.')
                    : sourceId(row) ? (catalog?.error || 'Uses your subscription. No API key required.') : '';
        }
    }

    async function refreshRow(row) {
        if (!sourceId(row) || destroyed) return;
        const key = catalogKey(row);
        if (requests.has(key) || catalogs.has(key)) return requests.get(key)?.promise;
        const query = new URLSearchParams({ source_id: sourceId(row) });
        if (row.account) query.set('credential_profile_id', row.account);
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), MODEL_CATALOG_TIMEOUT_MS);
        const promise = Promise.resolve().then(() => catalogFetch(`/api/model-catalog?${query}`, { cache: 'no-store', signal: controller.signal }))
            .then((data) => { if (!destroyed) {
                catalogs.set(key, { items: Array.isArray(data.items) ? data.items : [], error: data.errors?.length ? 'Model catalog could not be read; your selection is kept.' : '' });
                if (Array.isArray(data.model_sources)) sources = data.model_sources;
            } })
            .catch(() => { if (!destroyed) catalogs.set(key, { items: [], error: 'Model catalog could not be read; your selection is kept.' }); })
            .finally(() => { clearTimeout(timer); requests.delete(key); updateCatalogViews(); });
        requests.set(key, { promise, controller });
        return promise;
    }

    function bindRows(element) {
        for (const row of rows) {
            const node = element.querySelector(`[data-model-role="${row.id}"]`);
            node.querySelector('[data-model-role-source]').addEventListener('change', (event) => {
                row.source = event.target.value; row.model = ''; row.account = ''; row.context = 0;
                changed(); render();
                host()?.querySelector(`[data-model-role="${row.id}"] [data-model-role-model]`)?.focus();
            });
            const input = getDoc().getElementById(inputIdFor(row));
            input.addEventListener('input', () => {
                row.model = input.value;
                if (row.source === 'inherit' && row.model) row.source = effectiveSource(row);
                if (!row.model && !['main', 'fallback'].includes(row.slot.slot)) row.source = 'inherit';
                if (row.model.includes('::')) Object.assign(row, parseModelSource(row.model));
                changed(); updateCatalogViews();
                for (const item of rows) void refreshRow(item);
            });
            node.querySelector('[data-model-role-account]').addEventListener('change', (event) => {
                row.account = event.target.value; changed(); updateCatalogViews(); void refreshRow(row);
            });
            node.querySelector('[data-model-role-context]')?.addEventListener('input', (event) => {
                row.context = event.target.value; changed(); updateCatalogViews();
            });
            for (const [selector, delta] of [['[data-model-up]', -1], ['[data-model-down]', 1]]) {
                node.querySelector(selector)?.addEventListener('click', () => {
                    const index = rows.indexOf(row);
                    [rows[index], rows[index + delta]] = [rows[index + delta], rows[index]];
                    changed(); render();
                    host()?.querySelector(`[data-model-role="${row.id}"] ${selector}`)?.focus();
                });
            }
            node.querySelector('[data-model-remove]')?.addEventListener('click', () => {
                rows = rows.filter((entry) => entry !== row); changed(); render();
                host()?.querySelector('[data-model-add]')?.focus();
            });
        }
        element.querySelectorAll('[data-model-role-group]').forEach((group) => {
            const slot = slots.find((entry) => entry.slot === group.dataset.modelRoleGroup);
            group.querySelector('[data-model-local]')?.addEventListener('change', (event) => {
                rows.filter((row) => row.slot === slot).forEach((row) => { row.local = event.target.checked; });
                changed(); updateCatalogViews();
            });
            group.querySelector('[data-model-add]')?.addEventListener('click', () => {
                const row = rowState(slot, '', rows.filter((entry) => entry.slot === slot).length);
                row.account = ''; row.context = 0;
                rows.push(row); changed(); render();
                const added = host()?.querySelector(`[data-model-role="${row.id}"]`);
                revealNewRow(added, added?.querySelector('[data-model-role-model]'));
            });
        });
    }

    return {
        load(value, contract = {}) {
            settings = { ...value }; slots = contract.modelSlots || slots;
            providers = contract.providerProfiles || providers;
            rows = [];
            for (const slot of slots) {
                if (slot.slot === 'fallback') {
                    String(settings[slot.settingKey] || '').split(',').map((entry) => entry.trim()).filter(Boolean)
                        .forEach((entry, index) => rows.push(rowState(slot, entry, index)));
                } else rows.push(rowState(slot, settings[slot.settingKey]));
            }
            loaded = true; render();
        },
        mount() {
            if (!disposeStatus) disposeStatus = bindStatusSurface(store, { elementId: hostId, doc: getDoc,
                listener: updateCatalogViews });
            render();
        },
        adoptCatalog(data = {}) {
            apiItems = Array.isArray(data.items) ? data.items.filter((item) => !String(item.value || '').startsWith('claudexor::')) : apiItems;
            sources = Array.isArray(data.model_sources) ? data.model_sources : sources;
            if (Array.isArray(data.items)) {
                catalogs.clear();
                for (const row of rows) void refreshRow(row);
            }
            updateCatalogViews();
        },
        collect,
        validate() {
            const missing = rows.find((row) => row.slot.slot === 'main' && !row.model.trim());
            if (missing) return 'Choose the Main model.';
            if (rows.some((row) => row.slot.slot === 'fallback' && !row.model.trim())) return 'Choose a model for every fallback, or remove the empty row.';
            if (rows.some((row) => !Number.isSafeInteger(Number(row.context || 0)) || Number(row.context || 0) < 0)) return 'Context windows must be positive whole numbers, or Auto.';
            return '';
        },
        destroy() {
            destroyed = true; disposeStatus?.(); disposeStatus = null;
            for (const request of requests.values()) request.controller.abort();
        },
    };
}
