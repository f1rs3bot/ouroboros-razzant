import { apiFetch } from './api_client.js';
import { setInlineStatus } from './ui_helpers.js';
export const MODEL_CATALOG_TIMEOUT_MS = 25000;
let catalogRefreshSeq = 0;

function setCatalogStatus(statusEl, text, tone = 'muted') {
    setInlineStatus(statusEl, text, tone);
}

function broadcastCatalog(items, modelSources) {
    document.dispatchEvent(new CustomEvent('settings-model-catalog:updated', {
        detail: { items, model_sources: modelSources },
    }));
}

function fillCatalogDatalist(items, modelSources) {
    const list = document.getElementById('settings-model-catalog');
    if (list) {
        list.innerHTML = '';
        for (const item of items) {
            const option = document.createElement('option');
            option.value = item.value || item.id || '';
            option.label = item.label || item.provider || '';
            list.appendChild(option);
        }
    }
    broadcastCatalog(items, modelSources);
}

export async function refreshModelCatalog({ button } = {}) {
    const refreshSeq = ++catalogRefreshSeq;
    const statusEl = document.getElementById('settings-model-catalog-status');
    setCatalogStatus(statusEl, 'Refreshing model catalog...', 'muted');
    if (button) {
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
    }
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), MODEL_CATALOG_TIMEOUT_MS);

    try {
        const resp = await apiFetch('/api/model-catalog', {
            cache: 'no-store',
            signal: controller.signal,
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);

        const items = Array.isArray(data.items) ? data.items : [];
        const errors = Array.isArray(data.errors) ? data.errors : [];
        if (refreshSeq !== catalogRefreshSeq) {
            return { items, errors, stale: true };
        }
        fillCatalogDatalist(items, data.model_sources);

        if (errors.length && items.length) {
            setCatalogStatus(
                statusEl,
                `Loaded ${items.length} models. Some providers failed: ${errors.map((e) => e.provider_id).join(', ')}`,
                'warn',
            );
        } else if (errors.length) {
            setCatalogStatus(
                statusEl,
                `Model catalog unavailable right now: ${errors.map((e) => e.provider_id).join(', ')}`,
                'warn',
            );
        } else if (items.length) {
            setCatalogStatus(statusEl, `Loaded ${items.length} models.`, 'ok');
        } else {
            setCatalogStatus(statusEl, 'No provider catalogs available yet. This is optional.', 'muted');
        }
        return { items, errors };
    } catch (err) {
        if (refreshSeq !== catalogRefreshSeq) {
            return { items: [], errors: [{ provider_id: 'catalog', error: 'stale refresh' }], stale: true };
        }
        const message = err?.name === 'AbortError'
            ? `Timed out after ${Math.round(MODEL_CATALOG_TIMEOUT_MS / 1000)}s`
            : (err.message || err);
        fillCatalogDatalist([]);
        setCatalogStatus(
            statusEl,
            `Model catalog failed: ${message}. This is optional.`,
            'warn',
        );
        return { items: [], errors: [{ provider_id: 'catalog', error: String(message) }] };
    } finally {
        clearTimeout(timeoutId);
        if (button && refreshSeq === catalogRefreshSeq) {
            button.disabled = false;
            button.removeAttribute('aria-busy');
        }
    }
}
