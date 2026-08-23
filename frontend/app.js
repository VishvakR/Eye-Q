/**
 * Eye-Q Dashboard — Vanilla JavaScript Application
 *
 * Handles:
 *   - REST API calls for rules, alerts, cameras, and status
 *   - WebSocket connection for live event streaming
 *   - DOM rendering for rule cards, alert cards, and stats
 *   - Rule CRUD modal form
 *   - Alert filtering and pagination
 */

const API_BASE = window.location.origin;
const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/events`;

// ── State ───────────────────────────────────────────────────────

let ws = null;
let wsReconnectTimer = null;
let alerts = [];
let currentFilter = 'all';
let alertOffset = 0;
const ALERT_LIMIT = 50;
let statusPollTimer = null;

// ── Initialization ──────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadRules();
    loadAlerts();
    connectWebSocket();
    startStatusPolling();
    startCameraStream();

    document.getElementById('rule-form').addEventListener('submit', handleRuleSubmit);
});

// ── Live Camera Feed ────────────────────────────────────────────

const CAMERA_ID = 'default';
let streamFallbackTimer = null;

/**
 * Ensure the live feed is loading, degrading to still-frame polling if needed.
 *
 * The <img> carries its own `src` and inline `onerror` so the feed works even
 * if this file is cached or fails to load. This only re-arms the handler and
 * covers the case where the element somehow has no source.
 *
 * `multipart/x-mixed-replace` in an <img> is not supported uniformly across
 * browsers (Safari in particular fails to render it), and an <img> that errors
 * never retries on its own — so a single transient failure at page load would
 * otherwise leave a permanently broken panel.
 */
// How long to let the MJPEG stream prove it can render before giving up on it.
// Generous, because the first frame waits on camera open plus YOLO warm-up.
const STREAM_BLANK_TIMEOUT_MS = 8000;

function startCameraStream() {
    const img = document.getElementById('camera-stream');
    if (!img) return;

    img.onerror = () => {
        console.warn('MJPEG stream failed; falling back to still-frame polling.');
        useSnapshotFallback();
    };

    if (!img.getAttribute('src')) {
        img.src = `/api/stream/${CAMERA_ID}.mjpg`;
    }

    // A stream that connects but renders nothing is the failure onerror cannot
    // see: the response is open and valid, so the browser reports no error, and
    // the panel just stays blank. naturalWidth is the only honest signal that
    // no image ever arrived.
    setTimeout(() => {
        if (!streamFallbackTimer && img.naturalWidth === 0) {
            console.warn('MJPEG stream produced no image; falling back to polling.');
            useSnapshotFallback();
        }
    }, STREAM_BLANK_TIMEOUT_MS);
}

// Exposed on window so the inline onerror in index.html can reach it.
function useSnapshotFallback() {
    const img = document.getElementById('camera-stream');
    if (!img || streamFallbackTimer) return;

    // Drop the error handler so a momentarily unavailable frame does not
    // re-enter this path and stack up timers.
    img.onerror = null;

    const refresh = () => {
        // Cache-bust: without it the browser serves the first frame forever.
        img.src = `/api/stream/${CAMERA_ID}/frame.jpg?t=${Date.now()}`;
    };

    refresh();
    streamFallbackTimer = setInterval(refresh, 200);  // ~5 FPS
}
window.useSnapshotFallback = useSnapshotFallback;

// ── REST API Helpers ────────────────────────────────────────────

async function apiFetch(path, options = {}) {
    const url = `${API_BASE}${path}`;
    const defaults = {
        headers: { 'Content-Type': 'application/json' },
    };
    const config = { ...defaults, ...options };

    try {
        const res = await fetch(url, config);
        if (res.status === 204) return null;
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        return await res.json();
    } catch (err) {
        console.error(`API Error [${path}]:`, err);
        throw err;
    }
}

// ── Rules ───────────────────────────────────────────────────────

async function loadRules() {
    try {
        const rules = await apiFetch('/api/rules');
        renderRules(rules);
    } catch (err) {
        console.error('Failed to load rules:', err);
    }
}

// The sentinel rule owns alerts whose rule was deleted. It is an internal
// bookkeeping row, not something the user created, so keep it out of the list.
const SENTINEL_RULE_ID = 0;

function renderRules(rules) {
    const container = document.getElementById('rules-list');
    const clearBtn = document.getElementById('clear-rules-btn');

    rules = (rules || []).filter(r => r.id !== SENTINEL_RULE_ID);

    if (clearBtn) clearBtn.classList.toggle('hidden', rules.length === 0);

    if (rules.length === 0) {
        container.innerHTML = '<div class="empty-state">No rules configured. Create one to get started.</div>';
        return;
    }

    container.innerHTML = rules.map(rule => {
        const labels = JSON.parse(rule.target_labels || '[]');
        const labelBadges = labels.length > 0
            ? labels.map(l => `<span class="rule-badge">${l}</span>`).join('')
            : '<span class="rule-badge">all labels</span>';

        const statusClass = rule.enabled ? 'enabled' : 'disabled';
        const statusText = rule.enabled ? '● Active' : '○ Disabled';

        return `
            <div class="rule-card" id="rule-${rule.id}">
                <div class="rule-name">${escapeHtml(rule.name)}</div>
                <div class="rule-prompt">"${escapeHtml(rule.prompt)}"</div>
                <div class="rule-meta">
                    <span class="rule-badge ${statusClass}">${statusText}</span>
                    ${labelBadges}
                    <span class="rule-badge">conf ≥ ${rule.confidence}</span>
                    <span class="rule-badge">${rule.cooldown_s}s cooldown</span>
                </div>
                <div class="flex gap-2 mt-3">
                    <button onclick="editRule(${rule.id})" class="btn-secondary" style="font-size:0.7rem;padding:0.25rem 0.5rem;">Edit</button>
                    <button onclick="toggleRule(${rule.id}, ${rule.enabled ? 0 : 1})" class="btn-secondary" style="font-size:0.7rem;padding:0.25rem 0.5rem;">
                        ${rule.enabled ? 'Disable' : 'Enable'}
                    </button>
                    <button onclick="deleteRule(${rule.id})" class="btn-danger">Delete</button>
                </div>
            </div>
        `;
    }).join('');
}

function openRuleModal(editData = null) {
    const modal = document.getElementById('rule-modal');
    const title = document.getElementById('modal-title');
    const form = document.getElementById('rule-form');
    const errorDiv = document.getElementById('rule-error');
    errorDiv.classList.add('hidden');
    errorDiv.textContent = '';

    if (editData) {
        title.textContent = 'Edit Rule';
        document.getElementById('rule-edit-id').value = editData.id;
        document.getElementById('rule-name').value = editData.name;
        document.getElementById('rule-prompt').value = editData.prompt;
        document.getElementById('rule-camera').value = editData.camera_id;
        const labels = JSON.parse(editData.target_labels || '[]');
        document.getElementById('rule-labels').value = labels.join(', ');
        document.getElementById('rule-confidence').value = editData.confidence;
        document.getElementById('rule-cooldown').value = editData.cooldown_s;
    } else {
        title.textContent = 'New Alert Rule';
        form.reset();
        document.getElementById('rule-edit-id').value = '';
        document.getElementById('rule-camera').value = 'default';
        document.getElementById('rule-confidence').value = '0.25';
        document.getElementById('rule-cooldown').value = '5.0';
    }

    scopeWarningAccepted = false;
    document.getElementById('rule-scope-warning').classList.add('hidden');
    modal.classList.remove('hidden');
}

function closeRuleModal() {
    document.getElementById('rule-modal').classList.add('hidden');
}

// Set once the operator has chosen to keep a rule unscoped, so the warning
// does not re-appear on the resubmit it triggered.
let scopeWarningAccepted = false;

/**
 * Warn before saving a rule that asks about a specific object but watches all
 * of them.
 *
 * An empty label list means every COCO class wakes the VLM. That is right for
 * a scene-level question ("is anything blocking the exit?") and wrong for the
 * far more common object-specific one, where it spends seconds per inference
 * judging pallets and trucks against a question about people.
 *
 * @returns {Promise<boolean>} true to continue saving.
 */
async function confirmRuleScope(prompt, labels) {
    if (labels.length > 0 || scopeWarningAccepted) return true;

    let suggested = [];
    try {
        const res = await apiFetch(`/api/labels?prompt=${encodeURIComponent(prompt)}`);
        suggested = res.suggested || [];
    } catch {
        return true;  // Never block saving because a hint could not be fetched.
    }
    if (suggested.length === 0) return true;

    const box = document.getElementById('rule-scope-warning');
    box.innerHTML = `
        <p class="mb-2">
            This rule watches <strong>every object class</strong>, but its prompt is about
            <strong>${suggested.map(escapeHtml).join(', ')}</strong>.
            Trucks, pallets and chairs will all be judged against it &mdash; several
            seconds of inference each, spent on things you did not ask about.
        </p>
        <div class="flex gap-2">
            <button type="button" id="scope-fix" class="btn-primary text-xs px-2 py-1">
                Watch only ${suggested.map(escapeHtml).join(', ')}
            </button>
            <button type="button" id="scope-ignore" class="btn-secondary text-xs px-2 py-1">
                Watch every class
            </button>
        </div>`;
    box.classList.remove('hidden');

    return new Promise(resolve => {
        document.getElementById('scope-fix').onclick = () => {
            document.getElementById('rule-labels').value = suggested.join(', ');
            box.classList.add('hidden');
            resolve(false);  // Resubmit with the labels now filled in.
            document.getElementById('rule-form').requestSubmit();
        };
        document.getElementById('scope-ignore').onclick = () => {
            scopeWarningAccepted = true;
            box.classList.add('hidden');
            resolve(false);
            document.getElementById('rule-form').requestSubmit();
        };
    });
}

async function handleRuleSubmit(e) {
    e.preventDefault();

    const editId = document.getElementById('rule-edit-id').value;
    const labelsStr = document.getElementById('rule-labels').value;
    const labels = labelsStr ? labelsStr.split(',').map(l => l.trim()).filter(Boolean) : [];

    const promptText = document.getElementById('rule-prompt').value;
    if (!(await confirmRuleScope(promptText, labels))) return;

    const data = {
        camera_id: document.getElementById('rule-camera').value || 'default',
        name: document.getElementById('rule-name').value,
        prompt: document.getElementById('rule-prompt').value,
        target_labels: labels,
        confidence: parseFloat(document.getElementById('rule-confidence').value),
        cooldown_s: parseFloat(document.getElementById('rule-cooldown').value),
        enabled: true,
    };

    try {
        if (editId) {
            await apiFetch(`/api/rules/${editId}`, {
                method: 'PUT',
                body: JSON.stringify(data),
            });
        } else {
            await apiFetch('/api/rules', {
                method: 'POST',
                body: JSON.stringify(data),
            });
        }
        closeRuleModal();
        await loadRules();
    } catch (err) {
        const errorDiv = document.getElementById('rule-error');
        errorDiv.textContent = `Failed to save rule: ${err.message}`;
        errorDiv.classList.remove('hidden');
    }
}

async function editRule(ruleId) {
    try {
        const rule = await apiFetch(`/api/rules/${ruleId}`);
        openRuleModal(rule);
    } catch (err) {
        alert(`Failed to load rule: ${err.message}`);
    }
}

async function toggleRule(ruleId, enabled) {
    try {
        await apiFetch(`/api/rules/${ruleId}`, {
            method: 'PUT',
            body: JSON.stringify({ enabled: Boolean(enabled) }),
        });
        await loadRules();
    } catch (err) {
        alert(`Failed to toggle rule: ${err.message}`);
    }
}

async function deleteRule(ruleId) {
    if (!confirm('Delete this rule? This also removes associated alerts.')) return;
    try {
        await apiFetch(`/api/rules/${ruleId}`, { method: 'DELETE' });
        await loadRules();
    } catch (err) {
        alert(`Failed to delete rule: ${err.message}`);
    }
}

async function clearAllRules() {
    const count = document.querySelectorAll('#rules-list .rule-card').length;
    const subject = count === 1 ? '1 rule' : `all ${count} rules`;

    if (!confirm(`Delete ${subject}?\n\nAlert history is kept — past alerts stay in the feed under "Default (unmatched)".`)) {
        return;
    }

    try {
        const result = await apiFetch('/api/rules', { method: 'DELETE' });
        await loadRules();
        // Rule deletion reassigns alerts, so the feed's labels are now stale.
        alertOffset = 0;
        await loadAlerts();
        console.log(`Cleared ${result.deleted} rule(s).`);
    } catch (err) {
        alert(`Failed to clear rules: ${err.message}`);
    }
}

// ── Alerts ──────────────────────────────────────────────────────

async function loadAlerts() {
    try {
        const result = await apiFetch(`/api/alerts?limit=${ALERT_LIMIT}&offset=${alertOffset}`);
        alerts = result.alerts || [];
        renderAlerts(alerts);

        const loadMoreBtn = document.getElementById('load-more-btn');
        if (result.total > alertOffset + ALERT_LIMIT) {
            loadMoreBtn.classList.remove('hidden');
        } else {
            loadMoreBtn.classList.add('hidden');
        }
    } catch (err) {
        console.error('Failed to load alerts:', err);
    }
}

async function loadMoreAlerts() {
    alertOffset += ALERT_LIMIT;
    try {
        const result = await apiFetch(`/api/alerts?limit=${ALERT_LIMIT}&offset=${alertOffset}`);
        const newAlerts = result.alerts || [];
        alerts = [...alerts, ...newAlerts];
        renderAlerts(alerts);

        if (result.total <= alertOffset + ALERT_LIMIT) {
            document.getElementById('load-more-btn').classList.add('hidden');
        }
    } catch (err) {
        console.error('Failed to load more alerts:', err);
    }
}

function filterAlerts(filter) {
    currentFilter = filter;
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`filter-${filter}`).classList.add('active');
    renderAlerts(alerts);
}

function renderAlerts(alertList) {
    const container = document.getElementById('alerts-list');

    let filtered = alertList;
    if (currentFilter === 'violations') {
        filtered = alertList.filter(a => a.verdict === 1);
    } else if (currentFilter === 'clear') {
        filtered = alertList.filter(a => a.verdict === 0);
    }

    if (filtered.length === 0) {
        container.innerHTML = '<div class="empty-state">No alerts match the current filter.</div>';
        return;
    }

    container.innerHTML = filtered.map(alert => {
        const isViolation = alert.verdict === 1;
        const cardClass = isViolation ? 'violation' : 'clear';
        const verdictClass = isViolation ? 'positive' : 'negative';
        const verdictIcon = isViolation ? '🔴' : '🟢';
        const verdictText = isViolation ? 'VIOLATION' : 'CLEAR';

        const createdAt = alert.created_at || '';
        const timeStr = createdAt ? new Date(createdAt + 'Z').toLocaleTimeString() : '--';
        const dateStr = createdAt ? new Date(createdAt + 'Z').toLocaleDateString() : '';

        // frame_snapshot is a server-side path (data/snapshots/<uuid>.jpg);
        // the snapshots directory is mounted at /snapshots.
        const snapshot = alert.frame_snapshot
            ? `<img src="/snapshots/${escapeHtml(alert.frame_snapshot.split('/').pop())}"
                    alt="Alert evidence" class="alert-snapshot" loading="lazy">`
            : '';

        const ruleLabel = alert.rule_name
            ? escapeHtml(alert.rule_name)
            : `Rule #${alert.rule_id}`;

        // Which subject this verdict is about. Several alerts can now come from
        // one frame, and without the id they read as duplicates of each other
        // rather than as different people.
        const subject = alert.label
            ? escapeHtml(alert.label) + (alert.track_id >= 0 ? ` #${alert.track_id}` : '')
            : '';

        // The VLM's own p(yes), not the detector's box score. Null whenever the
        // verdict came from generate-and-parse, which produces no strength —
        // so the badge is omitted rather than shown as 0.
        const confident = typeof alert.confidence === 'number';
        const closeCall = confident && Math.abs(alert.confidence - 0.5) < 0.1;
        const confidenceBadge = confident
            ? `<span class="${closeCall ? 'alert-closecall' : ''}" title="${
                closeCall
                    ? 'Close call — the model was near the alert threshold'
                    : 'Probability the model assigned to Yes'
              }">p(yes) ${alert.confidence.toFixed(2)}${closeCall ? ' ⚠' : ''}</span>`
            : '';

        return `
            <div class="alert-card ${cardClass}">
                <div class="alert-header">
                    <span class="alert-verdict ${verdictClass}">
                        ${verdictIcon} ${verdictText}
                    </span>
                    <span class="alert-time">${timeStr} · ${dateStr}</span>
                </div>
                <div class="alert-body">
                    ${snapshot}
                    <div class="alert-response">${escapeHtml(alert.raw_response || '').substring(0, 200)}</div>
                </div>
                <div class="alert-meta">
                    <span>${ruleLabel}</span>
                    ${subject ? `<span>${subject}</span>` : ''}
                    <span>Camera: ${escapeHtml(alert.camera_id)}</span>
                    ${confidenceBadge}
                    <span>VLM: ${(alert.inference_ms || 0).toFixed(0)}ms</span>
                </div>
            </div>
        `;
    }).join('');
}

function addLiveAlert(alertData) {
    // Prepend to the list
    alerts.unshift(alertData);
    renderAlerts(alerts);
}

// ── Status Polling ──────────────────────────────────────────────

function startStatusPolling() {
    pollStatus();
    statusPollTimer = setInterval(pollStatus, 3000);
}

async function pollStatus() {
    try {
        const status = await apiFetch('/api/status');
        updateStats(status);
    } catch (err) {
        // Silently ignore — server may not be fully ready
    }
}

function updateStats(status) {
    // Buffer
    const bufOccupancy = status.buffer?.occupancy || 0;
    document.getElementById('stat-buffer').textContent = `${(bufOccupancy * 100).toFixed(0)}%`;
    document.getElementById('stat-buffer-bar').style.width = `${bufOccupancy * 100}%`;

    // YOLO FPS
    document.getElementById('stat-fps').textContent = (status.watchdog?.fps || 0).toFixed(1);

    // Detections
    document.getElementById('stat-detections').textContent = status.watchdog?.detections_published || 0;

    // VLM
    document.getElementById('stat-vlm-triggers').textContent = status.vlm?.triggers_processed || 0;

    // Violations
    document.getElementById('stat-violations').textContent = status.vlm?.verdicts_positive || 0;

    // Rules
    document.getElementById('stat-rules').textContent = status.database?.active_rules || 0;

    // Device
    document.getElementById('device-badge').textContent = (status.inference_device || 'CPU').toUpperCase();

    // Uptime
    const uptime = status.uptime_seconds || 0;
    document.getElementById('uptime-badge').textContent = `Uptime: ${formatUptime(uptime)}`;

    updateCameraStatus(status);
}

function updateCameraStatus(status) {
    const label = document.getElementById('camera-status');
    const errorOverlay = document.getElementById('stream-error');
    if (!label || !errorOverlay) return;

    if (status.pipeline_enabled === false) {
        label.textContent = 'pipeline disabled';
        errorOverlay.classList.remove('hidden');
        return;
    }

    const cam = (status.cameras || [])[0];
    if (cam) {
        label.textContent = `${cam.camera_id} · ${cam.connected ? 'online' : 'offline'} · ${(cam.fps || 0).toFixed(1)} FPS`;
        errorOverlay.classList.toggle('hidden', cam.connected);
    }
}

// ── WebSocket ───────────────────────────────────────────────────

function connectWebSocket() {
    if (ws) {
        ws.close();
    }

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('WebSocket connected');
        const indicator = document.getElementById('ws-indicator');
        indicator.innerHTML = `
            <span class="w-2 h-2 rounded-full bg-accent-green pulse-online"></span>
            <span class="text-accent-green">Live</span>
        `;

        // Send periodic pings
        if (ws._pingTimer) clearInterval(ws._pingTimer);
        ws._pingTimer = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 30000);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWSMessage(data);
        } catch (err) {
            console.error('WS message parse error:', err);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        const indicator = document.getElementById('ws-indicator');
        indicator.innerHTML = `
            <span class="w-2 h-2 rounded-full bg-accent-red animate-pulse"></span>
            <span class="text-text-muted">Disconnected</span>
        `;

        // Reconnect after 3 seconds
        if (wsReconnectTimer) clearTimeout(wsReconnectTimer);
        wsReconnectTimer = setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
}

function handleWSMessage(msg) {
    if (msg.type === 'pong') return;

    if (msg.type === 'alert') {
        addLiveAlert(msg.data);
    } else if (msg.type === 'status') {
        updateStats(msg.data);
    }
}

// ── Helpers ─────────────────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatUptime(seconds) {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(0)}m`;
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hrs}h ${mins}m`;
}
