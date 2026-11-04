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
let wsReconnectDelayMs = 1000;
const WS_RECONNECT_MAX_MS = 30000;
let wsHasConnected = false;
// The newest alerts, newest first, as one contiguous run of history — live
// pushes are prepended, "Load more" appends from offset alerts.length.
let alerts = [];
let currentFilter = 'all';
const ALERT_LIMIT = 50;
// Hours of uptime used to mean thousands of cards rebuilt on every verdict.
const MAX_ALERTS = 500;
let statusPollTimer = null;
let statusPollInFlight = false;

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

    // Request the next frame only once the previous one has finished loading
    // or failed. A fixed interval stacked requests whenever a frame took longer
    // than 200 ms, on exactly the slow links where the fallback is in use.
    const refresh = () => {
        // Cache-bust: without it the browser serves the first frame forever.
        img.src = `/api/stream/${CAMERA_ID}/frame.jpg?t=${Date.now()}`;
    };
    const scheduleNext = () => {
        streamFallbackTimer = setTimeout(refresh, 200);  // ~5 FPS at most
    };
    img.onload = scheduleNext;
    img.onerror = scheduleNext;

    streamFallbackTimer = true;
    refresh();
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
            throw new Error(describeError(err.detail) || `HTTP ${res.status}`);
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
        container.innerHTML = '<div class="empty-state">No rules yet. Add one to start asking questions about the feed.</div>';
        return;
    }

    container.innerHTML = rules.map(rule => {
        const labels = JSON.parse(rule.target_labels || '[]');
        const watching = labels.length > 0
            ? `Watches ${labels.map(escapeHtml).join(', ')}`
            : 'Watches every object';

        return `
            <div class="rule-card" id="rule-${rule.id}">
                <div class="rule-top">
                    <span class="rule-name">${escapeHtml(rule.name)}</span>
                    <span class="rule-state ${rule.enabled ? 'enabled' : ''}">${rule.enabled ? 'Active' : 'Disabled'}</span>
                </div>
                <div class="rule-prompt">${escapeHtml(rule.prompt)}</div>
                <div class="rule-meta">
                    <span>${watching}</span>
                    <span>Confidence ${Number(rule.confidence)}</span>
                    <span>${Number(rule.cooldown_s)} s cooldown</span>
                </div>
                <div class="rule-actions">
                    <button onclick="editRule(${Number(rule.id)})" class="btn-quiet">Edit</button>
                    <button onclick="toggleRule(${Number(rule.id)}, ${rule.enabled ? 0 : 1})" class="btn-quiet">${rule.enabled ? 'Disable' : 'Enable'}</button>
                    <button onclick="deleteRule(${Number(rule.id)})" class="btn-quiet btn-quiet-danger">Delete</button>
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
        title.textContent = 'Edit rule';
        document.getElementById('rule-edit-id').value = editData.id;
        document.getElementById('rule-name').value = editData.name;
        document.getElementById('rule-prompt').value = editData.prompt;
        document.getElementById('rule-camera').value = editData.camera_id;
        const labels = JSON.parse(editData.target_labels || '[]');
        document.getElementById('rule-labels').value = labels.join(', ');
        document.getElementById('rule-confidence').value = editData.confidence;
        document.getElementById('rule-cooldown').value = editData.cooldown_s;
    } else {
        title.textContent = 'New rule';
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
        <p>
            This rule watches every object, but the question is about
            <strong>${suggested.map(escapeHtml).join(', ')}</strong>. Trucks, pallets and chairs
            would each be judged against it, at several seconds of inference apiece.
        </p>
        <div class="notice-actions">
            <button type="button" id="scope-fix" class="btn-primary btn-small">
                Watch only ${suggested.map(escapeHtml).join(', ')}
            </button>
            <button type="button" id="scope-ignore" class="btn-secondary btn-small">
                Watch every object
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
    };
    // A new rule starts enabled. An edit leaves the enabled state alone: sending
    // `enabled: true` on every save silently re-armed rules the operator had
    // switched off.
    if (!editId) data.enabled = true;

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
        errorDiv.textContent = `The rule was not saved: ${err.message}`;
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
    if (!confirm('Delete this rule?\n\nIts alert history is kept under "Default (unmatched)".')) return;
    try {
        await apiFetch(`/api/rules/${ruleId}`, { method: 'DELETE' });
        await loadRules();
        await loadAlerts();  // the deleted rule's alerts were reassigned
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
        await loadAlerts();
        console.log(`Cleared ${result.deleted} rule(s).`);
    } catch (err) {
        alert(`Failed to clear rules: ${err.message}`);
    }
}

// ── Alerts ──────────────────────────────────────────────────────

async function loadAlerts() {
    try {
        const result = await apiFetch(`/api/alerts?limit=${ALERT_LIMIT}&offset=0`);
        alerts = result.alerts || [];
        renderAlerts(alerts);
        updateLoadMore(result.total);
    } catch (err) {
        console.error('Failed to load alerts:', err);
    }
}

async function loadMoreAlerts() {
    // Continue from what is already shown, live pushes included. A separate
    // offset counter ignored alerts prepended over the WebSocket, so every
    // live alert shifted the next page and "Load more" returned duplicates.
    try {
        const result = await apiFetch(`/api/alerts?limit=${ALERT_LIMIT}&offset=${alerts.length}`);
        const known = new Set(alerts.map(a => a.id));
        alerts = [...alerts, ...(result.alerts || []).filter(a => !known.has(a.id))];
        renderAlerts(alerts);
        updateLoadMore(result.total);
    } catch (err) {
        console.error('Failed to load more alerts:', err);
    }
}

function updateLoadMore(total) {
    document.getElementById('load-more-btn').classList.toggle('hidden', total <= alerts.length);
}

function filterAlerts(filter) {
    currentFilter = filter;
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`filter-${filter}`).classList.add('active');
    renderAlerts(alerts);
}

function matchesFilter(alert) {
    if (currentFilter === 'violations') return alert.verdict === 1;
    if (currentFilter === 'clear') return alert.verdict === 0;
    return true;
}

function renderAlerts(alertList) {
    const container = document.getElementById('alerts-list');
    const filtered = alertList.filter(matchesFilter);

    if (filtered.length === 0) {
        container.innerHTML = alertList.length === 0
            ? '<div class="empty-state">No alerts yet. They appear here as the model answers your rules.</div>'
            : '<div class="empty-state">No alerts match this filter.</div>';
        return;
    }

    container.innerHTML = filtered.map(alertCardHtml).join('');
}

// SQLite's datetime('now') is "YYYY-MM-DD HH:MM:SS" in UTC. Appending "Z" to
// that is not ISO 8601 — the space is — and Safari parses it as Invalid Date.
function parseDbTime(value) {
    if (!value) return null;
    const date = new Date(value.replace(' ', 'T') + (/[zZ]|[+-]\d\d:?\d\d$/.test(value) ? '' : 'Z'));
    return isNaN(date.getTime()) ? null : date;
}

function alertCardHtml(alert) {
    const isViolation = alert.verdict === 1;
    const cardClass = isViolation ? 'violation' : 'clear';
    const verdictClass = isViolation ? 'positive' : 'negative';
    const verdictText = isViolation ? 'Violation' : 'Clear';

    const created = parseDbTime(alert.created_at);
    const timeStr = created ? created.toLocaleTimeString() : '--';
    const fullTime = created ? created.toLocaleString() : '';

    // frame_snapshot is a server-side path (data/snapshots/<uuid>.jpg);
    // the snapshots directory is mounted at /snapshots.
    const snapshot = alert.frame_snapshot
        ? `<img src="/snapshots/${encodeURIComponent(alert.frame_snapshot.split('/').pop())}"
                alt="Alert evidence" class="alert-snapshot" loading="lazy">`
        : '';

    const ruleLabel = alert.rule_name
        ? escapeHtml(alert.rule_name)
        : `Rule #${Number(alert.rule_id)}`;

    // Which subject this verdict is about. Several alerts can now come from
    // one frame, and without the id they read as duplicates of each other
    // rather than as different people.
    const subject = alert.label
        ? escapeHtml(alert.label) + (alert.track_id >= 0 ? ` #${Number(alert.track_id)}` : '')
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
          }">p(yes) ${alert.confidence.toFixed(2)}${closeCall ? ', close call' : ''}</span>`
        : '';

    // A probed verdict's text is just "Yes" or "No", which the verdict word
    // already says. Show the model's text only when it says something more.
    const rawText = (alert.raw_response || '').trim();
    const response = /^(yes|no)[.!]?$/i.test(rawText) ? '' : rawText.substring(0, 200);

    return `
        <div class="alert-card ${cardClass}${snapshot ? '' : ' no-snapshot'}" data-alert-id="${Number(alert.id)}">
            ${snapshot}
            <div class="alert-main">
                <div class="alert-header">
                    <span class="alert-title"><span class="alert-verdict ${verdictClass}">${verdictText}</span>${ruleLabel}</span>
                    <span class="alert-time" title="${escapeHtml(fullTime)}">${timeStr}</span>
                </div>
                ${response ? `<div class="alert-response">${escapeHtml(response)}</div>` : ''}
                <div class="alert-meta">
                    ${subject ? `<span>${subject}</span>` : ''}
                    <span>Camera ${escapeHtml(alert.camera_id)}</span>
                    ${confidenceBadge}
                    <span>${Number(alert.inference_ms || 0).toFixed(0)} ms</span>
                </div>
            </div>
        </div>
    `;
}

function addLiveAlert(alertData) {
    if (alerts.some(a => a.id === alertData.id)) return;
    alerts.unshift(alertData);

    const container = document.getElementById('alerts-list');
    if (alerts.length > MAX_ALERTS) {
        alerts.length = MAX_ALERTS;
        document.getElementById('load-more-btn').classList.remove('hidden');
    }
    if (!matchesFilter(alertData)) return;

    // One node in, not the whole list rebuilt.
    const empty = container.querySelector('.empty-state');
    if (empty) empty.remove();
    container.insertAdjacentHTML('afterbegin', alertCardHtml(alertData));
    container.firstElementChild.classList.add('is-new');
    while (container.children.length > MAX_ALERTS) {
        container.lastElementChild.remove();
    }
}

// ── Status Polling ──────────────────────────────────────────────

function startStatusPolling() {
    pollStatus();
    statusPollTimer = setInterval(pollStatus, 3000);
    // Catch up at once when the tab comes back rather than showing stale
    // numbers for up to one interval.
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) pollStatus();
    });
}

async function pollStatus() {
    // A hidden tab has nobody reading it, and a slow response must not let
    // requests pile up behind one another.
    if (document.hidden || statusPollInFlight) return;
    statusPollInFlight = true;
    try {
        const status = await apiFetch('/api/status');
        updateStats(status);
    } catch (err) {
        // Silently ignore — server may not be fully ready
    } finally {
        statusPollInFlight = false;
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

    // Memo hit rate — whether load control is actually working. Hits are
    // counted per rule and judged subjects per subject, so this is a ratio of
    // reuse to total demand rather than a like-for-like percentage.
    const vlm = status.vlm || {};
    const hits = (vlm.instances_memoised || 0) + (vlm.instances_memoised_spatial || 0);
    const misses = (vlm.memo_miss_new_track || 0) + (vlm.memo_miss_ttl || 0)
        + (vlm.memo_miss_moved || 0);
    const looked = hits + misses;
    const rate = looked ? hits / looked : 0;
    document.getElementById('stat-memo').textContent = looked
        ? `${(rate * 100).toFixed(0)}%` : '--';
    document.getElementById('stat-memo-bar').style.width = `${rate * 100}%`;

    // Capture-to-verdict, which is what an operator actually waits through.
    const p95 = vlm.latency_p95_ms || 0;
    document.getElementById('stat-latency').textContent = p95
        ? `${(p95 / 1000).toFixed(1)}s` : '--';

    // Timeouts and failures. Without these a total VLM outage looks exactly
    // like a quiet scene: triggers simply stop incrementing, with no reason.
    const timeouts = vlm.triggers_timed_out || 0;
    const failures = (vlm.triggers_malformed || 0) + (vlm.triggers_frame_lost || 0)
        + (vlm.triggers_backend_error || 0) + (vlm.triggers_internal_error || 0);
    const timeoutEl = document.getElementById('stat-timeouts');
    timeoutEl.textContent = timeouts;
    timeoutEl.classList.toggle('is-bad', timeouts > 0);
    const failEl = document.getElementById('stat-failures');
    failEl.textContent = failures;
    failEl.classList.toggle('is-bad', failures > 0);

    // The one field that explains a silent dashboard.
    const down = vlm.backend_unavailable;
    const backendEl = document.getElementById('stat-backend');
    backendEl.textContent = down ? 'Down' : (vlm.is_running ? 'Running' : 'Off');
    backendEl.classList.toggle('is-bad', Boolean(down));

    // Device
    document.getElementById('device-badge').textContent = (status.inference_device || 'CPU').toUpperCase();

    // Uptime
    const uptime = status.uptime_seconds || 0;
    document.getElementById('uptime-badge').textContent = `Up ${formatUptime(uptime)}`;

    updateCameraStatus(status);
}

function updateCameraStatus(status) {
    const label = document.getElementById('camera-status');
    const errorOverlay = document.getElementById('stream-error');
    if (!label || !errorOverlay) return;

    if (status.pipeline_enabled === false) {
        label.textContent = 'Pipeline off';
        errorOverlay.classList.remove('hidden');
        return;
    }

    const cam = (status.cameras || [])[0];
    if (cam) {
        label.textContent = cam.connected
            ? `Camera ${cam.camera_id}, ${(cam.fps || 0).toFixed(1)} FPS`
            : `Camera ${cam.camera_id} is offline`;
        errorOverlay.classList.toggle('hidden', cam.connected);
    }
}

// ── WebSocket ───────────────────────────────────────────────────

function connectWebSocket() {
    if (ws) {
        // Detach first, or closing the old socket schedules a second reconnect.
        ws.onclose = null;
        clearInterval(ws._pingTimer);
        ws.close();
    }

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('WebSocket connected');
        setConnection(true);
        wsReconnectDelayMs = 1000;

        // Alerts broadcast while disconnected were never delivered. Reload
        // rather than leave a silent gap in the feed.
        if (wsHasConnected) loadAlerts();
        wsHasConnected = true;

        // Send periodic pings
        const socket = ws;
        socket._pingTimer = setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send('ping');
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

    ws.onclose = (event) => {
        console.log('WebSocket disconnected');
        // Each socket used to leave its ping interval running forever.
        clearInterval(event.target._pingTimer);
        setConnection(false);

        // Back off, so a server that is down for minutes is not hammered.
        if (wsReconnectTimer) clearTimeout(wsReconnectTimer);
        wsReconnectTimer = setTimeout(connectWebSocket, wsReconnectDelayMs);
        wsReconnectDelayMs = Math.min(wsReconnectDelayMs * 2, WS_RECONNECT_MAX_MS);
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
}

function setConnection(live) {
    const indicator = document.getElementById('ws-indicator');
    indicator.classList.toggle('is-live', live);
    indicator.lastElementChild.textContent = live ? 'Live' : 'Disconnected';
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

// Safe in text and in quoted attributes alike. The DOM-based version left
// quotes unescaped, which is only safe between tags.
function escapeHtml(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// FastAPI's 422 detail is a list of {loc, msg}; show the messages, not
// "[object Object]".
function describeError(detail) {
    if (!detail) return '';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
        return detail
            .map(d => `${(d.loc || []).filter(p => p !== 'body').join('.')}: ${d.msg}`)
            .join('; ');
    }
    return JSON.stringify(detail);
}

function formatUptime(seconds) {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(0)}m`;
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hrs}h ${mins}m`;
}
