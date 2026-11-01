/* FleetIQ dashboard — vanilla JS, no build step.
   All data comes from the FastAPI service at /api. */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  options: null,
  overview: null,
  models: [],
  priority: [],
  filter: "all",
  activeTab: "overview",
  health: null,
};

const DEMO_MAINTENANCE = {
  vehicle_id: 200055,
  make_and_model: "Isuzu NPR",
  vehicle_type: "Light Truck",
  year_of_manufacture: 2018,
  route_info: "Highway Freight",
  weather_conditions: "Hot",
  road_conditions: "Congested",
  communication_interface: "OBD-II/CAN",
  telematics_gateway: "Fleet_IoT_Node",
  edge_device_class: "Automotive_ECU",
  usage_hours: 10038.57,
  load_capacity: 10000,
  actual_load: 10105.3,
  engine_temperature: 124.26,
  tire_pressure: 23.88,
  fuel_consumption: 14.4,
  battery_status: 43.94,
  vibration_levels: 3.32,
  oil_quality: 48.59,
  failure_history: 10,
  anomalies_detected: 7,
  diagnostic_trouble_code_count: 8,
  can_message_rate_hz: 60.05,
  sensor_packet_loss_rate: 0.0196,
  historical_maintenance_cost: 2945.74,
  maintenance_type: "Electrical Repair",
  brake_condition: "Poor",
  service_note:
    "Driver reports brake vibration and an overheat smell near the wheel, ERR-431. Pads last replaced at 184,000 miles.",
};

/* ---------- Utilities ---------- */

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const fmtInt = (value) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(Number(value));

const fmtNum = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toFixed(digits);

const fmtMoney = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : `$${Number(value).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })}`;

const fmtPct = (value, digits = 1) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : `${(Number(value) * 100).toFixed(digits)}%`;

const shortId = (value) => (value ? String(value).slice(0, 10) : "—");

function timeAgo(iso) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

function formatMonth(value) {
  if (!value) return "";
  const [year, month] = String(value).split("-");
  const names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${names[Number(month) - 1] || month} '${String(year).slice(2)}`;
}

function toast(message, isError = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("is-error", isError);
  node.classList.add("is-visible");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("is-visible"), 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!response.ok) {
    const detail =
      body && typeof body === "object" && body.detail
        ? typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail)
        : text || `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return body;
}

function setStatus(id, ok, label) {
  const node = $(`#${id}`);
  if (!node) return;
  node.classList.toggle("is-ok", Boolean(ok));
  node.classList.toggle("is-bad", ok === false);
  node.innerHTML = `<i></i> ${esc(label)}`;
}

/* ---------- Navigation ---------- */

function activateTab(name) {
  const target = $(`#panel-${name}`) ? name : "overview";
  state.activeTab = target;
  $$(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === target));
  $$(".panel").forEach((panel) => panel.classList.toggle("is-active", panel.id === `panel-${target}`));
  if (location.hash !== `#${target}`) history.replaceState(null, "", `#${target}`);
  if (target === "priority") loadPriority();
  if (target === "monitoring") loadMonitoring();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setupNavigation() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
  $$("[data-goto]").forEach((button) =>
    button.addEventListener("click", () => activateTab(button.dataset.goto))
  );
  window.addEventListener("hashchange", () => activateTab(location.hash.replace("#", "") || "overview"));
}

/* ---------- Status, options, models ---------- */

async function refreshStatus() {
  try {
    const health = await api("/api/health");
    state.health = health;
    setStatus("status-api", true, "API");
    setStatus("status-db", health.database, health.database ? "Warehouse" : "Warehouse down");
    const modelCount = Object.values(health.models || {}).filter(Boolean).length;
    setStatus("status-models", modelCount === 2, `${modelCount}/2 models`);
    $("#status-version").textContent = `v${health.version}`;
    $("#footer-version").textContent = `v${health.version}`;
  } catch (error) {
    setStatus("status-api", false, "API down");
    setStatus("status-db", false, "Warehouse");
    setStatus("status-models", false, "Models");
    toast(`Cannot reach the API: ${error.message}`, true);
  }
}

async function loadModels() {
  try {
    state.models = await api("/api/models");
    renderRegistry();
    if (state.activeTab === "monitoring") renderModelCards();
  } catch (error) {
    toast(`Model registry unavailable: ${error.message}`, true);
  }
}

function fillSelect(id, values) {
  const select = $(`#${id}`);
  if (!select || !Array.isArray(values) || !values.length) return;
  const current = select.value;
  select.innerHTML = values.map((value) => `<option value="${esc(value)}">${esc(value)}</option>`).join("");
  if (values.includes(current)) select.value = current;
}

async function loadOptions() {
  try {
    state.options = await api("/api/analytics/options");
    const map = {
      "m-make": "make_and_model",
      "m-type": "vehicle_type",
      "m-route": "route_info",
      "m-weather": "weather_conditions",
      "m-road": "road_conditions",
      "m-comm": "communication_interface",
      "m-gateway": "telematics_gateway",
      "m-edge": "edge_device_class",
      "m-mtype": "maintenance_type",
      "f-origin": "origin_warehouse",
      "f-destination": "destination",
      "f-carrier": "carrier",
    };
    Object.entries(map).forEach(([id, key]) => fillSelect(id, state.options[key]));
    setSelect("m-make", DEMO_MAINTENANCE.make_and_model);
    setSelect("m-type", DEMO_MAINTENANCE.vehicle_type);
    setSelect("m-route", DEMO_MAINTENANCE.route_info);
    setSelect("m-weather", DEMO_MAINTENANCE.weather_conditions);
    setSelect("m-road", DEMO_MAINTENANCE.road_conditions);
    setSelect("m-comm", DEMO_MAINTENANCE.communication_interface);
    setSelect("m-gateway", DEMO_MAINTENANCE.telematics_gateway);
    setSelect("m-edge", DEMO_MAINTENANCE.edge_device_class);
    setSelect("m-mtype", DEMO_MAINTENANCE.maintenance_type);
    setSelect("f-origin", "Warehouse_MIA");
    setSelect("f-destination", "Atlanta");
    setSelect("f-carrier", "DHL");
    refreshBenchmark();
  } catch (error) {
    toast(`Could not load form options: ${error.message}`, true);
  }
}

function setSelect(id, value) {
  const select = $(`#${id}`);
  if (!select) return;
  const exists = Array.from(select.options).some((option) => option.value === value);
  if (exists) select.value = value;
}

async function ensureOptions() {
  // The demo loader can be clicked before the first options request finishes.
  // Waiting here prevents it from writing numeric fields while the selects are
  // still empty, which would leave the form on the default maintenance type.
  if (!state.options) await loadOptions();
}

/* ---------- Overview ---------- */

async function loadOverview() {
  try {
    const [overview, segments, trend, brakes, carriers] = await Promise.all([
      api("/api/analytics/overview"),
      api("/api/analytics/segments"),
      api("/api/analytics/maintenance-trend"),
      api("/api/analytics/brake-condition"),
      api("/api/analytics/carriers"),
    ]);
    state.overview = overview;
    renderHero(overview);
    renderKpis(overview);
    renderSegments(segments);
    renderTrend(trend);
    renderBrakes(brakes);
    renderCarriers(carriers);
  } catch (error) {
    toast(`Overview failed: ${error.message}`, true);
  }
}

function modelMetric(key, metric) {
  const model = state.models.find((item) => item.key === key);
  const value = model?.metrics?.[metric];
  return typeof value === "number" ? value : null;
}

function renderHero(overview) {
  const maintenance = overview.maintenance || {};
  const shipments = overview.shipments || {};
  $("#hero-fleet").textContent = fmtInt(maintenance.total_vehicles);
  $("#hero-rate").textContent = fmtPct(maintenance.maintenance_rate);
  $("#hero-mape").textContent = modelMetric("freight", "mape") !== null ? `${fmtNum(modelMetric("freight", "mape"), 2)}%` : "—";
  const auc = modelMetric("maintenance", "roc_auc");
  $("#hero-auc").textContent = auc !== null ? fmtNum(auc, 3) : "—";
  $("#hero-note").textContent = `${fmtInt(shipments.total_shipments)} priced shipments · ${fmtInt(maintenance.distinct_models)} vehicle models · refreshed ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

function renderKpis(overview) {
  const maintenance = overview.maintenance || {};
  const shipments = overview.shipments || {};
  const predictions = overview.predictions || {};
  const cards = [
    {
      label: "Vehicles in warehouse",
      value: fmtInt(maintenance.total_vehicles),
      meta: `${fmtInt(maintenance.distinct_models)} models · avg year ${fmtNum(maintenance.avg_model_year, 0)}`,
      delta: "fleet",
    },
    {
      label: "Maintenance due",
      value: fmtPct(maintenance.maintenance_rate),
      meta: `${fmtInt(maintenance.poor_brake_vehicles)} poor brakes`,
      delta: `${fmtInt(maintenance.repeat_failure_vehicles)} repeat failures`,
    },
    {
      label: "Freight spend",
      value: fmtMoney(shipments.total_spend, 0),
      meta: `${fmtMoney(shipments.avg_cost)} avg · ${fmtInt(shipments.avg_distance)} mi`,
      delta: "completed",
    },
    {
      label: "Prediction traffic",
      value: fmtInt(predictions.total),
      meta: `last 24h · ${predictions.by_endpoint?.length || 0} endpoints`,
      delta: predictions.by_endpoint?.[0]?.avg_latency_ms ? `${fmtNum(predictions.by_endpoint[0].avg_latency_ms, 1)} ms` : "online",
    },
  ];
  $("#kpi-grid").innerHTML = cards
    .map(
      (card) => `
      <article class="kpi-card">
        <div class="kpi-label">${esc(card.label)}</div>
        <div class="kpi-value">${esc(card.value)}</div>
        <div class="kpi-meta"><span>${esc(card.meta)}</span><span class="kpi-delta">${esc(card.delta)}</span></div>
      </article>`
    )
    .join("");
}

function barList(container, rows, options) {
  if (!rows || !rows.length) {
    container.innerHTML = '<p class="muted">No rows returned.</p>';
    return;
  }
  const values = rows.map((row) => Number(row[options.value]) || 0);
  const max = Math.max(...values, 1e-9);
  container.innerHTML = rows
    .map((row) => {
      const value = Number(row[options.value]) || 0;
      const width = Math.max(2, (value / max) * 100);
      return `
        <div class="bar-row">
          <span class="bar-label" title="${esc(row[options.label])}">${esc(row[options.label])}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${options.color || "var(--teal)"}"></div></div>
          <span class="bar-value">${esc(options.format(value))}</span>
        </div>`;
    })
    .join("");
}

function renderSegments(rows) {
  barList($("#segments-chart"), rows, {
    label: "vehicle_type",
    value: "risk_rate",
    format: (value) => fmtPct(value),
    color: "var(--accent)",
  });
}

function renderCarriers(rows) {
  const sorted = [...(rows || [])].sort((a, b) => (b.avg_cost || 0) - (a.avg_cost || 0));
  barList($("#carrier-chart"), sorted, {
    label: "carrier",
    value: "avg_cost",
    format: (value) => fmtMoney(value, 0),
    color: "var(--blue)",
  });
}

function renderBrakes(rows) {
  const container = $("#brake-cards");
  if (!rows || !rows.length) {
    container.innerHTML = '<p class="muted">No brake records.</p>';
    return;
  }
  container.innerHTML = rows
    .map((row) => {
      const rate = Number(row.risk_rate) || 0;
      const color = rate > 0.45 ? "var(--red)" : rate > 0.25 ? "var(--amber)" : "var(--teal)";
      return `
        <div class="brake-row">
          <span class="brake-name">${esc(row.brake_condition)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, rate * 100)}%;background:${color}"></div></div>
          <span class="brake-rate">${fmtPct(rate)}</span>
        </div>`;
    })
    .join("");
}

function renderTrend(rows) {
  const container = $("#trend-chart");
  if (!rows || rows.length < 2) {
    container.innerHTML = '<p class="muted">Not enough service history.</p>';
    return;
  }
  const width = 560;
  const height = 200;
  const pad = { left: 34, right: 14, top: 14, bottom: 26 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const events = rows.map((row) => Number(row.events) || 0);
  const risks = rows.map((row) => Number(row.risk_rate) || 0);
  const maxEvents = Math.max(...events, 1);
  const maxRisk = Math.max(...risks, 1e-6);
  const x = (index) => pad.left + (index / Math.max(rows.length - 1, 1)) * innerW;
  const yEvents = (value) => pad.top + innerH - (value / maxEvents) * innerH;
  const yRisk = (value) => pad.top + innerH - (value / maxRisk) * innerH;

  const eventPath = events.map((value, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${yEvents(value).toFixed(1)}`).join(" ");
  const areaPath = `${eventPath} L${x(rows.length - 1).toFixed(1)},${(pad.top + innerH).toFixed(1)} L${pad.left},${(pad.top + innerH).toFixed(1)} Z`;
  const riskPath = risks.map((value, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${yRisk(value).toFixed(1)}`).join(" ");
  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((fraction) => {
      const y = pad.top + innerH - fraction * innerH;
      return `<line class="grid-line" x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" />
              <text class="axis-text" x="${pad.left - 5}" y="${y + 3}" text-anchor="end">${fmtInt(maxEvents * fraction)}</text>`;
    })
    .join("");
  const labelIndexes = [0, Math.floor(rows.length / 2), rows.length - 1];
  const labels = labelIndexes
    .map((index) => `<text class="axis-text" x="${x(index)}" y="${height - 7}" text-anchor="middle">${esc(formatMonth(rows[index].month))}</text>`)
    .join("");
  const dots = events
    .map((value, index) => `<circle class="dot" cx="${x(index)}" cy="${yEvents(value)}" r="2.4"><title>${esc(formatMonth(rows[index].month))}: ${fmtInt(value)} events</title></circle>`)
    .join("");

  container.innerHTML = `
    <svg class="line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Monthly service volume and maintenance risk rate">
      ${grid}
      <path class="area" d="${areaPath}"></path>
      <path class="line" d="${eventPath}"></path>
      <path class="line-secondary" d="${riskPath}"></path>
      ${dots}
      ${labels}
    </svg>
    <div class="chart-legend">
      <span><i style="background:var(--teal)"></i> service events</span>
      <span><i style="background:var(--accent)"></i> maintenance rate (scaled)</span>
    </div>`;
}

function renderRegistry() {
  const container = $("#registry-card");
  if (!state.models.length) {
    container.innerHTML = '<p class="muted">No model metadata.</p>';
    return;
  }
  container.innerHTML = state.models
    .map((model) => {
      const ok = Boolean(model.available);
      const version = model.mlflow?.version || model.version || "local";
      return `
        <div class="registry-item">
          <div>
            <div class="registry-name">${esc(model.name || model.key)}</div>
            <div class="registry-meta">v${esc(version)} · run ${esc(shortId(model.run_id))}</div>
            <div class="registry-meta">${esc(model.trained_at ? new Date(model.trained_at).toLocaleString() : "not trained")}</div>
          </div>
          <span class="model-pill ${ok ? "ok" : "bad"}">${ok ? "champion" : "missing"}</span>
        </div>`;
    })
    .join("");
}

/* ---------- Maintenance form ---------- */

const RANGE_BINDINGS = [
  { range: "m-temp-range", number: "m-temp", output: "m-temp-out", format: (value) => `${value} °C` },
  { range: "m-vibration-range", number: "m-vibration", output: "m-vibration-out", format: (value) => fmtNum(value, 1) },
  { range: "m-battery-range", number: "m-battery", output: "m-battery-out", format: (value) => `${fmtNum(value, 0)}%` },
  { range: "m-oil-range", number: "m-oil", output: "m-oil-out", format: (value) => `${fmtNum(value, 0)} / 100` },
];

function paintRange(range) {
  const min = Number(range.min || 0);
  const max = Number(range.max || 100);
  const pct = max === min ? 0 : ((Number(range.value) - min) / (max - min)) * 100;
  range.style.background = `linear-gradient(90deg, var(--accent) 0%, var(--accent) ${pct}%, var(--line) ${pct}%)`;
}

function setupRanges() {
  RANGE_BINDINGS.forEach((binding) => {
    const range = $(`#${binding.range}`);
    const number = $(`#${binding.number}`);
    const output = $(`#${binding.output}`);
    if (!range || !number || !output) return;
    const sync = () => {
      number.value = range.value;
      output.textContent = binding.format(range.value);
      paintRange(range);
    };
    range.addEventListener("input", sync);
    sync();
  });
}

function syncRangesFromNumbers() {
  RANGE_BINDINGS.forEach((binding) => {
    const range = $(`#${binding.range}`);
    const number = $(`#${binding.number}`);
    const output = $(`#${binding.output}`);
    if (!range || !number || !output) return;
    const value = Number(number.value);
    if (Number.isFinite(value)) {
      range.value = String(value);
      output.textContent = binding.format(value);
      paintRange(range);
    }
  });
}

function value(id) {
  const node = $(`#${id}`);
  return node ? node.value : "";
}

function numberValue(id) {
  const raw = value(id);
  return raw === "" ? null : Number(raw);
}

function maintenancePayload() {
  return {
    vehicle_id: DEMO_MAINTENANCE.vehicle_id,
    make_and_model: value("m-make"),
    vehicle_type: value("m-type"),
    year_of_manufacture: Math.round(numberValue("m-year") || 2018),
    route_info: value("m-route"),
    weather_conditions: value("m-weather"),
    road_conditions: value("m-road"),
    communication_interface: value("m-comm"),
    telematics_gateway: value("m-gateway"),
    edge_device_class: value("m-edge"),
    usage_hours: numberValue("m-usage") || 0,
    load_capacity: numberValue("m-capacity") || 1,
    actual_load: numberValue("m-load") || 0,
    engine_temperature: numberValue("m-temp") || 0,
    tire_pressure: numberValue("m-tire") || 0,
    fuel_consumption: numberValue("m-fuel") || 0,
    battery_status: numberValue("m-battery") || 0,
    vibration_levels: numberValue("m-vibration") || 0,
    oil_quality: numberValue("m-oil") || 0,
    failure_history: Math.round(numberValue("m-failures") || 0),
    anomalies_detected: Math.round(numberValue("m-anomalies") || 0),
    diagnostic_trouble_code_count: Math.round(numberValue("m-dtc") || 0),
    can_message_rate_hz: numberValue("m-can") || 0,
    sensor_packet_loss_rate: numberValue("m-packet") || 0,
    historical_maintenance_cost: numberValue("m-histcost") || 0,
    maintenance_type: value("m-mtype"),
    brake_condition: value("m-brake"),
    service_note: value("m-note") || null,
  };
}

async function fillMaintenanceForm(record) {
  await ensureOptions();
  const map = {
    "m-make": record.make_and_model,
    "m-type": record.vehicle_type,
    "m-year": record.year_of_manufacture,
    "m-route": record.route_info,
    "m-weather": record.weather_conditions,
    "m-road": record.road_conditions,
    "m-comm": record.communication_interface,
    "m-gateway": record.telematics_gateway,
    "m-edge": record.edge_device_class,
    "m-usage": record.usage_hours,
    "m-capacity": record.load_capacity,
    "m-load": record.actual_load,
    "m-temp": record.engine_temperature,
    "m-tire": record.tire_pressure,
    "m-fuel": record.fuel_consumption,
    "m-battery": record.battery_status,
    "m-vibration": record.vibration_levels,
    "m-oil": record.oil_quality,
    "m-failures": record.failure_history,
    "m-anomalies": record.anomalies_detected,
    "m-dtc": record.diagnostic_trouble_code_count,
    "m-can": record.can_message_rate_hz,
    "m-packet": record.sensor_packet_loss_rate,
    "m-histcost": record.historical_maintenance_cost,
    "m-mtype": record.maintenance_type,
    "m-brake": record.brake_condition,
    "m-note": record.service_note,
  };
  Object.entries(map).forEach(([id, next]) => {
    const node = $(`#${id}`);
    if (!node) return;
    if (node.tagName === "SELECT") setSelect(id, next);
    else node.value = next ?? "";
  });
  syncRangesFromNumbers();
  toast("Loaded a high-risk vehicle from the warehouse.");
}

function gaugeSvg(probability, level) {
  const pct = Math.max(0, Math.min(1, Number(probability) || 0));
  const color =
    level === "critical" ? "var(--red)" : level === "high" ? "var(--accent)" : level === "moderate" ? "var(--amber)" : "var(--teal)";
  return `
    <svg class="gauge" viewBox="0 0 150 100" role="img" aria-label="Risk probability ${(pct * 100).toFixed(1)} percent">
      <path d="M 15 88 A 60 60 0 0 1 135 88" fill="none" stroke="var(--paper-3)" stroke-width="11" stroke-linecap="butt"></path>
      <path d="M 15 88 A 60 60 0 0 1 135 88" fill="none" stroke="${color}" stroke-width="11" stroke-linecap="butt"
            pathLength="100" stroke-dasharray="${(pct * 100).toFixed(2)} 100"></path>
      <text x="75" y="76" text-anchor="middle" style="font-family:var(--mono);font-size:23px" fill="var(--ink)">${(pct * 100).toFixed(1)}%</text>
      <text x="75" y="93" text-anchor="middle" style="font-family:var(--sans);font-size:8px;letter-spacing:1.4px" fill="var(--ink-faint)">PROBABILITY</text>
    </svg>`;
}

function riskChip(level) {
  return `<span class="risk-chip ${esc(level)}">${esc(level)}</span>`;
}

function renderSignals(signals) {
  if (!signals || !signals.length) return '<p class="muted small">No individual signal crossed the fleet threshold.</p>';
  const max = Math.max(...signals.map((signal) => Number(signal.impact) || 0), 1e-9);
  return `<div class="signal-list">${signals
    .map(
      (signal) => `
      <div class="signal-row">
        <div>
          <div class="signal-label">${esc(signal.label)}</div>
          <div class="signal-detail">${esc(signal.direction)} · ${esc(signal.detail)}</div>
          <div class="signal-bar"><i style="width:${Math.max(3, ((Number(signal.impact) || 0) / max) * 100)}%"></i></div>
        </div>
        <div class="signal-impact">${fmtNum(signal.impact, 2)}</div>
      </div>`
    )
    .join("")}</div>`;
}

function renderNote(note) {
  if (!note) return "";
  const chips = [];
  (note.error_codes || []).forEach((code) => chips.push(`<span class="note-chip">${esc(code)}</span>`));
  (note.part_numbers || []).forEach((part) => chips.push(`<span class="note-chip">${esc(part)}</span>`));
  if (note.mileage) chips.push(`<span class="note-chip">${fmtInt(note.mileage)} mi</span>`);
  (note.keywords || []).forEach((keyword) => chips.push(`<span class="note-chip">${esc(keyword)}</span>`));
  chips.push(`<span class="note-chip urgency-${esc(note.urgency)}">${esc(note.urgency)} urgency · ${fmtInt(note.urgency_score)}</span>`);
  return `
    <div class="divider"></div>
    <p class="eyebrow">Parsed service note</p>
    <div class="note-chips">${chips.join("")}</div>`;
}

function feedbackMarkup(predictionId, placeholder) {
  return `
    <div class="feedback-row">
      <span class="small">Observed outcome</span>
      <input type="number" id="feedback-value" placeholder="${esc(placeholder)}" step="any" />
      <button class="button ghost small" id="feedback-save" ${predictionId ? "" : "disabled"}>Record</button>
      <span class="feedback-note">${predictionId ? `Stored with prediction ${esc(shortId(predictionId))}` : "Prediction logging is disabled."}</span>
    </div>`;
}

function bindFeedback(predictionId, successMessage) {
  const button = $("#feedback-save");
  const input = $("#feedback-value");
  if (!button || !input || !predictionId) return;
  button.addEventListener("click", async () => {
    const actual = Number(input.value);
    if (!Number.isFinite(actual)) {
      toast("Enter a numeric observed outcome first.", true);
      return;
    }
    button.disabled = true;
    try {
      await api(`/api/feedback/${predictionId}`, {
        method: "POST",
        body: JSON.stringify({ actual_value: actual }),
      });
      toast(successMessage);
      input.value = "";
    } catch (error) {
      toast(`Feedback failed: ${error.message}`, true);
    } finally {
      button.disabled = false;
    }
  });
}

function renderMaintenanceResult(data) {
  const container = $("#maintenance-result");
  const level = data.risk.level;
  const note = data.service_note;
  container.innerHTML = `
    <div class="result-top">
      <div>
        <p class="eyebrow">Risk assessment</p>
        <h2 class="result-title">${data.risk.maintenance_required ? "Service recommended" : "Continue monitoring"}</h2>
        <div style="display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap">
          ${riskChip(level)}
          <span class="small">threshold ${fmtNum(data.risk.threshold, 6)}</span>
        </div>
      </div>
      ${gaugeSvg(data.risk.probability, level)}
    </div>
    <div class="result-action"><strong>Recommended action</strong>${esc(data.recommended_action)}</div>
    <div class="meta-grid">
      <div>Model<span>${esc(data.model.name)}</span></div>
      <div>Version<span>${esc(data.model.version)}</span></div>
      <div>Run<span>${esc(shortId(data.model.run_id))}</span></div>
      <div>Trained<span>${esc(data.model.trained_at ? new Date(data.model.trained_at).toLocaleDateString() : "—")}</span></div>
    </div>
    <div class="divider"></div>
    <p class="eyebrow">Why this score</p>
    ${renderSignals(data.signals)}
    ${renderNote(note)}
    ${feedbackMarkup(data.prediction_id, "0 / 1")}`;
  bindFeedback(data.prediction_id, "Outcome recorded. Monitoring will use it on the next refresh.");
  container.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function setupMaintenance() {
  setupRanges();
  const form = $("#maintenance-form");
  form.addEventListener("reset", () => setTimeout(syncRangesFromNumbers, 0));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#maintenance-submit");
    button.disabled = true;
    button.textContent = "Scoring…";
    try {
      await ensureOptions();
      const data = await api("/api/predict/maintenance", {
        method: "POST",
        body: JSON.stringify(maintenancePayload()),
      });
      renderMaintenanceResult(data);
    } catch (error) {
      toast(`Scoring failed: ${error.message}`, true);
    } finally {
      button.disabled = false;
      button.textContent = "Score vehicle";
    }
  });
  $("#load-example").addEventListener("click", () => fillMaintenanceForm(DEMO_MAINTENANCE));
}

/* ---------- Freight form ---------- */

function freightPayload() {
  return {
    origin_warehouse: value("f-origin"),
    destination: value("f-destination"),
    carrier: value("f-carrier"),
    shipment_date: value("f-date"),
    weight_kg: numberValue("f-weight") || 0,
    distance_miles: numberValue("f-distance") || 0,
  };
}

async function refreshBenchmark() {
  const origin = value("f-origin");
  const destination = value("f-destination");
  const container = $("#lane-benchmark-card");
  if (!origin || !destination) return;
  try {
    const data = await api(
      `/api/analytics/lane-benchmark?origin=${encodeURIComponent(origin)}&destination=${encodeURIComponent(destination)}`
    );
    container.innerHTML = `
      <p class="eyebrow">Lane benchmark</p>
      <p class="small">${esc(origin)} → ${esc(destination)} · ${fmtInt(data.shipments)} completed shipments</p>
      <div class="benchmark-grid">
        <div><strong>${fmtMoney(data.median_cost, 0)}</strong><span>median</span></div>
        <div><strong>${fmtMoney(data.p25_cost, 0)}</strong><span>p25</span></div>
        <div><strong>${fmtMoney(data.p75_cost, 0)}</strong><span>p75</span></div>
      </div>`;
  } catch {
    container.innerHTML = `
      <p class="eyebrow">Lane benchmark</p>
      <p class="muted small">No completed shipments observed for ${esc(origin)} → ${esc(destination)}. The model interval is the main guide.</p>`;
  }
}

function renderFreightResult(data) {
  const container = $("#freight-result");
  const interval = data.interval_usd || {};
  const low = Number(interval.low) || 0;
  const high = Number(interval.high) || 0;
  const estimate = Number(data.estimated_cost_usd) || 0;
  const spanLow = Math.min(low, estimate) * 0.85;
  const spanHigh = Math.max(high, estimate) * 1.12 || 1;
  const position = (value) => ((value - spanLow) / (spanHigh - spanLow)) * 100;
  const benchmark = data.lane_benchmark;
  let comparison = "No completed shipments exist for this lane, so the interval is the main guide.";
  if (benchmark && benchmark.median_cost) {
    const delta = ((estimate - benchmark.median_cost) / benchmark.median_cost) * 100;
    const direction = delta >= 0 ? "above" : "below";
    comparison = `The point estimate is ${Math.abs(delta).toFixed(1)}% ${direction} the observed lane median of ${fmtMoney(benchmark.median_cost)} across ${fmtInt(benchmark.shipments)} shipments.`;
  }
  container.innerHTML = `
    <div class="result-top">
      <div>
        <p class="eyebrow">Freight quote</p>
        <h2 class="result-title">Estimated cost</h2>
        <div class="quote-number">${fmtMoney(estimate)} <small>USD</small></div>
        <div class="quote-interval">90% interval ${fmtMoney(low)} – ${fmtMoney(high)}</div>
      </div>
      ${riskChip("low")}
    </div>
    <div class="interval-track">
      <div class="interval-fill" style="left:${Math.max(0, position(low))}%;width:${Math.max(1, position(high) - position(low))}%"></div>
      <div class="interval-marker" style="left:${Math.max(0, position(estimate))}%"></div>
    </div>
    <div class="interval-scale"><span>${fmtMoney(spanLow, 0)}</span><span>${fmtMoney(estimate, 0)}</span><span>${fmtMoney(spanHigh, 0)}</span></div>
    <div class="meta-grid" style="margin-top:16px">
      <div>Cost per mile<span>${data.cost_per_mile_usd ? `${fmtMoney(data.cost_per_mile_usd, 3)}/mi` : "—"}</span></div>
      <div>Quote valid until<span>${esc(data.quote_valid_until || "—")}</span></div>
      <div>Model<span>${esc(data.model.name)}</span></div>
      <div>Version<span>${esc(data.model.version)}</span></div>
    </div>
    <div class="result-action"><strong>Lane read</strong>${esc(comparison)}</div>
    ${benchmark ? `
      <p class="eyebrow">Observed lane</p>
      <div class="benchmark-grid">
        <div><strong>${fmtMoney(benchmark.median_cost, 0)}</strong><span>median</span></div>
        <div><strong>${fmtMoney(benchmark.p25_cost, 0)}</strong><span>p25</span></div>
        <div><strong>${fmtMoney(benchmark.p75_cost, 0)}</strong><span>p75</span></div>
      </div>` : ""}
    ${feedbackMarkup(data.prediction_id, "actual USD")}`;
  bindFeedback(data.prediction_id, "Actual cost recorded. Feedback metrics will update on refresh.");
  container.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function setupFreight() {
  const form = $("#freight-form");
  ["f-origin", "f-destination"].forEach((id) => $(`#${id}`).addEventListener("change", refreshBenchmark));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#freight-submit");
    button.disabled = true;
    button.textContent = "Quoting…";
    try {
      await ensureOptions();
      const data = await api("/api/predict/freight", {
        method: "POST",
        body: JSON.stringify(freightPayload()),
      });
      renderFreightResult(data);
    } catch (error) {
      toast(`Quote failed: ${error.message}`, true);
    } finally {
      button.disabled = false;
      button.textContent = "Quote shipment";
    }
  });
  $("#load-shipment").addEventListener("click", async () => {
    try {
      const rows = await api("/api/shipments/recent?limit=1");
      if (!rows.length) return;
      const row = rows[0];
      setSelect("f-origin", row.origin_warehouse);
      setSelect("f-destination", row.destination);
      setSelect("f-carrier", row.carrier);
      $("#f-date").value = row.shipment_date;
      $("#f-weight").value = row.weight_kg;
      $("#f-distance").value = row.distance_miles;
      refreshBenchmark();
      toast("Loaded the most recent completed shipment.");
    } catch (error) {
      toast(`Could not load a shipment: ${error.message}`, true);
    }
  });
  refreshBenchmark();
}

/* ---------- Priority queue ---------- */

async function loadPriority() {
  const body = $("#priority-table tbody");
  body.innerHTML = '<tr><td colspan="8" class="loading-row">Scoring the fleet…</td></tr>';
  try {
    state.priority = await api("/api/vehicles/priority?limit=30");
    renderPriority();
  } catch (error) {
    body.innerHTML = `<tr><td colspan="8" class="loading-row">${esc(error.message)}</td></tr>`;
    toast(`Priority queue failed: ${error.message}`, true);
  }
}

function renderPriority() {
  const body = $("#priority-table tbody");
  const rows = state.priority.filter((row) => state.filter === "all" || row.level === state.filter);
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="8" class="loading-row">No vehicles match this filter.</td></tr>';
    return;
  }
  body.innerHTML = rows
    .map((row) => {
      const probability = Number(row.probability) || 0;
      const observed = row.actual_maintenance_required === 1 ? "due" : row.actual_maintenance_required === 0 ? "clear" : "—";
      return `
        <tr data-vehicle="${esc(row.vehicle_id)}">
          <td><span class="mono">#${esc(row.vehicle_id)}</span><br><span class="small">${esc(row.make_and_model)}</span></td>
          <td>${esc(row.vehicle_type)}</td>
          <td class="num">${fmtInt(row.usage_hours)} h</td>
          <td>${esc(row.brake_condition)}</td>
          <td class="num">${fmtInt(row.failure_history)}</td>
          <td>${esc(row.top_signal || "—")}</td>
          <td>
            <div class="risk-inline">
              <div class="bar-track"><div class="bar-fill" style="width:${Math.max(3, probability * 100)}%"></div></div>
              <span class="mono">${(probability * 100).toFixed(1)}%</span>
            </div>
            <div style="margin-top:5px">${riskChip(row.level)}</div>
          </td>
          <td class="observed ${observed === "due" ? "yes" : observed === "clear" ? "no" : ""}">${observed}</td>
        </tr>`;
    })
    .join("");
}

function setupPriority() {
  $$("[data-filter]").forEach((button) =>
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      $$("[data-filter]").forEach((item) => item.classList.toggle("is-active", item === button));
      renderPriority();
    })
  );
  $("#rescore-queue").addEventListener("click", loadPriority);
  $("#priority-table").addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-vehicle]");
    if (row) openVehicle(row.dataset.vehicle);
  });
}

/* ---------- Vehicle dialog ---------- */

async function openVehicle(vehicleId) {
  const dialog = $("#vehicle-dialog");
  const body = $("#dialog-body");
  $("#dialog-title").textContent = `Vehicle #${vehicleId}`;
  body.innerHTML = '<p class="muted">Loading the full record…</p>';
  dialog.showModal();
  try {
    const data = await api(`/api/vehicles/${vehicleId}`);
    const record = data.record || {};
    const observed = data.observed || {};
    const prediction = data.prediction;
    const fields = [
      ["Make and model", record.make_and_model],
      ["Vehicle type", record.vehicle_type],
      ["Model year", record.year_of_manufacture],
      ["Route", record.route_info],
      ["Usage hours", fmtInt(record.usage_hours)],
      ["Load ratio", record.load_capacity ? fmtNum(record.actual_load / record.load_capacity, 2) : "—"],
      ["Engine temp", `${fmtNum(record.engine_temperature, 1)} °C`],
      ["Tire pressure", `${fmtNum(record.tire_pressure, 1)} psi`],
      ["Fuel", `${fmtNum(record.fuel_consumption, 1)} L/100km`],
      ["Battery", `${fmtNum(record.battery_status, 1)}%`],
      ["Vibration", fmtNum(record.vibration_levels, 2)],
      ["Oil quality", fmtNum(record.oil_quality, 1)],
      ["Failures", fmtInt(record.failure_history)],
      ["Anomalies", fmtInt(record.anomalies_detected)],
      ["DTC count", fmtInt(record.diagnostic_trouble_code_count)],
      ["CAN rate", `${fmtNum(record.can_message_rate_hz, 0)} Hz`],
      ["Packet loss", fmtPct(record.sensor_packet_loss_rate, 2)],
      ["Historical cost", fmtMoney(record.historical_maintenance_cost)],
      ["Last maintenance", record.maintenance_type],
      ["Brakes", record.brake_condition],
      ["Observed outcome", observed.maintenance_required === 1 ? "maintenance due" : "clear"],
      ["Observed cost", fmtMoney(observed.maintenance_cost)],
    ];
    const predictionHtml = prediction
      ? `
        <div class="divider"></div>
        <div class="result-top">
          <div>
            <p class="eyebrow">Live score</p>
            <h3 class="result-title">${prediction.risk.maintenance_required ? "Service recommended" : "Continue monitoring"}</h3>
            <div style="display:flex;gap:8px;align-items:center;margin-top:8px">${riskChip(prediction.risk.level)}<span class="small">threshold ${fmtNum(prediction.risk.threshold, 6)}</span></div>
          </div>
          ${gaugeSvg(prediction.risk.probability, prediction.risk.level)}
        </div>
        <div class="result-action"><strong>Recommended action</strong>${esc(prediction.recommended_action)}</div>
        <p class="eyebrow">Drivers</p>
        ${renderSignals(prediction.signals)}`
      : `<div class="divider"></div><p class="muted">${esc(data.error || "No prediction available.")}</p>`;
    body.innerHTML = `
      <div class="detail-grid">
        ${fields.map(([label, value]) => `<div><span>${esc(label)}</span><strong>${esc(value ?? "—")}</strong></div>`).join("")}
      </div>
      ${predictionHtml}`;
  } catch (error) {
    body.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
  }
}

function setupDialog() {
  const dialog = $("#vehicle-dialog");
  $("#dialog-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
}

/* ---------- Monitoring ---------- */

function metricPills(model) {
  const metrics = model.metrics || {};
  const wanted =
    model.key === "maintenance"
      ? ["roc_auc", "pr_auc", "f1", "brier"]
      : ["mae", "mape", "r2", "coverage_90"];
  const formatters = {
    roc_auc: (value) => `AUC ${fmtNum(value, 3)}`,
    pr_auc: (value) => `PR-AUC ${fmtNum(value, 3)}`,
    f1: (value) => `F1 ${fmtNum(value, 3)}`,
    brier: (value) => `Brier ${fmtNum(value, 4)}`,
    mae: (value) => `MAE ${fmtMoney(value)}`,
    mape: (value) => `MAPE ${fmtNum(value, 2)}%`,
    r2: (value) => `R² ${fmtNum(value, 3)}`,
    coverage_90: (value) => `Coverage ${fmtPct(value)}`,
  };
  const pills = wanted
    .filter((key) => typeof metrics[key] === "number")
    .map((key) => `<span class="metric-pill">${formatters[key](metrics[key])}</span>`);
  return pills.join("") || '<span class="muted small">No holdout metrics stored.</span>';
}

function renderModelCards() {
  const container = $("#model-cards");
  if (!state.models.length) {
    container.innerHTML = '<article class="card"><p class="muted">No model metadata available.</p></article>';
    return;
  }
  container.innerHTML = state.models
    .map((model) => {
      const ok = Boolean(model.available);
      return `
        <article class="card model-card">
          <div class="model-card-head">
            <div>
              <p class="eyebrow">${esc(model.key)}</p>
              <h2 class="model-title">${esc(model.name || model.key)}</h2>
            </div>
            <span class="model-pill ${ok ? "ok" : "bad"}">${ok ? "champion" : "unavailable"}</span>
          </div>
          <div class="metric-row">${ok ? metricPills(model) : `<span class="muted small">${esc(model.error || "Bundle not loaded")}</span>`}</div>
          <div class="model-foot">version ${esc(model.version || "local")} · run ${esc(shortId(model.run_id))}<br>${esc(model.trained_at ? new Date(model.trained_at).toLocaleString() : "not trained")}</div>
          ${ok ? `<div><a class="small" href="/api/models/${esc(model.key)}/card" target="_blank" rel="noreferrer">Open model card →</a></div>` : ""}
        </article>`;
    })
    .join("");
}

function renderDrift(payload) {
  const container = $("#drift-report");
  const reports = payload.reports || {};
  const blocks = Object.entries(reports).map(([key, report]) => {
    const status = report.status || "insufficient_data";
    const features = report.top_drifted || [];
    const maxPsi = Number(report.max_psi) || 0;
    const scale = Math.max(0.3, maxPsi, 0.05);
    const featureRows = features
      .map((feature) => {
        const psi = Number(feature.psi) || 0;
        return `
          <div class="drift-feature">
            <span title="${esc(feature.feature)}">${esc(feature.feature)}</span>
            <div class="bar-track"><div class="bar-fill ${esc(feature.status)}" style="width:${Math.min(100, (psi / scale) * 100)}%"></div></div>
            <span class="psi">${fmtNum(psi, 3)}</span>
          </div>`;
      })
      .join("");
    return `
      <div>
        <div class="drift-banner ${esc(status)}">
          <strong>${esc(key)} · ${esc(status.replace("_", " "))}</strong><br>
          ${fmtInt(report.window)} requests · max PSI ${report.max_psi === null ? "—" : fmtNum(report.max_psi, 3)} · watch ${fmtNum(report.thresholds?.watch, 2)} / alert ${fmtNum(report.thresholds?.alert, 2)}
        </div>
        ${featureRows || '<p class="muted small">Not enough logged requests to compute PSI. Score at least 25 records with this model.</p>'}
      </div>`;
  });
  container.innerHTML = blocks.join("") || '<p class="muted">No drift reports.</p>';
}

function renderTraffic(traffic) {
  const container = $("#traffic");
  const rows = (traffic.by_endpoint || [])
    .map(
      (row) => `
      <div class="traffic-row">
        <span>${esc(row.endpoint)}</span>
        <span class="mono">${fmtInt(row.count)} · ${fmtNum(row.avg_latency_ms, 1)} ms</span>
      </div>`
    )
    .join("");
  container.innerHTML = `
    <div class="traffic-row"><span>Total predictions</span><span class="mono">${fmtInt(traffic.total)}</span></div>
    ${rows || '<p class="muted small">No predictions logged in this window.</p>'}`;
}

function renderFeedback(feedback) {
  const container = $("#feedback-summary");
  const blocks = [`<div class="traffic-row"><span>Labelled predictions</span><span class="mono">${fmtInt(feedback.labeled_events)}</span></div>`];
  if (feedback.maintenance) {
    blocks.push(
      `<div class="traffic-row"><span>Maintenance accuracy @ 0.5</span><span class="mono">${fmtPct(feedback.maintenance.accuracy_at_0_5)}</span></div>`,
      `<div class="traffic-row"><span>Observed rate vs mean score</span><span class="mono">${fmtPct(feedback.maintenance.observed_rate)} / ${fmtPct(feedback.maintenance.mean_probability)}</span></div>`
    );
  }
  if (feedback.freight) {
    blocks.push(
      `<div class="traffic-row"><span>Freight feedback MAE</span><span class="mono">${fmtMoney(feedback.freight.mae)}</span></div>`,
      `<div class="traffic-row"><span>Freight feedback MAPE</span><span class="mono">${fmtNum(feedback.freight.mape, 2)}%</span></div>`
    );
  }
  if (!feedback.maintenance && !feedback.freight) {
    blocks.push('<p class="muted small">Attach an observed outcome to a prediction to populate this panel.</p>');
  }
  container.innerHTML = blocks.join("");
}

function renderEvents(events) {
  const body = $("#recent-events tbody");
  if (!events.length) {
    body.innerHTML = '<tr><td colspan="7" class="loading-row">No predictions logged yet.</td></tr>';
    return;
  }
  body.innerHTML = events
    .map((event) => {
      const score = event.risk_probability !== null && event.risk_probability !== undefined
        ? `${fmtPct(event.risk_probability)} risk`
        : event.predicted_cost !== null && event.predicted_cost !== undefined
        ? fmtMoney(event.predicted_cost)
        : "—";
      const labelled = event.actual_value !== null && event.actual_value !== undefined;
      const feedbackCell = labelled
        ? `<span class="observed">${fmtNum(event.actual_value, 2)}</span>`
        : `<input class="feedback-input" data-event="${esc(event.id)}" type="number" step="any" placeholder="actual" style="width:82px;padding:4px 6px;font-size:11px" />
           <button class="button ghost small feedback-save" data-event="${esc(event.id)}">Save</button>`;
      return `
        <tr>
          <td class="mono" title="${esc(event.created_at)}">${esc(timeAgo(event.created_at))}</td>
          <td>${esc(event.endpoint)}</td>
          <td>${esc(event.model_name)}<br><span class="mono">v${esc(event.model_version)}</span></td>
          <td class="num">${fmtNum(event.latency_ms, 1)} ms</td>
          <td class="num">${score}</td>
          <td class="observed">${labelled ? fmtNum(event.actual_value, 2) : "—"}</td>
          <td>${feedbackCell}</td>
        </tr>`;
    })
    .join("");
}

async function loadMonitoring() {
  renderModelCards();
  const model = $("#drift-model").value;
  const windowValue = $("#drift-window").value;
  try {
    const [drift, traffic, feedback, events] = await Promise.all([
      api(`/api/monitoring/drift?model=${encodeURIComponent(model)}&window=${encodeURIComponent(windowValue)}`),
      api("/api/monitoring/traffic?hours=24"),
      api("/api/monitoring/feedback"),
      api("/api/monitoring/events?limit=20"),
    ]);
    renderDrift(drift);
    renderTraffic(traffic);
    renderFeedback(feedback);
    renderEvents(events);
  } catch (error) {
    toast(`Monitoring failed: ${error.message}`, true);
  }
}

function setupMonitoring() {
  ["drift-model", "drift-window"].forEach((id) => $(`#${id}`).addEventListener("change", loadMonitoring));
  $("#refresh-monitoring").addEventListener("click", loadMonitoring);
  $("#recent-events").addEventListener("click", async (event) => {
    const button = event.target.closest(".feedback-save");
    if (!button) return;
    const id = button.dataset.event;
    const input = $(`.feedback-input[data-event="${id}"]`);
    const actual = Number(input?.value);
    if (!Number.isFinite(actual)) {
      toast("Enter the observed value before saving.", true);
      return;
    }
    button.disabled = true;
    try {
      await api(`/api/feedback/${id}`, {
        method: "POST",
        body: JSON.stringify({ actual_value: actual }),
      });
      toast("Feedback recorded.");
      await loadMonitoring();
    } catch (error) {
      toast(`Feedback failed: ${error.message}`, true);
      button.disabled = false;
    }
  });
}

/* ---------- Boot ---------- */

async function refreshAll() {
  const button = $("#refresh-all");
  button.classList.add("is-spinning");
  try {
    await refreshStatus();
    await Promise.all([loadOptions(), loadOverview(), loadModels()]);
    if (state.activeTab === "priority") await loadPriority();
    if (state.activeTab === "monitoring") await loadMonitoring();
    toast("Dashboard refreshed.");
  } finally {
    button.classList.remove("is-spinning");
  }
}

async function boot() {
  setupNavigation();
  setupMaintenance();
  setupFreight();
  setupPriority();
  setupDialog();
  setupMonitoring();
  $("#refresh-all").addEventListener("click", refreshAll);

  await refreshStatus();
  await Promise.all([loadOptions(), loadOverview(), loadModels()]);
  activateTab(location.hash.replace("#", "") || "overview");
}

document.addEventListener("DOMContentLoaded", boot);
