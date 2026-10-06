/**
 * AETHERIX BATTERY INTELLIGENCE ALGORITHM (ABIA)
 * Dashboard Client Controller & Reactive Visualizer
 */

let sohChartInstance = null;
let fingerprintChartInstance = null;
let telemetryChartInstance = null;
let indicesBarChartInstance = null;

document.addEventListener("DOMContentLoaded", async () => {
  await loadBatteryList();
  await loadBenchmarks();

  document.getElementById("batterySelector").addEventListener("change", (e) => {
    if (e.target.value) {
      loadBatteryData(e.target.value);
    }
  });

  document.getElementById("refreshBtn").addEventListener("click", () => {
    const sel = document.getElementById("batterySelector").value;
    if (sel) loadBatteryData(sel);
  });

  document.getElementById("exportBtn").addEventListener("click", () => {
    const sel = document.getElementById("batterySelector").value;
    if (sel) {
      window.open(`/api/battery/${sel}`, "_blank");
    }
  });
});

async function loadBatteryList() {
  try {
    const res = await fetch("/api/batteries");
    const batteries = await res.json();
    const selector = document.getElementById("batterySelector");
    selector.innerHTML = "";

    batteries.forEach((b, idx) => {
      const opt = document.createElement("option");
      opt.value = b.battery_id;
      opt.textContent = `${b.battery_id} — ${b.decision} (SOH: ${b.current_soh}%)`;
      selector.appendChild(opt);
    });

    if (batteries.length > 0) {
      selector.value = batteries[0].battery_id;
      await loadBatteryData(batteries[0].battery_id);
    }
  } catch (err) {
    console.error("Failed to load battery list:", err);
  }
}

async function loadBatteryData(batteryId) {
  try {
    const res = await fetch(`/api/battery/${batteryId}`);
    const data = await res.json();
    updateKPIs(data);
    renderCharts(data);
    updateExplainability(data);
  } catch (err) {
    console.error(`Failed to load data for ${batteryId}:`, err);
  }
}

function updateKPIs(data) {
  // Decision Badge
  const badge = document.getElementById("decisionBadge");
  const subtext = document.getElementById("decisionSubtext");
  const overridePill = document.getElementById("overridePill");

  badge.textContent = data.final_circularity_decision;
  badge.className = "decision-badge " + getDecisionClass(data.final_circularity_decision);
  subtext.textContent = `CSS: ${data.circularity_suitability_score}/100 • Risk: ${data.anomaly_risk}`;

  if (data.safety_override_triggered) {
    overridePill.style.display = "block";
  } else {
    overridePill.style.display = "none";
  }

  // KPIs
  document.getElementById("kpiCurrentSOH").textContent = `${data.current_soh}%`;
  document.getElementById("kpiFutureSOH").textContent = `${data.future_soh}%`;
  document.getElementById("sohBarFill").style.width = `${Math.min(100, Math.max(0, data.current_soh))}%`;

  document.getElementById("kpiABHI").textContent = `${data.abhi}`;
  document.getElementById("kpiDegRate").textContent = `${data.degradation_rate_pct_per_100_cycles}%`;

  document.getElementById("kpiMomentum").textContent = data.health_momentum;
  document.getElementById("kpiConfidence").textContent = `${data.confidence_pct}%`;

  document.getElementById("kpiRisk").textContent = data.anomaly_risk;
  document.getElementById("kpiCSS").textContent = data.circularity_suitability_score;
  document.getElementById("riskMeterFill").style.width = `${Math.min(100, Math.max(0, data.risk_score))}%`;
}

function getDecisionClass(decision) {
  switch (decision) {
    case "CONTINUE USING": return "decision-continue";
    case "NEEDS MONITORING": return "decision-monitoring";
    case "SECOND-LIFE SUITABLE": return "decision-second-life";
    case "REPAIR / SERVICE": return "decision-repair";
    case "RECYCLE": return "decision-recycle";
    case "HIGH-RISK / QUARANTINE": return "decision-quarantine";
    default: return "";
  }
}

function renderCharts(data) {
  const timeline = data.telemetry_timeline || [];

  // 1. SOH Trajectory Chart
  const ctxSoh = document.getElementById("sohChart").getContext("2d");
  if (sohChartInstance) sohChartInstance.destroy();

  const labels = timeline.map(p => `Cyc ${p.cycle}`);
  const sohValues = timeline.map(p => p.soh);

  // Append future forecast point
  const extLabels = [...labels, `Cyc ${timeline[timeline.length - 1]?.cycle + 10} (Forecast)`];
  const extSohValues = [...sohValues, null];
  const forecastSeries = new Array(sohValues.length).fill(null);
  forecastSeries[sohValues.length - 1] = sohValues[sohValues.length - 1];
  forecastSeries.push(data.future_soh);

  sohChartInstance = new Chart(ctxSoh, {
    type: "line",
    data: {
      labels: extLabels,
      datasets: [
        {
          label: "Historical SOH (%)",
          data: extSohValues,
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.1)",
          borderWidth: 2.5,
          tension: 0.3,
          fill: true,
          pointRadius: 1,
        },
        {
          label: "Predicted Future Horizon (%)",
          data: forecastSeries,
          borderColor: "#f59e0b",
          borderDash: [5, 5],
          backgroundColor: "transparent",
          borderWidth: 3,
          pointRadius: 4,
          pointBackgroundColor: "#f59e0b",
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8" } }
      },
      scales: {
        x: { ticks: { color: "#64748b", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { min: 50, max: 105, ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } }
      }
    }
  });

  // 2. Radar Fingerprint Chart
  const ctxFp = document.getElementById("fingerprintChart").getContext("2d");
  if (fingerprintChartInstance) fingerprintChartInstance.destroy();

  const fp = data.behaviour_fingerprint || {};
  const fpLabels = Object.keys(fp).map(k => k.replace(/_/g, " ").toUpperCase());
  const fpValues = Object.values(fp);

  fingerprintChartInstance = new Chart(ctxFp, {
    type: "radar",
    data: {
      labels: fpLabels,
      datasets: [{
        label: "Module Behaviour",
        data: fpValues,
        borderColor: "#10b981",
        backgroundColor: "rgba(16, 185, 129, 0.2)",
        borderWidth: 2,
        pointBackgroundColor: "#10b981",
        pointRadius: 3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: { display: false },
          grid: { color: "rgba(255,255,255,0.08)" },
          angleLines: { color: "rgba(255,255,255,0.08)" },
          pointLabels: { color: "#94a3b8", font: { size: 10 } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  // 3. Telemetry Dynamics (Voltage, Current, Temp)
  const ctxTelem = document.getElementById("telemetryChart").getContext("2d");
  if (telemetryChartInstance) telemetryChartInstance.destroy();

  telemetryChartInstance = new Chart(ctxTelem, {
    type: "line",
    data: {
      labels: timeline.map(p => `Cyc ${p.cycle}`),
      datasets: [
        {
          label: "Voltage (V)",
          data: timeline.map(p => p.voltage),
          borderColor: "#3b82f6",
          borderWidth: 1.5,
          yAxisID: "yV",
          tension: 0.2,
          pointRadius: 0,
        },
        {
          label: "Temperature (°C)",
          data: timeline.map(p => p.temperature),
          borderColor: "#ef4444",
          borderWidth: 1.5,
          yAxisID: "yT",
          tension: 0.2,
          pointRadius: 0,
        },
        {
          label: "Current (A)",
          data: timeline.map(p => p.current),
          borderColor: "#8b5cf6",
          borderWidth: 1.2,
          yAxisID: "yI",
          borderDash: [3, 3],
          pointRadius: 0,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8" } }
      },
      scales: {
        x: { ticks: { color: "#64748b", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.04)" } },
        yV: { position: "left", min: 2.5, max: 4.5, ticks: { color: "#3b82f6" }, grid: { color: "rgba(255,255,255,0.04)" } },
        yT: { position: "right", min: 10, max: 70, ticks: { color: "#ef4444" }, grid: { display: false } },
        yI: { display: false }
      }
    }
  });

  // 4. Health Indices Bar Chart
  const ctxBar = document.getElementById("indicesBarChart").getContext("2d");
  if (indicesBarChartInstance) indicesBarChartInstance.destroy();

  indicesBarChartInstance = new Chart(ctxBar, {
    type: "bar",
    data: {
      labels: ["VSI", "TSI", "CSI", "PSI", "DGI"],
      datasets: [{
        label: "Index Score (0-100)",
        data: [data.vsi, data.tsi, data.csi, data.psi, data.dgi],
        backgroundColor: [
          "#3b82f6",
          data.tsi < 60 ? "#ef4444" : "#10b981",
          "#8b5cf6",
          "#06b6d4",
          "#f59e0b"
        ],
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
        y: { min: 0, max: 100, ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } }
      }
    }
  });
}

function updateExplainability(data) {
  // Decision reasons
  const reasonsList = document.getElementById("decisionReasonsList");
  reasonsList.innerHTML = "";
  (data.decision_explanation || []).forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    reasonsList.appendChild(li);
  });

  // Risk drivers
  const driversList = document.getElementById("riskDriversList");
  driversList.innerHTML = "";
  (data.primary_risk_drivers || []).forEach(d => {
    const li = document.createElement("li");
    li.textContent = d;
    driversList.appendChild(li);
  });

  // Safety status box
  const safetyTitle = document.getElementById("safetyStatusTitle");
  const safetyDesc = document.getElementById("safetyStatusDesc");
  const box = document.getElementById("safetyStatusBox");

  if (data.safety_override_triggered) {
    box.style.borderColor = "#ef4444";
    box.style.background = "rgba(239, 68, 68, 0.15)";
    safetyTitle.style.color = "#ef4444";
    safetyTitle.textContent = "CRITICAL SAFETY OVERRIDE ACTIVE";
    safetyDesc.textContent = "Safety thresholds breached. Automated quarantine enforced.";
  } else {
    box.style.borderColor = "rgba(16, 185, 129, 0.4)";
    box.style.background = "rgba(16, 185, 129, 0.08)";
    safetyTitle.style.color = "#10b981";
    safetyTitle.textContent = "SAFETY STATUS: NOMINAL";
    safetyDesc.textContent = "Zero critical threshold violations detected across all sensors.";
  }
}

async function loadBenchmarks() {
  try {
    const res = await fetch("/api/benchmarks");
    const benchmarks = await res.json();
    const tbody = document.getElementById("benchmarkTableBody");
    tbody.innerHTML = "";

    // Add baseline models
    const baselines = benchmarks.baselines || {};
    for (const [name, stats] of Object.entries(baselines)) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong>${name.replace(/_/g, " ")}</strong></td>
        <td>${stats.test.MAE.toFixed(3)}</td>
        <td>${stats.test.RMSE.toFixed(3)}</td>
        <td>${stats.test.R2.toFixed(3)}</td>
        <td>${stats.test.Max_Error.toFixed(3)}</td>
        <td style="color: ${stats.overfitting_detected ? '#f59e0b' : '#10b981'}">${stats.generalization_diagnosis}</td>
      `;
      tbody.appendChild(row);
    }

    // Add ABIA Predictor (Highlighted)
    if (benchmarks.abia_predictor) {
      const abia = benchmarks.abia_predictor;
      const abiaRow = document.createElement("tr");
      abiaRow.className = "highlight-abia";
      abiaRow.innerHTML = `
        <td>⚡ <strong>ABIA Framework (Custom)</strong></td>
        <td><strong>${abia.test.MAE.toFixed(3)}</strong></td>
        <td><strong>${abia.test.RMSE.toFixed(3)}</strong></td>
        <td><strong>${abia.test.R2.toFixed(3)}</strong></td>
        <td><strong>${abia.test.Max_Error.toFixed(3)}</strong></td>
        <td style="color: #10b981"><strong>${abia.generalization_diagnosis}</strong></td>
      `;
      tbody.appendChild(abiaRow);
    }
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}
