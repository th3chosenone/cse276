/**
 * Last-Mile Delivery Route Optimiser — Frontend Application Logic
 * Replicates the complete Streamlit UI & workflow of app.py
 */

const API_BASE = ''; // Same-origin on Vercel

// ── Application State ────────────────────────────────────────────────────────
const state = {
  mapSource: 'procedural', // 'procedural' or 'upload'
  mapImageB64: null,
  labeledImageB64: null,
  roadMaskB64: null,
  seed: 42,
  depot: null,
  allDepotsCount: 0,
  nodes: [], // all detected nodes (depot + houses)
  selectedHouseIds: [],
  solveResults: null, // response from /api/optimize
  lastRunNodes: [],
  customStops: [],
  selectedViewAlgoKey: null,
};

// ── Algorithm Options ────────────────────────────────────────────────────────
const ALGO_OPTIONS = {
  nn: 'Nearest Neighbour',
  greedy: 'Cheapest Insertion',
  '2opt': '2-opt Local Search',
  hill: 'Hill Climbing',
  sa: 'Simulated Annealing',
  exact: 'Exact Brute-Force / Held-Karp (≤ 12 stops)',
};

// ── Formatting Helpers ───────────────────────────────────────────────────────
function formatMinutesToTime(minutes) {
  const mins = (Math.round(minutes) % 1440 + 1440) % 1440;
  const hr = Math.floor(mins / 60);
  const mn = mins % 60;
  return `${String(hr).padStart(2, '0')}:${String(mn).padStart(2, '0')}`;
}

function showLoading(text = 'Running optimization...') {
  const el = document.getElementById('loadingOverlay');
  document.getElementById('loadingText').textContent = text;
  el.classList.add('active');
}

function hideLoading() {
  document.getElementById('loadingOverlay').classList.remove('active');
}

function getSelectedAlgos() {
  const checked = document.querySelectorAll('.algo-pill input:checked');
  return Array.from(checked).map((cb) => cb.value);
}

// ── 1. Map Management & API Calls ────────────────────────────────────────────

async function generateProceduralMap(isNew = false) {
  const nHouses = parseInt(document.getElementById('nHousesSlider').value, 10);
  const seedInput = document.getElementById('seedInput');

  let chosenSeed = null;
  if (!isNew && seedInput.value.trim() !== '') {
    chosenSeed = parseInt(seedInput.value.trim(), 10);
  }

  showLoading(isNew ? 'Generating fresh procedural city map...' : 'Regenerating map with seed...');
  try {
    const res = await fetch(`${API_BASE}/api/generate-map?t=${Date.now()}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_houses: nHouses, seed: chosenSeed }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to generate map');
    }

    const data = await res.json();
    state.mapSource = 'procedural';
    state.mapImageB64 = data.map_image;
    state.labeledImageB64 = data.labeled_image;
    state.roadMaskB64 = data.road_mask;
    state.seed = data.seed;
    state.depot = data.depot;
    state.allDepotsCount = data.all_depots_count;
    seedInput.value = data.seed;

    // Build unified nodes array (Depot at index 0, followed by houses)
    const depotNode = {
      id: 'DEPOT',
      cx: data.depot ? data.depot.cx : 0,
      cy: data.depot ? data.depot.cy : 0,
      is_depot: true,
      time_window: [0, 1440],
      service_time: 0,
      address: 'Central Depot Hub',
    };
    state.nodes = [depotNode, ...data.houses];
    state.selectedHouseIds = data.houses.map((h) => h.id);
    state.solveResults = null;

    renderVerification();
    renderMetricsRow();
    renderStopSelector();
    renderStopEditorTable();
    renderMapTab();
    resetResultsTabs();
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    hideLoading();
  }
}

async function handleMapUpload(file) {
  if (!file) return;

  const roadType = document.getElementById('roadContrastSelect').value;
  const roadThresh = parseInt(document.getElementById('roadThreshSlider').value, 10);

  const formData = new FormData();
  formData.append('file', file);
  formData.append('road_type', roadType);
  formData.append('road_thresh', roadThresh);

  showLoading('Analyzing uploaded map & detecting delivery markers...');
  try {
    const res = await fetch(`${API_BASE}/api/upload-map?t=${Date.now()}`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to upload map');
    }

    const data = await res.json();
    state.mapSource = 'upload';
    state.mapImageB64 = data.map_image;
    state.labeledImageB64 = data.labeled_image;
    state.roadMaskB64 = data.road_mask;
    state.seed = null;
    state.depot = data.depot;
    state.allDepotsCount = data.all_depots_count;

    const depotNode = {
      id: 'DEPOT',
      cx: data.depot ? data.depot.cx : 0,
      cy: data.depot ? data.depot.cy : 0,
      is_depot: true,
      time_window: [0, 1440],
      service_time: 0,
      address: 'Central Depot Hub',
    };
    state.nodes = [depotNode, ...data.houses];
    state.selectedHouseIds = data.houses.map((h) => h.id);
    state.solveResults = null;

    renderVerification();
    renderMetricsRow();
    renderStopSelector();
    renderStopEditorTable();
    renderMapTab();
    resetResultsTabs();
  } catch (err) {
    alert(`Error uploading map: ${err.message}`);
  } finally {
    hideLoading();
  }
}

// ── 2. UI Renderers ──────────────────────────────────────────────────────────

function renderVerification() {
  const depotBox = document.getElementById('depotVerifyBox');
  const houseBox = document.getElementById('houseVerifyBox');

  if (state.allDepotsCount === 1) {
    depotBox.className = 'alert-box alert-success';
    depotBox.innerHTML = `🟢 <b>Verified Red Depot</b>: 1 Hub at <code>(${state.depot.cx}, ${state.depot.cy})</code>`;
  } else if (state.allDepotsCount > 1) {
    depotBox.className = 'alert-box alert-warning';
    depotBox.innerHTML = `⚠️ <b>Multiple Red Blobs</b>: ${state.allDepotsCount} found. Primary Hub selected by largest area at <code>(${state.depot.cx}, ${state.depot.cy})</code>.`;
  } else {
    depotBox.className = 'alert-box alert-error';
    depotBox.innerHTML = `❌ <b>No Red Depot Detected!</b> Upload an image with a red depot or click to place one.`;
  }

  const houseCount = state.nodes.filter((n) => !n.is_depot).length;
  houseBox.className = 'alert-box alert-info';
  houseBox.innerHTML = `🔵 <b>Delivery Houses Detected</b>: ${houseCount} stops`;
}

function renderMetricsRow() {
  const depotEl = document.getElementById('metricDepot');
  const stopsEl = document.getElementById('metricStops');
  const engineEl = document.getElementById('metricEngine');
  const statusEl = document.getElementById('metricStatus');

  const depot = state.nodes.find((n) => n.is_depot);
  depotEl.textContent = depot ? `Hub at (${depot.cx}, ${depot.cy})` : 'None';

  const houses = state.nodes.filter((n) => !n.is_depot);
  stopsEl.textContent = `${houses.length} Stops`;

  const scale = document.getElementById('gridScaleSelect').value;
  engineEl.textContent = `A* 8-Way (Scale: ${scale}x)`;

  statusEl.textContent = state.solveResults ? 'Optimized ✓' : 'Ready to Optimize';
}

function renderStopSelector() {
  const container = document.getElementById('stopChipsContainer');
  container.innerHTML = '';

  const houses = state.nodes.filter((n) => !n.is_depot);
  houses.forEach((h) => {
    const isChecked = state.selectedHouseIds.includes(h.id);
    const chip = document.createElement('label');
    chip.className = `stop-chip ${isChecked ? 'checked' : ''}`;
    chip.innerHTML = `
      <input type="checkbox" value="${h.id}" ${isChecked ? 'checked' : ''}>
      ${h.id} (${h.address || 'Stop'} | Window: ${formatMinutesToTime(h.time_window[0])} - ${formatMinutesToTime(h.time_window[1])})
    `;

    chip.querySelector('input').addEventListener('change', (e) => {
      if (e.target.checked) {
        if (!state.selectedHouseIds.includes(h.id)) state.selectedHouseIds.push(h.id);
        chip.classList.add('checked');
      } else {
        state.selectedHouseIds = state.selectedHouseIds.filter((id) => id !== h.id);
        chip.classList.remove('checked');
      }
    });

    container.appendChild(chip);
  });
}

function renderStopEditorTable() {
  const tbody = document.getElementById('stopEditorBody');
  tbody.innerHTML = '';

  const houses = state.nodes.filter((n) => !n.is_depot);
  houses.forEach((h) => {
    const tr = document.createElement('tr');
    tr.dataset.id = h.id;
    tr.innerHTML = `
      <td><b>${h.id}</b></td>
      <td><input type="text" class="edit-addr" value="${h.address || `Delivery Location ${h.id}`}" style="width:180px"></td>
      <td><input type="number" class="edit-x" value="${h.cx}" style="width:70px" min="0"></td>
      <td><input type="number" class="edit-y" value="${h.cy}" style="width:70px" min="0"></td>
      <td><input type="number" class="edit-early" value="${h.time_window[0]}" style="width:75px" min="0" max="1440"></td>
      <td><input type="number" class="edit-late" value="${h.time_window[1]}" style="width:75px" min="0" max="1440"></td>
      <td><input type="number" class="edit-svc" value="${h.service_time}" style="width:65px" min="1" max="120"></td>
    `;
    tbody.appendChild(tr);
  });
}

function applyUpdatedTimeWindows() {
  const tbody = document.getElementById('stopEditorBody');
  const rows = tbody.querySelectorAll('tr');

  rows.forEach((row) => {
    const id = row.dataset.id;
    const node = state.nodes.find((n) => n.id === id);
    if (!node) return;

    let eMin = parseInt(row.querySelector('.edit-early').value, 10);
    let lMin = parseInt(row.querySelector('.edit-late').value, 10);
    if (eMin > lMin) {
      alert(`Warning: ${id} earliest (${eMin}) > latest (${lMin}). Auto-swapped.`);
      const tmp = eMin;
      eMin = lMin;
      lMin = tmp;
    }

    node.address = row.querySelector('.edit-addr').value;
    node.cx = parseInt(row.querySelector('.edit-x').value, 10);
    node.cy = parseInt(row.querySelector('.edit-y').value, 10);
    node.time_window = [eMin, lMin];
    node.service_time = parseInt(row.querySelector('.edit-svc').value, 10);
  });

  state.solveResults = null;
  renderStopSelector();
  alert('Time windows and service times updated successfully!');
}

function addCustomStop() {
  const idInput = document.getElementById('newStopId');
  const addrInput = document.getElementById('newStopAddr');
  const xInput = document.getElementById('newStopX');
  const yInput = document.getElementById('newStopY');
  const earlyInput = document.getElementById('newStopEarly');
  const lateInput = document.getElementById('newStopLate');
  const svcInput = document.getElementById('newStopSvc');

  const id = idInput.value.trim() || `H${state.nodes.length}`;
  let early = parseInt(earlyInput.value, 10);
  let late = parseInt(lateInput.value, 10);
  if (early > late) {
    const tmp = early;
    early = late;
    late = tmp;
  }

  const newNode = {
    id,
    cx: parseInt(xInput.value, 10) || 300,
    cy: parseInt(yInput.value, 10) || 300,
    is_depot: false,
    area: 500.0,
    time_window: [early, late],
    service_time: parseInt(svcInput.value, 10) || 5,
    address: addrInput.value.trim() || `${state.nodes.length * 12 + 10} Market St`,
  };

  state.nodes.push(newNode);
  if (!state.selectedHouseIds.includes(id)) {
    state.selectedHouseIds.push(id);
  }

  state.solveResults = null;
  renderVerification();
  renderMetricsRow();
  renderStopSelector();
  renderStopEditorTable();
  alert(`Added stop '${id}' at (${newNode.cx}, ${newNode.cy}) with window ${formatMinutesToTime(early)} - ${formatMinutesToTime(late)}`);
}

function handleCsvImport(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const text = e.target.result;
      const lines = text.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
      if (lines.length < 2) throw new Error('CSV is empty or missing headers');

      const headers = lines[0].split(',').map((h) => h.trim().toLowerCase());
      const idIdx = headers.indexOf('id');
      const xIdx = headers.indexOf('x');
      const yIdx = headers.indexOf('y');
      const earlyIdx = headers.indexOf('earliest');
      const lateIdx = headers.indexOf('latest');
      const svcIdx = headers.indexOf('service_time');
      const addrIdx = headers.indexOf('address');

      let imported = 0;
      for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',').map((c) => c.trim());
        if (cols.length < 3) continue;

        const sid = idIdx >= 0 && cols[idIdx] ? cols[idIdx] : `H${state.nodes.length + 1}`;
        const sx = xIdx >= 0 ? parseInt(cols[xIdx], 10) : 200;
        const sy = yIdx >= 0 ? parseInt(cols[yIdx], 10) : 200;
        const early = earlyIdx >= 0 ? parseInt(cols[earlyIdx], 10) : 540;
        const late = lateIdx >= 0 ? parseInt(cols[lateIdx], 10) : 720;
        const svc = svcIdx >= 0 ? parseInt(cols[svcIdx], 10) : 5;
        const addr = addrIdx >= 0 ? cols[addrIdx] : sid;

        const node = {
          id: sid,
          cx: sx,
          cy: sy,
          is_depot: false,
          area: 500.0,
          time_window: [early, late],
          service_time: svc,
          address: addr,
        };
        state.nodes.push(node);
        if (!state.selectedHouseIds.includes(sid)) state.selectedHouseIds.push(sid);
        imported++;
      }

      state.solveResults = null;
      renderVerification();
      renderMetricsRow();
      renderStopSelector();
      renderStopEditorTable();
      alert(`Imported ${imported} stops from CSV!`);
    } catch (err) {
      alert(`Error importing CSV: ${err.message}`);
    }
  };
  reader.readAsText(file);
}

// ── 3. Tab Rendering ─────────────────────────────────────────────────────────

function renderMapTab() {
  const container = document.getElementById('mapDisplayContainer');
  const algoSelectWrap = document.getElementById('algoSelectWrap');
  const algoSelect = document.getElementById('viewAlgoSelect');
  const btnDownloadRoute = document.getElementById('btnDownloadRouteMap');
  const maskPreviewWrap = document.getElementById('maskPreviewWrap');
  const maskPreviewImg = document.getElementById('maskPreviewImg');

  const showMask = document.getElementById('showMaskCheckbox').checked;
  if (showMask && state.roadMaskB64) {
    maskPreviewWrap.style.display = 'block';
    maskPreviewImg.src = `data:image/png;base64,${state.roadMaskB64}`;
  } else {
    maskPreviewWrap.style.display = 'none';
  }

  if (state.solveResults && state.solveResults.route_images) {
    algoSelectWrap.style.display = 'block';
    btnDownloadRoute.style.display = 'inline-flex';

    // Populate algo dropdown if not already populated
    const keys = Object.keys(state.solveResults.route_images);
    algoSelect.innerHTML = '';
    keys.forEach((k) => {
      const opt = document.createElement('option');
      opt.value = k;
      const bRow = state.solveResults.benchmark_table.find((b) => b.key === k);
      const dist = bRow ? bRow.road_distance_px : 0;
      const score = bRow ? bRow.score : 0;
      opt.textContent = `${ALGO_OPTIONS[k] || k} (Road Dist: ${dist}px | Score: ${score})`;
      algoSelect.appendChild(opt);
    });

    if (!state.selectedViewAlgoKey || !keys.includes(state.selectedViewAlgoKey)) {
      state.selectedViewAlgoKey = state.solveResults.best_algo_key || keys[0];
    }
    algoSelect.value = state.selectedViewAlgoKey;

    const b64 = state.solveResults.route_images[state.selectedViewAlgoKey];
    container.innerHTML = `
      <img src="data:image/png;base64,${b64}" alt="Optimized Road Route">
      <p class="map-caption">Optimized Road Route: ${ALGO_OPTIONS[state.selectedViewAlgoKey] || state.selectedViewAlgoKey} (Starts & Ends at Verified Red Depot via A*)</p>
    `;
  } else if (state.labeledImageB64) {
    algoSelectWrap.style.display = 'none';
    btnDownloadRoute.style.display = 'none';
    container.innerHTML = `
      <img src="data:image/png;base64,${state.labeledImageB64}" alt="Detected Map Markers">
      <p class="map-caption">Detected Primary Red Depot (Index 0) and Blue Delivery Houses with Time Windows</p>
    `;
  }
}

function renderBenchmarkTab() {
  const container = document.getElementById('benchmarkTableBody');
  const bestCard = document.getElementById('bestHeuristicCard');
  const btnCsv = document.getElementById('btnExportBenchmarkCsv');

  if (!state.solveResults) {
    container.innerHTML = '<tr><td colspan="13" style="text-align:center;padding:24px;color:#777;">Click <b>"🚀 Optimize Delivery Route"</b> above to run and view benchmark metrics across all algorithms.</td></tr>';
    bestCard.style.display = 'none';
    btnCsv.style.display = 'none';
    return;
  }

  btnCsv.style.display = 'inline-flex';
  container.innerHTML = '';

  const tableData = state.solveResults.benchmark_table;
  tableData.forEach((r) => {
    const tr = document.createElement('tr');
    if (r.is_best_heuristic) tr.className = 'row-best-heuristic';
    else if (r.is_exact) tr.className = 'row-exact-ref';

    tr.innerHTML = `
      <td>${r.is_best_heuristic ? '🏆 ' : ''}<b>${r.algorithm}</b></td>
      <td>${r.road_distance_px}</td>
      <td>${r.drive_time_min}</td>
      <td>${r.wait_time_min}</td>
      <td>${r.lateness_min}</td>
      <td>${r.total_elapsed_min}</td>
      <td><b>${r.score}</b></td>
      <td>${r.feasibility_str}</td>
      <td>${r.late_stops}</td>
      <td>${r.max_lateness_min}</td>
      <td>${r.cpu_time_ms}</td>
      <td>${r.evaluations}</td>
      <td>${r.gap_vs_exact}</td>
    `;
    container.appendChild(tr);
  });

  // Render Best Heuristic Card
  const bestKey = state.solveResults.best_algo_key;
  const bestR = tableData.find((t) => t.key === bestKey);
  if (bestR) {
    let gapInfo = '';
    const exactScore = state.solveResults.exact_score;
    if (exactScore && exactScore > 0 && typeof bestR.score === 'number') {
      const gapPct = (((bestR.score - exactScore) / exactScore) * 100).toFixed(2);
      gapInfo = `&nbsp;|&nbsp; <b>Optimality Gap vs Exact:</b> <span style="color:#00C853; font-weight:700;">${gapPct}%</span>`;
    }

    bestCard.style.display = 'block';
    bestCard.innerHTML = `
      <b>🏆 Best Heuristic Solution:</b> <span style="color:#1E88E5; font-size:1.1rem; font-weight:700;">${bestR.algorithm}</span><br>
      <b>Starting Hub:</b> Verified Red Depot (Index 0) &nbsp;|&nbsp;
      <b>Road Distance:</b> ${bestR.road_distance_px} px (${bestR.drive_time_min} min drive) &nbsp;|&nbsp;
      <b>Waiting Time:</b> ${bestR.wait_time_min} min &nbsp;|&nbsp;
      <b>Total Mission Duration:</b> ${bestR.total_elapsed_min} min &nbsp;|&nbsp;
      <b>Lateness:</b> ${bestR.lateness_min} min &nbsp;|&nbsp;
      <b>Feasible:</b> ${bestR.feasible ? '✅ Yes' : '❌ No'} &nbsp;|&nbsp;
      <b>Solve Runtime:</b> ${bestR.cpu_time_ms} ms
      ${gapInfo}
    `;
  }
}

function renderConvergenceTab() {
  const chartWrap = document.getElementById('convergenceChartWrap');
  const summaryBody = document.getElementById('convergenceSummaryBody');

  if (!state.solveResults || !state.solveResults.convergence_chart) {
    chartWrap.innerHTML = '<p style="text-align:center;color:#777;padding:30px;">Run optimization to generate convergence tracking curves.</p>';
    summaryBody.innerHTML = '';
    return;
  }

  chartWrap.innerHTML = `
    <img src="data:image/png;base64,${state.solveResults.convergence_chart}" style="max-width:100%;border-radius:4px;" alt="Convergence Curves">
    <p class="map-caption">Evaluation-Based Convergence Comparison (Shared Evaluations Axis)</p>
  `;

  summaryBody.innerHTML = '';
  state.solveResults.convergence_summary.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><b>${row.algorithm}</b></td>
      <td>${row.starting_score}</td>
      <td>${row.final_best_score}</td>
      <td>${row.improvement_pts} pts (${row.improvement_pct}%)</td>
      <td>${row.evaluations}</td>
    `;
    summaryBody.appendChild(tr);
  });
}

function renderManifestTab() {
  const container = document.getElementById('manifestTableBody');
  const manifestHeader = document.getElementById('manifestHeader');
  const btnCsv = document.getElementById('btnExportManifestCsv');

  if (!state.solveResults || !state.solveResults.itinerary || !state.solveResults.itinerary.length) {
    container.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:24px;color:#777;">Run optimization to generate turn-by-turn delivery schedule.</td></tr>';
    manifestHeader.textContent = '';
    btnCsv.style.display = 'none';
    return;
  }

  btnCsv.style.display = 'inline-flex';
  const bestKey = state.solveResults.best_algo_key;
  manifestHeader.innerHTML = `<b>Optimal Route Tour (${ALGO_OPTIONS[bestKey] || bestKey}):</b> Complete delivery itinerary`;

  container.innerHTML = '';
  state.solveResults.itinerary.forEach((leg) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${leg.leg}</td>
      <td><b>${leg.from_to}</b></td>
      <td>${leg.destination_address}</td>
      <td>${leg.road_distance_px} px</td>
      <td>${leg.travel_time_min} min</td>
      <td>${leg.est_arrival}</td>
      <td>${leg.customer_window}</td>
      <td>${leg.status_badge}</td>
    `;
    container.appendChild(tr);
  });
}

function resetResultsTabs() {
  renderBenchmarkTab();
  renderConvergenceTab();
  renderManifestTab();
  document.getElementById('multiSeedTableBody').innerHTML = '<tr><td colspan="11" style="text-align:center;padding:24px;color:#777;">Run multi-seed validation study to see statistical performance across seeds.</td></tr>';
  document.getElementById('btnExportMultiSeedCsv').style.display = 'none';
}

// ── 4. Route Optimization ───────────────────────────────────────────────────

async function runOptimization() {
  const depot = state.nodes.find((n) => n.is_depot);
  if (!depot) {
    alert('No verified Red Depot found! Cannot optimize without a starting hub.');
    return;
  }

  if (state.selectedHouseIds.length < 1) {
    alert('Please select at least 1 delivery house to form a route.');
    return;
  }

  const selectedAlgos = getSelectedAlgos();
  if (!selectedAlgos.length) {
    alert('Please select at least 1 algorithm in the sidebar.');
    return;
  }

  const payload = {
    map_image: state.mapImageB64,
    seed: state.mapSource === 'procedural' ? state.seed : null,
    nodes: state.nodes,
    selected_ids: state.selectedHouseIds,
    algorithms: selectedAlgos,
    start_time: parseFloat(document.getElementById('startTimeInput').value) || 540,
    depot_close: parseFloat(document.getElementById('depotCloseInput').value) || 1080,
    speed: parseFloat(document.getElementById('speedSlider').value) || 60,
    hard_windows: document.getElementById('hardTwCheckbox').checked,
    tw_weight: parseFloat(document.getElementById('twWeightSlider').value) || 1.5,
    grid_scale: parseInt(document.getElementById('gridScaleSelect').value, 10) || 2,
    opt_seed: parseInt(document.getElementById('optSeedInput').value, 10) || 42,
    road_type: document.getElementById('roadContrastSelect').value,
    road_thresh: parseInt(document.getElementById('roadThreshSlider').value, 10) || 130,
  };

  showLoading('Extracting road network & computing A* shortest paths strictly from Red Depot...');
  try {
    const res = await fetch(`${API_BASE}/api/optimize?t=${Date.now()}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Optimization failed');
    }

    state.solveResults = await res.json();
    state.selectedViewAlgoKey = state.solveResults.best_algo_key;

    renderMetricsRow();
    renderMapTab();
    renderBenchmarkTab();
    renderConvergenceTab();
    renderManifestTab();
  } catch (err) {
    alert(`Optimization error: ${err.message}`);
  } finally {
    hideLoading();
  }
}

// ── 5. Multi-Seed Study ──────────────────────────────────────────────────────

async function runMultiSeedStudy() {
  if (!state.solveResults) {
    alert('Run optimization on a delivery map first to initialize the delivery network.');
    return;
  }

  const nSeeds = parseInt(document.getElementById('nSeedsSlider').value, 10) || 15;
  const payload = {
    map_image: state.mapImageB64,
    seed: state.mapSource === 'procedural' ? state.seed : null,
    nodes: state.nodes,
    selected_ids: state.selectedHouseIds,
    n_seeds: nSeeds,
    start_time: parseFloat(document.getElementById('startTimeInput').value) || 540,
    depot_close: parseFloat(document.getElementById('depotCloseInput').value) || 1080,
    speed: parseFloat(document.getElementById('speedSlider').value) || 60,
    hard_windows: document.getElementById('hardTwCheckbox').checked,
    tw_weight: parseFloat(document.getElementById('twWeightSlider').value) || 1.5,
    grid_scale: parseInt(document.getElementById('gridScaleSelect').value, 10) || 2,
    road_type: document.getElementById('roadContrastSelect').value,
    road_thresh: parseInt(document.getElementById('roadThreshSlider').value, 10) || 130,
  };

  showLoading(`Benchmarking across ${nSeeds} random seeds...`);
  try {
    const res = await fetch(`${API_BASE}/api/multi-seed?t=${Date.now()}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Multi-seed benchmark failed');
    }

    const data = await res.json();
    const tbody = document.getElementById('multiSeedTableBody');
    tbody.innerHTML = '';

    data.summary_rows.forEach((s) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><b>${s.algorithm}</b></td>
        <td>${s.runs}</td>
        <td>${s.mean_score}</td>
        <td>±${s.std_score}</td>
        <td>${s.best_score}</td>
        <td>${s.worst_score}</td>
        <td>${s.feasibility_pct}%</td>
        <td>${s.mean_distance}</td>
        <td>${s.mean_lateness}</td>
        <td>${s.mean_cpu_ms}</td>
        <td>${s.mean_gap_pct}</td>
      `;
      tbody.appendChild(tr);
    });

    document.getElementById('btnExportMultiSeedCsv').style.display = 'inline-flex';
    state._lastMultiSeedRows = data.summary_rows;
  } catch (err) {
    alert(`Multi-seed error: ${err.message}`);
  } finally {
    hideLoading();
  }
}

// ── 6. CSV Download Utilities ────────────────────────────────────────────────

function downloadCsv(filename, headers, rows) {
  const content = [headers.join(','), ...rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(','))].join('\n');
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function exportBenchmarkCsv() {
  if (!state.solveResults || !state.solveResults.benchmark_table) return;
  const headers = ['Algorithm', 'Road Distance (px)', 'Drive Time (min)', 'Wait Time (min)', 'Lateness (min)', 'Total Elapsed Time (min)', 'Combined Score', 'Feasible?', 'Late Stops', 'Max Lateness (min)', 'CPU Time (ms)', 'Evaluations', 'Optimality Gap vs Exact'];
  const rows = state.solveResults.benchmark_table.map((r) => [
    r.algorithm, r.road_distance_px, r.drive_time_min, r.wait_time_min, r.lateness_min, r.total_elapsed_min, r.score, r.feasible ? 'Yes' : 'No', r.late_stops, r.max_lateness_min, r.cpu_time_ms, r.evaluations, r.gap_vs_exact
  ]);
  downloadCsv('benchmark_results.csv', headers, rows);
}

function exportManifestCsv() {
  if (!state.solveResults || !state.solveResults.itinerary) return;
  const headers = ['Leg', 'From -> To', 'Destination Address', 'Road Dist (px)', 'Travel Time (min)', 'Est. Arrival', 'Customer Window', 'Status'];
  const rows = state.solveResults.itinerary.map((leg) => [
    leg.leg, leg.from_to, leg.destination_address, leg.road_distance_px, leg.travel_time_min, leg.est_arrival, leg.customer_window, leg.status_plain
  ]);
  downloadCsv('delivery_manifest.csv', headers, rows);
}

function exportMultiSeedCsv() {
  if (!state._lastMultiSeedRows) return;
  const headers = ['Algorithm', 'Runs', 'Mean Score', 'Std Dev', 'Best Score', 'Worst Score', 'Feasibility %', 'Mean Road Dist (px)', 'Mean Lateness (min)', 'Mean CPU (ms)', 'Mean Gap vs Exact'];
  const rows = state._lastMultiSeedRows.map((s) => [
    s.algorithm, s.runs, s.mean_score, s.std_score, s.best_score, s.worst_score, `${s.feasibility_pct}%`, s.mean_distance, s.mean_lateness, s.mean_cpu_ms, s.mean_gap_pct
  ]);
  downloadCsv('multiseed_validation_study.csv', headers, rows);
}

function downloadRouteMapPng() {
  if (!state.solveResults || !state.selectedViewAlgoKey) return;
  const b64 = state.solveResults.route_images[state.selectedViewAlgoKey];
  if (!b64) return;
  const a = document.createElement('a');
  a.href = `data:image/png;base64,${b64}`;
  a.download = `optimized_road_delivery_route_${state.selectedViewAlgoKey}.png`;
  a.click();
}

// ── 7. DOM Event Listeners & Initialization ──────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Map source radio buttons
  const proceduralControls = document.getElementById('proceduralMapControls');
  const uploadControls = document.getElementById('uploadMapControls');
  document.querySelectorAll('input[name="mapSource"]').forEach((radio) => {
    radio.addEventListener('change', (e) => {
      if (e.target.value === 'procedural') {
        proceduralControls.style.display = 'block';
        uploadControls.style.display = 'none';
        generateProceduralMap(false);
      } else {
        proceduralControls.style.display = 'none';
        uploadControls.style.display = 'block';
      }
    });
  });

  // Slider badges
  const linkSlider = (sliderId, badgeId, suffix = '') => {
    const slider = document.getElementById(sliderId);
    const badge = document.getElementById(badgeId);
    if (!slider || !badge) return;
    slider.addEventListener('input', () => {
      badge.textContent = `${slider.value}${suffix}`;
    });
  };

  linkSlider('nHousesSlider', 'nHousesBadge');
  linkSlider('speedSlider', 'speedBadge', ' px/m');
  linkSlider('twWeightSlider', 'twWeightBadge');
  linkSlider('roadThreshSlider', 'roadThreshBadge');
  linkSlider('nSeedsSlider', 'nSeedsBadge');

  // Subheader time updates
  const startInput = document.getElementById('startTimeInput');
  const closeInput = document.getElementById('depotCloseInput');
  const updateSubHeaderTimes = () => {
    const st = formatMinutesToTime(parseFloat(startInput.value) || 540);
    const cl = formatMinutesToTime(parseFloat(closeInput.value) || 1080);
    document.getElementById('subHeaderTimes').innerHTML = `Start: <b>${st}</b> &nbsp;|&nbsp; Depot Close: <b>${cl}</b>`;
  };
  startInput.addEventListener('input', updateSubHeaderTimes);
  closeInput.addEventListener('input', updateSubHeaderTimes);

  // Procedural map buttons
  document.getElementById('btnNewMap').addEventListener('click', () => generateProceduralMap(true));
  document.getElementById('btnRegenerateMap').addEventListener('click', () => generateProceduralMap(false));

  // File upload input
  document.getElementById('mapFileInput').addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleMapUpload(e.target.files[0]);
    }
  });

  // Algorithm pill toggle
  document.querySelectorAll('.algo-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      const cb = pill.querySelector('input');
      cb.checked = !cb.checked;
      pill.classList.toggle('active', cb.checked);
    });
  });

  // Multiselect All / Clear All
  document.getElementById('btnSelectAllStops').addEventListener('click', () => {
    state.selectedHouseIds = state.nodes.filter((n) => !n.is_depot).map((n) => n.id);
    renderStopSelector();
  });
  document.getElementById('btnClearAllStops').addEventListener('click', () => {
    state.selectedHouseIds = [];
    renderStopSelector();
  });

  // Expander sub-tabs
  document.querySelectorAll('.sub-tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.sub-tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.sub-tab-pane').forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.target).classList.add('active');
    });
  });

  // Manage stops actions
  document.getElementById('btnAddStop').addEventListener('click', addCustomStop);
  document.getElementById('btnApplyTimeWindows').addEventListener('click', applyUpdatedTimeWindows);
  document.getElementById('btnImportCsv').addEventListener('click', () => {
    const file = document.getElementById('stopsCsvInput').files[0];
    handleCsvImport(file);
  });

  // Main Tabs navigation
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.target).classList.add('active');
    });
  });

  // View algorithm route dropdown
  document.getElementById('viewAlgoSelect').addEventListener('change', (e) => {
    state.selectedViewAlgoKey = e.target.value;
    renderMapTab();
  });

  // Road mask preview checkbox
  document.getElementById('showMaskCheckbox').addEventListener('change', renderMapTab);

  // Big primary optimization button
  document.getElementById('btnOptimize').addEventListener('click', runOptimization);

  // Multi-seed study button
  document.getElementById('btnRunMultiSeed').addEventListener('click', runMultiSeedStudy);

  // Download / Export buttons
  document.getElementById('btnDownloadRouteMap').addEventListener('click', downloadRouteMapPng);
  document.getElementById('btnExportBenchmarkCsv').addEventListener('click', exportBenchmarkCsv);
  document.getElementById('btnExportManifestCsv').addEventListener('click', exportManifestCsv);
  document.getElementById('btnExportMultiSeedCsv').addEventListener('click', exportMultiSeedCsv);

  // Initial Load: generate default procedural map (n_houses=8, seed=42)
  generateProceduralMap(false);
});
