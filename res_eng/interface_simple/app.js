/* ============================================================
   Dermatology VLM Evaluation — app.js
   Expects these globals injected before this script:
     VLM_DATA, IMAGES, QUESTIONS, TASK_KEY, SK
   ============================================================ */

/* ===== State ===== */
const S = {
  vlm:  Object.keys(VLM_DATA)[0],
  idx:  0,
  R:    JSON.parse(localStorage.getItem(SK) || '{}'),
  zoom: 1, panX: 0, panY: 0,
  baseW: 0, baseH: 0,
  cropMode: false,
  isPanning: false,
  panStartX: 0, panStartY: 0,
  panOrigX: 0, panOrigY: 0
};

/* ===== Element refs ===== */
const imgEl       = document.getElementById('img');
const containerEl = document.getElementById('img-container');
const cropOverlay = document.getElementById('crop-overlay');
const cropSel     = document.getElementById('crop-sel');
const zoomLbl     = document.getElementById('zoom-lbl');
const evArea      = document.getElementById('evidence-area');
const evList      = document.getElementById('evidence-list');

/* ===== VLM tabs ===== */
const VLM_NAMES = Object.keys(VLM_DATA);
const tabsEl = document.getElementById('tabs');
VLM_NAMES.forEach(n => {
  const b = document.createElement('button');
  b.className = 'tab'; b.dataset.vlm = n;
  b.innerHTML = n + '<span class="dot"></span>';
  b.onclick = () => { collect(); S.vlm = n; render(); };
  tabsEl.appendChild(b);
});

/* ===== Question inputs ===== */
const surveyEl = document.getElementById('survey');
QUESTIONS.forEach((q, i) => {
  const inp = document.createElement('textarea');
  inp.id = 'q' + i; inp.placeholder = 'Your response...'; inp.rows = 4;
  surveyEl.appendChild(inp);
});
const qInputs = QUESTIONS.map((_, i) => document.getElementById('q' + i));

/* ===== Text cleaning ===== */
function cleanText(text) {
  if (!text) return '';
  let t = text;
  t = t.replace(/\*{3,}/g, '');
  t = t.replace(/\*\*(.+?)\*\*/gs, '$1');
  t = t.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/gs, '$1');
  t = t.replace(/\*/g, '');
  t = t.replace(/^#{1,6}\s*/gm, '');
  t = t.replace(/^[-_]{3,}\s*$/gm, '');
  t = t.replace(/\n{3,}/g, '\n\n');
  return t.trim();
}

/* ===== Data helpers ===== */
function row() { return VLM_DATA[S.vlm][S.idx]; }
function rkey(i) { return TASK_KEY + '_q' + (i + 1); }
function getR(i) { return S.R[S.vlm]?.[row().id]?.[rkey(i)] || ''; }
function setR(i, v) {
  if (!S.R[S.vlm]) S.R[S.vlm] = {};
  if (!S.R[S.vlm][row().id]) S.R[S.vlm][row().id] = {};
  S.R[S.vlm][row().id][rkey(i)] = v;
}
function collect() { qInputs.forEach((el, i) => setR(i, el.value)); }
function persist() { localStorage.setItem(SK, JSON.stringify(S.R)); }

function getCrops() {
  return S.R[S.vlm]?.[row().id]?.crops || [];
}
function setCrops(crops) {
  if (!S.R[S.vlm]) S.R[S.vlm] = {};
  if (!S.R[S.vlm][row().id]) S.R[S.vlm][row().id] = {};
  S.R[S.vlm][row().id].crops = crops;
}

function fmtId(id) {
  const m = id.match(/^(\d+)_combined$/);
  return m ? 'Lesion ' + m[1] : id;
}
function vlmDone(vlm) {
  const imgId = VLM_DATA[vlm][S.idx].id;
  const r = S.R[vlm]?.[imgId];
  if (!r) return false;
  return QUESTIONS.every((_, i) => {
    const v = r[TASK_KEY + '_q' + (i + 1)];
    return v && v.trim();
  });
}

/* ===== Image fit / transform ===== */
function fitImage() {
  if (!imgEl.naturalWidth) return;
  const cw = containerEl.clientWidth;
  const nw = imgEl.naturalWidth, nh = imgEl.naturalHeight;
  const maxH = window.innerHeight * 0.85;
  let dw = cw, dh = cw * (nh / nw);
  if (dh > maxH) { dh = maxH; dw = maxH * (nw / nh); }
  containerEl.style.height = dh + 'px';
  S.baseW = dw;
  S.baseH = dh;
  updateTransform();
}
function updateTransform() {
  const cw = containerEl.clientWidth, ch = containerEl.clientHeight;
  const cx = (cw - S.baseW * S.zoom) / 2 + S.panX;
  const cy = (ch - S.baseH * S.zoom) / 2 + S.panY;
  imgEl.style.width  = S.baseW + 'px';
  imgEl.style.height = S.baseH + 'px';
  imgEl.style.transformOrigin = '0 0';
  imgEl.style.transform = 'translate(' + cx + 'px,' + cy + 'px) scale(' + S.zoom + ')';
  zoomLbl.textContent = Math.round(S.zoom * 100) + '%';
}
function resetView() {
  S.zoom = 1; S.panX = 0; S.panY = 0;
  updateTransform();
}

/* ===== Zoom ===== */
function zoomBy(delta, cx, cy) {
  const oldZ = S.zoom;
  S.zoom = Math.min(Math.max(S.zoom * (1 + delta), 0.5), 8);
  if (cx !== undefined) {
    const cw = containerEl.clientWidth, ch = containerEl.clientHeight;
    const oldIx = (cw - S.baseW * oldZ) / 2 + S.panX;
    const oldIy = (ch - S.baseH * oldZ) / 2 + S.panY;
    const ix = (cx - oldIx) / oldZ;
    const iy = (cy - oldIy) / oldZ;
    const newIx = (cw - S.baseW * S.zoom) / 2 + S.panX;
    const newIy = (ch - S.baseH * S.zoom) / 2 + S.panY;
    S.panX += cx - (newIx + ix * S.zoom);
    S.panY += cy - (newIy + iy * S.zoom);
  }
  updateTransform();
}
containerEl.addEventListener('wheel', e => {
  e.preventDefault();
  const r = containerEl.getBoundingClientRect();
  zoomBy(e.deltaY < 0 ? 0.15 : -0.15, e.clientX - r.left, e.clientY - r.top);
}, { passive: false });
document.getElementById('btn-zin').onclick  = () => zoomBy(0.25);
document.getElementById('btn-zout').onclick = () => zoomBy(-0.25);
document.getElementById('btn-zreset').onclick = resetView;

/* ===== Pan (default mode) ===== */
containerEl.addEventListener('mousedown', e => {
  if (S.cropMode || e.button !== 0) return;
  S.isPanning = true;
  S.panStartX = e.clientX; S.panStartY = e.clientY;
  S.panOrigX = S.panX; S.panOrigY = S.panY;
  containerEl.classList.add('panning');
  e.preventDefault();
});
window.addEventListener('mousemove', e => {
  if (!S.isPanning) return;
  S.panX = S.panOrigX + (e.clientX - S.panStartX);
  S.panY = S.panOrigY + (e.clientY - S.panStartY);
  updateTransform();
});
window.addEventListener('mouseup', () => {
  if (S.isPanning) { S.isPanning = false; containerEl.classList.remove('panning'); }
});

/* ===== Crop mode ===== */
const btnCrop = document.getElementById('btn-crop');
let cropStart = null;

btnCrop.onclick = () => {
  S.cropMode = !S.cropMode;
  btnCrop.classList.toggle('active', S.cropMode);
  cropOverlay.style.display = S.cropMode ? 'block' : 'none';
  containerEl.classList.toggle('crop-active', S.cropMode);
};

function screenToImgNorm(sx, sy) {
  const cw = containerEl.clientWidth, ch = containerEl.clientHeight;
  const imgCx = (cw - S.baseW * S.zoom) / 2 + S.panX;
  const imgCy = (ch - S.baseH * S.zoom) / 2 + S.panY;
  const ix = (sx - imgCx) / (S.baseW * S.zoom);
  const iy = (sy - imgCy) / (S.baseH * S.zoom);
  return { x: Math.max(0, Math.min(1, ix)), y: Math.max(0, Math.min(1, iy)) };
}

cropOverlay.addEventListener('mousedown', e => {
  if (!S.cropMode) return;
  e.preventDefault(); e.stopPropagation();
  const r = containerEl.getBoundingClientRect();
  cropStart = { sx: e.clientX - r.left, sy: e.clientY - r.top };
  cropSel.style.display = 'none';
});
cropOverlay.addEventListener('mousemove', e => {
  if (!cropStart) return;
  const r = containerEl.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const x1 = Math.min(cropStart.sx, mx), y1 = Math.min(cropStart.sy, my);
  const x2 = Math.max(cropStart.sx, mx), y2 = Math.max(cropStart.sy, my);
  cropSel.style.display = 'block';
  cropSel.style.left = x1 + 'px'; cropSel.style.top = y1 + 'px';
  cropSel.style.width = (x2 - x1) + 'px'; cropSel.style.height = (y2 - y1) + 'px';
});
cropOverlay.addEventListener('mouseup', e => {
  if (!cropStart) return;
  const r = containerEl.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const p1 = screenToImgNorm(cropStart.sx, cropStart.sy);
  const p2 = screenToImgNorm(mx, my);
  cropSel.style.display = 'none';
  cropStart = null;
  const x = Math.min(p1.x, p2.x), y = Math.min(p1.y, p2.y);
  const w = Math.abs(p2.x - p1.x), h = Math.abs(p2.y - p1.y);
  if (w < 0.01 || h < 0.01) return;
  addCropEvidence({ x, y, w, h });
  S.cropMode = false;
  btnCrop.classList.remove('active');
  cropOverlay.style.display = 'none';
  containerEl.classList.remove('crop-active');
});

/* ===== Evidence management ===== */
let cachedImg = null, cachedImgId = null;

function ensureCachedImg(cb) {
  const id = row().id;
  if (cachedImgId === id && cachedImg && cachedImg.complete) { cb(); return; }
  cachedImg = new Image();
  cachedImg.onload = () => { cachedImgId = id; cb(); };
  cachedImg.src = IMAGES[id];
}

function addCropEvidence(crop) {
  const crops = [...getCrops(), crop];
  setCrops(crops);
  persist();
  renderEvidence();
  const ta = qInputs[0];
  if (ta) {
    const marker = '[Evidence ' + crops.length + '] ';
    const pos = ta.selectionStart || ta.value.length;
    ta.value = ta.value.slice(0, pos) + marker + ta.value.slice(pos);
    ta.focus();
    ta.selectionStart = ta.selectionEnd = pos + marker.length;
    collect(); persist();
  }
  showSaved();
}

function removeCropEvidence(idx) {
  const oldLen = getCrops().length;
  const crops = getCrops().filter((_, i) => i !== idx);
  setCrops(crops);
  const ta = qInputs[0];
  if (ta) {
    let text = ta.value;
    const removed = idx + 1;
    text = text.replace(new RegExp('\\[Evidence\\s+' + removed + '\\]\\s?', 'g'), '');
    for (let n = removed + 1; n <= oldLen; n++) {
      text = text.replace(new RegExp('\\[Evidence\\s+' + n + '\\]', 'g'), '[Evidence ' + (n - 1) + ']');
    }
    ta.value = text;
    collect();
  }
  persist();
  renderEvidence();
}

function renderEvidence() {
  const crops = getCrops();
  evArea.style.display = crops.length ? '' : 'none';
  evList.innerHTML = '';
  if (!crops.length) return;

  ensureCachedImg(() => {
    const nw = cachedImg.naturalWidth, nh = cachedImg.naturalHeight;
    crops.forEach((c, idx) => {
      const cropPxW = c.w * nw, cropPxH = c.h * nh;
      const maxThumb = 100;
      const sc = Math.min(maxThumb / cropPxW, maxThumb / cropPxH, 1);
      const dispW = Math.max(1, cropPxW * sc);
      const dispH = Math.max(1, cropPxH * sc);

      const div = document.createElement('div');
      div.className = 'evidence-item';

      const wrapper = document.createElement('div');
      wrapper.className = 'ev-crop';
      wrapper.style.width  = dispW + 'px';
      wrapper.style.height = dispH + 'px';

      const thumb = document.createElement('img');
      thumb.src = IMAGES[row().id];
      thumb.style.width  = (nw * sc) + 'px';
      thumb.style.height = (nh * sc) + 'px';
      thumb.style.left   = (-c.x * nw * sc) + 'px';
      thumb.style.top    = (-c.y * nh * sc) + 'px';

      wrapper.appendChild(thumb);

      const lbl = document.createElement('div');
      lbl.className = 'ev-label';
      lbl.textContent = 'Evidence ' + (idx + 1);

      const rm = document.createElement('button');
      rm.className = 'ev-remove';
      rm.innerHTML = '&times;';
      rm.onclick = () => { removeCropEvidence(idx); };

      div.appendChild(wrapper);
      div.appendChild(lbl);
      div.appendChild(rm);
      evList.appendChild(div);
    });
  });
}

/* ===== Edit AI response ===== */
document.getElementById('btn-copy').onclick = () => {
  const resp = document.getElementById('resp').textContent;
  if (qInputs.length > 0) { qInputs[0].value = resp; qInputs[0].focus(); collect(); persist(); }
  showSaved('\u2713 Copied');
};

/* ===== Status ===== */
let saveTimer;
function showSaved(msg) {
  const el = document.getElementById('status');
  el.textContent = msg || '\u2713 Saved';
  el.classList.add('show');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => el.classList.remove('show'), 1500);
}

/* ===== Render ===== */
function render() {
  const r = row(), data = VLM_DATA[S.vlm];
  imgEl.onload = fitImage;
  imgEl.src = IMAGES[r.id];
  resetView();
  document.getElementById('img-id').textContent = fmtId(r.id);
  document.getElementById('prog').textContent = 'Lesion ' + (S.idx + 1) + ' of ' + data.length;
  document.getElementById('btn-prev').disabled = S.idx === 0;
  document.getElementById('btn-next').disabled = S.idx === data.length - 1;
  document.querySelectorAll('.tab').forEach(t => {
    const v = t.dataset.vlm;
    t.classList.toggle('active', v === S.vlm);
    t.querySelector('.dot').classList.toggle('done', vlmDone(v));
  });
  document.getElementById('resp').textContent = cleanText(r[TASK_KEY]);
  qInputs.forEach((el, i) => { el.value = getR(i); });
  renderEvidence();
}

/* ===== Navigation ===== */
document.getElementById('btn-prev').onclick = () => { collect(); persist(); if (S.idx > 0) { S.idx--; render(); } };
document.getElementById('btn-next').onclick = () => { collect(); persist(); if (S.idx < VLM_DATA[S.vlm].length - 1) { S.idx++; render(); } };

/* ===== Sync: text edits remove crops ===== */
function syncCropsFromText() {
  const ta = qInputs[0];
  if (!ta) return;
  const crops = getCrops();
  if (!crops.length) return;
  const present = new Set();
  const re = /\[Evidence\s+(\d+)\]/g;
  let m;
  while ((m = re.exec(ta.value)) !== null) present.add(parseInt(m[1]));
  if (present.size === crops.length) return;
  const kept = [];
  const oldToNew = {};
  for (let i = 0; i < crops.length; i++) {
    if (present.has(i + 1)) { oldToNew[i + 1] = kept.length + 1; kept.push(crops[i]); }
  }
  setCrops(kept);
  let text = ta.value;
  for (const [old, nw] of Object.entries(oldToNew)) {
    if (+old !== nw) text = text.replace(new RegExp('\\[Evidence\\s+' + old + '\\]', 'g'), '[Evidence ' + nw + ']');
  }
  ta.value = text;
  collect(); persist();
  renderEvidence();
}

/* ===== Auto-save ===== */
document.getElementById('survey').addEventListener('input', () => {
  collect(); persist();
  syncCropsFromText();
  showSaved('\u2713 Auto-saved');
});

/* ===== Export CSV ===== */
document.getElementById('btn-exp').onclick = () => {
  collect(); persist();
  const qHeaders = QUESTIONS.map((_, i) => 'Q' + (i + 1));
  const qKeys = QUESTIONS.map((_, i) => rkey(i));
  let csv = ['id', 'model', 'response', ...qHeaders, 'crop_coordinates'].join(',') + '\n';
  Object.keys(VLM_DATA).forEach(vlm => {
    const rows = VLM_DATA[vlm], resp = S.R[vlm] || {};
    rows.forEach(r => {
      const a = resp[r.id] || {};
      const crops = a.crops ? JSON.stringify(a.crops) : '';
      const vals = [r.id, vlm, r[TASK_KEY], ...qKeys.map(c => a[c] || ''), crops];
      csv += vals.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(',') + '\n';
    });
  });
  const blob = new Blob([csv], { type: 'text/csv' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'vlm_evaluation_responses.csv';
  document.body.appendChild(link); link.click(); document.body.removeChild(link);
};

/* ===== Keyboard ===== */
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft') document.getElementById('btn-prev').click();
  if (e.key === 'ArrowRight') document.getElementById('btn-next').click();
  if (e.key === 'Escape' && S.cropMode) {
    S.cropMode = false; btnCrop.classList.remove('active');
    cropOverlay.style.display = 'none'; containerEl.classList.remove('crop-active');
    cropStart = null; cropSel.style.display = 'none';
  }
});

window.addEventListener('resize', fitImage);
render();
