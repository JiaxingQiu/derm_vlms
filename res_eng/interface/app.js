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
  inp.id = 'q' + i; inp.placeholder = 'Other feedback...'; inp.rows = 4;
  inp.dataset.cropKey = 'q' + i + '_crops';
  surveyEl.appendChild(inp);
  const evDiv = document.createElement('div');
  evDiv.className = 'field-evidence';
  evDiv.id = 'q' + i + '-ev';
  surveyEl.appendChild(evDiv);
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

/* ===== Per-field crop storage ===== */
function getFieldCrops(cropKey) {
  return S.R[S.vlm]?.[row().id]?.[cropKey] || [];
}
function setFieldCrops(cropKey, crops) {
  if (!S.R[S.vlm]) S.R[S.vlm] = {};
  if (!S.R[S.vlm][row().id]) S.R[S.vlm][row().id] = {};
  S.R[S.vlm][row().id][cropKey] = crops;
}

/* ===== Review-row state helpers ===== */
function getItemR(key) { return S.R[S.vlm]?.[row().id]?.[key] || ''; }
function setItemR(key, val) {
  if (!S.R[S.vlm]) S.R[S.vlm] = {};
  if (!S.R[S.vlm][row().id]) S.R[S.vlm][row().id] = {};
  S.R[S.vlm][row().id][key] = val;
}

/**
 * Render an interactive check/maybe/cross list.
 */
function renderItemList(containerId, sectionId, items, prefix, numbered, rerender) {
  const listEl = document.getElementById(containerId);
  const sectionEl = document.getElementById(sectionId);

  if (!items.length) {
    sectionEl.style.display = 'none';
    return;
  }
  sectionEl.style.display = '';
  listEl.innerHTML = '';

  items.forEach((text, idx) => {
    const key = prefix + '_' + idx;
    const status = getItemR(key);
    const correction = getItemR(key + '_correction') || '';

    const rowDiv = document.createElement('div');
    rowDiv.className = 'review-row';
    if (status) rowDiv.classList.add('rev-' + status);

    if (numbered) {
      const numEl = document.createElement('span');
      numEl.className = 'rev-num';
      numEl.textContent = (idx + 1) + '.';
      rowDiv.appendChild(numEl);
    }

    const textEl = document.createElement('span');
    textEl.className = 'rev-text';
    textEl.textContent = text;
    rowDiv.appendChild(textEl);

    const actions = document.createElement('div');
    actions.className = 'rev-actions';

    const btnOk = document.createElement('button');
    btnOk.className = 'rev-btn rev-ok' + (status === 'correct' ? ' active' : '');
    btnOk.innerHTML = '&#x2713;';
    btnOk.dataset.tip = 'Correct';
    const clearCorrection = () => {
      setItemR(key + '_correction', '');
      setFieldCrops(key + '_crops', []);
    };

    btnOk.onclick = () => {
      setItemR(key, status === 'correct' ? '' : 'correct');
      clearCorrection();
      persist(); rerender(); showSaved();
    };

    const btnMaybe = document.createElement('button');
    btnMaybe.className = 'rev-btn rev-maybe' + (status === 'maybe' ? ' active' : '');
    btnMaybe.innerHTML = '?';
    btnMaybe.dataset.tip = 'Maybe';
    btnMaybe.onclick = () => {
      setItemR(key, status === 'maybe' ? '' : 'maybe');
      clearCorrection();
      persist(); rerender(); showSaved();
    };

    const btnNo = document.createElement('button');
    btnNo.className = 'rev-btn rev-no' + (status === 'incorrect' ? ' active' : '');
    btnNo.innerHTML = '&#x2717;';
    btnNo.dataset.tip = 'Incorrect';
    btnNo.onclick = () => {
      setItemR(key, status === 'incorrect' ? '' : 'incorrect');
      if (status === 'incorrect') clearCorrection();
      persist(); rerender(); showSaved();
    };

    actions.appendChild(btnOk);
    actions.appendChild(btnNo);
    actions.appendChild(btnMaybe);
    rowDiv.appendChild(actions);
    listEl.appendChild(rowDiv);

    if (status === 'incorrect') {
      const corrDiv = document.createElement('div');
      corrDiv.className = 'rev-correction';

      const inp = document.createElement('input');
      inp.type = 'text';
      inp.placeholder = 'What should it say instead?';
      inp.value = correction;
      const cropKey = key + '_crops';
      inp.dataset.cropKey = cropKey;
      inp.oninput = () => {
        setItemR(key + '_correction', inp.value);
        syncFieldCrops(inp);
        persist(); showSaved('\u2713 Auto-saved');
      };
      corrDiv.appendChild(inp);

      const evDiv = document.createElement('div');
      evDiv.className = 'field-evidence';
      corrDiv.appendChild(evDiv);
      renderFieldEvidenceInto(evDiv, cropKey, inp);

      listEl.appendChild(corrDiv);
    }
  });
}

function renderDiagnoses() {
  renderItemList('diag-list', 'diag-section', row().diagnosis_list || [], 'diag', true, renderDiagnoses);
}

function renderDescription() {
  renderItemList('desc-list', 'desc-section', row().sentence_list || [], 'desc', false, renderDescription);
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

/* ===== Track last-focused text input ===== */
let lastFocusedInput = null;
document.addEventListener('focusin', e => {
  if (e.target.tagName === 'TEXTAREA' || (e.target.tagName === 'INPUT' && e.target.type === 'text')) {
    lastFocusedInput = e.target;
  }
});

/* ===== Per-field evidence ===== */
function addCropEvidence(crop) {
  const ta = lastFocusedInput || qInputs[0];
  if (!ta) return;
  const cropKey = ta.dataset.cropKey || 'q0_crops';
  const crops = [...getFieldCrops(cropKey), crop];
  setFieldCrops(cropKey, crops);

  const marker = '[ev ' + crops.length + '] ';
  const pos = ta.selectionStart || ta.value.length;
  ta.value = ta.value.slice(0, pos) + marker + ta.value.slice(pos);
  ta.focus();
  ta.selectionStart = ta.selectionEnd = pos + marker.length;

  if (ta === qInputs[0]) collect();
  persist();

  const evDiv = findEvidenceDiv(ta);
  if (evDiv) renderFieldEvidenceInto(evDiv, cropKey, ta);
  showSaved();
}

function findEvidenceDiv(field) {
  if (field.id && document.getElementById(field.id + '-ev')) {
    return document.getElementById(field.id + '-ev');
  }
  const parent = field.parentElement;
  if (parent) return parent.querySelector('.field-evidence');
  return null;
}

function removeFieldCrop(cropKey, idx, fieldEl, evDiv) {
  floatPrev.style.display = 'none';
  const oldLen = getFieldCrops(cropKey).length;
  const crops = getFieldCrops(cropKey).filter((_, i) => i !== idx);
  setFieldCrops(cropKey, crops);

  if (fieldEl) {
    let text = fieldEl.value;
    const removed = idx + 1;
    text = text.replace(new RegExp('\\[ev\\s+' + removed + '\\]\\s?', 'g'), '');
    for (let n = removed + 1; n <= oldLen; n++) {
      text = text.replace(new RegExp('\\[ev\\s+' + n + '\\]', 'g'), '[ev ' + (n - 1) + ']');
    }
    fieldEl.value = text;
    if (fieldEl === qInputs[0]) collect();
  }
  persist();
  if (evDiv) renderFieldEvidenceInto(evDiv, cropKey, fieldEl);
  showSaved();
}

function renderFieldEvidenceInto(container, cropKey, fieldEl) {
  const crops = getFieldCrops(cropKey);
  container.innerHTML = '';
  if (!crops.length) return;

  const imgSrc = IMAGES[row().id];
  const im = new Image();
  im.onload = () => {
    const nw = im.naturalWidth, nh = im.naturalHeight;
    crops.forEach((c, idx) => {
      const cropPxW = c.w * nw, cropPxH = c.h * nh;
      const maxThumb = 48;
      const sc = Math.min(maxThumb / cropPxW, maxThumb / cropPxH, 1);
      const dispW = Math.max(1, cropPxW * sc);
      const dispH = Math.max(1, cropPxH * sc);

      const div = document.createElement('div');
      div.className = 'evidence-item';
      div.dataset.prevSrc = imgSrc;
      div.dataset.prevCrop = JSON.stringify(c);
      div.dataset.prevNw = nw;
      div.dataset.prevNh = nh;

      const wrapper = document.createElement('div');
      wrapper.className = 'ev-crop';
      wrapper.style.width  = dispW + 'px';
      wrapper.style.height = dispH + 'px';

      const thumb = document.createElement('img');
      thumb.src = imgSrc;
      thumb.style.width  = (nw * sc) + 'px';
      thumb.style.height = (nh * sc) + 'px';
      thumb.style.left   = (-c.x * nw * sc) + 'px';
      thumb.style.top    = (-c.y * nh * sc) + 'px';
      wrapper.appendChild(thumb);

      const lbl = document.createElement('div');
      lbl.className = 'ev-label';
      lbl.textContent = 'Ev ' + (idx + 1);

      const rm = document.createElement('button');
      rm.className = 'ev-remove';
      rm.innerHTML = '&times;';
      rm.onclick = () => { removeFieldCrop(cropKey, idx, fieldEl, container); };

      div.appendChild(wrapper);
      div.appendChild(lbl);
      div.appendChild(rm);
      container.appendChild(div);
    });
  };
  im.src = imgSrc;
}

/* ===== Floating preview on hover ===== */
const floatPrev = document.createElement('div');
floatPrev.className = 'ev-float-preview';
document.body.appendChild(floatPrev);

document.addEventListener('mouseenter', e => {
  const item = e.target.closest('.evidence-item');
  if (!item || !item.dataset.prevCrop) return;
  const c = JSON.parse(item.dataset.prevCrop);
  const nw = +item.dataset.prevNw, nh = +item.dataset.prevNh;
  const cropPxW = c.w * nw, cropPxH = c.h * nh;
  const prevMax = 240;
  const psc = Math.min(prevMax / cropPxW, prevMax / cropPxH, 1);
  const pW = Math.max(1, cropPxW * psc);
  const pH = Math.max(1, cropPxH * psc);

  floatPrev.style.width = pW + 'px';
  floatPrev.style.height = pH + 'px';
  floatPrev.innerHTML = '';
  const img = document.createElement('img');
  img.src = item.dataset.prevSrc;
  img.style.width  = (nw * psc) + 'px';
  img.style.height = (nh * psc) + 'px';
  img.style.left   = (-c.x * nw * psc) + 'px';
  img.style.top    = (-c.y * nh * psc) + 'px';
  floatPrev.appendChild(img);

  const rect = item.getBoundingClientRect();
  let top = rect.top - pH - 8;
  if (top < 4) top = rect.bottom + 8;
  let left = rect.left + rect.width / 2 - pW / 2;
  left = Math.max(4, Math.min(left, window.innerWidth - pW - 4));
  floatPrev.style.top = top + 'px';
  floatPrev.style.left = left + 'px';
  floatPrev.style.display = 'block';
}, true);

document.addEventListener('mouseleave', e => {
  const item = e.target.closest('.evidence-item');
  if (item) floatPrev.style.display = 'none';
}, true);

/* ===== Sync: text edits → remove orphaned crops ===== */
function syncFieldCrops(fieldEl) {
  const cropKey = fieldEl.dataset.cropKey;
  if (!cropKey) return;
  const crops = getFieldCrops(cropKey);
  if (!crops.length) return;

  const present = new Set();
  const re = /\[ev\s+(\d+)\]/g;
  let m;
  while ((m = re.exec(fieldEl.value)) !== null) present.add(parseInt(m[1]));
  if (present.size === crops.length) return;

  const kept = [];
  const oldToNew = {};
  for (let i = 0; i < crops.length; i++) {
    if (present.has(i + 1)) { oldToNew[i + 1] = kept.length + 1; kept.push(crops[i]); }
  }
  setFieldCrops(cropKey, kept);

  let text = fieldEl.value;
  for (const [old, nw] of Object.entries(oldToNew)) {
    if (+old !== nw) text = text.replace(new RegExp('\\[ev\\s+' + old + '\\]', 'g'), '[ev ' + nw + ']');
  }
  fieldEl.value = text;
  if (fieldEl === qInputs[0]) collect();
  persist();

  const evDiv = findEvidenceDiv(fieldEl);
  if (evDiv) renderFieldEvidenceInto(evDiv, cropKey, fieldEl);
}

/* ===== Atomic delete of [ev N] markers ===== */
document.addEventListener('keydown', e => {
  const el = e.target;
  if (!(el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && el.type === 'text'))) return;
  if (e.key !== 'Backspace' && e.key !== 'Delete') return;

  const text = el.value;
  const pos = el.selectionStart;
  const selEnd = el.selectionEnd;
  if (pos !== selEnd) return;

  const re = /\[ev\s+\d+\]/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const start = m.index, end = m.index + m[0].length;
    const hit = (e.key === 'Backspace')
      ? (pos > start && pos <= end)
      : (pos >= start && pos < end);
    if (hit) {
      e.preventDefault();
      el.value = text.slice(0, start) + text.slice(end);
      el.selectionStart = el.selectionEnd = start;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      return;
    }
  }
});


/* ===== Confidence slider ===== */
const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');

function getConf() { return S.R[S.vlm]?.[row().id]?.confidence ?? ''; }
function setConf(v) {
  if (!S.R[S.vlm]) S.R[S.vlm] = {};
  if (!S.R[S.vlm][row().id]) S.R[S.vlm][row().id] = {};
  S.R[S.vlm][row().id].confidence = v;
}

confSlider.addEventListener('input', () => {
  const v = confSlider.value;
  confVal.textContent = v;
  setConf(v);
  persist();
  showSaved('\u2713 Auto-saved');
});

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
  document.getElementById('img-id').textContent = 'Lesion ' + (S.idx + 1) + ' of ' + data.length;
  document.getElementById('btn-prev').disabled = S.idx === 0;
  document.getElementById('btn-next').disabled = S.idx === data.length - 1;
  document.querySelectorAll('.tab').forEach(t => {
    const v = t.dataset.vlm;
    t.classList.toggle('active', v === S.vlm);
    t.querySelector('.dot').classList.toggle('done', vlmDone(v));
  });
  const savedConf = getConf();
  if (savedConf !== '') { confSlider.value = savedConf; confVal.textContent = savedConf; }
  else { confSlider.value = 5; confVal.textContent = '5'; }

  renderDiagnoses();
  renderDescription();
  qInputs.forEach((el, i) => {
    el.value = getR(i);
    const evDiv = document.getElementById('q' + i + '-ev');
    if (evDiv) renderFieldEvidenceInto(evDiv, el.dataset.cropKey, el);
  });
}

/* ===== Navigation ===== */
document.getElementById('btn-prev').onclick = () => { collect(); persist(); if (S.idx > 0) { S.idx--; render(); } };
document.getElementById('btn-next').onclick = () => { collect(); persist(); if (S.idx < VLM_DATA[S.vlm].length - 1) { S.idx++; render(); } };

/* ===== Auto-save ===== */
document.getElementById('survey').addEventListener('input', () => {
  collect(); persist();
  qInputs.forEach(el => syncFieldCrops(el));
  showSaved('\u2713 Auto-saved');
});

/* ===== Export CSV ===== */
document.getElementById('btn-exp').onclick = () => {
  collect(); persist();
  const qHeaders = QUESTIONS.map((_, i) => 'Q' + (i + 1));
  const qKeys = QUESTIONS.map((_, i) => rkey(i));
  let csv = ['id', 'model', 'response', ...qHeaders, 'confidence', 'diagnosis_judgments', 'description_judgments', 'crop_coordinates'].join(',') + '\n';
  Object.keys(VLM_DATA).forEach(vlm => {
    const rows = VLM_DATA[vlm], resp = S.R[vlm] || {};
    rows.forEach(r => {
      const a = resp[r.id] || {};
      const allCrops = [];
      Object.keys(a).forEach(k => {
        if (k.endsWith('_crops') && Array.isArray(a[k])) allCrops.push(...a[k]);
      });
      const buildJudgments = (list, prefix) => {
        const j = {};
        (list || []).forEach((text, i) => {
          const st = a[prefix + '_' + i] || '';
          const corr = a[prefix + '_' + i + '_correction'] || '';
          if (st) j[i] = { text: text, status: st, correction: corr };
        });
        return Object.keys(j).length ? JSON.stringify(j) : '';
      };
      const diagStr = buildJudgments(r.diagnosis_list, 'diag');
      const descStr = buildJudgments(r.sentence_list, 'desc');
      const cropStr = allCrops.length ? JSON.stringify(allCrops) : '';
      const conf = a.confidence ?? '';
      const vals = [r.id, vlm, r[TASK_KEY], ...qKeys.map(c => a[c] || ''), conf, diagStr, descStr, cropStr];
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
