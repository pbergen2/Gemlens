// GemLens UI: uploads, Python analysis, report visibility, and image callouts.
// Detection and scoring live in engine.py; this file only renders its results.
const $ = id => document.getElementById(id);
const state = { front: null, back: null, side: 'front', busy: false, report: null, selectedIssue: null };
let server = false;
let worker;
let sequence = 0;
const pending = new Map();

const serverReady = fetch('/api/health')
  .then(async response => {
    if (response.ok && response.headers.get('content-type')?.includes('application/json')) {
      server = (await response.json()).runtime === 'python-server';
    }
  })
  .catch(() => {})
  .then(() => {
    if (server) $('privacy').textContent = 'Photos are processed here and are not saved.';
  });

// The hosted site runs the same Python code in a background worker.
function invoke(operation, payload, buffer) {
  if (!worker) {
    worker = new Worker('/worker.js');
    worker.onmessage = ({ data }) => {
      const job = pending.get(data.id);
      if (!job) return;
      clearTimeout(job.timer);
      pending.delete(data.id);
      data.error ? job.reject(new Error(data.error)) : job.resolve(data.result);
    };
    worker.onerror = () => stopWorker('The inspection engine could not load. Please try again.');
  }
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => stopWorker('Analysis timed out. Check your connection and try again.'), 120000);
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ id, operation, payload, buffer }, buffer ? [buffer] : []);
  });
}

function stopWorker(reason) {
  for (const job of pending.values()) {
    clearTimeout(job.timer);
    job.reject(new Error(reason));
  }
  pending.clear();
  worker?.terminate();
  worker = null;
}

function clearReport(clearMeasurements = true) {
  state.report = null;
  state.selectedIssue = null;
  if (clearMeasurements) {
    for (const side of ['front', 'back']) if (state[side]) state[side].result = null;
  }
  renderReport();
}

function select(side) {
  state.side = side;
  state.selectedIssue = null;
  for (const name of ['front', 'back']) {
    $(name + 'Tab').classList.toggle('active', name === side);
    $(name + 'Tab').setAttribute('aria-pressed', String(name === side));
  }
  renderImage();
}

function drawBox(ctx, box, w, h, color, dash = false) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash(dash ? [5, 4] : []);
  ctx.strokeRect(box[0] * w, box[1] * h, (box[2] - box[0]) * w, (box[3] - box[1]) * h);
  ctx.setLineDash([]);
}

// Circle the full detected patch, then route a line to a readable label.
// Labels are stacked in columns to avoid collisions, rather than sitting
// directly on top of another defect. Selecting a list item isolates a callout.
function drawIssues(ctx, regions, w, h) {
  const sides = [[], []];
  regions.forEach((region, index) => {
    if (state.selectedIssue !== null && state.selectedIssue !== index) return;
    sides[region.x < 0.5 ? 0 : 1].push({ region, index });
  });
  sides.forEach((items, column) => {
    items.sort((a, b) => a.region.y - b.region.y);
    const gap = 37;
    let previousY = 5;
    items.forEach(({ region, index }, order) => {
      const box = region.bbox || [region.x - 0.01, region.y - 0.01, region.x + 0.01, region.y + 0.01];
      const x = region.x * w;
      const y = region.y * h;
      const rx = Math.max(6, (box[2] - box[0]) * w / 2 + 3);
      const ry = Math.max(6, (box[3] - box[1]) * h / 2 + 3);
      const labelY = Math.max(previousY, Math.min(y - 13, h - 35 - (items.length - 1 - order) * gap));
      previousY = labelY + gap;
      const labelWidth = 154;
      const labelX = column === 0 ? 9 : w - labelWidth - 9;
      ctx.strokeStyle = '#ffbd76';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(column === 0 ? labelX : labelX + labelWidth, labelY + 15);
      ctx.stroke();
      ctx.fillStyle = 'rgba(29, 27, 25, 0.94)';
      ctx.fillRect(labelX, labelY, labelWidth, 32);
      ctx.font = '600 11px system-ui';
      ctx.fillStyle = '#ffd29c';
      ctx.fillText((index + 1) + '. ' + capitalize(region.severity || 'Possible') + ' whitening', labelX + 6, labelY + 13);
      ctx.font = '10px system-ui';
      ctx.fillStyle = '#ded6cd';
      ctx.fillText('Suspected · ' + (region.location || 'border'), labelX + 6, labelY + 26);
    });
  });
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function renderImage() {
  const item = state[state.side];
  $('uploadEmpty').hidden = !!item;
  $('imageView').hidden = !item;
  $('imageActions').hidden = !item;
  $('cropControls').hidden = !item?.result?.bbox;
  $('analyze').disabled = state.busy || !state.front;
  for (const side of ['front', 'back']) {
    $(side + 'Status').textContent = state[side] ? 'Added' : side === 'front' ? 'Required' : 'Required for grade';
  }
  const list = $('issueList');
  list.replaceChildren();
  list.hidden = true;
  if (!item) return;

  const canvas = $('preview');
  const w = 420;
  const h = Math.round(item.canvas.height / item.canvas.width * w);
  // Draw at 2x resolution for crisp outlines, without altering analysis pixels.
  canvas.width = w * 2;
  canvas.height = h * 2;
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);
  ctx.drawImage(item.displayCanvas || item.canvas, 0, 0, w, h);
  $('imageCaption').textContent = item.name;
  if (!item.result?.bbox) return;
  const box = item.crop || item.result.bbox;
  ['L', 'T', 'R', 'B'].forEach((key, i) => { $('crop' + key).value = Math.round(box[i] * 100); });
  const regions = item.result.regions || [];
  if ($('overlay').checked) {
    drawBox(ctx, box, w, h, '#a9f085');
    if (item.result.inner_bbox && !item.crop) drawBox(ctx, item.result.inner_bbox, w, h, '#e7ca73', true);
    drawIssues(ctx, regions, w, h);
  }
  list.hidden = !regions.length;
  regions.forEach((region, index) => {
    const button = document.createElement('button');
    button.className = 'issue-button';
    button.setAttribute('aria-pressed', String(state.selectedIssue === index));
    button.textContent = (index + 1) + '. ' + capitalize(region.severity || 'Possible') + ' whitening';
    const detail = document.createElement('small');
    detail.textContent = 'Suspected · ' + capitalize(region.location || 'border') + ' · select to isolate';
    button.append(detail);
    button.onclick = () => {
      state.selectedIssue = state.selectedIssue === index ? null : index;
      $('overlay').checked = true;
      renderImage();
      $('issueList').children[index]?.focus({ preventScroll: true });
    };
    list.append(button);
  });
}

function imageCanvas(image, maxWidth, maxHeight) {
  const scale = Math.min(maxWidth / image.width, maxHeight / image.height, 1);
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(image.width * scale);
  canvas.height = Math.round(image.height * scale);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#171e19';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas;
}

async function addFile(file) {
  if (!file || state.busy) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return message('Please choose a JPG, PNG or WebP image.');
  if (file.size > 15 * 1024 * 1024) return message('That photo is larger than 15 MB.');
  const side = state.side;
  try {
    const image = await createImageBitmap(file, { imageOrientation: 'from-image' });
    try {
      if (image.width * image.height > 24000000) throw new Error('Please choose a photo smaller than 24 megapixels.');
      if (Math.min(image.width, image.height) < 150) throw new Error('Use a photo at least 150 pixels wide and tall.');
      state[side] = {
        canvas: imageCanvas(image, 420, 560),
        displayCanvas: imageCanvas(image, 1260, 1680),
        file, name: file.name, originalHeight: image.height, result: null, crop: null
      };
    } finally {
      image.close();
    }
    if (side === 'front') clearReference();
    clearReport();
    message('');
    renderImage();
  } catch (error) {
    message(error.message || 'This photo could not be opened.');
  }
}

function message(text) { $('message').textContent = text; }

function setBusy(busy) {
  state.busy = busy;
  for (const id of ['frontTab', 'backTab', 'replace', 'remove', 'choose', 'applyCrop', 'file', 'cropL', 'cropT', 'cropR', 'cropB']) {
    $(id).disabled = busy;
  }
  $('analyze').disabled = busy || !state.front;
  $('analyze').textContent = busy ? 'Inspecting…' : 'Inspect card';
}

async function inspect() {
  if (state.busy || !state.front) return;
  setBusy(true);
  clearReport();
  renderImage();
  message('Preparing the inspection. The first scan may take a moment…');
  try {
    await serverReady;
    if (server && !state.back) throw new Error('Add the back photo to get a trained grade estimate.');
    const reports = [];
    for (const side of ['front', 'back']) {
      const item = state[side];
      if (!item) continue;
      message('Inspecting ' + side + '…');
      let result;
      if (server) {
        const blob = await new Promise(resolve => item.canvas.toBlob(resolve, 'image/png'));
        const params = new URLSearchParams({ side, original_height: item.originalHeight });
        if (item.crop) params.set('crop', JSON.stringify(item.crop));
        const response = await fetch('/api/analyze?' + params, {
          method: 'POST', headers: { 'Content-Type': 'image/png' }, body: blob
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || 'Inspection failed.');
        result = body;
      } else {
        const pixels = item.canvas.getContext('2d').getImageData(0, 0, item.canvas.width, item.canvas.height);
        result = await invoke('analyze', {
          width: item.canvas.width, height: item.canvas.height, side,
          original_height: item.originalHeight, crop: item.crop
        }, pixels.data.buffer);
      }
      item.result = result;
      reports.push(result);
    }
    if (server) {
      message('Estimating the grade from front and back details…');
      const params = new URLSearchParams({ front_bytes: state.front.file.size });
      if (state.front.crop) params.set('front_crop', JSON.stringify(state.front.crop));
      if (state.back.crop) params.set('back_crop', JSON.stringify(state.back.crop));
      const response = await fetch('/api/grade?' + params, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
        body: new Blob([state.front.file, state.back.file]), signal: AbortSignal.timeout(120000)
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Could not estimate the grade.');
      state.report = body;
      for (const report of body.reports) state[report.side].result = report;
    } else {
      state.report = await invoke('summarize', reports);
    }
    const invalid = reports.find(report => !report.valid);
    select(invalid?.side || (state[state.side] ? state.side : 'front'));
    renderReport();
    if (state.report.grade_range || Number.isFinite(state.report.predicted_grade)) {
      message('Review the circled areas. Select an issue to isolate it.');
      $('reportTitle').focus({ preventScroll: true });
    } else {
      const reasons = reports.flatMap(report => report.valid ? report.warnings : [report.reason]);
      message('No grade assigned. ' + [...new Set(reasons)].join('\n'));
      if (invalid?.bbox) $('cropControls').open = true;
    }
  } catch (error) {
    clearReport(false);
    renderImage();
    message(error.message);
  } finally {
    setBusy(false);
  }
}

function renderReport() {
  const report = state.report;
  const range = report?.grade_range;
  const trained = Number.isFinite(report?.predicted_grade);
  const visible = trained || (Array.isArray(range) && range.length === 2);
  $('resultsPanel').hidden = !visible;
  $('workspace').classList.toggle('has-report', visible);
  $('download').hidden = !visible;
  $('criteriaRows').replaceChildren();
  $('findings').replaceChildren();
  if (!visible) return;
  const valid = report.reports.filter(side => side.valid);
  $('gradeHeading').textContent = trained ? 'ESTIMATED GRADE' : 'HEURISTIC GRADE RANGE';
  $('grade').replaceChildren(document.createTextNode(trained ? report.predicted_grade.toFixed(1) : range.join('–')));
  const suffix = document.createElement('span');
  suffix.textContent = '/ 10';
  $('grade').append(suffix);
  $('gradeLabel').textContent = trained ? 'Trained model · front and back' : 'Rule-based inspection · trained model unavailable in browser-only mode';
  $('confidence').textContent = trained ? 'Experimental · not a certified grade' : 'Low confidence · heuristic';
  $('sideCount').textContent = report.coverage;
  $('reportStatus').textContent = 'ESTIMATE';
  for (const [key, title] of [['centering', 'Centering'], ['edges', 'Edges'], ['corners', 'Corners'], ['surface', 'Surface']]) {
    const row = document.createElement('div');
    row.className = 'criterion';
    const detail = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = title;
    const small = document.createElement('small');
    let score = 'Manual check';
    small.textContent = 'Not scored · inspect under angled light';
    if (key !== 'surface') {
      const values = valid.map(side => key === 'centering' ? side.centering?.score : side[key]).filter(value => value != null);
      score = values.length ? Math.min(...values).toFixed(1) + ' / 10' : 'Unmeasured';
      small.textContent = values.length ? 'Lowest measured side' : 'Could not measure reliably';
      if (key === 'centering' && values.length) {
        small.textContent = valid.filter(side => side.centering).map(side =>
          capitalize(side.side) + ': L/R ' + side.centering.left_right.join('/') + ' · T/B ' + side.centering.top_bottom.join('/')
        ).join(' | ');
      }
    }
    detail.append(strong, small);
    const scoreElement = document.createElement('span');
    scoreElement.className = 'score';
    scoreElement.textContent = score;
    row.append(detail, scoreElement);
    $('criteriaRows').append(row);
  }
  $('findings').hidden = false;
  const heading = document.createElement('h3');
  heading.textContent = 'Inspection notes';
  const notes = [];
  if (trained) notes.push('The overall estimate comes from the trained model. Criterion scores and orange markers are separate rule-based checks, not explanations of its prediction. Accuracy on everyday phone photos has not been independently established.');
  if (!state.back) notes.push('Add the back for a more complete inspection.');
  report.reports.forEach(side => side.warnings?.forEach(note => notes.push(capitalize(side.side) + ': ' + note)));
  notes.push('Green: card boundary. Dashed gold: inner border. Orange circles and labels: suspected whitening.');
  const list = document.createElement('ul');
  notes.forEach(note => {
    const li = document.createElement('li');
    li.textContent = note;
    list.append(li);
  });
  $('findings').append(heading, list);
}

// Form events. Every changed photo or boundary invalidates the previous grade.
$('frontTab').onclick = () => select('front');
$('backTab').onclick = () => select('back');
$('choose').onclick = $('replace').onclick = () => { $('file').value = ''; $('file').click(); };
$('file').onchange = event => addFile(event.target.files[0]);
$('remove').onclick = () => { if (state.side === 'front') clearReference(); state[state.side] = null; clearReport(); renderImage(); message(''); };
$('overlay').onchange = renderImage;
$('analyze').onclick = $('applyCrop').onclick = inspect;
for (const event of ['dragenter', 'dragover']) $('dropzone').addEventListener(event, e => {
  e.preventDefault();
  if (!state.busy) $('dropzone').classList.add('dragging');
});
for (const event of ['dragleave', 'drop']) $('dropzone').addEventListener(event, e => {
  e.preventDefault();
  $('dropzone').classList.remove('dragging');
});
$('dropzone').addEventListener('drop', event => addFile(event.dataTransfer.files[0]));
for (const key of ['L', 'T', 'R', 'B']) $('crop' + key).oninput = () => {
  const values = ['L', 'T', 'R', 'B'].map(name => Number($('crop' + name).value) / 100);
  if (values[2] - values[0] < 0.2 || values[3] - values[1] < 0.2) {
    clearReport(false);
    message('Keep the boundary at least 20% wide and tall.');
    return;
  }
  state[state.side].crop = values;
  // Keep only the old boundary for editing; never show stale defect locations.
  state[state.side].result = { bbox: values, regions: [] };
  clearReport(false);
  renderImage();
  message('Boundary changed. Reinspect to update the grade.');
};

// Help popup: hover, keyboard focus, touch, Escape and outside-click dismissal.
const help = $('photoHelp');
const helpButton = $('photoHelpToggle');
let pinnedHelp = false;
function showHelp(open) {
  $('photoHelpPopup').hidden = !open;
  helpButton.setAttribute('aria-expanded', String(open));
}
help.addEventListener('pointerenter', event => { if (event.pointerType === 'mouse') showHelp(true); });
help.addEventListener('pointerleave', () => { if (!pinnedHelp) showHelp(false); });
helpButton.addEventListener('focus', () => showHelp(true));
helpButton.onclick = () => { pinnedHelp = !pinnedHelp; showHelp(pinnedHelp); };
help.addEventListener('focusout', event => { if (!help.contains(event.relatedTarget) && !pinnedHelp) showHelp(false); });
document.addEventListener('pointerdown', event => {
  if (!help.contains(event.target)) { pinnedHelp = false; showHelp(false); }
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') { pinnedHelp = false; showHelp(false); }
});

$('download').onclick = () => {
  if (!state.report?.grade_range && !Number.isFinite(state.report?.predicted_grade)) return;
  const blob = new Blob([JSON.stringify({ ...state.report, reference_card: reference, created_at: new Date().toISOString() }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'gemlens-inspection.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
renderReport();

// Optional catalog lookup. A generation counter discards stale searches/selections.
let reference = null;
let catalogGeneration = 0;
let catalogSearch = null;
function clearReference() {
  catalogGeneration++;
  reference = null;
  $('catalogSelected').hidden = true;
  $('referenceImage').removeAttribute('src');
  $('catalogResults').replaceChildren();
  $('catalogMore').hidden = true;
  $('catalogStatus').textContent = '';
}
async function catalogRequest(path) {
  await serverReady;
  if (!server) throw new Error('Card lookup needs the Python server. You can still inspect your photo.');
  const response = await fetch(path, { signal: AbortSignal.timeout(15000) });
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Could not look up this card.');
  return body;
}
async function searchCatalog(page = 1) {
  const generation = ++catalogGeneration;
  if (page === 1) catalogSearch = { q: $('catalogQuery').value.trim(), language: $('catalogLanguage').value };
  if (!catalogSearch || catalogSearch.q.length < 2) return;
  const search = { ...catalogSearch };
  $('catalogStatus').textContent = 'Looking up cards…';
  $('catalogMore').hidden = true;
  $('catalogResults').replaceChildren();
  try {
    const result = await catalogRequest('/api/cards?' + new URLSearchParams({ ...search, page }));
    if (generation !== catalogGeneration) return;
    $('catalogStatus').textContent = result.cards.length ? 'Select the matching printing below.' : 'No physical cards found. Try another name or language.';
    for (const card of result.cards) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'catalog-card';
      if (card.thumbnail) {
        const img = document.createElement('img');
        img.src = card.thumbnail; img.alt = ''; img.loading = 'lazy'; img.referrerPolicy = 'no-referrer';
        img.onerror = () => { img.hidden = true; };
        button.append(img);
      }
      const label = document.createElement('span');
      label.textContent = `${card.name} · ${card.id}`;
      button.append(label);
      button.onclick = () => chooseReference(card.id, search.language);
      $('catalogResults').append(button);
    }
    $('catalogMore').hidden = !result.has_more;
    $('catalogMore').onclick = () => searchCatalog(page + 1);
  } catch (error) {
    if (generation === catalogGeneration) $('catalogStatus').textContent = error.name === 'TimeoutError' ? 'Lookup timed out. Please retry.' : error.message;
  }
}
async function chooseReference(id, language) {
  const generation = ++catalogGeneration;
  $('catalogStatus').textContent = 'Loading reference…';
  try {
    const card = await catalogRequest('/api/cards/' + encodeURIComponent(id) + '?' + new URLSearchParams({ language }));
    if (generation !== catalogGeneration) return;
    reference = { ...card, language, match_method: 'user_selected' };
    $('referenceName').textContent = `${card.name} · ${card.set || card.set_id || card.id} · #${card.number || card.id}`;
    $('referenceImage').hidden = !card.image;
    if (card.image) $('referenceImage').src = card.image;
    else $('referenceImage').removeAttribute('src');
    $('referenceImage').alt = card.name + ' catalog reference';
    $('referenceImage').onerror = () => { $('referenceImage').hidden = true; };
    $('catalogSelected').hidden = false;
    $('catalogResults').replaceChildren();
    $('catalogMore').hidden = true;
    $('catalogStatus').textContent = 'Reference selected. Confirm the set and artwork match your photo.';
  } catch (error) {
    if (generation === catalogGeneration) $('catalogStatus').textContent = error.message;
  }
}
$('catalogForm').onsubmit = event => { event.preventDefault(); searchCatalog(); };
$('catalogClear').onclick = clearReference;
