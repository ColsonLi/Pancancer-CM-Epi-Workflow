const byId = id => document.getElementById(id);
const canvas = byId('umap'), message = byId('plot-message');
const grouping = byId('grouping'), cohort = byId('cohort'), search = byId('legend-search');
const legend = byId('legend'), resetGroups = byId('show-all'), countText = byId('visible-count');
const number = new Intl.NumberFormat('en');
const selected = new Set();
const camera = {x: 0, y: 0, zoom: 1};
let data, plot, bounds, drag = null, scheduled = false;

function colorForSubtype(index, parent) {
  const hues = [217, 30, 153, 358, 282, 14, 326, 75, 190];
  const ring = (index % 7) - 3;
  const h = ((hues[parent] + ring * 10) % 360 + 360) % 360;
  const s = .62, l = .39 + .07 * ((index * 3) % 4);
  const a = s * Math.min(l, 1 - l);
  const f = n => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return '#' + [f(0), f(8), f(4)].map(value => value.toString(16).padStart(2, '0')).join('');
}

const rgb = hex => [1, 3, 5].map(k => parseInt(hex.slice(k, k + 2), 16));

function renderer(xy, size) {
  const gl = canvas.getContext('webgl', {alpha: false, antialias: false});
  if (!gl) {
    const ctx = canvas.getContext('2d', {alpha: false});
    if (!ctx) throw new Error('No drawing context available');
    let style;
    return {
      colors(value) { style = value; },
      draw(center, scale, pointSize) {
        const {width: w, height: h} = canvas;
        const frame = ctx.createImageData(w, h);
        const pixels = new Uint32Array(frame.data.buffer);
        pixels.fill(0xfffdfcfa);
        const radius = Math.max(0, Math.floor((pointSize - 1) / 2));
        for (let i = 0; i < size; i++) {
          const off = i * 4;
          if (!style[off + 3]) continue;
          const x = Math.round((xy[2 * i] - center.x) * scale + w / 2);
          const y = Math.round((center.y - xy[2 * i + 1]) * scale + h / 2);
          if (x < radius || y < radius || x >= w - radius || y >= h - radius) continue;
          const c = 0xff000000 | style[off + 2] << 16 | style[off + 1] << 8 | style[off];
          for (let dy = -radius; dy <= radius; dy++) {
            pixels.fill(c, (y + dy) * w + x - radius, (y + dy) * w + x + radius + 1);
          }
        }
        ctx.putImageData(frame, 0, 0);
      },
    };
  }
  const shader = (type, source) => {
    const item = gl.createShader(type);
    gl.shaderSource(item, source);
    gl.compileShader(item);
    if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
    return item;
  };
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, `
    attribute vec2 a_xy;
    attribute vec4 a_color;
    uniform vec2 u_center;
    uniform vec2 u_scale;
    uniform float u_size;
    varying vec4 v_color;
    void main() {
      gl_Position = vec4((a_xy - u_center) * u_scale, 0.0, 1.0);
      gl_PointSize = u_size;
      v_color = a_color;
    }
  `));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec4 v_color;
    void main() {
      if (v_color.a == 0.0 || length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
      gl_FragColor = v_color;
    }
  `));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  const pos = gl.createBuffer(), col = gl.createBuffer();
  const attr = (field, buffer, components, type, normalized) => {
    const loc = gl.getAttribLocation(program, field);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, components, type, normalized, 0, 0);
  };
  attr('a_xy', pos, 2, gl.FLOAT, false);
  gl.bufferData(gl.ARRAY_BUFFER, xy, gl.STATIC_DRAW);
  attr('a_color', col, 4, gl.UNSIGNED_BYTE, true);
  const uniforms = Object.fromEntries(['center', 'scale', 'size'].map(key => [key, gl.getUniformLocation(program, `u_${key}`)]));
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(250 / 255, 252 / 255, 253 / 255, 1);
  return {
    colors(value) {
      gl.bindBuffer(gl.ARRAY_BUFFER, col);
      gl.bufferData(gl.ARRAY_BUFFER, value, gl.DYNAMIC_DRAW);
    },
    draw(center, scale, pointSize) {
      const {width, height} = canvas;
      gl.viewport(0, 0, width, height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(uniforms.center, center.x, center.y);
      gl.uniform2f(uniforms.scale, 2 * scale / width, 2 * scale / height);
      gl.uniform1f(uniforms.size, pointSize);
      gl.drawArrays(gl.POINTS, 0, size);
    },
  };
}

function cssScale() {
  return .86 * Math.min(canvas.clientWidth / (bounds[1][0] - bounds[0][0]), canvas.clientHeight / (bounds[1][1] - bounds[0][1])) * camera.zoom;
}

function drawSoon() {
  if (!plot || scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(canvas.clientWidth * ratio), h = Math.round(canvas.clientHeight * ratio);
    if (!w || !h) return;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    plot.draw(camera, cssScale() * ratio, Math.min(5.7, 1.7 + .7 * Math.log2(camera.zoom)) * ratio);
  });
}

function resetCamera() {
  if (!bounds) return;
  camera.x = (bounds[0][0] + bounds[1][0]) / 2;
  camera.y = (bounds[0][1] + bounds[1][1]) / 2;
  camera.zoom = 1;
  drawSoon();
}

function zoom(factor, x = canvas.clientWidth / 2, y = canvas.clientHeight / 2) {
  if (!plot) return;
  const s = cssScale(), dx = x - canvas.clientWidth / 2, dy = y - canvas.clientHeight / 2;
  camera.zoom = Math.max(.6, Math.min(30, camera.zoom * factor));
  const t = cssScale();
  camera.x += dx / s - dx / t;
  camera.y -= dy / s - dy / t;
  drawSoon();
}

function pan(dx, dy) {
  if (!plot) return;
  const scale = cssScale();
  camera.x -= dx / scale; camera.y += dy / scale;
  drawSoon();
}

function updatePlot(keepLegend = false) {
  const field = data.fields[grouping.value];
  const labels = data.codes[grouping.value], statuses = data.codes.status;
  const status = cohort.value === 'all' ? -1 : data.fields.status.labels.indexOf(cohort.value);
  const counts = new Uint32Array(field.labels.length), colors = field.colors.map(rgb);
  const style = new Uint8Array(data.count * 4);
  let shown = 0;
  for (let i = 0; i < data.count; i++) {
    if (status >= 0 && statuses[i] !== status) continue;
    const code = labels[i];
    counts[code]++;
    if (selected.size && !selected.has(code)) continue;
    const color = colors[code], k = i * 4;
    style[k] = color[0]; style[k + 1] = color[1]; style[k + 2] = color[2]; style[k + 3] = 195;
    shown++;
  }
  plot.colors(style);
  data.groupCounts = counts;
  countText.textContent = `${number.format(shown)} of ${number.format(data.count)} cells shown`;
  message.hidden = shown !== 0;
  if (!shown) message.textContent = 'No cells match the current filters.';
  resetGroups.disabled = !selected.size;
  if (keepLegend) {
    for (const button of legend.querySelectorAll('button')) button.setAttribute('aria-pressed', selected.has(+button.dataset.code));
  } else renderLegend();
  drawSoon();
}

function renderLegend() {
  const field = data.fields[grouping.value];
  const term = search.value.trim().toLowerCase();
  const fragment = document.createDocumentFragment();
  field.labels.forEach((label, code) => {
    if (term && !label.toLowerCase().includes(term)) return;
    const item = document.createElement('button');
    item.type = 'button'; item.className = 'legend-item'; item.dataset.code = code;
    item.setAttribute('aria-pressed', selected.has(code));
    for (const [klass, text] of [['swatch', ''], ['legend-label', label], ['legend-count', number.format(data.groupCounts[code])]]) {
      const span = document.createElement('span');
      span.className = klass; span.textContent = text;
      if (klass === 'swatch') { span.style.backgroundColor = field.colors[code]; span.setAttribute('aria-hidden', 'true'); }
      item.append(span);
    }
    item.addEventListener('click', () => {
      if (selected.has(code)) selected.delete(code); else selected.add(code);
      updatePlot(true);
    });
    fragment.append(item);
  });
  if (!fragment.childNodes.length) {
    const empty = document.createElement('p');
    empty.className = 'secondary empty'; empty.textContent = 'No matching groups.'; fragment.append(empty);
  }
  legend.replaceChildren(fragment);
}

canvas.addEventListener('pointerdown', e => {
  if (!plot || drag !== null || e.button !== 0) return;
  drag = {id: e.pointerId, x: e.clientX, y: e.clientY};
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', e => {
  if (drag?.id !== e.pointerId) return;
  pan(e.clientX - drag.x, e.clientY - drag.y);
  drag.x = e.clientX; drag.y = e.clientY;
});
for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(type, e => {
  if (drag?.id === e.pointerId) drag = null;
});
canvas.addEventListener('wheel', e => {
  if (!plot) return;
  e.preventDefault();
  const box = canvas.getBoundingClientRect();
  zoom(Math.exp(-.0016 * Math.max(-240, Math.min(240, e.deltaY * (e.deltaMode === 1 ? 15 : 1)))), e.clientX - box.left, e.clientY - box.top);
}, {passive: false});
canvas.addEventListener('keydown', e => {
  const steps = {ArrowLeft: [-28, 0], ArrowRight: [28, 0], ArrowUp: [0, -28], ArrowDown: [0, 28]};
  if (steps[e.key]) pan(...steps[e.key]);
  else if (['+', '='].includes(e.key)) zoom(1.35);
  else if (e.key === '-') zoom(1 / 1.35);
  else if (e.key === '0') resetCamera();
  else return;
  e.preventDefault();
});
byId('zoom-in').addEventListener('click', () => zoom(1.35));
byId('zoom-out').addEventListener('click', () => zoom(1 / 1.35));
byId('reset').addEventListener('click', resetCamera);
new ResizeObserver(drawSoon).observe(canvas);

async function init() {
  const files = await Promise.all([fetch('./atlas.json'), fetch('./atlas.bin')]);
  if (files.some(r => !r.ok)) throw new Error('The atlas data could not be loaded');
  const [manifest, buffer] = await Promise.all([files[0].json(), files[1].arrayBuffer()]);
  const n = manifest.count, length = n * (8 + manifest.fieldOrder.length);
  if (buffer.byteLength !== length) throw new Error('The atlas data were incomplete');
  bounds = manifest.bounds;
  manifest.fields.cell_subtype.colors = manifest.fields.cell_subtype.parents.map((parent, i) => colorForSubtype(i, parent));
  data = {...manifest, codes: Object.fromEntries(manifest.fieldOrder.map((name, i) => [name, new Uint8Array(buffer, (8 + i) * n, n)]))};
  plot = renderer(new Float32Array(buffer, 0, 2 * n), n);
  updatePlot();
  resetCamera();
  for (const element of [grouping, cohort, search]) element.disabled = false;
  grouping.addEventListener('change', () => { selected.clear(); search.value = ''; updatePlot(); });
  cohort.addEventListener('change', () => updatePlot());
  resetGroups.addEventListener('click', () => { selected.clear(); updatePlot(true); });
  search.addEventListener('input', renderLegend);
}

init().catch(error => {
  console.error(error);
  message.hidden = false;
  message.textContent = 'The atlas could not start. Refresh the page or try another browser.';
  countText.textContent = 'Atlas unavailable';
});
