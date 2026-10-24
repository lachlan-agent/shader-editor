// Graph: a small graphing calculator in a view window. Each row is a
// math.js expression in x (`sin(x) * fract(x)`, `y = …`) or a constant
// binding (`a = 0.5`) that the rows below can use; curves are drawn on a 2D
// canvas with pan, zoom and a hover readout. Same syntax and functions as
// the calculator, evaluated in doubles (see getPlotMath) so a thousand
// samples per curve per frame stay cheap.

import { getPlotMath, libHas, type Math } from './mathlib';
import type { ScratchHost, ScratchPad } from './scratch';

interface GraphView { cx: number; cy: number; kx: number; ky: number }   // centre in units, px per unit
interface GraphState { rows: string[]; view: GraphView; lock: boolean }

const DEFAULT_VIEW = (): GraphView => ({ cx: 0, cy: 0, kx: 40, ky: 40 });
const MAX_ROWS = 12;
const MAX_DPR = 2;
const ASSIGN = /^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.*)$/s;
const CURVE_VARS = new Set(['x', 'y']);
// Curve colours cycle through theme variables so every theme reads well.
const CURVE_COLORS = ['--accent', '--ok', '--warn', '--err', '--accent2', '--text'];

type Compiled = { evaluate(scope: Record<string, unknown>): unknown };

interface Row {
  src: string;
  el: HTMLElement;
  input: HTMLInputElement;
  swatch: HTMLElement;
  value: HTMLElement;
  // Set by `rebuild`: a curve to draw, a constant, or an error.
  curve: Compiled | null;
  name: string | null;
  error: string | null;
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** Tick spacing in units: a 1-2-5 step near `px` pixels apart at scale k. */
function niceStep(k: number, px: number): number {
  const raw = px / k;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  return (r <= 1 ? 1 : r <= 2 ? 2 : r <= 5 ? 5 : 10) * mag;
}

/** Axis label: short, no float noise. */
function label(v: number, step: number): string {
  if (Math.abs(v) < step * 1e-9) return '0';
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  const s = v.toFixed(Math.min(digits, 12));
  return Math.abs(v) >= 1e6 || Math.abs(v) < 1e-4 ? v.toExponential(digits > 0 ? 2 : 0).replace('e+', 'e') : s;
}

function readout(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  return a !== 0 && (a >= 1e7 || a < 1e-4) ? v.toExponential(5) : String(+v.toPrecision(7));
}

export function makeGraph(host: ScratchHost): ScratchPad {
  const el = document.createElement('div');
  el.className = 'pad pad-graph';
  el.innerHTML = `
    <div class="graph-rows"></div>
    <div class="graph-plot">
      <canvas></canvas>
      <div class="graph-hud" hidden></div>
      <div class="graph-tools">
        <button class="float-btn graph-lock" title="Lock a 1:1 aspect ratio (same scale on both axes)">1:1</button>
        <button class="float-btn graph-reset" title="Reset the view (double-click the graph also resets)">⌂</button>
      </div>
    </div>`;
  const rowsEl = el.querySelector('.graph-rows') as HTMLElement;
  const plot = el.querySelector('.graph-plot') as HTMLElement;
  const cv = el.querySelector('canvas') as HTMLCanvasElement;
  const hud = el.querySelector('.graph-hud') as HTMLElement;
  const lockBtn = el.querySelector('.graph-lock') as HTMLButtonElement;
  const resetBtn = el.querySelector('.graph-reset') as HTMLButtonElement;
  const ctx = cv.getContext('2d')!;

  /* ── state ── */
  let state: GraphState = { rows: ['sin(x)'], view: DEFAULT_VIEW(), lock: true };
  try {
    const p = JSON.parse(host.data || '{}');
    if (Array.isArray(p.rows)) {
      const v = p.view ?? {};
      const num = (x: unknown, d: number) => (typeof x === 'number' && Number.isFinite(x) ? x : d);
      state = {
        rows: p.rows.filter((r: unknown) => typeof r === 'string').slice(0, MAX_ROWS),
        view: { cx: num(v.cx, 0), cy: num(v.cy, 0), kx: Math.max(1e-300, num(v.kx, 40)), ky: Math.max(1e-300, num(v.ky, 40)) },
        lock: p.lock !== false,
      };
    }
  } catch { /* fresh */ }
  if (!state.rows.length) state.rows = [''];

  function persist(): void {
    host.onData(JSON.stringify(state));
  }

  /* ── rows ── */
  const rows: Row[] = [];
  let math: Math | null = null;
  const ready = getPlotMath().then(m => { math = m; rebuild(); });
  ready.catch(e => { hud.hidden = false; hud.textContent = `math.js failed to load: ${(e as Error).message}`; });

  function makeRow(src: string): Row {
    const r = document.createElement('div');
    r.className = 'graph-row';
    r.innerHTML = `
      <span class="graph-swatch"></span>
      <input type="text" spellcheck="false" autocomplete="off" placeholder="y = f(x)   or   a = 0.5">
      <span class="graph-val"></span>
      <button class="float-btn graph-del" title="Remove">✕</button>`;
    const row: Row = {
      src, el: r,
      input: r.querySelector('input') as HTMLInputElement,
      swatch: r.querySelector('.graph-swatch') as HTMLElement,
      value: r.querySelector('.graph-val') as HTMLElement,
      curve: null, name: null, error: null,
    };
    row.input.value = src;
    row.input.addEventListener('input', () => {
      row.src = row.input.value;
      state.rows = rows.map(x => x.src);
      rebuild();
      persist();
    });
    row.input.addEventListener('keydown', (e: KeyboardEvent) => {
      const i = rows.indexOf(row);
      if (e.key === 'Enter') {
        // Enter on the last row adds one; elsewhere it moves down.
        if (i === rows.length - 1) addRow('', true); else rows[i + 1].input.focus();
        e.preventDefault();
      } else if (e.key === 'ArrowDown' && i < rows.length - 1) { rows[i + 1].input.focus(); e.preventDefault(); }
      else if (e.key === 'ArrowUp' && i > 0) { rows[i - 1].input.focus(); e.preventDefault(); }
      else if (e.key === 'Backspace' && !row.input.value && rows.length > 1) { removeRow(row); rows[Math.max(0, i - 1)].input.focus(); e.preventDefault(); }
      e.stopPropagation();
    });
    (r.querySelector('.graph-del') as HTMLButtonElement).addEventListener('click', () => removeRow(row));
    return row;
  }

  function addRow(src: string, focus = false): void {
    if (rows.length >= MAX_ROWS) return;
    const row = makeRow(src);
    rows.push(row);
    rowsEl.insertBefore(row.el, addBtn);
    state.rows = rows.map(x => x.src);
    if (focus) row.input.focus();
    rebuild();
  }

  function removeRow(row: Row): void {
    if (rows.length === 1) { row.input.value = ''; row.src = ''; }
    else { rows.splice(rows.indexOf(row), 1); row.el.remove(); }
    state.rows = rows.map(x => x.src);
    rebuild();
    persist();
  }

  const addBtn = document.createElement('button');
  addBtn.className = 'float-btn graph-add';
  addBtn.title = 'Add a row (Enter on the last row also adds one)';
  addBtn.textContent = '+';
  addBtn.addEventListener('click', () => addRow('', true));
  rowsEl.appendChild(addBtn);
  for (const src of state.rows) addRow(src);

  /** Does the expression mention x (so it is a curve, not a constant)? */
  function usesX(node: import('mathjs').MathNode): boolean {
    let found = false;
    node.traverse(n => { if (math!.isSymbolNode(n) && n.name === 'x') found = true; });
    return found;
  }

  /** Names math.js would otherwise read as units (t, m…) must be bound or built in. */
  function checkSymbols(node: import('mathjs').MathNode, scope: Record<string, unknown>): void {
    node.traverse(n => {
      if (!math!.isSymbolNode(n)) return;
      if (CURVE_VARS.has(n.name) || n.name in scope || libHas(math!, n.name)) return;
      throw new Error(`unknown ${n.name}`);
    });
  }

  // Constants are evaluated once here; curves are compiled for `sample`.
  const scope: Record<string, unknown> = {};
  let curveCount = 0;

  function rebuild(): void {
    if (!math) return;
    for (const k of Object.keys(scope)) delete scope[k];
    curveCount = 0;
    for (const row of rows) {
      row.curve = null; row.name = null; row.error = null;
      row.value.textContent = '';
      row.el.classList.remove('bad', 'curve', 'const');
      row.swatch.style.background = '';
      const src = row.src.trim();
      if (!src) { row.el.title = ''; continue; }
      try {
        const m = ASSIGN.exec(src);
        let name = m ? m[1] : null;
        let expr = m ? m[2] : src;
        if (name === 'y') name = null;
        if (!expr.trim()) { row.el.title = ''; row.value.textContent = name ? '= ?' : ''; continue; }   // `y =` with nothing after it
        if (name && (name === 'x' || typeof (math as unknown as Record<string, unknown>)[name] === 'function')) {
          throw new Error(`${name} is built in and cannot be assigned`);
        }
        const node = math.parse(expr);
        checkSymbols(node, scope);
        const compiled = node.compile();
        if (usesX(node)) {
          row.curve = compiled;
          row.name = name;
          row.swatch.style.background = `var(${CURVE_COLORS[curveCount % CURVE_COLORS.length]})`;
          curveCount++;
          row.el.classList.add('curve');
        } else {
          const v = compiled.evaluate(scope);
          if (name) scope[name] = v;
          row.el.classList.add('const');
          row.value.textContent = (name ? '= ' : '') + (isFiniteNumber(v) ? readout(v) : math.format(v, { precision: 7 }));
          if (!name && isFiniteNumber(v)) {
            // A bare number is a horizontal line.
            row.curve = { evaluate: () => v };
            row.swatch.style.background = `var(${CURVE_COLORS[curveCount % CURVE_COLORS.length]})`;
            curveCount++;
            row.el.classList.add('curve');
          }
        }
        row.el.title = '';
      } catch (e) {
        row.error = (e as Error).message;
        row.el.classList.add('bad');
        row.el.title = row.error;
      }
    }
    draw();
  }

  /* ── view ── */
  let W = 0, H = 0;   // CSS px
  const v = state.view;
  const toPx = (x: number, y: number): [number, number] => [W / 2 + (x - v.cx) * v.kx, H / 2 - (y - v.cy) * v.ky];
  const toUnit = (px: number, py: number): [number, number] => [v.cx + (px - W / 2) / v.kx, v.cy - (py - H / 2) / v.ky];

  function setLock(on: boolean): void {
    state.lock = on;
    lockBtn.classList.toggle('active', on);
    if (on) { const k = Math.sqrt(v.kx * v.ky); v.kx = v.ky = k; }
  }
  setLock(state.lock);
  lockBtn.addEventListener('click', () => { setLock(!state.lock); draw(); persist(); });
  resetBtn.addEventListener('click', () => { Object.assign(v, DEFAULT_VIEW()); draw(); persist(); });

  /** Zoom by fx, fy about a screen point. */
  function zoomAt(px: number, py: number, fx: number, fy: number): void {
    if (state.lock) fx = fy = Math.sqrt(fx * fy);
    const [ux, uy] = toUnit(px, py);
    v.kx = Math.min(Math.max(v.kx * fx, 1e-300), 1e300);
    v.ky = Math.min(Math.max(v.ky * fy, 1e-300), 1e300);
    // Keep the unit under the pointer where it was.
    v.cx = ux - (px - W / 2) / v.kx;
    v.cy = uy + (py - H / 2) / v.ky;
  }

  /* ── drawing ── */
  const css = (name: string): string => getComputedStyle(el).getPropertyValue(name).trim();
  let frame = 0;
  function draw(): void {
    if (frame) return;
    frame = requestAnimationFrame(() => { frame = 0; paint(); });
  }

  // Each curve is sampled once per CSS pixel column, the samples kept for the hover readout.
  const samples: Array<{ row: Row; ys: Float64Array; color: string }> = [];

  function paint(): void {
    const dpr = Math.min(devicePixelRatio, MAX_DPR);
    W = cv.clientWidth; H = cv.clientHeight;
    if (W === 0 || H === 0) return;
    const pw = Math.max(1, W * dpr | 0), ph = Math.max(1, H * dpr | 0);
    if (cv.width !== pw || cv.height !== ph) { cv.width = pw; cv.height = ph; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const colText = css('--text-dim'), colGrid = css('--border'), colAxis = css('--text');
    const curveColors = CURVE_COLORS.map(css);
    ctx.font = '10px ' + (css('font-family') || 'monospace');

    // Grid: major lines at a 1-2-5 step, minor lines at a fifth of it.
    const sx = niceStep(v.kx, 80), sy = niceStep(v.ky, 80);
    const [x0, y1] = toUnit(0, 0), [x1, y0] = toUnit(W, H);
    ctx.lineWidth = 1;
    ctx.strokeStyle = colGrid;
    // Ticks are walked by integer index, not by `x += step`: far from the
    // origin at a deep zoom the step falls below the ULP of x and adding it
    // would never move x, so that loop never ends. The count is bounded too,
    // in case the view is ever degenerate.
    const ticks = (lo: number, hi: number, step: number, fn: (value: number) => void): void => {
      if (!(step > 0)) return;
      const i0 = Math.ceil(lo / step), i1 = Math.floor(hi / step);
      if (!(i1 - i0 < 10000)) return;
      for (let i = i0; i <= i1; i++) fn(i * step);
    };
    const gridLines = (stepX: number, stepY: number): void => {
      ctx.beginPath();
      ticks(x0, x1, stepX, x => { const px = (toPx(x, 0)[0] | 0) + 0.5; ctx.moveTo(px, 0); ctx.lineTo(px, H); });
      ticks(y0, y1, stepY, y => { const py = (toPx(0, y)[1] | 0) + 0.5; ctx.moveTo(0, py); ctx.lineTo(W, py); });
      ctx.stroke();
    };
    ctx.globalAlpha = 0.4;
    gridLines(sx / 5 * v.kx >= 10 ? sx / 5 : 0, sy / 5 * v.ky >= 10 ? sy / 5 : 0);
    ctx.globalAlpha = 1;
    gridLines(sx, sy);

    // Axes, with labels that stay on screen when the axis is off it.
    const [ax, ay] = toPx(0, 0);
    ctx.strokeStyle = colAxis; ctx.globalAlpha = 0.5;
    ctx.beginPath();
    if (ax >= 0 && ax <= W) { ctx.moveTo((ax | 0) + 0.5, 0); ctx.lineTo((ax | 0) + 0.5, H); }
    if (ay >= 0 && ay <= H) { ctx.moveTo(0, (ay | 0) + 0.5); ctx.lineTo(W, (ay | 0) + 0.5); }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = colText;
    const labelY = Math.min(Math.max(ay, 12), H - 4) + (ay < H - 16 ? 12 : -4);
    const labelX = Math.min(Math.max(ax, 4), W - 30) + 4;
    ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ticks(x0, x1, sx, x => {
      if (Math.abs(x) < sx * 1e-9) return;
      const [px] = toPx(x, 0);
      ctx.fillText(label(x, sx), px, labelY);
    });
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ticks(y0, y1, sy, y => {
      if (Math.abs(y) < sy * 1e-9) return;
      const [, py] = toPx(0, y);
      ctx.fillText(label(y, sy), labelX, py);
    });

    // Curves.
    samples.length = 0;
    if (!math) return;
    ctx.lineWidth = 1.5; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    let ci = 0;
    for (const row of rows) {
      if (!row.curve) continue;
      const color = curveColors[ci % curveColors.length]; ci++;
      const ys = new Float64Array(W + 1);
      const eval1 = (x: number): number => {
        scope.x = x;
        try { const r = row.curve!.evaluate(scope); return isFiniteNumber(r) ? r : NaN; } catch { return NaN; }
      };
      for (let px = 0; px <= W; px++) ys[px] = eval1(toUnit(px, 0)[0]);
      samples.push({ row, ys, color });

      ctx.strokeStyle = color;
      ctx.beginPath();
      let pen = false;
      let prevPy = NaN;
      for (let px = 0; px <= W; px++) {
        const y = ys[px];
        if (!Number.isFinite(y)) { pen = false; continue; }
        let py = H / 2 - (y - v.cy) * v.ky;
        // A tall jump is a discontinuity unless the midpoint lies between its ends
        // (a steep line passes that test; fract, tan and 1/x fail it).
        if (pen && Math.abs(py - prevPy) > 20) {
          const ym = eval1(toUnit(px - 0.5, 0)[0]);
          const prevY = ys[px - 1];
          const between = Number.isFinite(ym) && ym >= Math.min(prevY, y) && ym <= Math.max(prevY, y);
          if (!between) pen = false;
        }
        // Clamp far-off points so the canvas never sees absurd coordinates.
        py = Math.min(Math.max(py, -2 * H), 3 * H);
        if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
        prevPy = py;
      }
      ctx.stroke();
    }
    if (hover) drawHover();
  }

  /* ── hover readout ── */
  let hover: [number, number] | null = null;
  function drawHover(): void {
    if (!hover || !samples.length) { hud.hidden = true; return; }
    const [px] = hover;
    const col = Math.min(Math.max(Math.round(px), 0), W);
    const [x] = toUnit(col, 0);
    ctx.save();
    ctx.strokeStyle = css('--text-dim'); ctx.globalAlpha = 0.5; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(col + 0.5, 0); ctx.lineTo(col + 0.5, H); ctx.stroke();
    ctx.restore();
    const lines = [`x = ${readout(x)}`];
    for (const s of samples) {
      const y = s.ys[col];
      if (!Number.isFinite(y)) { lines.push(`${s.row.name ?? 'y'} = —`); continue; }
      const [, py] = toPx(0, y);
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(col, py, 3, 0, Math.PI * 2); ctx.fill();
      lines.push(`${s.row.name ?? 'y'} = ${readout(y)}`);
    }
    hud.hidden = false;
    hud.textContent = lines.join('\n');
    // Keep the readout out from under the pointer.
    hud.classList.toggle('left', px > W / 2);
  }

  /* ── gestures: drag to pan, wheel to zoom (Shift: x only, Ctrl: y only), pinch, double-click to reset ── */
  const pointers = new Map<number, [number, number]>();
  let dragged = false;
  let pinch: { d: number; cx: number; cy: number } | null = null;
  const pos = (e: PointerEvent): [number, number] => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

  cv.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    cv.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, pos(e));
    dragged = false;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), cx: (a[0] + b[0]) / 2, cy: (a[1] + b[1]) / 2 };
    }
    e.preventDefault();
  });
  cv.addEventListener('pointermove', e => {
    const p = pos(e);
    if (!pointers.has(e.pointerId)) {
      if (e.pointerType === 'mouse') { hover = p; draw(); }
      return;
    }
    const prev = pointers.get(e.pointerId)!;
    pointers.set(e.pointerId, p);
    if (pointers.size === 2 && pinch) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      const cx = (a[0] + b[0]) / 2, cy = (a[1] + b[1]) / 2;
      if (pinch.d > 0 && d > 0) zoomAt(cx, cy, d / pinch.d, d / pinch.d);
      v.cx -= (cx - pinch.cx) / v.kx; v.cy += (cy - pinch.cy) / v.ky;
      pinch = { d, cx, cy };
      dragged = true;
    } else if (pointers.size === 1) {
      const dx = p[0] - prev[0], dy = p[1] - prev[1];
      if (dragged || Math.hypot(dx, dy) > 2) {
        dragged = true;
        v.cx -= dx / v.kx; v.cy += dy / v.ky;
        hover = e.pointerType === 'mouse' ? p : null;
      }
    }
    draw();
  });
  const up = (e: PointerEvent) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);
    pinch = null;
    if (dragged) persist();
    else if (e.pointerType !== 'mouse') { hover = hover ? null : pos(e); draw(); }   // tap toggles the readout
  };
  cv.addEventListener('pointerup', up);
  cv.addEventListener('pointercancel', up);
  cv.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse' && !pointers.size) { hover = null; draw(); } });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const f = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.002));
    const [px, py] = pos(e as unknown as PointerEvent);
    if (e.shiftKey) { setLock(false); zoomAt(px, py, f, 1); }
    else if (e.ctrlKey) { setLock(false); zoomAt(px, py, 1, f); }
    else zoomAt(px, py, f, f);
    hover = [px, py];
    draw();
    persist();
  }, { passive: false });
  cv.addEventListener('dblclick', () => { Object.assign(v, DEFAULT_VIEW()); draw(); persist(); });
  cv.addEventListener('contextmenu', e => e.preventDefault());

  /* ── lifecycle ── */
  const ro = new ResizeObserver(() => draw());
  ro.observe(plot);
  // Themes set CSS variables on <html>; repaint when they change.
  const mo = new MutationObserver(() => draw());
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['style'] });
  el.addEventListener('click', e => { if (e.target === rowsEl) rows[rows.length - 1]?.input.focus(); });

  return {
    el,
    destroy() {
      ro.disconnect(); mo.disconnect();
      if (frame) cancelAnimationFrame(frame);
      el.remove();
    },
  };
}
