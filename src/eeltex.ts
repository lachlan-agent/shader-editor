// A LaTeX maths pad driven by eeltex (github.com/ParticleCat314/eeltex) compiled
// to WebAssembly. The pad is a list of formula lines. The wasm module keeps one
// editor per instance, so a pad runs a single instance: the active line lives
// in it (edited MathQuill-style, or through the LaTeX field underneath) and the
// other lines are shown from their last rendered SVG.
//
// Keys: Enter new line below (Shift+Enter a new matrix row) · Backspace on an empty line removes it ·
// ↑/↓ cross lines once the formula has nowhere further to go.

import type { ScratchHost, ScratchPad } from './scratch';
import { evaluateTex, toTex, trivialTex } from './mathcalc';
import { getMath, type Math, type Scope } from './mathlib';

interface Exports {
  memory: WebAssembly.Memory;
  alloc(len: number): number;
  init(fontSize: number): number;
  key(code: number, shift: number, word: number): number;
  insert(ch: number): number;
  paste(): number;
  set_tex(): number;
  click(x: number, y: number, shift: number): number;
  select_at(x: number, y: number): number;
  copy(): number;
  out_ptr(which: number): number;
  out_len(which: number): number;
}

interface Info {
  tex: string;
  error: string | null;
  pending: string | null;
  unknown: string[];
  caret: [number, number, number, number] | null;
  selection: [number, number, number, number] | null;
}

const KEY_CODES: Record<string, number> = {
  ArrowLeft: 1, ArrowRight: 2, ArrowUp: 3, ArrowDown: 4, Home: 5, End: 6,
  Backspace: 7, Delete: 8, Escape: 12, Enter: 13,
};
const FONT_SIZE = 30;
const SVG_NS = 'http://www.w3.org/2000/svg';

let modulePromise: Promise<WebAssembly.Module> | null = null;

function loadModule(): Promise<WebAssembly.Module> {
  if (!modulePromise) {
    const url = `${import.meta.env.BASE_URL}eeltex.wasm`;
    modulePromise = (WebAssembly.compileStreaming
      ? WebAssembly.compileStreaming(fetch(url))
      : fetch(url).then(r => r.arrayBuffer()).then(b => WebAssembly.compile(b))
    ).catch(e => { modulePromise = null; throw e; });
  }
  return modulePromise;
}

const encoder = new TextEncoder(), decoder = new TextDecoder();

/** Saved form: JSON `{ lines: string[] }`; a bare string is one line from older saves. */
function parseLines(data: string): string[] {
  if (!data) return [''];
  try {
    const p = JSON.parse(data);
    if (p && Array.isArray(p.lines) && p.lines.every((l: unknown) => typeof l === 'string')) {
      return p.lines.length ? p.lines : [''];
    }
  } catch { /* not JSON: a single formula */ }
  return [data];
}

export interface MathPadOptions {
  /** Evaluate each line (scalars and matrices) and render its result beside it. */
  evaluate?: boolean;
  /** Hide the LaTeX source field; the formula is edited in place only. */
  noSource?: boolean;
}

/** Instantiate the shared module; `env.now` is the only import. */
async function instantiate(): Promise<Exports> {
  const mod = await loadModule();
  const { exports } = await WebAssembly.instantiate(mod, { env: { now: () => performance.now() } });
  return exports as unknown as Exports;
}

export function makeMathPad(host: ScratchHost, opts: MathPadOptions = {}): ScratchPad {
  const el = document.createElement('div');
  el.className = 'pad pad-math' + (opts.evaluate ? ' pad-texcalc' : '') + (opts.noSource ? ' no-source' : '');
  el.innerHTML = `
    <div class="math-stage" tabindex="-1">
      <div class="math-loading">loading eeltex…</div>
      <div class="math-lines"></div>
      <textarea class="math-keys" autocapitalize="off" autocomplete="off" spellcheck="false" aria-label="Type maths"></textarea>
    </div>
    <div class="math-status"></div>
    <textarea class="math-src" rows="1" spellcheck="false" autocomplete="off" autocapitalize="off" placeholder="LaTeX for the selected line"></textarea>`;
  const stage  = el.querySelector('.math-stage')  as HTMLElement;
  const list   = el.querySelector('.math-lines')  as HTMLElement;
  const keys   = el.querySelector('.math-keys')   as HTMLTextAreaElement;
  const status = el.querySelector('.math-status') as HTMLElement;
  const src    = el.querySelector('.math-src')    as HTMLTextAreaElement;

  let wasm: Exports | null = null;
  let info: Info | null = null;
  let alive = true;

  // Results get their own instance so rendering them never disturbs the editor.
  let renderer: Exports | null = null;
  let math: Math | null = null;
  const resultSvgs = new Map<string, string>();
  function renderTex(tex: string): string {
    if (!renderer) return '';
    const cached = resultSvgs.get(tex);
    if (cached) return cached;
    const bytes = encoder.encode(tex);
    const ptr = renderer.alloc(bytes.length);
    new Uint8Array(renderer.memory.buffer, ptr, bytes.length).set(bytes);
    renderer.set_tex();
    const svg = decoder.decode(new Uint8Array(renderer.memory.buffer, renderer.out_ptr(0), renderer.out_len(0)));
    if (resultSvgs.size > 200) resultSvgs.clear();
    resultSvgs.set(tex, svg);
    return svg;
  }

  const lines = parseLines(host.data);
  const svgs: string[] = lines.map(() => '');
  const rows: HTMLElement[] = [];
  let active = 0;

  /* ── wasm boundary ── */
  function writeInput(text: string): void {
    const bytes = encoder.encode(text);
    const ptr = wasm!.alloc(bytes.length);
    new Uint8Array(wasm!.memory.buffer, ptr, bytes.length).set(bytes);
  }
  function readOutput(which: number): string {
    const ptr = wasm!.out_ptr(which), len = wasm!.out_len(which);
    return decoder.decode(new Uint8Array(wasm!.memory.buffer, ptr, len));
  }
  function call(fn: () => number): boolean {
    if (!wasm) return false;
    const changed = !!fn();
    if (changed) refresh();
    return changed;
  }

  /* ── lines ── */
  const activeSvg = (): SVGSVGElement | null => rows[active]?.querySelector(':scope > .math-formula > svg') ?? null;

  function fitSvg(svg: SVGSVGElement, room: number, scale = 1): void {
    const w = parseFloat(svg.getAttribute('width') ?? '0') * scale;
    const h = parseFloat(svg.getAttribute('height') ?? '0') * scale;
    const s = Math.max(0.05, Math.min(1, w > 0 ? room / w : 1));
    svg.style.width = `${w * s}px`;
    svg.style.height = `${h * s}px`;
  }
  function fitRow(row: HTMLElement): void {
    const formula = row.querySelector<SVGSVGElement>(':scope > .math-formula > svg');
    const out = row.querySelector<HTMLElement>(':scope > .math-result');
    const result = out?.querySelector<SVGSVGElement>(':scope > svg') ?? null;
    const cs = getComputedStyle(row);
    const room = row.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    // The result box is padding + "=" + gap wider than its SVG (see the CSS).
    const chrome = out ? out.getBoundingClientRect().width - (result?.getBoundingClientRect().width ?? 0) : 0;
    if (result) fitSvg(result, room * 0.55 - chrome, 0.8);
    // An error message has no SVG to scale: it keeps a third of the row and wraps.
    const used = result ? result.getBoundingClientRect().width + chrome + 6 : out ? Math.min(out.offsetWidth, room / 3) + 14 : 0;
    if (formula) fitSvg(formula, Math.max(40, room - used));
  }

  function setRowSvg(i: number, svg: string): void {
    svgs[i] = svg;
    const row = rows[i];
    const holder = row.querySelector('.math-formula') as HTMLElement;
    holder.innerHTML = svg;
    holder.querySelector('svg')?.setAttribute('aria-hidden', 'true');
    row.classList.toggle('empty', lines[i].trim() === '');
    fitRow(row);
  }

  /** Re-evaluate every line in order (names defined above are visible below). */
  function evaluateAll(): void {
    if (!opts.evaluate || !renderer || !math) return;
    const env: Scope = new Map();
    lines.forEach((tex, i) => {
      const row = rows[i];
      let out = row.querySelector('.math-result') as HTMLElement | null;
      if (!tex.trim() || trivialTex(tex)) { out?.remove(); fitRow(row); return; }
      if (!out) { out = document.createElement('div'); out.className = 'math-result'; row.appendChild(out); }
      try {
        const r = evaluateTex(math!, tex, env);
        const resultTex = toTex(math!, r.value);
        if (out.dataset.tex !== resultTex) {
          out.dataset.tex = resultTex;
          out.classList.remove('err');
          out.innerHTML = `<span class="eq">=</span>${renderTex(resultTex)}`;
          out.querySelector('svg')?.setAttribute('aria-hidden', 'true');
        }
      } catch (e) {
        // Lines that are not arithmetic (prose maths, free symbols) show nothing.
        const msg = (e as Error).message;
        const quiet = /^unknown |^Undefined symbol|^Undefined function|unexpected end|expected a name/.test(msg);
        if (quiet || i !== active) { out.remove(); fitRow(row); return; }
        out.dataset.tex = '';
        out.className = 'math-result err';
        out.textContent = msg;
      }
      fitRow(row);
    });
  }

  function makeRow(): HTMLElement {
    const row = document.createElement('div');
    row.className = 'math-line';
    row.innerHTML = '<div class="math-formula"></div>';
    return row;
  }
  const renumber = (): void => rows.forEach((r, i) => { r.dataset.i = String(i); });

  function buildRows(): void {
    list.innerHTML = '';
    rows.length = 0;
    lines.forEach((_, i) => {
      const row = makeRow();
      list.appendChild(row);
      rows.push(row);
      if (svgs[i]) setRowSvg(i, svgs[i]);
    });
    renumber();
    markActive();
    evaluateAll();
  }

  function markActive(): void {
    // The key sink lives inside the active row: typing scrolls the caret into
    // view, so this keeps the browser scrolling to the line being edited
    // rather than to the top of the stage.
    if (rows[active] && keys.parentElement !== rows[active]) {
      const hadFocus = document.activeElement === keys;   // moving a node blurs it
      rows[active].appendChild(keys);
      if (hadFocus) focusKeys();
    }
    rows[active]?.scrollIntoView({ block: 'nearest' });
    rows.forEach((r, i) => {
      r.classList.toggle('active', i === active);
      // A line that just lost focus keeps its rendered SVG but not the caret/selection.
      if (i !== active) r.querySelectorAll(':scope > .math-formula .ov').forEach(g => g.remove());
    });
  }

  function persist(): void {
    host.onData(JSON.stringify({ lines }));
  }

  /** Put line `i` into the editor; its SVG and info then come from `refresh`. */
  function loadLine(i: number): void {
    active = Math.max(0, Math.min(i, lines.length - 1));
    markActive();
    if (!wasm) return;
    writeInput(lines[active]);
    wasm.set_tex();
    refresh();
  }

  function rect(parent: Element, cls: string, x: number, y: number, w: number, h: number): void {
    const r = document.createElementNS(SVG_NS, 'rect');
    r.setAttribute('class', cls);
    r.setAttribute('x', String(x)); r.setAttribute('y', String(y));
    r.setAttribute('width', String(Math.max(w, 0))); r.setAttribute('height', String(Math.max(h, 0)));
    parent.appendChild(r);
  }
  // Selection goes under the glyphs, the caret above them.
  function drawOverlay(): void {
    const svg = activeSvg();
    if (!svg || !info) return;
    svg.querySelectorAll('.ov').forEach(g => g.remove());
    if (info.selection) {
      const under = document.createElementNS(SVG_NS, 'g');
      under.setAttribute('class', 'ov');
      rect(under, 'selection', ...info.selection);
      svg.insertBefore(under, svg.firstChild);
    }
    if (info.caret) {
      const over = document.createElementNS(SVG_NS, 'g');
      over.setAttribute('class', 'ov');
      const [x, y, t, h] = info.caret;
      rect(over, 'caret', x - t / 2, y, t, h);
      svg.appendChild(over);
    }
  }

  function growSrc(): void {
    src.style.height = 'auto';
    src.style.height = `${src.scrollHeight + src.offsetHeight - src.clientHeight}px`;
  }

  function refresh(): void {
    if (!wasm || !alive) return;
    info = JSON.parse(readOutput(1)) as Info;
    stage.querySelector('.math-loading')?.remove();
    lines[active] = info.tex;
    setRowSvg(active, readOutput(0));
    if (document.activeElement !== src) {
      src.value = info.tex;
      src.classList.remove('bad');
      growSrc();
    }
    let text = '', bad = false;
    if (info.error) { text = info.error; bad = true; }
    else if (info.pending != null) text = `\\${info.pending} — Enter to insert, Esc to cancel`;
    else if (info.unknown.length) { text = `unknown: ${info.unknown.join(', ')}`; bad = true; }
    else if (lines.length === 1 && info.tex === '') text = opts.evaluate
      ? 'type maths: x = \\frac{1}{2}, A^{-1}, \\det A, \\begin{pmatrix}…  ·  Enter for a new line'
      : 'type maths · Enter for a new line';
    status.textContent = text;
    status.classList.toggle('bad', bad);
    evaluateAll();
    drawOverlay();
    persist();
  }

  // Rows are edited in place rather than rebuilt: emptying the list would
  // shrink the stage and reset its scroll position.
  function insertLine(): void {
    lines.splice(active + 1, 0, '');
    svgs.splice(active + 1, 0, '');
    const row = makeRow();
    rows[active].after(row);
    rows.splice(active + 1, 0, row);
    renumber();
    loadLine(active + 1);
  }
  function removeLine(): void {
    if (lines.length <= 1) return;
    lines.splice(active, 1);
    svgs.splice(active, 1);
    const hadFocus = document.activeElement === keys;
    stage.appendChild(keys);   // the sink sits in this row: removing it would blur the sink
    rows[active].remove();
    rows.splice(active, 1);
    renumber();
    loadLine(Math.max(0, active - 1));
    if (hadFocus) focusKeys();
    call(() => wasm!.key(6, 0, 0));   // End: land after the line above, as a text editor would
  }

  /* ── input ── */
  function scenePoint(e: MouseEvent): [number, number] | null {
    const ctm = activeSvg()?.getScreenCTM();
    if (!ctm) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    return [p.x, p.y];
  }
  const focusKeys = () => keys.focus({ preventScroll: true });

  keys.addEventListener('keydown', (e: KeyboardEvent) => {
    e.stopPropagation();   // the page's own shortcuts stay out of the formula
    if (!wasm) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod) {
      const k = e.key.toLowerCase();
      if (k === 'z' && !e.shiftKey) { e.preventDefault(); call(() => wasm!.key(10, 0, 0)); return; }
      if (k === 'y' || (k === 'z' && e.shiftKey)) { e.preventDefault(); call(() => wasm!.key(11, 0, 0)); return; }
      if (k === 'a') { e.preventDefault(); call(() => wasm!.key(9, 0, 0)); return; }
      if (k === 'c' || k === 'x') {
        if (wasm.copy()) {
          e.preventDefault();
          navigator.clipboard?.writeText(readOutput(2)).catch(() => {});
          if (k === 'x') call(() => wasm!.key(7, 0, 0));
        }
        return;
      }
      if (k === 'v') return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      if (info?.pending != null) { call(() => wasm!.key(14, 0, 0)); return; }   // finish a raw \command
      if (e.shiftKey) { call(() => wasm!.key(13, 0, 0)); return; }   // a new row inside the formula (matrices, cases…)
      insertLine();
      return;
    }
    if (e.key === 'Backspace' && !e.shiftKey && info?.tex === '' && lines.length > 1) {
      e.preventDefault();
      removeLine();
      return;
    }
    const code = KEY_CODES[e.key];
    if (code) {
      e.preventDefault();
      // eeltex reports Up/Down as a change even when the caret stays put, so
      // compare the caret itself: no movement means hop to the next line.
      const before = JSON.stringify([info?.caret, info?.selection]);
      call(() => wasm!.key(code, e.shiftKey ? 1 : 0, mod || e.altKey ? 1 : 0));
      const after = JSON.stringify([info?.caret, info?.selection]);
      if (before === after && !e.shiftKey) {
        if (e.key === 'ArrowUp' && active > 0) loadLine(active - 1);
        else if (e.key === 'ArrowDown' && active < lines.length - 1) loadLine(active + 1);
      }
    } else if (e.key === 'Tab') {
      e.preventDefault();
      call(() => wasm!.key(e.shiftKey ? 1 : 2, 0, 0));
    }
  });
  keys.addEventListener('input', () => {
    const text = keys.value;
    keys.value = '';
    for (const ch of text) call(() => wasm!.insert(ch.codePointAt(0)!));
  });
  keys.addEventListener('paste', (e: ClipboardEvent) => {
    e.preventDefault();
    const text = e.clipboardData?.getData('text/plain');
    if (text && wasm) { writeInput(text); call(() => wasm!.paste()); }
  });

  let dragging = false;
  stage.addEventListener('pointerdown', (e: PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    focusKeys();
    if (!wasm) return;
    const row = (e.target as HTMLElement).closest<HTMLElement>('.math-line');
    if (row) {
      const i = Number(row.dataset.i);
      if (i !== active) loadLine(i);
    } else if (e.target === stage || e.target === list) {
      // Below the last line: land at the end of it.
      if (active !== lines.length - 1) loadLine(lines.length - 1);
      call(() => wasm!.key(6, 0, 0));
      return;
    }
    const p = scenePoint(e);
    if (!p) return;
    call(() => wasm!.click(p[0], p[1], e.shiftKey ? 1 : 0));
    dragging = true;
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointerup', () => { dragging = false; });
  stage.addEventListener('pointercancel', () => { dragging = false; });
  stage.addEventListener('lostpointercapture', () => { dragging = false; });
  stage.addEventListener('pointermove', (e: PointerEvent) => {
    if (!dragging) return;
    const p = scenePoint(e);
    if (p) call(() => wasm!.click(p[0], p[1], 1));
  });
  stage.addEventListener('dblclick', (e: MouseEvent) => {
    const p = scenePoint(e);
    if (p) call(() => wasm!.select_at(p[0], p[1]));
  });

  let srcTimer: ReturnType<typeof setTimeout> | null = null;
  src.addEventListener('keydown', e => e.stopPropagation());
  src.addEventListener('input', () => {
    growSrc();
    if (srcTimer) clearTimeout(srcTimer);
    srcTimer = setTimeout(() => {
      if (!wasm) return;
      writeInput(src.value);
      const ok = wasm.set_tex();
      refresh();
      src.classList.toggle('bad', !ok);
    }, 120);
  });
  src.addEventListener('blur', () => {
    if (info && !src.classList.contains('bad')) src.value = info.tex;
    growSrc();
  });
  // The source field is hidden while the pad is idle (CSS), and a hidden
  // textarea measures as 0: size it again once hover or focus reveals it.
  const regrow = () => requestAnimationFrame(growSrc);
  el.addEventListener('mouseenter', regrow);
  el.addEventListener('focusin', regrow);

  const ro = new ResizeObserver(() => rows.forEach(fitRow));
  ro.observe(list);

  buildRows();
  Promise.all([instantiate(), opts.evaluate ? instantiate() : null, opts.evaluate ? getMath() : null])
    .then(([editor, results, m]) => {
      if (!alive) return;
      wasm = editor;
      math = m;
      writeInput('');
      if (!wasm.init(FONT_SIZE)) throw new Error('embedded font failed to load');
      if (results) {
        results.alloc(0);   // init reads the input buffer: start it empty
        if (!results.init(FONT_SIZE)) throw new Error('embedded font failed to load');
        renderer = results;
      }
      stage.querySelector('.math-loading')?.remove();
      // Render every line once for its static SVG, then settle on the first.
      lines.forEach((tex, i) => {
        writeInput(tex);
        wasm!.set_tex();
        setRowSvg(i, readOutput(0));
      });
      loadLine(0);
    })
    .catch((err: Error) => {
      const loading = stage.querySelector('.math-loading');
      if (loading) loading.textContent = `eeltex failed to load: ${err.message}`;
      status.classList.add('bad');
    });

  return {
    el,
    destroy() {
      alive = false;
      ro.disconnect();
      if (srcTimer) clearTimeout(srcTimer);
      el.remove();
    },
  };
}
