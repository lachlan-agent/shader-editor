// Scratch pads: small tools that live in a view window beside the shader.
// Each pad owns one element and reports its persisted state through `onData`.

import { getMath, evalStatement, formatValue, helpGroups, SYNTAX, PRECISION, type Math, type Scope } from './mathlib';
import { makeGraph } from './graph';

export interface ScratchPad {
  el: HTMLElement;
  /** Called when the window is closed or switched to something else. */
  destroy(): void;
}

export type ScratchKind = 'calc' | 'graph' | 'picture';

export interface ScratchHost {
  data: string;                        // whatever this pad saved last time
  onData: (data: string) => void;      // persist (the host debounces storage)
}

/* ── Calculator: a REPL over math.js ─────────────────────── */
// See mathlib.ts for the numeric core and the extension functions.

interface CalcState { lines: Array<[string, string, boolean]>; vars: Record<string, string> }

function makeCalc(host: ScratchHost): ScratchPad {
  const el = document.createElement('div');
  el.className = 'pad pad-calc';
  el.innerHTML = `
    <div class="calc-log"></div>
    <div class="calc-help"></div>
    <div class="calc-in">
      <span class="calc-prompt">›</span>
      <input type="text" spellcheck="false" autocomplete="off" placeholder="sin(pi/4) · x = 2^10 · bits(0.1) · 0xFF & ~0x0F · [1,2;3,4]^-1">
      <button class="float-btn calc-help-btn" title="Functions and syntax">?</button>
    </div>`;
  const log   = el.querySelector('.calc-log') as HTMLElement;
  const help  = el.querySelector('.calc-help') as HTMLElement;
  const input = el.querySelector('input') as HTMLInputElement;
  const helpBtn = el.querySelector('.calc-help-btn') as HTMLButtonElement;

  // Help panel: one table per group; click an entry to paste its signature.
  for (const g of [{ title: 'Syntax', items: SYNTAX.map(([sig, doc]) => ({ name: '', sig, doc })) }, ...helpGroups()]) {
    const h = document.createElement('div'); h.className = 'calc-help-title'; h.textContent = g.title;
    help.appendChild(h);
    for (const it of g.items) {
      const row = document.createElement('div'); row.className = 'calc-help-row';
      const a = document.createElement('span'); a.className = 'calc-help-sig'; a.textContent = it.sig;
      const b = document.createElement('span'); b.className = 'calc-help-doc'; b.textContent = it.doc;
      row.append(a, b);
      if (it.name) row.addEventListener('click', () => {
        const at = input.selectionStart ?? input.value.length;
        input.value = input.value.slice(0, at) + it.sig + input.value.slice(at);
        input.focus();
        const open = at + it.sig.indexOf('(') + 1;
        input.setSelectionRange(open, open);
      });
      help.appendChild(row);
    }
  }
  function toggleHelp(on = !el.classList.contains('show-help')): void {
    el.classList.toggle('show-help', on);
    helpBtn.classList.toggle('active', on);
  }
  helpBtn.addEventListener('click', () => toggleHelp());

  let state: CalcState = { lines: [], vars: {} };
  try {
    const p = JSON.parse(host.data || '{}');
    if (Array.isArray(p.lines)) {
      const ok = (l: unknown): l is [string, string, boolean] => Array.isArray(l) && typeof l[0] === 'string' && typeof l[1] === 'string';
      const lines = (p.lines as unknown[]).filter(ok).map(l => [l[0], l[1], !!l[2]] as [string, string, boolean]);
      if (lines.length !== p.lines.length) console.warn(`calculator: dropped ${p.lines.length - lines.length} corrupt saved line(s)`);
      const vars = p.vars && typeof p.vars === 'object' && !Array.isArray(p.vars) ? p.vars as Record<string, unknown> : {};
      state = { lines, vars: Object.fromEntries(Object.entries(vars).filter(([, v]) => typeof v === 'string')) as Record<string, string> };
    }
  } catch { /* fresh */ }
  const history: string[] = state.lines.map(l => l[0]);
  let histPos = history.length;

  function addLine(src: string, out: string, bad: boolean): void {
    const row = document.createElement('div');
    row.className = 'calc-row' + (bad ? ' bad' : '');
    const a = document.createElement('div'); a.className = 'calc-src'; a.textContent = src;
    const b = document.createElement('div'); b.className = 'calc-out'; b.textContent = out;
    row.append(a, b);
    a.addEventListener('click', () => { input.value = src; input.focus(); });
    b.addEventListener('click', () => { if (!bad) navigator.clipboard?.writeText(out).catch(() => {}); });
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
  }
  for (const [s, o, bad] of state.lines) addLine(s, o, bad);

  // Variables persist as math.js expression strings (any type round-trips).
  let math: Math | null = null;
  const scope: Scope = new Map();
  const ready = getMath().then(m => {
    math = m;
    for (const [k, v] of Object.entries(state.vars)) {
      try { scope.set(k, m.evaluate(v)); } catch { /* a value this build cannot read */ }
    }
  });
  ready.catch(e => addLine('math.js', `failed to load: ${(e as Error).message}`, true));

  function persist(): void {
    if (math) {
      state.vars = {};
      for (const [k, v] of scope) if (typeof v !== 'function') state.vars[k] = math.format(v, { precision: PRECISION });
    }
    if (state.lines.length > 200) state.lines.splice(0, state.lines.length - 200);
    host.onData(JSON.stringify(state));
  }

  async function run(src: string): Promise<void> {
    src = src.trim();
    if (!src) return;
    if (src === 'clear') { state.lines = []; log.innerHTML = ''; persist(); return; }
    if (src === 'help' || src === '?') { toggleHelp(true); return; }
    try { await ready; } catch { return; }   // the load failure is already in the log
    let out: string, bad = false;
    try {
      const r = evalStatement(math!, src, scope);
      if (typeof r.value !== 'string') scope.set('ans', r.value);
      out = (r.name ? `${r.name} = ` : '') + formatValue(math!, r.value);
    } catch (e) {
      out = (e as Error).message; bad = true;
    }
    state.lines.push([src, out, bad]);
    addLine(src, out, bad);
    history.push(src); histPos = history.length;
    persist();
  }

  input.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'Enter') { const v = input.value; input.value = ''; void run(v); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { if (histPos > 0) { histPos--; input.value = history[histPos]; } e.preventDefault(); }
    else if (e.key === 'ArrowDown') { histPos = Math.min(histPos + 1, history.length); input.value = history[histPos] ?? ''; e.preventDefault(); }
    else if (e.key === 'Escape' && el.classList.contains('show-help')) { toggleHelp(false); e.preventDefault(); }
    e.stopPropagation();
  });
  el.addEventListener('click', (e: MouseEvent) => { if (e.target === el || e.target === log) input.focus(); });

  return { el, destroy() { el.remove(); } };
}

/* ── Image: a reference picture ──────────────────────────── */
// Stored as a data URL, shrunk so a few of them still fit in localStorage.
// A PNG or JPEG within the budget is kept byte for byte; anything else is
// re-encoded, PNG when the source may have alpha, JPEG otherwise.
const IMG_MAX = 1280;
const IMG_BYTES = 1_500_000;

function shrink(img: HTMLImageElement, type: string): string {
  const s = Math.min(1, IMG_MAX / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(img.naturalWidth * s));
  c.height = Math.max(1, Math.round(img.naturalHeight * s));
  c.getContext('2d')!.drawImage(img, 0, 0, c.width, c.height);
  return type === 'image/jpeg' ? c.toDataURL('image/jpeg', 0.88) : c.toDataURL('image/png');
}

function readDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = () => reject(r.error ?? new Error('could not read the file'));
    r.readAsDataURL(blob);
  });
}

function makeImage(host: ScratchHost): ScratchPad {
  const el = document.createElement('div');
  el.className = 'pad pad-image';
  el.innerHTML = `
    <img alt="" draggable="false">
    <div class="pad-drop">
      <div>Drop an image, paste, or</div>
      <button class="btn">Choose file…</button>
    </div>
    <button class="float-btn pad-clear" title="Remove image">✕</button>
    <input type="file" accept="image/*" hidden>`;
  const img   = el.querySelector('img') as HTMLImageElement;
  const drop  = el.querySelector('.pad-drop') as HTMLElement;
  const pick  = el.querySelector('.btn') as HTMLButtonElement;
  const clear = el.querySelector('.pad-clear') as HTMLButtonElement;
  const file  = el.querySelector('input') as HTMLInputElement;

  function show(src: string): void {
    img.src = src;
    el.classList.toggle('has-image', !!src);
  }
  function report(msg: string): void {
    drop.firstElementChild!.textContent = msg;
    el.classList.remove('has-image');
  }
  function load(blob: Blob): void {
    const url = URL.createObjectURL(blob);
    const probe = new Image();
    const keep = (blob.type === 'image/png' || blob.type === 'image/jpeg') && blob.size <= IMG_BYTES;
    probe.onload = () => {
      URL.revokeObjectURL(url);
      const small = Math.max(probe.naturalWidth, probe.naturalHeight) <= IMG_MAX;
      const done = (data: string) => { show(data); host.onData(data); };
      if (keep && small) readDataUrl(blob).then(done, e => report(`Could not read the image: ${(e as Error).message}`));
      else done(shrink(probe, blob.type));
    };
    probe.onerror = () => {
      URL.revokeObjectURL(url);
      report(`Could not decode the image (${blob.type || 'unknown type'})`);
    };
    probe.src = url;
  }
  show(host.data);

  pick.addEventListener('click', () => file.click());
  file.addEventListener('change', () => { const f = file.files?.[0]; if (f) load(f); file.value = ''; });
  el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('over'); });
  el.addEventListener('dragleave', () => el.classList.remove('over'));
  el.addEventListener('drop', e => {
    e.preventDefault(); el.classList.remove('over');
    const f = e.dataTransfer?.files?.[0];
    if (f && f.type.startsWith('image/')) load(f);
  });
  el.tabIndex = 0;
  el.addEventListener('paste', e => {
    const item = Array.from(e.clipboardData?.items ?? []).find(i => i.type.startsWith('image/'));
    const f = item?.getAsFile();
    if (f) { e.preventDefault(); load(f); }
  });
  el.addEventListener('click', e => { if (e.target === el || e.target === img || e.target === drop) el.focus(); });
  clear.addEventListener('click', () => { show(''); host.onData(''); });

  return { el, destroy() { el.remove(); } };
}

export function makeScratch(kind: ScratchKind, host: ScratchHost): ScratchPad {
  switch (kind) {
    case 'calc':  return makeCalc(host);
    case 'graph': return makeGraph(host);
    case 'picture': return makeImage(host);
  }
}
