#!/usr/bin/env node
// Checks the scratch-pad calculators against SymPy, and the control annotations.
//
//   npm test
//
// test/cases.py prints the cases with their SymPy values; this bundles the two
// evaluators with esbuild (no DOM needed: both are pure) and compares.
// Consecutive cases of one kind share bindings, like lines in a pad.

import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const cases = JSON.parse(execFileSync('python3', ['-I', join(here, 'cases.py')], { encoding: 'utf8' }));

const out = mkdtempSync(join(tmpdir(), 'shader-calc-test-'));
try {
  execFileSync('npx', ['esbuild', join(root, 'src/mathlib.ts'), join(root, 'src/mathcalc.ts'), join(root, 'src/controls.ts'),
    '--bundle', '--platform=node', '--format=esm', '--log-level=error', `--outdir=${out}`], { cwd: root, stdio: 'inherit' });
  var { getMath, getPlotMath, evalStatement } = await import(pathToFileURL(join(out, 'mathlib.js')));
  var { evaluateTex } = await import(pathToFileURL(join(out, 'mathcalc.js')));
  var controlsMod = await import(pathToFileURL(join(out, 'controls.js')));
} finally {
  // Keep the bundle around until the imports resolved; then it is not needed.
  process.on('exit', () => rmSync(out, { recursive: true, force: true }));
}

const REL = 1e-12, ABS = 1e-300;
const close = (a, b) => a === b || Math.abs(a - b) <= Math.max(ABS, REL * Math.max(Math.abs(a), Math.abs(b)));

const math = await getMath();

// math.js values to plain JS: BigNumber → number, Matrix → rows.
function plain(v) {
  if (math.isBigNumber(v)) return v.toNumber();
  if (math.isMatrix(v)) { const a = v.toArray().map(r => (Array.isArray(r) ? r.map(plain) : [plain(r)])); return a; }
  if (math.isComplex(v)) return v.im === 0 ? v.re : `${v.re}+${v.im}i`;
  return v;
}

const SPECIAL = { inf: Infinity, '-inf': -Infinity, nan: NaN };

function same(got, expect) {
  if (typeof expect === 'string' && expect in SPECIAL) expect = SPECIAL[expect];
  if (typeof expect === 'string' && expect.startsWith('[[')) return JSON.stringify(got) === JSON.stringify(JSON.parse(expect));
  if (typeof expect === 'string' || typeof got === 'string') return got === expect;
  if (Array.isArray(expect)) {
    const rows = got;
    if (!Array.isArray(rows) || rows.length !== expect.length) return false;
    return rows.every((r, i) => r.length === expect[i].length && r.every((x, j) => close(x, expect[i][j])));
  }
  if (typeof got !== 'number') return false;
  if (Number.isNaN(expect)) return Number.isNaN(got);
  return close(got, expect);
}

const show = v => (typeof v === 'number' && !isFinite(v) ? String(v) : JSON.stringify(v));

const calcVars = new Map();
const texEnv = new Map();
let pass = 0;
const failures = [];

for (const c of cases) {
  let got, error = null;
  try {
    const raw = c.kind === 'calc' ? evalStatement(math, c.src, calcVars).value : evaluateTex(math, c.src, texEnv).value;
    if (c.kind === 'calc' && typeof raw !== 'string') calcVars.set('ans', raw);
    got = plain(raw);
  } catch (e) {
    error = e.message;
  }
  const wantError = c.expect && c.expect.error === true;
  const ok = wantError ? error !== null : error === null && same(got, c.expect);
  if (ok) pass++;
  else failures.push(`${c.kind.padEnd(4)} ${c.src}\n       want ${wantError ? 'an error' : show(c.expect)}\n       got  ${error ? `error: ${error}` : show(got)}`);
}

// The graph's number-mode instance: plain doubles in and out, same extensions.
const plotMath = await getPlotMath();
const plotScope = { x: 2.5 };
for (const [src, want] of [['fract(x)', 0.5], ['sin(x) + sat(x)', Math.sin(2.5) + 1], ['smoothstep(0, 5, x)', 0.5], ['x * epsilon', 2.5 * 2 ** -23]]) {
  let got;
  try { got = plotMath.parse(src).compile().evaluate(plotScope); } catch (e) { got = `error: ${e.message}`; }
  if (typeof got === 'number' && close(got, want)) pass++;
  else failures.push(`plot ${src}\n       want ${show(want)}\n       got  ${show(got)}`);
}

import { checkControls } from './controls.mjs';
const ctl = checkControls(controlsMod);
for (const f of ctl) failures.push(`controls: ${f}`);
pass += 0;

for (const f of failures) console.log(`FAIL ${f}`);
console.log(`${pass}/${cases.length + 4} calculator cases passed, controls ${ctl.length ? `${ctl.length} failed` : 'ok'}`);
process.exit(failures.length ? 1 : 0);
