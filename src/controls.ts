// Live controls: a `const` with an annotation becomes a uniform driven from
// the Uniforms view, and the literal in the source is the saved value.
//
//   const float speed = 1.0;                 // @slider(0.0, 5.0)
//   const int   steps = 64;                  // @slider(1, 256)
//   const vec2  centre = vec2(0.5, 0.5);     // @slider(0, 1)
//   const vec3  tint = vec3(1.0, 0.5, 0.2);  // @color
//   const bool  wire = false;                // @toggle
//
// The rewrite keeps every line in place, so error line numbers still match.

export type ControlType = 'float' | 'int' | 'vec2' | 'vec3' | 'vec4' | 'bool';
export type ControlKind = 'slider' | 'color' | 'toggle';

export interface Control {
  name: string;
  type: ControlType;
  kind: ControlKind;
  min: number;
  max: number;
  step: number;
  value: number[];   // bools are 0/1
  line: number;      // 0-based line of the declaration
}

const DECL_RE = /^(\s*const\s+)(float|int|vec2|vec3|vec4|bool)(\s+)(\w+)(\s*=\s*)([^;]+?)(\s*;)(.*?\/\/\s*@(slider|color|toggle)\b(?:\s*\(([^)]*)\))?.*)$/;

const LANES: Record<ControlType, number> = { float: 1, int: 1, vec2: 2, vec3: 3, vec4: 4, bool: 1 };

/** A whole GLSL numeric literal (1.0, .5, 1e-3f, 0x1F, -2) and nothing else; null otherwise. */
function parseNumber(s: string): number | null {
  const t = s.trim();
  const hex = /^([+-]?)0[xX]([0-9a-fA-F]+)[uU]?$/.exec(t);
  if (hex) return (hex[1] === '-' ? -1 : 1) * parseInt(hex[2], 16);
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:[fF]|lf|LF|[uU])?$/.test(t)) return null;
  const v = parseFloat(t);
  return Number.isFinite(v) ? v : null;
}

function parseLiteral(type: ControlType, text: string): number[] | null {
  const t = text.trim();
  if (type === 'bool') return t === 'true' ? [1] : t === 'false' ? [0] : null;
  if (type === 'float' || type === 'int') { const v = parseNumber(t); return v === null ? null : [v]; }
  const m = new RegExp(`^${type}\\s*\\(([^()]*)\\)$`).exec(t);
  if (!m) return null;
  const parts = m[1].split(',').map(parseNumber);
  if (parts.some(p => p === null)) return null;
  const n = LANES[type];
  if (parts.length === 1) return Array(n).fill(parts[0] as number);
  return parts.length === n ? (parts as number[]) : null;
}

/** The code part of a line, without comments. */
function codeOf(line: string): string {
  return line.replace(/\/\*.*?\*\//g, ' ').replace(/\/\/.*$/, '').replace(/\/\*.*$/, '');
}

/** The declaration match for one line; null when the line holds more than one `const` (left alone, G7). */
function matchDecl(line: string): RegExpExecArray | null {
  const m = DECL_RE.exec(line);
  if (!m) return null;
  return (codeOf(line).match(/\bconst\b/g) ?? []).length === 1 ? m : null;
}

/** Finds annotated consts and returns the source with them turned into uniforms. */
export function rewriteControls(code: string): { code: string; controls: Control[]; problems: string[] } {
  const controls: Control[] = [];
  const problems: string[] = [];
  const lines = code.split('\n');
  let depth = 0;        // brace depth: only global consts become uniforms
  let inComment = false;
  const count = (src: string) => { for (const ch of src) { if (ch === '{') depth++; else if (ch === '}') depth = Math.max(0, depth - 1); } };
  const out = lines.map((line, i) => {
    const atTop = depth === 0 && !inComment;
    // Brace depth for the lines after this one, ignoring comments.
    let rest = line;
    for (;;) {
      if (inComment) { const e = rest.indexOf('*/'); if (e < 0) { rest = ''; break; } rest = rest.slice(e + 2); inComment = false; }
      const lc = rest.indexOf('//'), bc = rest.indexOf('/*');
      if (bc >= 0 && (lc < 0 || bc < lc)) { count(rest.slice(0, bc)); rest = rest.slice(bc + 2); inComment = true; continue; }
      count(lc >= 0 ? rest.slice(0, lc) : rest);
      break;
    }
    if (!atTop) return line;
    const m = matchDecl(line);
    if (!m) return line;
    const type = m[2] as ControlType, name = m[4], kind = m[9] as ControlKind;
    const value = parseLiteral(type, m[6]);
    if (!value) return line;
    if (kind === 'color' && !(type === 'vec3' || type === 'vec4')) return line;
    if (kind === 'toggle' && type !== 'bool') return line;
    if (kind === 'slider' && type === 'bool') return line;
    const rawArgs = (m[10] ?? '').split(',').map(s => s.trim()).filter(s => s !== '');
    const args = rawArgs.map(parseNumber);
    if (args.some(a => a === null)) { problems.push(`line ${i + 1}: @${kind}(${m[10]}) has a non-numeric argument`); return line; }
    let min = 0, max = 1, step = type === 'int' ? 1 : 0;
    if (kind === 'slider') {
      const a = args as number[];
      if (a.length >= 2) { min = Math.min(a[0], a[1]); max = Math.max(a[0], a[1]); }
      else if (a.length === 1) { max = a[0]; }
      else { const top = Math.max(...value.map(Math.abs), 1); max = top * 2; min = value.some(v => v < 0) ? -top * 2 : 0; }
      if (!(max > min)) max = min + 1;   // @slider(1, 1): an empty range would give a zero step
      if (a.length >= 3 && a[2] > 0) step = a[2];
    }
    controls.push({ name, type, kind, min, max, step, value, line: i });
    // Same length is not needed; same line count is.
    return `${m[1].replace(/const/, 'uniform')}${type}${m[3]}${name};${' '.repeat(Math.max(1, m[5].length + m[6].length + m[7].length - 1))}${m[8]}`;
  });
  return { code: out.join('\n'), controls, problems };
}

function fmtFloat(v: number): string {
  if (!Number.isFinite(v)) return '0.0';
  let s = +v.toFixed(4) === v ? String(+v.toFixed(4)) : v.toPrecision(6);
  s = s.replace(/^(-?)\./, '$10.');
  if (!/[.e]/.test(s)) s += '.0';
  return s;
}

/** The GLSL literal for a control's current value. */
export function formatLiteral(c: Control): string {
  if (c.type === 'bool') return c.value[0] ? 'true' : 'false';
  if (c.type === 'int') return String(Math.round(c.value[0]));
  if (c.type === 'float') return fmtFloat(c.value[0]);
  return `${c.type}(${c.value.map(fmtFloat).join(', ')})`;
}

/** The declaration line with its literal replaced; null if the line no longer matches. */
export function replaceLiteral(line: string, c: Control): string | null {
  const m = matchDecl(line);
  if (!m || m[4] !== c.name) return null;
  return `${m[1]}${m[2]}${m[3]}${m[4]}${m[5]}${formatLiteral(c)}${m[7]}${m[8]}`;
}

/** Cheap identity for "has the set of controls changed", ignoring values. */
export function controlSignature(cs: Control[]): string {
  return cs.map(c => `${c.name}:${c.type}:${c.kind}:${c.min}:${c.max}:${c.step}`).join('|');
}
