// Translates the LaTeX subset eeltex edits into a math.js expression, which
// mathlib then evaluates (BigNumber scalars, matrices, complex numbers).
//
//   \frac{a}{b}  \sqrt[n]{x}  x^{2}  x_{1}  \sin x  \ln(x)  \pi  e  |x|  \det A
//   \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}   A^{T}  A^{-1}  AB  A \cdot B
//   x = …  defines a name for the lines below.
//
// `toTex` writes a result back as LaTeX for rendering.

import { evalStatement, formatValue, type Math, type Scope } from './mathlib';

/* ── tokens ── */
type Tok =
  | { t: 'num'; v: string }
  | { t: 'sym'; v: string }      // one letter
  | { t: 'cmd'; v: string }      // \name
  | { t: 'op'; v: string };      // punctuation

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  const re = /\s*(?:(\d+\.?\d*|\.\d+)|([A-Za-z])|\\([A-Za-z]+|\\|[,;:!{}|.])|([{}()\[\]|^_+\-*/=&,!']))/y;
  let i = 0;
  while (i < src.length) {
    re.lastIndex = i;
    const m = re.exec(src);
    if (!m || m.index !== i) {
      const rest = src.slice(i);
      if (/^\s*$/.test(rest)) break;
      throw new Error(`unexpected '${rest.trim()[0]}'`);
    }
    i = re.lastIndex;
    if (m[1] !== undefined && src[i] === '.' && m[1].includes('.')) throw new Error(`unexpected '.' after ${m[1]}`);   // 1.2.3
    if (m[1] !== undefined) out.push({ t: 'num', v: m[1] });
    else if (m[2] !== undefined) out.push({ t: 'sym', v: m[2] });
    else if (m[3] !== undefined) out.push({ t: 'cmd', v: m[3] });
    else if (m[4] !== undefined) out.push({ t: 'op', v: m[4] });
  }
  return out;
}

const FUNCS: Record<string, string> = {
  sin: 'sin', cos: 'cos', tan: 'tan', arcsin: 'asin', arccos: 'acos', arctan: 'atan',
  sinh: 'sinh', cosh: 'cosh', tanh: 'tanh', exp: 'exp', ln: 'log', log: 'log10', lg: 'log2',
  det: 'det', tr: 'trace', min: 'min', max: 'max', gcd: 'gcd',
};
/** Functions that take a comma-separated list: \min(a, b), \max{a, b}, \gcd(a, b). */
const MULTI = new Set(['min', 'max', 'gcd']);
/** \sin^{-1} x is arcsin, not 1/sin x. */
const INVERSE: Record<string, string> = { sin: 'asin', cos: 'acos', tan: 'atan', sinh: 'asinh', cosh: 'acosh', tanh: 'atanh' };
const CONSTS: Record<string, string> = { pi: 'pi', infty: 'Infinity', tau: 'tau', phi: 'phi' };
// Greek letters and friends are names, not commands.
const LETTERS = new Set(('alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi rho sigma tau upsilon phi varphi chi psi omega '
  + 'Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega ell hbar').split(' '));
const SKIP = new Set(['left', 'right', 'quad', 'qquad', ',', ';', ':', '!', 'displaystyle', 'textstyle', 'mathbf', 'boldsymbol', 'vec', 'hat', 'bar', 'tilde']);
// `\hat{x}_1` is the name x_1: an accent over a name is transparent, and the subscript follows the group.
const ACCENTS = new Set(['mathbf', 'boldsymbol', 'vec', 'hat', 'bar', 'tilde']);
// `\mathrm{name}` and `\text{name}` are multi-letter names (bindable, like x).
const WORD_CMDS = new Set(['mathrm', 'text', 'mathit', 'textit', 'mathsf', 'mathtt']);
const MATRIX_ENVS = new Set(['matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Bmatrix']);

/** A LaTeX name as a math.js identifier: \alpha_1 → alpha_1, \mathrm{rate} → rate. */
function ident(name: string): string {
  return name.replace(/\\mathrm\{(\w+)\}/g, '$1').replace(/\\/g, '').replace(/[{}]/g, '')
    .replace(/-/g, '_minus_').replace(/\+/g, '_plus_').replace(/[^\w]/g, '_');   // a_{n-1} is one name, not a_n - 1
}

/* ── parser: LaTeX → math.js source ── */
class Parser {
  private i = 0;
  private absDepth = 0;
  constructor(private toks: Tok[]) {}

  private peek(): Tok | undefined { return this.toks[this.i]; }
  private is(t: Tok['t'], v: string): boolean { const k = this.peek(); return !!k && k.t === t && k.v === v; }
  private isOp(v: string): boolean { return this.is('op', v); }
  private take(): Tok {
    const t = this.toks[this.i++];
    if (!t) throw new Error('unexpected end');
    return t;
  }
  private expectOp(v: string): void {
    if (!this.isOp(v)) throw new Error(`expected '${v}'`);
    this.i++;
  }

  /** Returns math.js source and, for `name = expr`, the bound name. */
  statement(): { name: string | null; src: string } {
    const save = this.i;
    const name = this.name();
    if (name && this.isOp('=') && this.i < this.toks.length - 1) {
      this.i++;
      const src = this.expr();
      this.end();
      return { name: ident(name), src: `${ident(name)} = ${src}` };
    }
    this.i = save;
    const src = this.expr();
    // `expr = ` with nothing after it is fine: "2+2 =" asks for the answer.
    if (this.isOp('=') && this.i === this.toks.length - 1) this.i++;
    this.end();
    return { name: null, src };
  }
  private end(): void { if (this.i < this.toks.length) throw new Error('trailing input'); }

  /** A symbol, Greek letter or \mathrm{word}, with an optional subscript folded in. */
  private name(): string | null {
    const t = this.peek();
    if (!t) return null;
    let base: string;
    if (t.t === 'sym') { base = t.v; this.i++; }
    else if (t.t === 'cmd' && LETTERS.has(t.v)) { base = `\\${t.v}`; this.i++; }
    else if (t.t === 'cmd' && ACCENTS.has(t.v) && this.toks[this.i + 1]?.t === 'op' && this.toks[this.i + 1].v === '{') {
      const save = this.i;
      this.i += 2;
      const inner = this.name();
      if (!inner || !this.isOp('}')) { this.i = save; return null; }
      this.i++;
      base = inner;
    }
    else if (t.t === 'cmd' && WORD_CMDS.has(t.v)) {
      const j = this.i + 1;
      const open = this.toks[j];
      if (!(open?.t === 'op' && open.v === '{')) return null;
      let k = j + 1, word = '';
      while (this.toks[k]?.t === 'sym') word += (this.toks[k++] as { v: string }).v;
      const close = this.toks[k];
      if (!word || !(close?.t === 'op' && close.v === '}')) return null;
      this.i = k + 1;
      base = word.length === 1 ? word : `\\mathrm{${word}}`;
    }
    else return null;
    if (this.isOp('_')) {
      this.i++;
      base += '_' + this.rawGroup();
    }
    return base;
  }
  /** Subscript text: `{…}` verbatim-ish, or one token. */
  private rawGroup(): string {
    if (this.isOp('{')) {
      this.i++;
      let s = '';
      let depth = 0;
      for (;;) {
        const t = this.take();
        if (t.t === 'op' && t.v === '{') depth++;
        if (t.t === 'op' && t.v === '}') { if (depth === 0) break; depth--; }
        s += t.t === 'cmd' ? `\\${t.v}` : String(t.v);
      }
      return s;
    }
    const t = this.take();
    return t.t === 'cmd' ? `\\${t.v}` : String(t.v);
  }

  private expr(): string {
    let v = this.term();
    for (;;) {
      if (this.isOp('+')) { this.i++; v = `${v} + ${this.term()}`; }
      else if (this.isOp('-')) { this.i++; v = `${v} - ${this.term()}`; }
      else if (this.is('cmd', 'pm')) { this.i++; v = `${v} + ${this.term()}`; }
      else return v;
    }
  }
  private startsFactor(): boolean {
    const t = this.peek();
    if (!t) return false;
    if (t.t === 'num' || t.t === 'sym') return true;
    if (t.t === 'op') {
      if (t.v === '|') return this.absDepth === 0;     // inside |…| a bar closes it
      return t.v === '(' || t.v === '[' || t.v === '{';
    }
    return !['pm', 'cdot', 'times', 'div', 'right', '\\', '&', 'end'].includes(t.v);
  }
  private term(): string {
    let v = this.unary();
    for (;;) {
      if (this.is('cmd', 'cdot') || this.is('cmd', 'times') || this.isOp('*')) { this.i++; v = `${v} * ${this.unary()}`; }
      else if (this.is('cmd', 'div') || this.isOp('/')) { this.i++; v = `${v} / ${this.unary()}`; }
      else if (this.startsFactor()) v = `${v} * ${this.unary()}`;     // juxtaposition
      else return v;
    }
  }
  private unary(): string {
    if (this.isOp('-')) { this.i++; return `-${this.unary()}`; }
    if (this.isOp('+')) { this.i++; return this.unary(); }
    return this.power();
  }
  private power(): string {
    let v = this.atom();
    for (;;) {
      if (this.isOp('^')) {
        this.i++;
        if (this.isTranspose()) { v = `transpose(${v})`; continue; }
        v = `(${v})^(${this.arg()})`;
      } else if (this.isOp("'")) {
        this.i++;
        v = `transpose(${v})`;
      } else if (this.isOp('!')) {
        this.i++;
        v = `factorial(${v})`;
      } else return v;
    }
  }
  /** ^T, ^\top, ^{T}, ^{\top}: consumed if present. */
  private isTranspose(): boolean {
    const isT = (t: Tok | undefined) => !!t && ((t.t === 'sym' && t.v === 'T') || (t.t === 'cmd' && (t.v === 'top' || t.v === 'intercal')));
    if (isT(this.peek())) { this.i++; return true; }
    if (this.isOp('{') && isT(this.toks[this.i + 1]) && this.toks[this.i + 2]?.t === 'op' && this.toks[this.i + 2].v === '}') {
      this.i += 3; return true;
    }
    return false;
  }
  /** A `{…}` group or a single atom, as script arguments are: `\frac13` is 1/3, `x^23` is x² · 3. */
  private arg(): string {
    if (this.isOp('{')) { this.i++; const v = this.expr(); this.expectOp('}'); return v; }
    if (this.isOp('-')) { this.i++; return `-${this.arg()}`; }
    const t = this.peek();
    if (t?.t === 'num' && t.v.length > 1) {
      this.toks.splice(this.i, 1, { t: 'num', v: t.v[0] }, { t: 'num', v: t.v.slice(1) });
      if (this.toks[this.i + 1].v.startsWith('.')) throw new Error(`unexpected '.' after ${t.v[0]}`);
    }
    return this.atom();
  }
  private group(open: string, close: string): string {
    this.expectOp(open);
    const v = this.expr();
    if (this.is('cmd', 'right')) this.i++;
    this.expectOp(close);
    return `(${v})`;
  }
  private abs(): string {
    this.absDepth++;
    const v = this.expr();
    this.absDepth--;
    return `absdet(${v})`;
  }

  private atom(): string {
    const t = this.peek();
    if (!t) throw new Error('unexpected end');
    if (t.t === 'num') { this.i++; return t.v; }
    if (t.t === 'op') {
      if (t.v === '(') return this.group('(', ')');
      if (t.v === '[') return this.group('[', ']');
      if (t.v === '{') { this.i++; const v = this.expr(); this.expectOp('}'); return `(${v})`; }
      if (t.v === '|') {
        this.i++;
        const v = this.abs();
        if (this.is('cmd', 'right')) this.i++;
        this.expectOp('|');
        return v;
      }
      throw new Error(`unexpected '${t.v}'`);
    }
    if (t.t === 'cmd') {
      if (WORD_CMDS.has(t.v)) {
        const save = this.i;
        const name = this.name();
        if (name) return ident(name);
        this.i = save + 1;
        return this.atom();   // \mathrm{2} etc.: transparent
      }
      if (t.v === 'left') {
        this.i++;
        const d = this.take();
        const close = d.t === 'op' && d.v === '(' ? ')' : d.t === 'op' && d.v === '[' ? ']' : d.t === 'op' && d.v === '|' ? '|' : d.t === 'cmd' && d.v === '{' ? '}' : null;
        if (!close) throw new Error('unsupported \\left delimiter');
        const v = close === '|' ? this.abs() : `(${this.expr()})`;
        if (!this.is('cmd', 'right')) throw new Error('missing \\right');
        this.i++;
        const e = this.take();
        if (!((e.t === 'op' && e.v === close) || (e.t === 'cmd' && (e.v === close || e.v === '.')))) throw new Error('mismatched \\right');
        return v;
      }
      if (ACCENTS.has(t.v)) { const save = this.i; const name = this.name(); if (name) return ident(name); this.i = save; }
      if (SKIP.has(t.v)) { this.i++; return this.atom(); }   // \vec{v}, spacing: transparent
      if (t.v === 'frac' || t.v === 'dfrac' || t.v === 'tfrac') { this.i++; const a = this.arg(); const b = this.arg(); return `((${a}) / (${b}))`; }
      if (t.v === 'sqrt') {
        this.i++;
        if (this.isOp('[')) { this.i++; const n = this.expr(); this.expectOp(']'); return `nthRoot(${this.arg()}, ${n})`; }
        return `sqrt(${this.arg()})`;
      }
      if (t.v === 'begin') {
        this.i++;
        const env = this.envName();
        if (!MATRIX_ENVS.has(env)) throw new Error(`unsupported environment ${env}`);
        const m = this.matrix(env);
        return env === 'vmatrix' ? `det(${m})` : m;
      }
      if (t.v === 'operatorname') { this.i++; return this.applyFn(this.rawGroup()); }
      if (t.v in FUNCS) { this.i++; return this.applyFn(t.v); }
      if (t.v in CONSTS) { this.i++; return CONSTS[t.v]; }
      if (LETTERS.has(t.v)) return this.lookup();
      throw new Error(`unknown \\${t.v}`);
    }
    return this.lookup();
  }
  private lookup(): string {
    const name = this.name();
    if (!name) throw new Error('expected a name');
    return ident(name);
  }
  private applyFn(name: string): string {
    let f = FUNCS[name];
    if (!f) throw new Error(`unknown function ${name}`);
    // \sin^2 x, \log_2 x
    let p: string | null = null, base: string | null = null;
    if (this.isOp('^')) { this.i++; p = this.arg(); }
    if (this.isOp('_')) { this.i++; base = this.arg(); }
    if (this.isOp('^') && p === null) { this.i++; p = this.arg(); }
    if (p === '-1' && name in INVERSE) { f = INVERSE[name]; p = null; }
    if (MULTI.has(name) && (this.isOp('(') || this.isOp('{') || this.isOp('[') || this.is('cmd', 'left'))) {
      return `${f}(${this.argList().join(', ')})`;
    }
    const x = this.fnArg();
    let r = base !== null && (name === 'log' || name === 'ln') ? `log(${x}, ${base})` : `${f}(${x})`;
    if (p !== null) r = `(${r})^(${p})`;
    return r;
  }
  /** `(a, b)`, `{a, b}`, `[a, b]` or `\left( a, b \right)`: comma-separated arguments. */
  private argList(): string[] {
    const left = this.is('cmd', 'left');
    if (left) this.i++;
    const d = this.take();
    const close = d.t === 'op' && d.v === '(' ? ')' : d.t === 'op' && d.v === '[' ? ']' : d.t === 'op' && d.v === '{' ? '}' : null;
    if (!close) throw new Error('expected an argument list');
    const args = [this.expr()];
    while (this.isOp(',')) { this.i++; args.push(this.expr()); }
    if (left) { if (!this.is('cmd', 'right')) throw new Error('missing \\right'); this.i++; }
    this.expectOp(close);
    return args;
  }
  /** A bare function argument runs to the next operator or function: \sin 2\pi x is sin(2πx). */
  private fnArg(): string {
    if (this.isOp('(') || this.is('cmd', 'left')) return this.atom();
    let v = this.unary();
    while (this.startsFactor()) {
      const t = this.peek()!;
      if (t.t === 'cmd' && (t.v in FUNCS || t.v === 'operatorname')) break;
      v = `${v} * ${this.unary()}`;
    }
    return v;
  }
  private envName(): string {
    this.expectOp('{');
    let s = '';
    for (;;) { const t = this.take(); if (t.t === 'op' && t.v === '}') break; s += t.v; }
    return s;
  }
  private matrix(env: string): string {
    const rows: string[][] = [[]];
    for (;;) {
      const t = this.peek();
      if (!t) throw new Error('unterminated matrix');
      if (t.t === 'cmd' && t.v === 'end') { this.i++; if (this.envName() !== env) throw new Error('mismatched \\end'); break; }
      if (t.t === 'op' && t.v === '&') { this.i++; continue; }
      if (t.t === 'cmd' && t.v === '\\') { this.i++; rows.push([]); continue; }
      rows[rows.length - 1].push(this.expr());
    }
    if (rows[rows.length - 1].length === 0) rows.pop();
    const cols = rows[0]?.length ?? 0;
    if (!cols || rows.some(r => r.length !== cols)) throw new Error('ragged matrix');
    return `[${rows.map(r => `[${r.join(', ')}]`).join(', ')}]`;
  }
}

/* ── public ── */
const MATRIX_DIGITS = 6;

/** The math.js source for a LaTeX line. */
export function texToMath(tex: string): { name: string | null; src: string } {
  return new Parser(tokenize(tex)).statement();
}

/** Names the LaTeX layer needs in the math.js scope (installed once per scope). */
export function prepareScope(math: Math, scope: Scope): void {
  if (scope.has('absdet')) return;
  // |x| is abs for scalars and the determinant for matrices.
  scope.set('absdet', (v: unknown) => (math.isMatrix(v) ? math.det(v) : math.abs(v as number)));
}

export function evaluateTex(math: Math, tex: string, scope: Scope): { name: string | null; value: unknown } {
  prepareScope(math, scope);
  const { src } = texToMath(tex);
  const r = evalStatement(math, src, scope);
  if (math.isMatrix(r.value)) {
    for (const x of r.value.toArray().flat(2)) {
      if (math.isComplex(x)) continue;   // complex matrices are fine
      if (!math.isNumeric(x)) throw new Error(`matrix entry is not a number (${math.typeOf(x)})`);
      if (!isFinite(math.number(x as number))) throw new Error(`matrix has a non-finite entry (${formatValue(math, x)})`);
    }
  }
  return r;
}

export function numTex(math: Math, v: unknown, precision?: number): string {
  const s = formatValue(math, v, precision);
  if (s === 'NaN') return '\\mathrm{NaN}';
  if (s === 'Infinity') return '\\infty';
  if (s === '-Infinity') return '-\\infty';
  const m = /^(-?[\d.]+)e([+-]?\d+)$/.exec(s);
  if (m) return `${m[1]}\\times10^{${+m[2]}}`;
  return s;
}

export function toTex(math: Math, v: unknown): string {
  if (math.isMatrix(v)) {
    const a = v.toArray() as unknown[];
    const rows = (Array.isArray(a[0]) ? a : a.map(x => [x])) as unknown[][];
    // Entries are shown short so a rendered matrix stays readable; scalars keep the full 15 digits.
    return `\\begin{pmatrix}${rows.map(r => r.map(x => numTex(math, x, MATRIX_DIGITS)).join(' & ')).join(' \\\\ ')}\\end{pmatrix}`;
  }
  if (math.isComplex(v)) {
    const re = numTex(math, v.re), im = numTex(math, Math.abs(v.im));
    if (v.re === 0) return `${v.im < 0 ? '-' : ''}${im === '1' ? '' : im}i`;
    return `${re} ${v.im < 0 ? '-' : '+'} ${im === '1' ? '' : im}i`;
  }
  if (typeof v === 'string') return `\\text{${v}}`;
  return numTex(math, v);
}

/** True when a result would only echo the input (a bare number). */
export function trivialTex(tex: string): boolean {
  return /^\s*-?\d*\.?\d*\s*=?\s*$/.test(tex);
}
