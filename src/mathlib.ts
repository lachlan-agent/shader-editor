// The numeric core shared by the calculators: math.js configured for
// BigNumber arithmetic at 64 significant digits, extended with the functions
// a shader or low-level graphics author reaches for. Loaded on first use so
// the editor bundle stays small.
//
// Results below 1e-54 in magnitude are treated as zero: that is where the
// rounding noise of a 64-digit evaluation lives (sin(2π) ≈ 6e-64), far
// beneath anything a graphics value can be (f32 min ≈ 1e-38).

import type { MathJsInstance, BigNumber } from 'mathjs';

export type Math = MathJsInstance;

export const PRECISION = 64;
const CHOP = 1e-54;

export interface Help { name: string; sig: string; doc: string }
export interface Group { title: string; items: Help[] }

let instance: Promise<Math> | null = null;
let plotInstance: Promise<Math> | null = null;

/** The configured math.js instance; one import, reused by every pad. */
export function getMath(): Promise<Math> {
  if (!instance) {
    instance = import('mathjs').then(({ create, all }) => {
      const math = create(all, { number: 'BigNumber', precision: PRECISION });
      installExtensions(math);
      return math;
    }).catch(e => { instance = null; throw e; });
  }
  return instance;
}

/**
 * The same library in plain doubles, for work that evaluates an expression
 * thousands of times a frame (the graph): a BigNumber sin costs ~0.2 ms, a
 * double one ~1 µs. Same syntax and extension functions as the calculators.
 */
export function getPlotMath(): Promise<Math> {
  if (!plotInstance) {
    plotInstance = import('mathjs').then(({ create, all }) => {
      const math = create(all, { number: 'number' });
      installExtensions(math);
      return math;
    }).catch(e => { plotInstance = null; throw e; });
  }
  return plotInstance;
}

/* ── extensions ──────────────────────────────────────────── */
// Each group documents math.js built-ins worth knowing and adds our own.
// `fn` entries are imported; `override` replaces a math.js function of the
// same name (hex/bin/oct: upper-case, 32-bit unsigned).

type Fn = (...a: number[]) => number | string;
interface Def extends Help { fn?: Fn; override?: boolean }

const clamp = (x: number, a: number, b: number) => Math.min(Math.max(x, a), b);
const u32 = (x: number) => (x >>> 0);
const hex = (x: number, pad = 0) => '0x' + u32(x).toString(16).toUpperCase().padStart(pad, '0');
const scratchF32 = new Float32Array(1);
const scratchU32 = new Uint32Array(scratchF32.buffer);
const f32bits = (x: number) => { scratchF32[0] = x; return scratchU32[0]; };
const bitsF32 = (b: number) => { scratchU32[0] = u32(b); return scratchF32[0]; };

// IEEE half: round-to-nearest-even, like a GPU store.
function f16bits(x: number): number {
  const b = f32bits(x);
  const sign = (b >>> 16) & 0x8000;
  const exp = ((b >>> 23) & 0xff) - 127 + 15;
  let mant = b & 0x7fffff;
  if (((b >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);   // inf / nan
  if (exp >= 31) return sign | 0x7c00;                                            // overflow
  if (exp <= 0) {                                                                 // subnormal or zero
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1))) h++;
    return sign | h;
  }
  let h = sign | (exp << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
  return h;
}
function bitsF16(h: number): number {
  h = h & 0xffff;
  const sign = h & 0x8000 ? -1 : 1, exp = (h >>> 10) & 0x1f, mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}
const srgbEnc = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
const srgbDec = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const byte = (x: number) => clamp(Math.round(x * 255), 0, 255);
const hex2 = (c: number) => clamp(Math.round(c), 0, 255).toString(16).padStart(2, '0');
const smooth = (a: number, b: number, x: number) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

export const GROUPS: Array<{ title: string; defs: Def[] }> = [
  { title: 'Arithmetic', defs: [
    { name: 'abs', sig: 'abs(x)', doc: 'absolute value' },
    { name: 'sign', sig: 'sign(x)', doc: '-1, 0 or 1' },
    { name: 'floor', sig: 'floor(x)', doc: 'round down' },
    { name: 'ceil', sig: 'ceil(x)', doc: 'round up' },
    { name: 'round', sig: 'round(x [, digits])', doc: 'round to nearest' },
    { name: 'trunc', sig: 'trunc(x)', doc: 'drop the fraction (fix is the same)', fn: Math.trunc },
    { name: 'fract', sig: 'fract(x)', doc: 'x - floor(x)', fn: x => x - Math.floor(x) },
    { name: 'mod', sig: 'mod(a, b)', doc: 'GLSL mod, always ≥ 0 for b > 0; a % b is the same' },
    { name: 'min', sig: 'min(a, b, …)', doc: 'smallest' },
    { name: 'max', sig: 'max(a, b, …)', doc: 'largest' },
    { name: 'clamp', sig: 'clamp(x, lo, hi)', doc: 'limit to a range', fn: clamp },
    { name: 'sat', sig: 'sat(x)', doc: 'clamp(x, 0, 1)', fn: x => clamp(x, 0, 1) },
    { name: 'sqrt', sig: 'sqrt(x)', doc: 'square root' },
    { name: 'cbrt', sig: 'cbrt(x)', doc: 'cube root' },
    { name: 'nthRoot', sig: 'nthRoot(x, n)', doc: 'n-th root' },
    { name: 'inversesqrt', sig: 'inversesqrt(x)', doc: '1 / sqrt(x)', fn: x => 1 / Math.sqrt(x) },
    { name: 'pow', sig: 'pow(x, y)', doc: 'x^y' },
    { name: 'exp', sig: 'exp(x)', doc: 'e^x' },
    { name: 'exp2', sig: 'exp2(x)', doc: '2^x', fn: x => 2 ** x },
    { name: 'log', sig: 'log(x [, base])', doc: 'natural log, or any base' },
    { name: 'ln', sig: 'ln(x)', doc: 'natural log', fn: Math.log },
    { name: 'log2', sig: 'log2(x)', doc: 'base-2 log' },
    { name: 'log10', sig: 'log10(x)', doc: 'base-10 log' },
    { name: 'hypot', sig: 'hypot(x, y, …)', doc: 'vector length' },
    { name: 'length', sig: 'length(x, y, …)', doc: 'vector length', fn: Math.hypot },
    { name: 'factorial', sig: 'factorial(n)', doc: 'n! (postfix ! works too)' },
    { name: 'gcd', sig: 'gcd(a, b)', doc: 'greatest common divisor' },
    { name: 'lcm', sig: 'lcm(a, b)', doc: 'least common multiple' },
  ] },
  { title: 'Trigonometry', defs: [
    { name: 'sin', sig: 'sin(x)', doc: 'radians' }, { name: 'cos', sig: 'cos(x)', doc: '' }, { name: 'tan', sig: 'tan(x)', doc: '' },
    { name: 'asin', sig: 'asin(x)', doc: '' }, { name: 'acos', sig: 'acos(x)', doc: '' }, { name: 'atan', sig: 'atan(x)', doc: '' },
    { name: 'atan2', sig: 'atan2(y, x)', doc: 'angle of (x, y)' },
    { name: 'sinh', sig: 'sinh(x)', doc: '' }, { name: 'cosh', sig: 'cosh(x)', doc: '' }, { name: 'tanh', sig: 'tanh(x)', doc: '' },
    { name: 'radians', sig: 'radians(deg)', doc: 'degrees to radians', fn: d => d * Math.PI / 180 },
    { name: 'degrees', sig: 'degrees(rad)', doc: 'radians to degrees', fn: r => r * 180 / Math.PI },
  ] },
  { title: 'Interpolation', defs: [
    { name: 'mix', sig: 'mix(a, b, t)', doc: 'linear blend (lerp is the same)', fn: (a, b, t) => a + (b - a) * t },
    { name: 'lerp', sig: 'lerp(a, b, t)', doc: 'linear blend', fn: (a, b, t) => a + (b - a) * t },
    { name: 'unlerp', sig: 'unlerp(a, b, x)', doc: 'where x sits between a and b, 0..1', fn: (a, b, x) => (x - a) / (b - a) },
    { name: 'remap', sig: 'remap(x, a, b, c, d)', doc: 'map x from [a,b] onto [c,d]', fn: (x, a, b, c, d) => c + (d - c) * (x - a) / (b - a) },
    { name: 'step', sig: 'step(edge, x)', doc: '0 below edge, else 1', fn: (edge, x) => (x < edge ? 0 : 1) },
    { name: 'smoothstep', sig: 'smoothstep(a, b, x)', doc: 'Hermite ramp from a to b', fn: smooth },
  ] },
  { title: 'Floats', defs: [
    { name: 'f32', sig: 'f32(x)', doc: 'round to single precision', fn: Math.fround },
    { name: 'f16', sig: 'f16(x)', doc: 'round to half precision', fn: x => bitsF16(f16bits(x)) },
    { name: 'bits', sig: 'bits(x)', doc: 'IEEE-754 single bits of x, as hex', fn: x => hex(f32bits(x), 8) },
    { name: 'bits16', sig: 'bits16(x)', doc: 'IEEE-754 half bits of x, as hex', fn: x => hex(f16bits(x), 4) },
    { name: 'fromBits', sig: 'fromBits(u)', doc: 'the float with single bits u', fn: bitsF32 },
    { name: 'fromBits16', sig: 'fromBits16(u)', doc: 'the float with half bits u', fn: bitsF16 },
    { name: 'ulp', sig: 'ulp(x)', doc: 'spacing of f32 neighbours at x', fn: x => { const b = f32bits(Math.abs(x)); return bitsF32(b + 1) - bitsF32(b); } },
    { name: 'eps', sig: 'eps(x)', doc: 'f32 relative precision: ulp(x) / x', fn: x => { const b = f32bits(Math.abs(x)); return (bitsF32(b + 1) - bitsF32(b)) / Math.abs(x); } },
  ] },
  { title: 'Integers & bits', defs: [
    { name: 'hex', sig: 'hex(x [, digits])', doc: 'unsigned 32-bit as hex', fn: (x, d) => hex(x, d ?? 0), override: true },
    { name: 'bin', sig: 'bin(x [, digits])', doc: 'unsigned 32-bit as binary', fn: (x, d) => '0b' + u32(x).toString(2).padStart(d ?? 0, '0'), override: true },
    { name: 'oct', sig: 'oct(x)', doc: 'unsigned 32-bit as octal', fn: x => '0o' + u32(x).toString(8), override: true },
    { name: 'u32', sig: 'u32(x)', doc: 'wrap to unsigned 32-bit', fn: u32 },
    { name: 'i32', sig: 'i32(x)', doc: 'wrap to signed 32-bit', fn: x => (x | 0) },
    { name: 'u16', sig: 'u16(x)', doc: 'wrap to 16 bits', fn: x => u32(x) & 0xffff },
    { name: 'u8', sig: 'u8(x)', doc: 'wrap to 8 bits', fn: x => u32(x) & 0xff },
    { name: 'bxor', sig: 'bxor(a, b)', doc: 'bitwise exclusive or (a ^| b; ^ is power)', fn: (a, b) => u32(a ^ b) },
    { name: 'popcnt', sig: 'popcnt(x)', doc: 'set bits of the low 32', fn: x => { let n = u32(x), c = 0; while (n) { n &= n - 1; c++; } return c; } },
    { name: 'clz', sig: 'clz(x)', doc: 'leading zero bits of 32', fn: x => Math.clz32(x) },
    { name: 'ctz', sig: 'ctz(x)', doc: 'trailing zero bits', fn: x => { const n = u32(x); return n ? 31 - Math.clz32(n & -n) : 32; } },
    { name: 'rol', sig: 'rol(x, n)', doc: 'rotate 32 bits left', fn: (x, n) => u32((x << (n & 31)) | (u32(x) >>> (32 - (n & 31)))) },
    { name: 'ror', sig: 'ror(x, n)', doc: 'rotate 32 bits right', fn: (x, n) => u32((u32(x) >>> (n & 31)) | (x << (32 - (n & 31)))) },
    { name: 'bit', sig: 'bit(n)', doc: '1 << n', fn: n => u32(1 << n) },
    { name: 'mask', sig: 'mask(n)', doc: 'lowest n bits set', fn: n => (n >= 32 ? 0xffffffff : u32((1 << n) - 1)) },
    { name: 'ispow2', sig: 'ispow2(x)', doc: '1 if x is a power of two', fn: x => (x > 0 && (u32(x) & (u32(x) - 1)) === 0 ? 1 : 0) },
    { name: 'nextpow2', sig: 'nextpow2(x)', doc: 'smallest power of two ≥ x', fn: x => (x <= 1 ? 1 : 2 ** Math.ceil(Math.log2(x))) },
    { name: 'ilog2', sig: 'ilog2(x)', doc: 'floor(log2(x))', fn: x => Math.floor(Math.log2(x)) },
    { name: 'align', sig: 'align(x, a)', doc: 'round x up to a multiple of a', fn: (x, a) => Math.ceil(x / a) * a },
  ] },
  { title: 'Colour', defs: [
    { name: 'byte', sig: 'byte(x)', doc: '0..1 to 0..255, rounded', fn: byte },
    { name: 'norm', sig: 'norm(n)', doc: '0..255 to 0..1 (norm([x, y]) is a vector length)' },
    { name: 'srgb', sig: 'srgb(linear)', doc: 'linear to sRGB-encoded, 0..1', fn: srgbEnc },
    { name: 'linear', sig: 'linear(srgb)', doc: 'sRGB-encoded to linear, 0..1', fn: srgbDec },
    { name: 'luma', sig: 'luma(r, g, b)', doc: 'Rec.709 luminance of linear rgb', fn: (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b },
    { name: 'rgb', sig: 'rgb(r, g, b)', doc: '0..1 channels to #RRGGBB', fn: (r, g, b) => '#' + [r, g, b].map(c => hex2(byte(c))).join('').toUpperCase() },
    { name: 'rgb8', sig: 'rgb8(r, g, b)', doc: '0..255 channels to #RRGGBB', fn: (r, g, b) => '#' + [r, g, b].map(hex2).join('').toUpperCase() },
    { name: 'pack', sig: 'pack(r, g, b [, a])', doc: '0..1 channels into 0xAABBGGRR (GL byte order)', fn: (r, g, b, a) => hex((byte(a ?? 1) << 24) | (byte(b) << 16) | (byte(g) << 8) | byte(r), 8) },
    { name: 'unpack', sig: 'unpack(u)', doc: '0xAABBGGRR back to r, g, b, a', fn: u => [u & 255, (u >>> 8) & 255, (u >>> 16) & 255, (u >>> 24) & 255].map(c => String(+(c / 255).toPrecision(12))).join(', ') },
    { name: 'kelvin', sig: 'kelvin(T)', doc: 'blackbody tint as #RRGGBB (1000..40000 K)', fn: k => {
      const t = clamp(k, 1000, 40000) / 100;
      const r = t <= 66 ? 255 : 329.698727446 * (t - 60) ** -0.1332047592;
      const g = t <= 66 ? 99.4708025861 * Math.log(t) - 161.1195681661 : 288.1221695283 * (t - 60) ** -0.0755148492;
      const b = t >= 66 ? 255 : t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
      return '#' + [r, g, b].map(hex2).join('').toUpperCase();
    } },
  ] },
  { title: 'Geometry', defs: [
    { name: 'dist', sig: 'dist(x1, y1, x2, y2)', doc: 'distance between two points', fn: (x1, y1, x2, y2) => Math.hypot(x2 - x1, y2 - y1) },
    { name: 'dot', sig: 'dot([ax, ay], [bx, by])', doc: 'dot product' },
    { name: 'cross', sig: 'cross([ax, ay, az], [bx, by, bz])', doc: 'cross product' },
    { name: 'fov', sig: 'fov(deg, aspect)', doc: 'projection scales: 1/tan(fov/2) for y, and x', fn: (d, a) => { const y = 1 / Math.tan(d * Math.PI / 360); return `y ${+y.toPrecision(10)}  x ${+(y / a).toPrecision(10)}`; } },
    { name: 'aspect', sig: 'aspect(w, h)', doc: 'w / h', fn: (w, h) => w / h },
    { name: 'texel', sig: 'texel(size)', doc: '1 / size: one texel in uv', fn: s => 1 / s },
    { name: 'mip', sig: 'mip(w, h)', doc: 'mip levels for a w × h texture', fn: (w, h) => Math.floor(Math.log2(Math.max(w, h))) + 1 },
  ] },
  { title: 'Matrices', defs: [
    { name: 'det', sig: 'det(A)', doc: 'determinant; A = [1, 2; 3, 4]' },
    { name: 'inv', sig: 'inv(A)', doc: 'inverse (A^-1 is the same)' },
    { name: 'transpose', sig: 'transpose(A)', doc: "transpose (A' is the same)" },
    { name: 'trace', sig: 'trace(A)', doc: 'sum of the diagonal' },
    { name: 'identity', sig: 'identity(n)', doc: 'n × n identity' },
    { name: 'zeros', sig: 'zeros(m, n)', doc: 'm × n zero matrix' },
    { name: 'size', sig: 'size(A)', doc: 'dimensions' },
  ] },
];

export const SYNTAX: Array<[string, string]> = [
  ['0x1F  0b1010  0o17', 'hex, binary, octal literals'],
  ['x = 2^10', 'assign (then use x); ans is the last result'],
  ['+ - * / % ^ !', 'arithmetic; ^ is power, % is mod, ! factorial'],
  ['& | ^| << >> ~', 'bitwise, arbitrary width like Python; u32(x) wraps'],
  ['[1, 2; 3, 4]  [1, 2, 3]', 'matrix (rows by ;) and vector'],
  ['2 + 3i', 'complex numbers'],
  ['1e-3  2.5e6', 'exponent notation'],
  ['pi tau e phi i inf', 'constants; epsilon fltmax fltmin for f32'],
  ['64 significant digits', 'BigNumber arithmetic; shown to 15'],
  ['help · clear', 'this list in the log · wipe the log'],
  ['↑ ↓', 'walk the history; click a line to reuse it, a result to copy it'],
];

export const CONSTS: Record<string, number> = {
  inf: Infinity, epsilon: 2 ** -23, fltmax: 3.4028234663852886e38, fltmin: 1.1754943508222875e-38,
};
/** Names the user may not rebind. */
const RESERVED = new Set(['pi', 'e', 'tau', 'phi', 'i', 'Infinity', 'NaN', 'true', 'false', 'null', ...Object.keys(CONSTS)]);

export function helpGroups(): Group[] {
  return GROUPS.map(g => ({ title: g.title, items: g.defs.map(({ name, sig, doc }) => ({ name, sig, doc })) }));
}

function installExtensions(math: Math): void {
  // Our functions think in JS doubles: unwrap BigNumber arguments and wrap
  // numeric results back, so later arithmetic stays at full precision.
  // (A number-mode instance keeps doubles throughout.)
  const big = math.config({}).number === 'BigNumber';
  const num = (v: number): unknown => (big ? math.bignumber(v) : v);
  const isVec = (a: unknown) => math.isMatrix(a) || Array.isArray(a);
  const wrap = (name: string, fn: Fn) => {
    const call = (...args: unknown[]): unknown => {
      // One vector argument maps element-wise (the others broadcast); more is ambiguous.
      const vecs = args.filter(isVec);
      if (vecs.length > 1) throw new Error(`${name}: only one argument may be a vector`);
      if (vecs.length === 1) {
        const at = args.indexOf(vecs[0]);
        return math.map(vecs[0] as number[], ((x: unknown) => call(...args.slice(0, at), x, ...args.slice(at + 1))) as (x: number) => number);
      }
      const nums = args.map(a => {
        if (typeof a === 'number') return a;
        if (math.isBigNumber(a)) return math.number(a);
        throw new Error(`${name} expects numbers, not ${math.typeOf(a)}`);
      });
      const r = fn(...nums);
      return typeof r === 'number' ? num(r) : r;
    };
    return call;
  };
  const defs: Record<string, unknown> = {};
  for (const g of GROUPS) for (const d of g.defs) if (d.fn) defs[d.name] = wrap(d.name, d.fn);
  for (const [k, v] of Object.entries(CONSTS)) defs[k] = num(v);
  // norm(v) stays the vector length; norm(n) on a number is 0..255 → 0..1.
  const vecNorm = math.norm;
  defs.norm = (v: unknown) => (math.isMatrix(v) || Array.isArray(v) ? vecNorm(v as number[]) : math.divide(v as BigNumber, 255));
  // math.js also has units (byte, t, radians…); our names win.
  math.import(defs, { override: true });
}

/* ── evaluation ──────────────────────────────────────────── */
export type Scope = Map<string, unknown>;

/** Values indistinguishable from rounding noise become exactly zero. */
export function chop(math: Math, v: unknown): unknown {
  if (math.isBigNumber(v)) return v.abs().lt(CHOP) ? math.bignumber(0) : v;
  if (typeof v === 'number') return Math.abs(v) < CHOP ? 0 : v;
  if (math.isMatrix(v)) return v.map(x => chop(math, x));
  if (math.isComplex(v)) {
    // Complex arithmetic runs in doubles, so its noise floor is higher.
    const scale = Math.max(Math.abs(v.re), Math.abs(v.im), 1) * 1e-15;
    if (Math.abs(v.re) < scale) v.re = 0;
    if (Math.abs(v.im) < scale) v.im = 0;
    return v.im === 0 ? math.bignumber(v.re) : v;
  }
  return v;
}

/** Housekeeping exports of the math.js instance that are not values an expression may use. */
const NOT_VALUES = new Set(['config', 'import', 'create', 'typed', 'expression', 'version', 'on', 'off', 'once', 'emit', 'chain', 'evaluate', 'parse', 'compile']);

/** Does the library bind `name` to something an expression can use: a function or a numeric constant? */
export function libHas(math: Math, name: string): boolean {
  if (NOT_VALUES.has(name) || !Object.prototype.hasOwnProperty.call(math, name)) return false;
  const v = (math as unknown as Record<string, unknown>)[name];
  return typeof v === 'function' || typeof v === 'number' || math.isBigNumber(v) || math.isComplex(v);
}

/** math.js resolves an unknown name as a unit (t, m, byte…); we want an error instead. */
function checkSymbols(math: Math, node: import('mathjs').MathNode, scope: Scope): void {
  node.traverse((n, path, parent) => {
    if (!math.isSymbolNode(n)) return;
    if (parent && math.isAssignmentNode(parent) && path === 'object') return;
    const name = n.name;
    if (scope.has(name) || libHas(math, name)) return;
    throw new Error(`unknown ${name}`);
  });
}

const ASSIGN = /^\s*([A-Za-z_]\w*)\s*=(?!=)/;

/** Evaluate one statement in a scope. Rejects rebinding functions and constants. */
export function evalStatement(math: Math, src: string, scope: Scope): { name: string | null; value: unknown } {
  src = src.replace(/[\s;]+$/, '');   // a trailing `;` would wrap the value in a ResultSet
  const m = ASSIGN.exec(src);
  const name = m ? m[1] : null;
  if (name && (RESERVED.has(name) || typeof (math as unknown as Record<string, unknown>)[name] === 'function')) {
    throw new Error(`${name} is built in and cannot be assigned`);
  }
  const node = math.parse(src);
  checkSymbols(math, node, scope);
  const raw = node.compile().evaluate(scope);
  if (math.isResultSet(raw)) throw new Error('one statement per line');
  const value = chop(math, raw);
  if (name) scope.set(name, value);
  return { name, value };
}

/** Human formatting: 15 significant digits, exponent only when it earns it. */
export function formatValue(math: Math, v: unknown, precision = 15): string {
  if (typeof v === 'string') return v;
  return math.format(v, { precision, lowerExp: -7, upperExp: 15 });
}
