import {
  EditorView, keymap, lineNumbers, highlightActiveLineGutter,
  highlightSpecialChars, drawSelection, dropCursor,
  rectangularSelection, crosshairCursor, highlightActiveLine,
  hoverTooltip, showTooltip, Tooltip,
} from '@codemirror/view';
import { EditorState, Compartment, StateField, StateEffect } from '@codemirror/state';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import {
  StreamLanguage, syntaxHighlighting, HighlightStyle, StringStream,
  indentOnInput, bracketMatching, foldGutter, foldKeymap,
} from '@codemirror/language';
import { tags } from '@lezer/highlight';
import { vim, Vim } from '@replit/codemirror-vim';
import { rewriteControls, replaceLiteral, controlSignature, type Control } from './controls';
import {
  autocompletion, completionKeymap, closeBrackets, closeBracketsKeymap,
  CompletionContext, CompletionResult,
} from '@codemirror/autocomplete';
import { searchKeymap, highlightSelectionMatches } from '@codemirror/search';
import { lintKeymap, lintGutter, setDiagnostics, Diagnostic } from '@codemirror/lint';
import { lookupGlslDoc, GlslDoc } from './glslDocs';
import { makeScratch, type ScratchKind, type ScratchPad } from './scratch';
import { makeMathPad } from './eeltex';
import { formatGlsl } from './glslFormat';

/* ── Default project ─────────────────────────────────────── */
// A Gray–Scott reaction–diffusion system: Buffer A steps the simulation
// (sampling itself on iChannel0) and the Image pass colours the result.
const DEFAULT_SHADER = `// Reaction–diffusion — the Image pass only colours Buffer A.
// The simulation itself lives in the "Buf A" tab. Click and drag to stir.

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float v = texture(iChannel0, uv).y;   // concentration of chemical v

    vec3 col = mix(vec3(0.02, 0.03, 0.08), vec3(0.10, 0.45, 0.70), smoothstep(0.0, 0.2, v));
    col = mix(col, vec3(1.00, 0.85, 0.55), smoothstep(0.2, 0.4, v));

    fragColor = vec4(col, 1.0);
}
`;

const DEFAULT_BUFFER_A = `// Gray–Scott reaction–diffusion: u + 2v -> 3v, with u fed in and v drained.
// Each texel stores (1 - u, v), so an all-black buffer is the quiet state
// u = 1, v = 0. Try FEED/KILL pairs like 0.037/0.060, 0.029/0.057, 0.025/0.060.

const float DU   = 1.0;     // diffusion rate of u
const float DV   = 0.5;     // diffusion rate of v
const float FEED = 0.055;   // how fast u is replenished
const float KILL = 0.062;   // how fast v is removed

vec2 state(ivec2 p) {
    ivec2 size = ivec2(iResolution.xy);
    vec2 s = texelFetch(iChannel0, (p + size) % size, 0).xy;   // wrap around edges
    return vec2(1.0 - s.x, s.y);
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 p = ivec2(fragCoord);
    vec2 c = state(p);

    // 3x3 Laplacian: edges weigh 0.2, corners 0.05.
    vec2 lap = -c
        + 0.20 * (state(p + ivec2( 1, 0)) + state(p + ivec2(-1, 0))
                + state(p + ivec2( 0, 1)) + state(p + ivec2( 0,-1)))
        + 0.05 * (state(p + ivec2( 1, 1)) + state(p + ivec2(-1, 1))
                + state(p + ivec2( 1,-1)) + state(p + ivec2(-1,-1)));

    float u = c.x, v = c.y;
    float uvv = u * v * v;
    u += DU * lap.x - uvv + FEED * (1.0 - u);
    v += DV * lap.y + uvv - (FEED + KILL) * v;

    // Scatter seed droplets on the first frames, and whenever the buffer was
    // just (re)allocated, e.g. after a resize — fresh textures read alpha 0.
    if (iFrame < 2 || texelFetch(iChannel0, p, 0).a < 0.5) {
        vec2 cell = floor(fragCoord / 48.0);
        vec2 centre = (cell + 0.25 + 0.5 * vec2(hash(cell), hash(cell + 7.0))) * 48.0;
        bool seed = hash(cell + 3.0) < 0.4 && length(fragCoord - centre) < 6.0;
        u = seed ? 0.5 : 1.0;
        v = seed ? 0.25 : 0.0;
    }

    // Mouse drag paints in more v.
    if (iMouse.z > 0.0 && length(fragCoord - iMouse.xy) < 10.0) v = 0.5;

    fragColor = vec4(clamp(vec2(1.0 - u, v), 0.0, 1.0), 0.0, 1.0);
}
`;

/** Starter code for the Common tab. */
const DEFAULT_COMMON = `// Common — pasted into every pass ahead of its own code, so constants
// and helper functions defined here can be used anywhere.
// Uniforms like iTime work here; the iChannels do not.

const float PI = 3.14159265359;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}
`;

/** Starter code for a newly added buffer: the wiring hint and an empty mainImage. */
function defaultBufferShader(name: string): string {
  return `// Buffer ${name} — renders to an offscreen framebuffer.
// Read it from another pass (or from itself, for feedback) by wiring an
// iChannel to "Buffer ${name}" in the strip under the preview.

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = vec4(uv, 0.0, 1.0);
}
`;
}

/** What "New" opens: the smallest thing that visibly works. */
const BLANK_SHADER = `void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
    fragColor = vec4(col, 1.0);
}
`;

/* ── WebGL setup ─────────────────────────────────────────── */
const canvas = document.getElementById('glcanvas') as HTMLCanvasElement;
const glOrNull = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: false });

if (!glOrNull) {
  document.getElementById('toolbar')!.insertAdjacentHTML('beforeend', '<span class="pill err">WebGL2 not supported</span>');
  throw new Error('WebGL2 not supported');
}

// Rebind so every closure sees the narrowed non-null type.
const gl: WebGL2RenderingContext = glOrNull;

const CUBE_FACES = [
  { target: gl.TEXTURE_CUBE_MAP_POSITIVE_X, label: '+X' },
  { target: gl.TEXTURE_CUBE_MAP_NEGATIVE_X, label: '-X' },
  { target: gl.TEXTURE_CUBE_MAP_POSITIVE_Y, label: '+Y' },
  { target: gl.TEXTURE_CUBE_MAP_NEGATIVE_Y, label: '-Y' },
  { target: gl.TEXTURE_CUBE_MAP_POSITIVE_Z, label: '+Z' },
  { target: gl.TEXTURE_CUBE_MAP_NEGATIVE_Z, label: '-Z' },
] as const;

const VERT_SRC = `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
`;

const FRAG_STATIC_PREFIX = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp samplerCube;

uniform float iTime;
uniform float iTimeDelta;
uniform float iFrameRate;
uniform int   iFrame;
uniform vec3  iResolution;
uniform vec4  iMouse;
uniform vec4  iDate;
uniform float iSampleRate;
uniform float iChannelTime[4];
uniform vec3  iChannelResolution[4];
`;

// Error logs name source 0 for a pass's own code and source 1 for Common;
// the #line directives below also restart line numbers at each.
const COMMON_SOURCE = 1;

/** Pass code as compiled: annotated consts become uniforms. Off for export, which keeps the source as written. */
function passCode(code: string, rewrite: boolean): string {
  return rewrite ? rewriteControls(code).code : code;
}

/** Common code, ready to splice in after the uniforms, or '' if it's off. */
function commonBlock(rewrite = true): string {
  return commonPass.enabled ? `#line 1 ${COMMON_SOURCE}\n${passCode(commonPass.code, rewrite)}\n` : '';
}

function fragPrefix(channels: Channel[], rewrite = true): string {
  const decls = channels
    .map((c, i) => `uniform ${c.kind === 'cube' ? 'samplerCube' : 'sampler2D'} iChannel${i};`)
    .join('\n');
  return FRAG_STATIC_PREFIX + commonBlock(rewrite) + decls + '\nout vec4 _fragOut;\n';
}

/** Full fragment source for a pass, as handed to the compiler. */
function fragSource(userCode: string, channels: Channel[], rewrite = true, image = false): string {
  return fragPrefix(channels, rewrite) + '#line 1 0\n' + passCode(userCode, rewrite) + (image ? IMAGE_SUFFIX : BUFFER_SUFFIX);
}

// As on Shadertoy: fragColor starts as opaque black, so a mainImage that
// writes only .rgb still gets alpha 1; the Image pass always ends opaque,
// buffers keep whatever alpha was written.
const BUFFER_SUFFIX = `
void main() { vec4 color = vec4(0.0, 0.0, 0.0, 1.0); mainImage(color, gl_FragCoord.xy); _fragOut = color; }
`;
const IMAGE_SUFFIX = `
void main() { vec4 color = vec4(0.0, 0.0, 0.0, 1.0); mainImage(color, gl_FragCoord.xy); _fragOut = vec4(color.xyz, 1.0); }
`;

let quadBuf: WebGLBuffer | null = null;
let vao: WebGLVertexArrayObject | null = null;

function initQuad(): void {
  const verts = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
  quadBuf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
}

// Shows a buffer's texture on screen, for previews that look at a pass
// other than Image.
const BLIT_FRAG = `#version 300 es
precision highp float;
uniform sampler2D u_tex;
uniform vec2 u_res;
out vec4 o;
void main() { o = vec4(texture(u_tex, gl_FragCoord.xy / u_res).rgb, 1.0); }
`;
let blitProg: WebGLProgram | null = null;
let blitRes: WebGLUniformLocation | null = null;

function ensureBlit(): WebGLProgram | null {
  if (blitProg) return blitProg;
  const vs = compileShader(VERT_SRC, gl.VERTEX_SHADER).sh;
  const fs = compileShader(BLIT_FRAG, gl.FRAGMENT_SHADER).sh;
  if (!vs || !fs) return null;
  blitProg = linkProgram(vs, fs).prog;
  if (blitProg) {
    gl.useProgram(blitProg);
    gl.uniform1i(gl.getUniformLocation(blitProg, 'u_tex'), 0);
    blitRes = gl.getUniformLocation(blitProg, 'u_res');
  }
  return blitProg;
}

function compileShader(src: string, type: number): { sh: WebGLShader | null; log: string | null } {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = cleanLog(gl.getShaderInfoLog(sh));
    gl.deleteShader(sh);
    return { sh: null, log };
  }
  return { sh, log: null };
}

function linkProgram(vs: WebGLShader, fs: WebGLShader): { prog: WebGLProgram | null; log: string | null } {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.bindAttribLocation(prog, 0, 'a_pos');
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = cleanLog(gl.getProgramInfoLog(prog));
    gl.deleteProgram(prog);
    return { prog: null, log };
  }
  return { prog, log: null };
}

/** Some drivers end info logs with a NUL and blank lines, which rendered as a stray glyph. */
function cleanLog(log: string | null): string | null {
  if (log == null) return null;
  const s = log.replace(/[\0\r]/g, '').trimEnd();
  return s || 'compile failed (no details from the driver)';
}

/** Line numbers already match the editor; just say which tab they're in. */
function labelErrors(log: string): string {
  return log.replace(new RegExp(`\\b${COMMON_SOURCE}:(\\d+):`, 'g'), 'Common:$1:');
}

/**
 * Compile Common on its own, so a mistake there is reported once rather
 * than once per pass. Returns the error log, or null if it compiles.
 */
function checkCommon(): string | null {
  if (!commonPass.enabled) return null;
  const { sh, log } = compileShader(FRAG_STATIC_PREFIX + commonBlock() + 'void main() {}\n',
                                    gl.FRAGMENT_SHADER);
  if (!sh) return labelErrors(log ?? '');
  gl.deleteShader(sh);
  return null;
}

function buildProgram(userCode: string, channels: Channel[], image: boolean): { prog: WebGLProgram | null; log: string | null } {
  const fragSrc = fragSource(userCode, channels, true, image);

  const { sh: vs, log: vsLog } = compileShader(VERT_SRC, gl.VERTEX_SHADER);
  if (!vs) return { prog: null, log: 'Vertex shader error:\n' + vsLog };

  const { sh: fs, log: fsLog } = compileShader(fragSrc, gl.FRAGMENT_SHADER);
  if (!fs) {
    gl.deleteShader(vs);
    return { prog: null, log: labelErrors(fsLog ?? '') };
  }

  const { prog, log: linkLog } = linkProgram(vs, fs);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!prog) return { prog: null, log: linkLog };

  return { prog, log: null };
}

/* ── Uniform locations cache ─────────────────────────────── */
type Uniforms = Record<string, WebGLUniformLocation | null>;

function cacheUniforms(prog: WebGLProgram, extra: string[] = []): Uniforms {
  const names = ['iTime','iTimeDelta','iFrameRate','iFrame','iResolution','iMouse','iDate','iSampleRate',
                  'iChannelResolution',
                  'iChannel0','iChannel1','iChannel2','iChannel3', ...extra];
  const u: Uniforms = {};
  names.forEach(n => { u[n] = gl.getUniformLocation(prog, n); });
  return u;
}

/* ── Passes and channels ─────────────────────────────────── */
const NUM_SLOTS = 4;
const NUM_BUFFERS = 4;
const BUFFER_NAMES = ['A', 'B', 'C', 'D'] as const;

/** What an iChannel of a given pass is wired to. */
type Filter = 'linear' | 'nearest' | 'mipmap';
type Wrap = 'clamp' | 'repeat';
type BufFormat = 'rgba8' | 'rgba16f' | 'rgba32f';
const FILTERS: Filter[] = ['linear', 'nearest', 'mipmap'];
const WRAPS: Wrap[] = ['clamp', 'repeat'];
const BUF_FORMATS: Array<[BufFormat, string]> = [['rgba8', '8-bit'], ['rgba16f', '16-bit float'], ['rgba32f', '32-bit float']];

interface Channel {
  kind: '2d' | 'cube' | 'buffer';
  buf: number;                            // source buffer index when kind === 'buffer'
  filter: Filter;                         // sampling, applied when the texture is bound
  wrap: Wrap;
  vflip: boolean;                         // 2D images: upload flipped (GL's origin is bottom-left)
  img: HTMLImageElement | null;           // the uploaded image, kept so vflip can re-upload it
  tex: WebGLTexture | null;               // uploaded 2D image
  imgUrl: string | null;                  // thumbnail for the slot
  texW: number;                           // source size, for iChannelResolution
  texH: number;
  cubeTex: WebGLTexture | null;
  cubeFaceUrls: Array<string | null>;
  cubeFaceLoaded: boolean[];
  generation: number;                     // bumped on reset, so late image loads can be dropped
}

function makeChannel(): Channel {
  return {
    kind: '2d', buf: 0, filter: 'linear', wrap: 'repeat', vflip: false, img: null, tex: null, imgUrl: null, texW: 0, texH: 0, cubeTex: null,
    cubeFaceUrls: Array(6).fill(null), cubeFaceLoaded: Array(6).fill(false), generation: 0,
  };
}

function channelDescription(c: Channel): string {
  if (c.kind === 'buffer') return `Buffer ${BUFFER_NAMES[c.buf]}`;
  if (c.kind === 'cube')   return c.cubeFaceLoaded.some(Boolean) ? 'cubemap' : 'cubemap (empty)';
  return c.tex ? '2D texture' : '2D texture (empty)';
}
/** Sampling settings that differ from the defaults, for labels and the export header. */
function samplingNote(c: Channel): string {
  if (c.kind === 'cube') return '';
  const parts = [];
  if (c.filter !== 'linear') parts.push(c.filter);
  if (c.wrap !== (c.kind === 'buffer' ? 'clamp' : 'repeat')) parts.push(c.wrap);   // relative to the kind's default
  if (c.vflip && c.kind === '2d') parts.push('vflip');
  return parts.length ? ` [${parts.join(', ')}]` : '';
}

/**
 * A render pass. `target` is null for the Image pass, which draws to the
 * canvas; buffer passes draw into their own ping-ponged framebuffer.
 */
interface Pass {
  id: string;                 // 'image' | 'common' | 'bufA'..'bufD'
  name: string;               // tab label
  bufIndex: number;           // -1 for Image and Common, else 0..3
  enabled: boolean;
  code: string;               // authoritative doc when this pass is not in the editor
  state: EditorState | null;  // preserved editor state (incl. undo history)
  program: WebGLProgram | null;
  uniforms: Uniforms;
  channels: Channel[];
  controls: Control[];        // annotated consts, driven from the Uniforms view
  format: BufFormat;          // render target precision (buffers only)
  error: string | null;
}

function makePass(id: string, name: string, bufIndex: number, code: string): Pass {
  return {
    id, name, bufIndex, enabled: bufIndex < 0, code, state: null,
    program: null, uniforms: {}, controls: [], format: 'rgba32f',
    channels: Array.from({ length: NUM_SLOTS }, makeChannel),
    error: null,
  };
}

const imagePass = makePass('image', 'Image', -1, DEFAULT_SHADER);
const bufferPasses = BUFFER_NAMES.map((n, i) =>
  makePass(`buf${n}`, `Buf ${n}`, i, i === 0 ? DEFAULT_BUFFER_A : defaultBufferShader(n)));   // the example lives in Buf A

/** Shared code, not a render pass: no program, channels or target of its own. */
const commonPass = makePass('common', 'Common', -1, DEFAULT_COMMON);
commonPass.enabled = false;

// Default wiring: Buffer A feeds back into itself and is shown by Image.
/** Point a channel at a buffer. A slot newly made a buffer takes Shadertoy's buffer defaults: linear, clamp. */
function wireToBuffer(c: Channel, buf: number): void {
  if (c.kind !== 'buffer') { c.filter = 'linear'; c.wrap = 'clamp'; }
  c.kind = 'buffer'; c.buf = buf;
}
bufferPasses[0].enabled = true;
wireToBuffer(bufferPasses[0].channels[0], 0);
wireToBuffer(imagePass.channels[0], 0);

/* ── Project persistence ─────────────────────────────────── */
// The working project (code, which passes are on, how channels are wired)
// is autosaved as you type, so a reload picks up where you left off.
// Uploaded images aren't kept: their slots come back empty but keep their type.
// On top of that sits a library of named snapshots, saved explicitly.
const PROJECT_KEY = 'shaderEditor.project';
const LIBRARY_KEY = 'shaderEditor.library';

interface SavedChannel { kind: Channel['kind']; buf: number; filter?: Filter; wrap?: Wrap; vflip?: boolean }
interface SavedPass { code: string; enabled: boolean; channels: SavedChannel[]; format?: BufFormat }
/** Window layout saved with a shader: every window (docked or floating) and the editor pane. */
interface SavedLayout {
  views: ViewSpec[];
  editorFloating: boolean; editorRect: FloatRect | null;
}
interface SavedProject {
  version: 1;
  active: string;
  passes: Record<string, SavedPass>;
  layout?: SavedLayout;
  name?: string;            // working project only: its title
  libId?: string | null;    // working project only: the library entry it came from
}

interface LibraryEntry { id: string; name: string; updated: number; project: SavedProject; }

const allPasses = (): Pass[] => [imagePass, commonPass, ...bufferPasses];

/** Everything about the passes worth keeping, from their current state. */
function snapshotPasses(): Record<string, SavedPass> {
  const passes: Record<string, SavedPass> = {};
  for (const p of allPasses()) {
    passes[p.id] = {
      code: p.code,
      enabled: p.enabled,
      channels: p.channels.map(c => ({ kind: c.kind, buf: c.buf, filter: c.filter, wrap: c.wrap, vflip: c.vflip })),
      ...(p.bufIndex >= 0 ? { format: p.format } : {}),
    };
  }
  return passes;
}

/** The project as first constructed above: what "Restore example" brings back. */
const EXAMPLE_PROJECT: SavedProject = { version: 1, active: 'image', passes: snapshotPasses() };

/** An Image-only project with the smallest working shader. */
function blankProject(): SavedProject {
  const p = structuredClone(EXAMPLE_PROJECT);
  p.passes.image.code = BLANK_SHADER;
  p.passes.image.channels.forEach(c => { c.kind = '2d'; c.buf = 0; });
  for (const b of bufferPasses) {   // no reaction–diffusion lurking in Buf A
    p.passes[b.id].code = defaultBufferShader(BUFFER_NAMES[b.bufIndex]);
    p.passes[b.id].enabled = false;
    p.passes[b.id].channels.forEach(c => { c.kind = '2d'; c.buf = 0; });
  }
  return p;
}

function isSavedProject(x: unknown): x is SavedProject {
  const p = x as SavedProject | null;
  return !!p && p.version === 1 && typeof p.passes === 'object' && p.passes !== null;
}

/**
 * Write a saved project over the passes. Only touches code, enabled flags
 * and channel wiring; the caller resets editor states, programs and textures.
 */
function applyPasses(project: SavedProject): void {
  for (const p of allPasses()) {
    const sp = project.passes[p.id];
    if (!sp) continue;
    if (typeof sp.code === 'string') p.code = sp.code;
    if (p !== imagePass && typeof sp.enabled === 'boolean') p.enabled = sp.enabled;
    p.format = BUF_FORMATS.some(([f]) => f === sp.format) ? sp.format as BufFormat : 'rgba32f';
    if (!Array.isArray(sp.channels)) continue;
    sp.channels.slice(0, NUM_SLOTS).forEach((sc, i) => {
      const c = p.channels[i];
      if (sc?.kind === '2d' || sc?.kind === 'cube' || sc?.kind === 'buffer') c.kind = sc.kind;
      if (Number.isInteger(sc?.buf) && sc.buf >= 0 && sc.buf < NUM_BUFFERS) c.buf = sc.buf;
      c.filter = FILTERS.includes(sc?.filter as Filter) ? sc.filter as Filter : 'linear';
      c.wrap = WRAPS.includes(sc?.wrap as Wrap) ? sc.wrap as Wrap : 'repeat';
      c.vflip = sc?.vflip === true;
    });
  }
}

let projectName = 'Untitled';
let projectLibId: string | null = null;

let saveTimer: ReturnType<typeof setTimeout> | null = null;

function saveProject(): void {
  if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
  syncActivePassCode();
  const project: SavedProject = {
    version: 1, active: activePass.id, passes: snapshotPasses(), layout: currentLayout(),
    name: projectName, libId: projectLibId,
  };
  try { localStorage.setItem(PROJECT_KEY, JSON.stringify(project)); } catch { /* full or blocked */ }
  updateShaderButton();
}

function scheduleSave(): void {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(saveProject, 500);
}

/** Apply the autosaved working project over the defaults. Returns the pass to open on. */
let initialLayoutRaw: unknown = null;   // the working project's layout, parsed once the helpers below exist

function loadProject(): Pass {
  let project: SavedProject;
  try {
    const raw = localStorage.getItem(PROJECT_KEY);
    if (!raw) return imagePass;
    project = JSON.parse(raw);
    if (!isSavedProject(project)) return imagePass;
  } catch { return imagePass; }

  applyPasses(project);
  if (typeof project.name === 'string' && project.name.trim()) projectName = project.name.trim();
  if (typeof project.libId === 'string') projectLibId = project.libId;
  initialLayoutRaw = project.layout;
  const active = allPasses().find(p => p.id === project.active);
  return active && active.enabled ? active : imagePass;
}

const initialPass = loadProject();

/* ── Shader library ──────────────────────────────────────── */
function loadLibrary(): LibraryEntry[] {
  try {
    const raw = localStorage.getItem(LIBRARY_KEY);
    if (!raw) return [];
    const list = JSON.parse(raw);
    if (!Array.isArray(list)) return [];
    return list.filter((e: LibraryEntry) =>
      e && typeof e.id === 'string' && typeof e.name === 'string' && isSavedProject(e.project));
  } catch { return []; }
}

let library: LibraryEntry[] = loadLibrary();
// A library entry no longer there (cleared in another tab, say) is just unsaved.
if (projectLibId && !library.some(e => e.id === projectLibId)) projectLibId = null;

function persistLibrary(): boolean {
  try { localStorage.setItem(LIBRARY_KEY, JSON.stringify(library)); return true; }
  catch { return false; }
}

function libraryEntry(): LibraryEntry | null {
  return projectLibId ? library.find(e => e.id === projectLibId) ?? null : null;
}

/** Does the working project differ from the library entry it was opened from? */
function projectDirty(): boolean {
  syncActivePassCode();
  const now = JSON.stringify(snapshotPasses());
  const entry = libraryEntry();
  if (entry) return entry.name !== projectName || now !== JSON.stringify(entry.project.passes);
  // Never saved: only worth asking about once it's no longer a stock template.
  return now !== JSON.stringify(EXAMPLE_PROJECT.passes)
      && now !== JSON.stringify(blankProject().passes);
}

const newId = (): string =>
  (crypto.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`);

/** Save the working project into the library, under its own entry or a new one. */
function saveToLibrary(asCopy = false): void {
  syncActivePassCode();
  let name = projectName.trim() || 'Untitled';
  // A copy always gets its own name; a first save of something still called
  // "Untitled" deserves a real one.
  if (asCopy || (!libraryEntry() && name === 'Untitled')) {
    const typed = prompt(asCopy ? 'Name the copy' : 'Name this shader',
                         asCopy && name !== 'Untitled' ? `${name} copy` : name);
    if (typed === null) return;
    name = typed.trim() || 'Untitled';
  }
  const project: SavedProject = { version: 1, active: activePass.id, passes: snapshotPasses(), layout: currentLayout() };
  const existing = asCopy ? null : libraryEntry();
  const entry: LibraryEntry = { id: existing?.id ?? newId(), name, updated: Date.now(), project };
  // Commit to memory only once storage has taken it, so the two never disagree.
  const before = library;
  library = existing ? library.map(e => (e === existing ? entry : e)) : [entry, ...library];
  if (!persistLibrary()) {
    library = before;
    alert('Could not save: this browser\'s storage is full or blocked.');
    return;
  }
  projectName = name;
  projectLibId = entry.id;
  saveProject();
  renderShaderList();
  flashShaderButton('Saved');
}

/** Replace the working project in place: code, wiring, editor states, buffers. */
function openProject(project: SavedProject, name: string, libId: string | null): void {
  // Uploaded images belong to the old project; drop them with everything else.
  for (const p of allPasses()) {
    for (let i = 0; i < NUM_SLOTS; i++) resetChannel(p.channels[i]);
    p.state = null;
    p.error = null;
    p.code = p === imagePass ? DEFAULT_SHADER
           : p === commonPass ? DEFAULT_COMMON
           : defaultBufferShader(BUFFER_NAMES[p.bufIndex]);
    p.enabled = p === imagePass;
  }
  applyPasses(project);
  for (const p of bufferPasses) {
    if (p.enabled) continue;
    if (p.program) { gl.deleteProgram(p.program); p.program = null; }
    disposeTarget(p.bufIndex);
  }
  projectName = name;
  projectLibId = libId;

  const active = allPasses().find(p => p.id === project.active);
  activePass = active && active.enabled ? active : imagePass;
  activePass.state = makeEditorState(activePass.code);
  editor.setState(activePass.state);

  iTime = 0; iFrame = -1; iTimeDelta = 0; lastTimestamp = null;
  clearTargets();
  updatePassTabs();
  rebuildChannelStrip();
  if (compileTimer) { clearTimeout(compileTimer); compileTimer = null; }
  // Windows first: tryCompile autosaves the working project with whatever
  // layout is on screen, and that must be this shader's, not the last one's.
  const layout = parseLayout(project.layout);
  if (layout) applySavedLayout(layout);   // shaders saved before layouts keep the windows as they are
  tryCompile();
  editor.focus();
}

/** Ask before throwing away unsaved work; true means go ahead. */
function confirmDiscard(): boolean {
  if (!projectDirty()) return true;
  const entry = libraryEntry();
  const what = entry ? `unsaved changes to "${entry.name}"` : `the unsaved shader "${projectName}"`;
  return confirm(`Discard ${what}?`);
}

/** Buffers run first, in order, then Image. */
const passes: Pass[] = [...bufferPasses, imagePass];

function activeBufferPasses(): Pass[] { return bufferPasses.filter(p => p.enabled); }
function livePasses(): Pass[] { return [...activeBufferPasses(), imagePass]; }

/* ── Buffer render targets ───────────────────────────────── */
// Rendering to half-float needs an extension; fall back to 8-bit if absent.
const floatRT = gl.getExtension('EXT_color_buffer_float');
const floatLinear = !!gl.getExtension('OES_texture_float_linear');

/** The format a buffer actually gets: float formats need the extension. */
function effectiveFormat(f: BufFormat): BufFormat { return floatRT ? f : 'rgba8'; }
function rtInternal(f: BufFormat): number { return f === 'rgba32f' ? gl.RGBA32F : f === 'rgba16f' ? gl.RGBA16F : gl.RGBA8; }
function rtType(f: BufFormat): number { return f === 'rgba32f' ? gl.FLOAT : f === 'rgba16f' ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE; }

interface RenderTarget {
  fbo: WebGLFramebuffer;
  read: WebGLTexture;    // last completed frame — what shaders sample
  write: WebGLTexture;   // currently being drawn into
  w: number;
  h: number;
  format: BufFormat;
  mips: boolean;         // a consumer samples it with mipmaps: regenerate after each draw
}

const targets: Array<RenderTarget | null> = Array(NUM_BUFFERS).fill(null);

function makeRTTexture(w: number, h: number, format: BufFormat): WebGLTexture {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, rtInternal(format), w, h, 0, gl.RGBA, rtType(format), null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return t;
}

function ensureTarget(i: number, w: number, h: number): RenderTarget {
  const existing = targets[i];
  const format = effectiveFormat(bufferPasses[i].format);
  if (existing && existing.w === w && existing.h === h && existing.format === format) return existing;
  const t: RenderTarget = {
    fbo: gl.createFramebuffer()!,
    read: makeRTTexture(w, h, format),
    write: makeRTTexture(w, h, format),
    w, h, format, mips: false,
  };
  if (existing) {
    // A resize keeps what the buffer held, pixel for pixel over the overlap
    // (Shadertoy's ResizeBuffer does the same); new pixels start at zero.
    copyTexture(existing.read, existing.w, existing.h, t.read, w, h, t.fbo, false);
    copyTexture(existing.write, existing.w, existing.h, t.write, w, h, t.fbo, false);
    if (existing.mips) { t.mips = true; gl.bindTexture(gl.TEXTURE_2D, t.read); gl.generateMipmap(gl.TEXTURE_2D); gl.bindTexture(gl.TEXTURE_2D, null); }
    gl.deleteTexture(existing.read);
    gl.deleteTexture(existing.write);
    gl.deleteFramebuffer(existing.fbo);
  }
  targets[i] = t;
  return t;
}

function disposeTarget(i: number): void {
  const t = targets[i];
  if (!t) return;
  gl.deleteTexture(t.read);
  gl.deleteTexture(t.write);
  gl.deleteFramebuffer(t.fbo);
  targets[i] = null;
}

/** Wipe buffer contents to transparent black (0,0,0,0), as Shadertoy does at frame 0 — feedback shaders need this on reset. */
function clearTargets(): void {
  const prevFbo = gl.getParameter(gl.FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
  for (const t of targets) {
    if (!t) continue;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
    for (const tex of [t.read, t.write]) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, prevFbo);
}

// Copies one buffer texture into another: pixel for pixel over the
// overlap (a resize, like Shadertoy's ResizeBuffer) or stretched to fit
// (a screenshot at another size).
const COPY_FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D u_tex;
uniform vec2 u_dst;
uniform int u_stretch;
out vec4 o;
void main() {
  o = u_stretch == 1 ? texture(u_tex, gl_FragCoord.xy / u_dst) : texelFetch(u_tex, ivec2(gl_FragCoord.xy), 0);
}
`;
let copyProg: WebGLProgram | null = null;
let copyDst: WebGLUniformLocation | null = null, copyStretch: WebGLUniformLocation | null = null;
function copyTexture(src: WebGLTexture, sw: number, sh: number, dst: WebGLTexture, dw: number, dh: number, fbo: WebGLFramebuffer, stretch: boolean): void {
  if (!copyProg) {
    const vs = compileShader(VERT_SRC, gl.VERTEX_SHADER).sh, fs = compileShader(COPY_FRAG, gl.FRAGMENT_SHADER).sh;
    if (!vs || !fs) return;
    copyProg = linkProgram(vs, fs).prog;
    if (!copyProg) return;
    gl.useProgram(copyProg);
    gl.uniform1i(gl.getUniformLocation(copyProg, 'u_tex'), 0);
    copyDst = gl.getUniformLocation(copyProg, 'u_dst');
    copyStretch = gl.getUniformLocation(copyProg, 'u_stretch');
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, dst, 0);
  gl.viewport(0, 0, stretch ? dw : Math.min(sw, dw), stretch ? dh : Math.min(sh, dh));
  gl.useProgram(copyProg);
  gl.uniform2f(copyDst, dw, dh);
  gl.uniform1i(copyStretch, stretch ? 1 : 0);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, src);
  const min = gl.getTexParameter(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER) as number;
  const mag = gl.getTexParameter(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER) as number;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, min);   // a live buffer keeps its own filter
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, mag);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

/** A buffer's current contents at another size, stretched; for a screenshot that must not disturb the live buffers. */
function stretchedCopy(t: RenderTarget, w: number, h: number): RenderTarget {
  const c: RenderTarget = { fbo: gl.createFramebuffer()!, read: makeRTTexture(w, h, t.format), write: makeRTTexture(w, h, t.format), w, h, format: t.format, mips: false };
  copyTexture(t.read, t.w, t.h, c.read, w, h, c.fbo, true);
  return c;
}
function disposeCopy(c: RenderTarget): void {
  gl.deleteTexture(c.read); gl.deleteTexture(c.write); gl.deleteFramebuffer(c.fbo);
}

/* ── Textures ────────────────────────────────────────────── */
const WHITE_TEX = gl.createTexture()!;
gl.bindTexture(gl.TEXTURE_2D, WHITE_TEX);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0,0,0,255]));

const BLACK_CUBE_TEX = gl.createTexture()!;
gl.bindTexture(gl.TEXTURE_CUBE_MAP, BLACK_CUBE_TEX);
for (const { target } of CUBE_FACES) {
  gl.texImage2D(target, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0,0,0,255]));
}
gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);

function uploadTexture(ch: Channel, img: HTMLImageElement): void {
  if (!ch.tex) ch.tex = gl.createTexture();
  ch.img = img;
  gl.bindTexture(gl.TEXTURE_2D, ch.tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, ch.vflip ? 1 : 0);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
  gl.generateMipmap(gl.TEXTURE_2D);   // filter and wrap are set when the texture is bound
  gl.bindTexture(gl.TEXTURE_2D, null);
  ch.texW = img.naturalWidth;
  ch.texH = img.naturalHeight;
}

/**
 * Filter and wrap for a 2D texture, from a channel's settings, as a sampler
 * object bound to the channel's unit. A sampler overrides the texture's own
 * parameters, so one buffer can be read with different filters on two
 * channels; drawPass() unbinds them after drawing, so the texture parameters
 * set by copyTexture() and the blit still apply there.
 */
const samplers = new Map<string, WebGLSampler>();
function bindSampling(unit: number, c: Channel, filterable: boolean, hasMips: boolean): void {
  const filter = c.filter === 'mipmap' && !hasMips ? 'linear' : c.filter;
  const linear = filterable;   // 32-bit float textures filter only with OES_texture_float_linear
  const min = filter === 'nearest' || !linear ? gl.NEAREST : filter === 'mipmap' ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR;
  const mag = filter === 'nearest' || !linear ? gl.NEAREST : gl.LINEAR;
  const wrap = c.wrap === 'clamp' ? gl.CLAMP_TO_EDGE : gl.REPEAT;
  const key = `${min}/${mag}/${wrap}`;
  let smp = samplers.get(key);
  if (!smp) {
    smp = gl.createSampler()!;
    gl.samplerParameteri(smp, gl.TEXTURE_MIN_FILTER, min);
    gl.samplerParameteri(smp, gl.TEXTURE_MAG_FILTER, mag);
    gl.samplerParameteri(smp, gl.TEXTURE_WRAP_S, wrap);
    gl.samplerParameteri(smp, gl.TEXTURE_WRAP_T, wrap);
    samplers.set(key, smp);
  }
  gl.bindSampler(unit, smp);
}

function uploadCubeFace(ch: Channel, faceIdx: number, img: HTMLImageElement): void {
  if (!ch.cubeTex) {
    ch.cubeTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, ch.cubeTex);
    for (const { target } of CUBE_FACES) {
      gl.texImage2D(target, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0,0,0,255]));
    }
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  } else {
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, ch.cubeTex);
  }
  gl.texImage2D(CUBE_FACES[faceIdx].target, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);
  ch.cubeFaceLoaded[faceIdx] = true;
  ch.texW = img.naturalWidth;   // face size, for iChannelResolution
  ch.texH = img.naturalHeight;
}

/* ── Themes ──────────────────────────────────────────────── */
interface Theme {
  id: string;
  name: string;
  dark: boolean;
  ui: {
    bg0: string; bg1: string; bg2: string; bg3: string; border: string;
    accent: string; accent2: string; err: string; warn: string; ok: string;
    text: string; textDim: string; preview: string; hl: string; sel: string;
  };
  syn: {
    keyword: string; type: string; builtin: string; variable: string; comment: string;
    number: string; string: string; operator: string; meta: string; atom: string; bracket: string;
  };
}

const THEMES: Theme[] = [
  {
    id: 'midnight', name: 'Midnight', dark: true,
    ui: {
      bg0: '#0d0f14', bg1: '#13161e', bg2: '#1a1e28', bg3: '#222736', border: '#2a2f3d',
      accent: '#5c8af7', accent2: '#7c5cf7', err: '#e06c75', warn: '#e5c07b', ok: '#98c379',
      text: '#abb2bf', textDim: '#5c6370', preview: '#000000',
      hl: 'rgba(255,255,255,0.03)', sel: 'rgba(92,138,247,0.22)',
    },
    syn: {
      keyword: '#c678dd', type: '#e5c07b', builtin: '#56b6c2', variable: '#abb2bf',
      comment: '#5c6370', number: '#d19a66', string: '#98c379', operator: '#abb2bf',
      meta: '#5c6370', atom: '#d19a66', bracket: '#abb2bf',
    },
  },
  {
    id: 'dracula', name: 'Dracula', dark: true,
    ui: {
      bg0: '#21222c', bg1: '#282a36', bg2: '#343746', bg3: '#3d4152', border: '#44475a',
      accent: '#bd93f9', accent2: '#ff79c6', err: '#ff5555', warn: '#f1fa8c', ok: '#50fa7b',
      text: '#f8f8f2', textDim: '#6272a4', preview: '#191a21',
      hl: 'rgba(255,255,255,0.05)', sel: 'rgba(189,147,249,0.28)',
    },
    syn: {
      keyword: '#ff79c6', type: '#8be9fd', builtin: '#50fa7b', variable: '#f8f8f2',
      comment: '#6272a4', number: '#bd93f9', string: '#f1fa8c', operator: '#ff79c6',
      meta: '#6272a4', atom: '#bd93f9', bracket: '#f8f8f2',
    },
  },
  {
    id: 'nord', name: 'Nord', dark: true,
    ui: {
      bg0: '#2e3440', bg1: '#333a47', bg2: '#3b4252', bg3: '#434c5e', border: '#4c566a',
      accent: '#88c0d0', accent2: '#b48ead', err: '#bf616a', warn: '#ebcb8b', ok: '#a3be8c',
      text: '#d8dee9', textDim: '#7b88a1', preview: '#242933',
      hl: 'rgba(255,255,255,0.04)', sel: 'rgba(136,192,208,0.25)',
    },
    syn: {
      keyword: '#81a1c1', type: '#8fbcbb', builtin: '#88c0d0', variable: '#d8dee9',
      comment: '#616e88', number: '#b48ead', string: '#a3be8c', operator: '#81a1c1',
      meta: '#616e88', atom: '#b48ead', bracket: '#d8dee9',
    },
  },
  {
    id: 'gruvbox', name: 'Gruvbox Dark', dark: true,
    ui: {
      bg0: '#1d2021', bg1: '#282828', bg2: '#32302f', bg3: '#3c3836', border: '#504945',
      accent: '#83a598', accent2: '#d3869b', err: '#fb4934', warn: '#fabd2f', ok: '#b8bb26',
      text: '#ebdbb2', textDim: '#928374', preview: '#1b1b1b',
      hl: 'rgba(255,255,255,0.04)', sel: 'rgba(131,165,152,0.28)',
    },
    syn: {
      keyword: '#fb4934', type: '#fabd2f', builtin: '#8ec07c', variable: '#ebdbb2',
      comment: '#928374', number: '#d3869b', string: '#b8bb26', operator: '#fe8019',
      meta: '#928374', atom: '#d3869b', bracket: '#ebdbb2',
    },
  },
  {
    id: 'monokai', name: 'Monokai', dark: true,
    ui: {
      bg0: '#1e1f1c', bg1: '#272822', bg2: '#2f302a', bg3: '#3e3d32', border: '#49483e',
      accent: '#66d9ef', accent2: '#ae81ff', err: '#f92672', warn: '#e6db74', ok: '#a6e22e',
      text: '#f8f8f2', textDim: '#75715e', preview: '#131410',
      hl: 'rgba(255,255,255,0.05)', sel: 'rgba(102,217,239,0.22)',
    },
    syn: {
      keyword: '#f92672', type: '#66d9ef', builtin: '#a6e22e', variable: '#f8f8f2',
      comment: '#75715e', number: '#ae81ff', string: '#e6db74', operator: '#f92672',
      meta: '#75715e', atom: '#ae81ff', bracket: '#f8f8f2',
    },
  },
  {
    id: 'tokyo-night', name: 'Tokyo Night', dark: true,
    ui: {
      bg0: '#16161e', bg1: '#1a1b26', bg2: '#20212e', bg3: '#292b3d', border: '#3b3d57',
      accent: '#7aa2f7', accent2: '#bb9af7', err: '#f7768e', warn: '#e0af68', ok: '#9ece6a',
      text: '#c0caf5', textDim: '#565f89', preview: '#101014',
      hl: 'rgba(255,255,255,0.04)', sel: 'rgba(122,162,247,0.25)',
    },
    syn: {
      keyword: '#bb9af7', type: '#2ac3de', builtin: '#7dcfff', variable: '#c0caf5',
      comment: '#565f89', number: '#ff9e64', string: '#9ece6a', operator: '#89ddff',
      meta: '#565f89', atom: '#ff9e64', bracket: '#c0caf5',
    },
  },
  {
    id: 'catppuccin-mocha', name: 'Catppuccin', dark: true,
    ui: {
      bg0: '#181825', bg1: '#1e1e2e', bg2: '#252538', bg3: '#313244', border: '#45475a',
      accent: '#89b4fa', accent2: '#cba6f7', err: '#f38ba8', warn: '#f9e2af', ok: '#a6e3a1',
      text: '#cdd6f4', textDim: '#6c7086', preview: '#11111b',
      hl: 'rgba(255,255,255,0.04)', sel: 'rgba(137,180,250,0.25)',
    },
    syn: {
      keyword: '#cba6f7', type: '#f9e2af', builtin: '#94e2d5', variable: '#cdd6f4',
      comment: '#6c7086', number: '#fab387', string: '#a6e3a1', operator: '#89dceb',
      meta: '#6c7086', atom: '#fab387', bracket: '#cdd6f4',
    },
  },
  {
    id: 'rose-pine', name: 'Rosé Pine', dark: true,
    ui: {
      bg0: '#16141f', bg1: '#191724', bg2: '#1f1d2e', bg3: '#26233a', border: '#403d52',
      accent: '#c4a7e7', accent2: '#ebbcba', err: '#eb6f92', warn: '#f6c177', ok: '#9ccfd8',
      text: '#e0def4', textDim: '#6e6a86', preview: '#0f0d17',
      hl: 'rgba(255,255,255,0.04)', sel: 'rgba(196,167,231,0.22)',
    },
    syn: {
      keyword: '#5ca7c4', type: '#9ccfd8', builtin: '#ebbcba', variable: '#e0def4',
      comment: '#6e6a86', number: '#c4a7e7', string: '#f6c177', operator: '#908caa',
      meta: '#6e6a86', atom: '#c4a7e7', bracket: '#908caa',
    },
  },
  {
    id: 'everforest', name: 'Everforest', dark: true,
    ui: {
      bg0: '#272e33', bg1: '#2d353b', bg2: '#343f44', bg3: '#3d484d', border: '#475258',
      accent: '#a7c080', accent2: '#7fbbb3', err: '#e67e80', warn: '#dbbc7f', ok: '#a7c080',
      text: '#d3c6aa', textDim: '#859289', preview: '#1e2326',
      hl: 'rgba(255,255,255,0.04)', sel: 'rgba(167,192,128,0.22)',
    },
    syn: {
      keyword: '#e67e80', type: '#dbbc7f', builtin: '#7fbbb3', variable: '#d3c6aa',
      comment: '#859289', number: '#d699b6', string: '#a7c080', operator: '#e69875',
      meta: '#859289', atom: '#d699b6', bracket: '#d3c6aa',
    },
  },
  {
    id: 'solarized-light', name: 'Solarized Light', dark: false,
    ui: {
      bg0: '#fdf6e3', bg1: '#f5eeda', bg2: '#eee8d5', bg3: '#e4ddc8', border: '#d6cfb8',
      accent: '#268bd2', accent2: '#6c71c4', err: '#dc322f', warn: '#b58900', ok: '#859900',
      text: '#586e75', textDim: '#93a1a1', preview: '#eee8d5',
      hl: 'rgba(0,0,0,0.05)', sel: 'rgba(38,139,210,0.22)',
    },
    syn: {
      keyword: '#859900', type: '#b58900', builtin: '#268bd2', variable: '#586e75',
      comment: '#93a1a1', number: '#d33682', string: '#2aa198', operator: '#859900',
      meta: '#93a1a1', atom: '#cb4b16', bracket: '#586e75',
    },
  },
  {
    id: 'paper', name: 'Paper', dark: false,
    ui: {
      bg0: '#ffffff', bg1: '#f6f8fa', bg2: '#eaeef2', bg3: '#dfe3e8', border: '#d0d7de',
      accent: '#0969da', accent2: '#8250df', err: '#cf222e', warn: '#9a6700', ok: '#1a7f37',
      text: '#1f2328', textDim: '#656d76', preview: '#f6f8fa',
      hl: 'rgba(0,0,0,0.04)', sel: 'rgba(9,105,218,0.18)',
    },
    syn: {
      keyword: '#cf222e', type: '#953800', builtin: '#8250df', variable: '#1f2328',
      comment: '#6e7781', number: '#0550ae', string: '#0a3069', operator: '#cf222e',
      meta: '#6e7781', atom: '#0550ae', bracket: '#1f2328',
    },
  },
  {
    id: 'ayu-light', name: 'Ayu Light', dark: false,
    ui: {
      bg0: '#fcfcfc', bg1: '#f3f4f5', bg2: '#e9eaeb', bg3: '#dcdfe2', border: '#d2d5d9',
      accent: '#399ee6', accent2: '#ff9940', err: '#f07171', warn: '#f2ae49', ok: '#86b300',
      text: '#5c6166', textDim: '#8a9199', preview: '#f3f4f5',
      hl: 'rgba(0,0,0,0.04)', sel: 'rgba(57,158,230,0.18)',
    },
    syn: {
      keyword: '#fa8d3e', type: '#55b4d4', builtin: '#f2ae49', variable: '#5c6166',
      comment: '#8a9199', number: '#a37acc', string: '#86b300', operator: '#ed9366',
      meta: '#8a9199', atom: '#a37acc', bracket: '#5c6166',
    },
  },
  {
    id: 'catppuccin-latte', name: 'Latte', dark: false,
    ui: {
      bg0: '#eff1f5', bg1: '#e6e9ef', bg2: '#dce0e8', bg3: '#ccd0da', border: '#bcc0cc',
      accent: '#1e66f5', accent2: '#8839ef', err: '#d20f39', warn: '#df8e1d', ok: '#40a02b',
      text: '#4c4f69', textDim: '#8c8fa1', preview: '#e6e9ef',
      hl: 'rgba(0,0,0,0.04)', sel: 'rgba(30,102,245,0.15)',
    },
    syn: {
      keyword: '#8839ef', type: '#df8e1d', builtin: '#179299', variable: '#4c4f69',
      comment: '#9ca0b0', number: '#fe640b', string: '#40a02b', operator: '#04a5e5',
      meta: '#9ca0b0', atom: '#fe640b', bracket: '#4c4f69',
    },
  },
  // Shared with typst-editor and musicTest3: generated from their palettes.
  {
    id: 'chalk', name: 'Chalk', dark: false,
    ui: {
      bg0: '#ffffff', bg1: '#ffffff', bg2: '#f1f2f5', bg3: '#e5e9ed', border: '#dfe3e8',
      accent: '#1d4ed8', accent2: '#6d28d9', err: '#c0392b', warn: '#8a6d00', ok: '#197d4b',
      text: '#14171a', textDim: '#8c949d', preview: '#f1f3f5', hl: 'rgba(0,0,0,0.04)', sel: '#cfe3ff',
    },
    syn: {
      keyword: '#a21c67', type: '#8a6d00', builtin: '#1d4ed8', variable: '#14171a',
      comment: '#8c949d', number: '#b45309', string: '#197d4b', operator: '#5a626b',
      meta: '#0e7490', atom: '#b45309', bracket: '#5a626b',
    },
  },
  {
    id: 'newsprint', name: 'Newsprint', dark: false,
    ui: {
      bg0: '#fafbfb', bg1: '#f4f5f6', bg2: '#e7e9ec', bg3: '#dde0e4', border: '#d7dbdf',
      accent: '#275ea8', accent2: '#6b4baf', err: '#b23a32', warn: '#7d6a12', ok: '#2a7048',
      text: '#1a1c1e', textDim: '#878d94', preview: '#e8eaec', hl: 'rgba(0,0,0,0.04)', sel: '#d3e2f7',
    },
    syn: {
      keyword: '#9a3670', type: '#7d6a12', builtin: '#275ea8', variable: '#1a1c1e',
      comment: '#878d94', number: '#a2600f', string: '#2a7048', operator: '#565b61',
      meta: '#17697f', atom: '#a2600f', bracket: '#565b61',
    },
  },
  {
    id: 'vellum', name: 'Vellum', dark: false,
    ui: {
      bg0: '#fffcf4', bg1: '#fbf6ec', bg2: '#f0e8d9', bg3: '#e8deca', border: '#e3d8c2',
      accent: '#2e4a7d', accent2: '#6a4a82', err: '#a83a2c', warn: '#7e6512', ok: '#3f6b35',
      text: '#2b261e', textDim: '#9a9080', preview: '#f1e9d9', hl: 'rgba(0,0,0,0.04)', sel: '#e4dcc4',
    },
    syn: {
      keyword: '#8e3a63', type: '#7e6512', builtin: '#2e4a7d', variable: '#2b261e',
      comment: '#9a9080', number: '#9c5b14', string: '#3f6b35', operator: '#6a6052',
      meta: '#2a6a6a', atom: '#9c5b14', bracket: '#6a6052',
    },
  },
  {
    id: 'gruvbox-light', name: 'Gruvbox Light', dark: false,
    ui: {
      bg0: '#f9f5d7', bg1: '#fbf1c7', bg2: '#eaddb6', bg3: '#ddcda9', border: '#d5c4a1',
      accent: '#076678', accent2: '#8f3f71', err: '#9d0006', warn: '#b57614', ok: '#79740e',
      text: '#3c3836', textDim: '#928374', preview: '#f2e5bc', hl: 'rgba(0,0,0,0.04)', sel: '#d5c4a1',
    },
    syn: {
      keyword: '#b16286', type: '#b57614', builtin: '#076678', variable: '#3c3836',
      comment: '#928374', number: '#af3a03', string: '#79740e', operator: '#665c54',
      meta: '#427b58', atom: '#af3a03', bracket: '#665c54',
    },
  },
  {
    id: 'rose-pine-dawn', name: 'Rosé Pine Dawn', dark: false,
    ui: {
      bg0: '#fffaf3', bg1: '#faf4ed', bg2: '#eee8e4', bg3: '#e4dfdd', border: '#dfdad9',
      accent: '#286983', accent2: '#907aa9', err: '#b4637a', warn: '#ea9d34', ok: '#478567',
      text: '#575279', textDim: '#9893a5', preview: '#f2e9e1', hl: 'rgba(0,0,0,0.04)', sel: '#dcd7e8',
    },
    syn: {
      keyword: '#b45c87', type: '#ea9d34', builtin: '#286983', variable: '#575279',
      comment: '#9893a5', number: '#d7827e', string: '#478567', operator: '#797593',
      meta: '#56949f', atom: '#d7827e', bracket: '#797593',
    },
  },
  {
    id: 'tangerine', name: 'Tangerine', dark: false,
    ui: {
      bg0: '#ffffff', bg1: '#fff7ef', bg2: '#f9e8d7', bg3: '#f4ddc5', border: '#f1d6ba',
      accent: '#e8620c', accent2: '#7b3fc4', err: '#d7263d', warn: '#a87a00', ok: '#2e8b3a',
      text: '#2a1a10', textDim: '#a08670', preview: '#fdecd9', hl: 'rgba(0,0,0,0.04)', sel: '#ffd8b0',
    },
    syn: {
      keyword: '#c2187a', type: '#a87a00', builtin: '#2457c5', variable: '#2a1a10',
      comment: '#a08670', number: '#e8620c', string: '#2e8b3a', operator: '#6e5240',
      meta: '#0b8a9a', atom: '#e8620c', bracket: '#6e5240',
    },
  },
  {
    id: 'sorbet', name: 'Sorbet', dark: false,
    ui: {
      bg0: '#ffffff', bg1: '#fdf5fa', bg2: '#f4e4ef', bg3: '#eed7e7', border: '#ead0e2',
      accent: '#d6336c', accent2: '#7048e8', err: '#d6336c', warn: '#a07a00', ok: '#2b8a3e',
      text: '#2b1830', textDim: '#a08aa4', preview: '#f6e6f1', hl: 'rgba(0,0,0,0.04)', sel: '#f7cdea',
    },
    syn: {
      keyword: '#c2255c', type: '#a07a00', builtin: '#3b5bdb', variable: '#2b1830',
      comment: '#a08aa4', number: '#e8590c', string: '#2b8a3e', operator: '#6c5272',
      meta: '#0c8599', atom: '#e8590c', bracket: '#6c5272',
    },
  },
  {
    id: 'nocturne', name: 'Nocturne', dark: true,
    ui: {
      bg0: '#0c0d0f', bg1: '#121316', bg2: '#1d1f23', bg3: '#25282d', border: '#2a2d33',
      accent: '#79c0ff', accent2: '#c792ea', err: '#ff7b72', warn: '#e3b341', ok: '#7ee787',
      text: '#e6e8eb', textDim: '#6b717a', preview: '#0c0d0f', hl: 'rgba(255,255,255,0.04)', sel: '#2c3a52',
    },
    syn: {
      keyword: '#ff7bc8', type: '#e3b341', builtin: '#79c0ff', variable: '#e6e8eb',
      comment: '#6b717a', number: '#ffa657', string: '#7ee787', operator: '#9ba1a9',
      meta: '#56d4bc', atom: '#ffa657', bracket: '#9ba1a9',
    },
  },
  {
    id: 'slate', name: 'Slate', dark: true,
    ui: {
      bg0: '#11141a', bg1: '#171a21', bg2: '#21262f', bg3: '#292f3b', border: '#2e3441',
      accent: '#7fb2f0', accent2: '#bfa0ee', err: '#f2777a', warn: '#e6c86e', ok: '#8cd98c',
      text: '#dde3ec', textDim: '#69727f', preview: '#11141a', hl: 'rgba(255,255,255,0.04)', sel: '#2f3e57',
    },
    syn: {
      keyword: '#ee8ac6', type: '#e6c86e', builtin: '#7fb2f0', variable: '#dde3ec',
      comment: '#69727f', number: '#f5a25d', string: '#8cd98c', operator: '#949dad',
      meta: '#6bd3c6', atom: '#f5a25d', bracket: '#949dad',
    },
  },
  {
    id: 'oxide', name: 'Oxide', dark: true,
    ui: {
      bg0: '#130f0d', bg1: '#1a1613', bg2: '#27211c', bg3: '#302923', border: '#362e27',
      accent: '#e8925b', accent2: '#c49bd1', err: '#f07e6e', warn: '#dfb25a', ok: '#a6c07a',
      text: '#ede3d8', textDim: '#796e62', preview: '#130f0d', hl: 'rgba(255,255,255,0.04)', sel: '#4a3a2a',
    },
    syn: {
      keyword: '#dd8aa8', type: '#dfb25a', builtin: '#8fb3d9', variable: '#ede3d8',
      comment: '#796e62', number: '#e8925b', string: '#a6c07a', operator: '#a89a8b',
      meta: '#79bfb0', atom: '#e8925b', bracket: '#a89a8b',
    },
  },
  {
    id: 'ember', name: 'Ember', dark: true,
    ui: {
      bg0: '#140c0a', bg1: '#1b1210', bg2: '#291b17', bg3: '#35231d', border: '#3b2720',
      accent: '#ff8f2e', accent2: '#c98fff', err: '#ff6150', warn: '#ffcc3d', ok: '#b8d84b',
      text: '#f7e6d6', textDim: '#8c705f', preview: '#140c0a', hl: 'rgba(255,255,255,0.04)', sel: '#50301f',
    },
    syn: {
      keyword: '#ff62a5', type: '#ffcc3d', builtin: '#6db3ff', variable: '#f7e6d6',
      comment: '#8c705f', number: '#ff8f2e', string: '#b8d84b', operator: '#c9a893',
      meta: '#52d3c6', atom: '#ff8f2e', bracket: '#c9a893',
    },
  },
  {
    id: 'lava', name: 'Lava', dark: true,
    ui: {
      bg0: '#100707', bg1: '#170c0c', bg2: '#281413', bg3: '#351a18', border: '#3d1d1b',
      accent: '#ff4d3d', accent2: '#d290ff', err: '#ff4d3d', warn: '#ffc03c', ok: '#c4da4e',
      text: '#ffeae2', textDim: '#92695f', preview: '#100707', hl: 'rgba(255,255,255,0.04)', sel: '#5c2018',
    },
    syn: {
      keyword: '#ff5290', type: '#ffc03c', builtin: '#72aaff', variable: '#ffeae2',
      comment: '#92695f', number: '#ff7b25', string: '#c4da4e', operator: '#d4a798',
      meta: '#55d7ca', atom: '#ff7b25', bracket: '#d4a798',
    },
  },
  {
    id: 'marigold', name: 'Marigold', dark: true,
    ui: {
      bg0: '#120e05', bg1: '#1a1508', bg2: '#29220f', bg3: '#342b14', border: '#3b3117',
      accent: '#ffc21a', accent2: '#c497e2', err: '#f2633f', warn: '#ffd43f', ok: '#aacd4b',
      text: '#f6edd3', textDim: '#877b57', preview: '#120e05', hl: 'rgba(255,255,255,0.04)', sel: '#4f4116',
    },
    syn: {
      keyword: '#f0719c', type: '#ffd43f', builtin: '#7aa9ea', variable: '#f6edd3',
      comment: '#877b57', number: '#f7a02b', string: '#aacd4b', operator: '#c4b58c',
      meta: '#61c6b2', atom: '#f7a02b', bracket: '#c4b58c',
    },
  },
  {
    id: 'sunset', name: 'Sunset', dark: true,
    ui: {
      bg0: '#170e1b', bg1: '#1f1424', bg2: '#2e1d35', bg3: '#392442', border: '#40284a',
      accent: '#ff7a59', accent2: '#c994ff', err: '#ff5c7a', warn: '#ffd166', ok: '#9ee57c',
      text: '#fbe8e1', textDim: '#8f7088', preview: '#170e1b', hl: 'rgba(255,255,255,0.04)', sel: '#532c55',
    },
    syn: {
      keyword: '#ff6ad5', type: '#ffd166', builtin: '#7da9ff', variable: '#fbe8e1',
      comment: '#8f7088', number: '#ff9152', string: '#9ee57c', operator: '#cca8ba',
      meta: '#5fe0d6', atom: '#ff9152', bracket: '#cca8ba',
    },
  },
  {
    id: 'carnival', name: 'Carnival', dark: true,
    ui: {
      bg0: '#0c0a17', bg1: '#12101f', bg2: '#1e1b36', bg3: '#282448', border: '#2d2952',
      accent: '#f472b6', accent2: '#a78bfa', err: '#ff4f6d', warn: '#fde047', ok: '#4ade80',
      text: '#edecff', textDim: '#6f6ba0', preview: '#0c0a17', hl: 'rgba(255,255,255,0.04)', sel: '#383270',
    },
    syn: {
      keyword: '#f472b6', type: '#fde047', builtin: '#60a5fa', variable: '#edecff',
      comment: '#6f6ba0', number: '#ff9f43', string: '#4ade80', operator: '#aba8d8',
      meta: '#22d3ee', atom: '#ff9f43', bracket: '#aba8d8',
    },
  },
  {
    id: 'synthwave', name: 'Synthwave', dark: true,
    ui: {
      bg0: '#130a27', bg1: '#1a1033', bg2: '#291c4b', bg3: '#35255d', border: '#3c2a68',
      accent: '#ff7edb', accent2: '#c792ea', err: '#fe4450', warn: '#fede5d', ok: '#72f1b8',
      text: '#f6eaff', textDim: '#7e6aa8', preview: '#130a27', hl: 'rgba(255,255,255,0.04)', sel: '#4b2e7c',
    },
    syn: {
      keyword: '#ff7edb', type: '#fede5d', builtin: '#6c9cff', variable: '#f6eaff',
      comment: '#7e6aa8', number: '#ff8b39', string: '#72f1b8', operator: '#bba5df',
      meta: '#36f9f6', atom: '#ff8b39', bracket: '#bba5df',
    },
  },
  {
    id: 'lagoon', name: 'Lagoon', dark: true,
    ui: {
      bg0: '#0a1416', bg1: '#0e1a1d', bg2: '#17292e', bg3: '#1d353b', border: '#213c42',
      accent: '#ffa94d', accent2: '#b197fc', err: '#ff6b6b', warn: '#ffe066', ok: '#69db7c',
      text: '#e3f7f4', textDim: '#608b87', preview: '#0a1416', hl: 'rgba(255,255,255,0.04)', sel: '#214e54',
    },
    syn: {
      keyword: '#f783ac', type: '#ffe066', builtin: '#74c0fc', variable: '#e3f7f4',
      comment: '#608b87', number: '#ffa94d', string: '#69db7c', operator: '#9dc6c1',
      meta: '#3bc9db', atom: '#ffa94d', bracket: '#9dc6c1',
    },
  },
  {
    id: 'solarized-dark', name: 'Solarized Dark', dark: true,
    ui: {
      bg0: '#00212b', bg1: '#002b36', bg2: '#053642', bg3: '#093f4c', border: '#0b4451',
      accent: '#268bd2', accent2: '#6c71c4', err: '#dc322f', warn: '#b58900', ok: '#859900',
      text: '#93a1a1', textDim: '#657b83', preview: '#00212b', hl: 'rgba(255,255,255,0.04)', sel: '#0b4451',
    },
    syn: {
      keyword: '#d33682', type: '#b58900', builtin: '#268bd2', variable: '#93a1a1',
      comment: '#657b83', number: '#cb4b16', string: '#859900', operator: '#839496',
      meta: '#2aa198', atom: '#cb4b16', bracket: '#839496',
    },
  },
  {
    id: 'contrast', name: 'Contrast', dark: true,
    ui: {
      bg0: '#000000', bg1: '#000000', bg2: '#282828', bg3: '#484848', border: '#5a5a5a',
      accent: '#7fb8ff', accent2: '#d0a8ff', err: '#ff6b6b', warn: '#ffe066', ok: '#6bff8f',
      text: '#ffffff', textDim: '#a8a8a8', preview: '#000000', hl: 'rgba(255,255,255,0.04)', sel: '#2a4a7a',
    },
    syn: {
      keyword: '#ff8fd0', type: '#ffe066', builtin: '#7fb8ff', variable: '#ffffff',
      comment: '#a8a8a8', number: '#ffb35c', string: '#6bff8f', operator: '#d0d0d0',
      meta: '#5ee8e8', atom: '#ffb35c', bracket: '#d0d0d0',
    },
  },
];

const THEME_BY_ID = new Map(THEMES.map(t => [t.id, t]));

/* ── Settings ────────────────────────────────────────────── */
// Vim needs a physical Esc key — default it off on touch-primary devices.
const hasFinePointer = window.matchMedia('(pointer: fine)').matches;

interface FloatRect { x: number; y: number; w: number; h: number }

interface Settings {
  theme: string;
  resScale: number;   // framebuffer multiplier
  swapped: boolean;   // preview pane first
  vimMode: boolean;
  compileDelay: number;     // ms of typing silence before auto-compile
  editorFloating: boolean;  // editor pane undocked into a floating window
  editorRect: FloatRect | null;
  editorGlass: GlassMode;   // floating editor background: see applyEditorGlass
  glassTint: number;        // 0..1 background opacity (tint and glass)
  glassFrost: number;       // px of backdrop blur (glass)
  views: ViewSpec[];        // the windows: pass views (the preview among them) and tools
  shotSize: string;         // screenshot size: '1x', '2x'… or 'WxH'
}

// `pass` is a pass id, 'uniforms' for the live uniform table, or a tool id
// ('calc', 'graph', 'texcalc', 'math', 'picture'); `data` holds what each tool pad
// saved, keyed by tool id, so switching kinds never shows another pad's content.
// Tool ids must never collide with pass ids ('image', 'bufA'…). `rect` is the
// floating position, kept while docked; w = 0 means "never floated yet".
// `kind`: a 'preview' carries the playback bar and the channel strip and shows
// passes only; a 'view' is just a selector over passes and tools.
type WinKind = 'preview' | 'view';
interface ViewSpec { pass: string; rect: FloatRect; kind?: WinKind; docked?: boolean; data?: Record<string, string> }
const DOCKED_IMAGE = (): ViewSpec => ({ pass: 'image', kind: 'preview', docked: true, rect: { x: 0, y: 0, w: 0, h: 0 } });
type GlassMode = 'solid' | 'tint' | 'clear' | 'glass';
const GLASS_MODES: GlassMode[] = ['solid', 'tint', 'clear', 'glass'];
const isGlassMode = (v: unknown): v is GlassMode => GLASS_MODES.includes(v as GlassMode);
const clamp01 = (v: number): number => Math.min(Math.max(v, 0), 1);
const UNIFORMS_VIEW = 'uniforms';
const TOOL_VIEWS: Array<[id: string, name: string]> = [
  [UNIFORMS_VIEW, 'Uniforms'], ['calc', 'Calculator'], ['graph', 'Graph'], ['texcalc', 'Math calc (LaTeX)'], ['math', 'Math notes (LaTeX)'], ['picture', 'Image'],
];
const isToolView = (id: string): boolean => TOOL_VIEWS.some(([t]) => t === id);
/** Room a tool needs to be usable; a window grows to this when switched to it. */
const TOOL_MIN_SIZE: Record<string, [w: number, h: number]> = {
  [UNIFORMS_VIEW]: [380, 320], calc: [320, 240], graph: [340, 300], texcalc: [360, 260], math: [360, 260], picture: [320, 240],
};

// 0 would give a degenerate framebuffer, so the low end stops just above it.
const RES_MIN = 0.01;
const RES_MAX = 1;

function clampRes(v: number): number {
  return Math.min(Math.max(v, RES_MIN), RES_MAX);
}

const DELAY_MIN = 0;
const DELAY_MAX = 5000;

function clampDelay(v: number): number {
  return Math.round(Math.min(Math.max(v, DELAY_MIN), DELAY_MAX));
}

const SETTINGS_KEY = 'shaderEditor.settings';

function isRect(r: unknown): r is FloatRect {
  const o = r as Record<string, unknown> | null;
  return !!o && ['x', 'y', 'w', 'h'].every(k => typeof o[k] === 'number' && isFinite(o[k] as number));
}
const copyRect = (r: FloatRect): FloatRect => ({ x: r.x, y: r.y, w: r.w, h: r.h });

function defaultSettings(): Settings {
  return {
    theme: 'midnight', resScale: 1, swapped: false, vimMode: hasFinePointer,
    compileDelay: 600, editorFloating: false, editorRect: null,
    editorGlass: 'solid', glassTint: 0.6, glassFrost: 8,
    views: [DOCKED_IMAGE()], shotSize: '1x',
  };
}

/** '2x' scales the preview size; '1920x1080' is absolute. Null when unparsable. */
function parseShotSize(spec: string, w: number, h: number): [number, number] | null {
  const m = /^\s*(\d+(?:\.\d+)?)\s*(x|×)?\s*(\d+)?\s*$/i.exec(spec);
  if (!m) return null;
  const a = parseFloat(m[1]);
  if (m[3] !== undefined) return [Math.round(a), parseInt(m[3], 10)].every(n => n >= 1 && n <= 16384) ? [Math.round(a), parseInt(m[3], 10)] : null;
  if (!(a > 0 && a <= 16)) return null;
  return [Math.max(1, Math.round(w * a)), Math.max(1, Math.round(h * a))];
}

function loadSettings(): Settings {
  const s = defaultSettings();
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return s;
    const p = JSON.parse(raw) as Partial<Settings>;
    if (typeof p.theme === 'string' && THEME_BY_ID.has(p.theme)) s.theme = p.theme;
    if (typeof p.resScale === 'number' && isFinite(p.resScale)) s.resScale = clampRes(p.resScale);
    if (typeof p.swapped === 'boolean') s.swapped = p.swapped;
    if (typeof p.vimMode === 'boolean') s.vimMode = p.vimMode;
    if (typeof p.compileDelay === 'number' && isFinite(p.compileDelay)) s.compileDelay = clampDelay(p.compileDelay);
    if (typeof p.editorFloating === 'boolean') s.editorFloating = p.editorFloating;
    if (isRect(p.editorRect)) s.editorRect = copyRect(p.editorRect);
    if (isGlassMode(p.editorGlass)) s.editorGlass = p.editorGlass;
    if (typeof p.glassTint === 'number' && isFinite(p.glassTint)) s.glassTint = clamp01(p.glassTint);
    if (typeof p.glassFrost === 'number' && isFinite(p.glassFrost)) s.glassFrost = Math.min(Math.max(p.glassFrost, 0), 40);
    if (typeof p.shotSize === 'string' && parseShotSize(p.shotSize, 1, 1)) s.shotSize = p.shotSize;
    if (Array.isArray(p.views)) s.views = parseViews(p.views, p as LegacyPreview);
  } catch {
    /* corrupt or unavailable storage — fall back to defaults */
  }
  return s;
}

/** Fields of layouts and settings from before the preview was a window. */
interface LegacyPreview { mainView?: unknown; floating?: unknown; floatRect?: unknown }

/**
 * Saved view specs, validated, with pre-rename tool ids migrated. A save from
 * when the preview was its own pane becomes a window in front of the rest:
 * docked where the pane was, or floating where it had been undocked to.
 */
function parseViews(raw: unknown[], legacy: LegacyPreview = {}): ViewSpec[] {
      const views = (raw as Array<Partial<ViewSpec> | null>)
        .filter((v): v is Partial<ViewSpec> & { pass: string; rect: FloatRect } => !!v && typeof v.pass === 'string' && isRect(v.rect))
        .map(v => {
          // Saves from before renames: an 'image' view with data was the picture
          // pad; the plain-text notes pad became a maths notepad (its text is dropped).
          const raw = v.data as unknown;
          const pass = v.pass === 'image' && typeof raw === 'string' ? 'picture' : v.pass === 'text' ? 'math' : v.pass;
          const data: Record<string, string> = {};
          if (typeof raw === 'string') { if (v.pass !== 'text') data[pass] = raw; }   // one-string saves
          else if (raw && typeof raw === 'object') {
            for (const [k, d] of Object.entries(raw as Record<string, unknown>)) if (typeof d === 'string') data[k] = d;
          }
          // Before kinds existed, the docked window was the preview.
          const kind: WinKind = v.kind === 'preview' || v.kind === 'view' ? v.kind : v.docked === true && !isToolView(pass) ? 'preview' : 'view';
          return { pass, rect: copyRect(v.rect), kind, docked: v.docked === true, ...(Object.keys(data).length ? { data } : {}) };
        });
      if (typeof legacy.mainView === 'string') {
        views.unshift({ pass: legacy.mainView, kind: 'preview', docked: legacy.floating !== true,
                        rect: isRect(legacy.floatRect) ? copyRect(legacy.floatRect) : { x: 0, y: 0, w: 0, h: 0 } });
      }
      return views;
}

function parseLayout(raw: unknown): SavedLayout | null {
  const l = raw as (Partial<SavedLayout> & LegacyPreview) | null | undefined;
  if (!l || typeof l !== 'object' || !Array.isArray(l.views)) return null;
  return {
    editorFloating: l.editorFloating === true,
    editorRect: isRect(l.editorRect) ? copyRect(l.editorRect) : null,
    views: parseViews(l.views, l),
  };
}

function saveSettings(): void {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* ignore */ }
  // The working project carries the layout too, and wins over settings on
  // load: keep it current, or a reload reverts to the windows as they were
  // at the last code edit.
  scheduleSave();
}

const settings: Settings = loadSettings();
{
  const l = parseLayout(initialLayoutRaw);   // from loadProject() above
  if (l) {
    settings.editorFloating = l.editorFloating;
    settings.editorRect = l.editorRect ?? settings.editorRect;
    settings.views = l.views;
  }
}

/* ── Resize ──────────────────────────────────────────────── */
// The GL canvas is never shown: every window copies the frame out of it. Its
// size is the primary pass window's (first docked, else the oldest floating),
// so iResolution follows that window and the others stretch to fit.

// Phones report 3x; past 2x the extra pixels cost far more than they show.
const MAX_DPR = 2;

let sizeOverride: [number, number] | null = null;   // set while capturing a screenshot
let shotOnCopies = false;                           // that screenshot is at another size, on copies of the buffers
let resText = '0 × 0';                              // the res readout, for windows made after the last change

function resizeCanvas(): void {
  const scale = settings.resScale * Math.min(devicePixelRatio, MAX_DPR);
  const pv = primaryView();
  if (!sizeOverride && !pv) return;   // no pass window: keep rendering at the last size
  const w = sizeOverride ? sizeOverride[0] : Math.max(1, pv!.cv.clientWidth  * scale | 0);
  const h = sizeOverride ? sizeOverride[1] : Math.max(1, pv!.cv.clientHeight * scale | 0);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    const suffix = settings.resScale === 1 ? '' : ` @ ${Math.round(settings.resScale * 100)}%`;
    resText = `${w} × ${h}${suffix}`;
    setReadout('res', resText);
    if (!sizeOverride) refreshShotReadout();   // "2x" means something new at the new size
  }
}

/** Write a status-bar readout into every pass window. */
function setReadout(cls: string, text: string): void {
  for (const v of views) {
    const el = v.status?.querySelector<HTMLElement>(`.${cls}`);
    if (el && el.textContent !== text) el.textContent = text;
  }
}

/* ── State ───────────────────────────────────────────────── */
let iTime = 0, iTimeDelta = 0, iFrame = -1;   // iFrame is the frame being drawn; -1 = none yet, so the first is 0 as on Shadertoy
let fpsValue = 60;                             // measured frame rate, for iFrameRate
let lastTimestamp: number | null = null;
let paused = false;
let stepFrames = 0;      // frames still to advance while paused (step button, iFrame edits)
let rafRunning = false;
let startTime = 0;
let iMouse: [number, number, number, number] = [0, 0, 0, 0];
let autoCompile = true;
let compileTimer: ReturnType<typeof setTimeout> | null = null;

const errLog   = document.getElementById('error-log')   as HTMLElement;
const errTitle = document.getElementById('error-title') as HTMLElement;
const errBody  = document.getElementById('error-body')  as HTMLElement;
const status   = document.getElementById('compile-status') as HTMLElement;

function setError(msg: string): void {
  renderErrorBody(msg);
  errLog.classList.toggle('has-errors', !!msg);
  status.textContent = msg ? 'ERR' : 'OK';
  status.className = msg ? 'pill err' : 'pill ok';
}

/**
 * The banner text, with every line the compiler located turned into a link
 * to that line of that tab. tryCompile() heads each pass's log with its name
 * when more than one pass is live; a lone Image log has no heading.
 */
function renderErrorBody(msg: string): void {
  errBody.replaceChildren();
  if (!msg) return;
  let pass: Pass | null = null;
  const lines = msg.split('\n');
  lines.forEach((line, i) => {
    const head = /^(.+):$/.exec(line);
    if (head) pass = allPasses().find(p => p.name === head[1]) ?? pass;
    const m = /^(?:ERROR|WARNING): (0|Common):(\d+):/.exec(line);
    const target = m ? (m[1] === 'Common' ? commonPass : pass ?? imagePass) : null;
    if (m && target) {
      const a = document.createElement('span');
      a.className = 'err-link';
      a.textContent = line;
      a.title = `Go to ${target.name} line ${m[2]}`;
      a.addEventListener('click', () => gotoLine(target, Number(m[2])));
      errBody.appendChild(a);
    } else {
      errBody.appendChild(document.createTextNode(line));
    }
    if (i < lines.length - 1) errBody.appendChild(document.createTextNode('\n'));
  });
}

/** Put the cursor at the start of line `n` (1-based) of a pass, switching tabs if needed. */
function gotoLine(p: Pass, n: number): void {
  if (p !== activePass) switchPass(p);
  const doc = editor.state.doc;
  const line = doc.line(Math.min(Math.max(n, 1), doc.lines));
  editor.dispatch({ selection: { anchor: line.from }, effects: EditorView.scrollIntoView(line.from, { y: 'center' }) });
  editor.focus();
}

/**
 * Lines of the active tab that the compiler complained about, as lint
 * diagnostics. WebGL logs look like `ERROR: 0:12: 'x' : undeclared identifier`;
 * labelErrors() has already turned the Common source's `1:` into `Common:`.
 */
function activeDiagnostics(): Diagnostic[] {
  const out: Diagnostic[] = [];
  const log = activePass === commonPass ? commonPass.error : activePass.error;
  if (!log) return out;
  const prefix = activePass === commonPass ? 'Common' : '0';
  const re = new RegExp(`^(ERROR|WARNING): ${prefix}:(\\d+):\\s*(.*)$`, 'gm');
  const doc = editor.state.doc;
  for (const m of log.matchAll(re)) {
    const n = Math.min(Math.max(Number(m[2]), 1), doc.lines);
    const line = doc.line(n);
    out.push({
      from: line.from, to: line.to,
      severity: m[1] === 'ERROR' ? 'error' : 'warning',
      message: m[3].trim(),
    });
  }
  return out;
}

function refreshDiagnostics(): void {
  editor.dispatch(setDiagnostics(editor.state, activeDiagnostics()));
}

/**
 * Compile every live pass. A pass that fails keeps its last good program, so
 * the preview keeps running whatever last worked while you fix the error.
 */
/**
 * Carries a pass's control objects across a recompile. The control rows in
 * the Uniforms view close over these objects and mutate `value` as a slider
 * moves, while drawPass reads whatever the pass holds now; replacing the
 * objects on every compile would leave the rows driving dead copies. So a
 * control that survives (same name, type and kind) keeps its identity and
 * takes the freshly parsed fields, including the literal's value.
 */
let controlsStale = false;
function adoptControls(previous: Control[], fresh: Control[]): Control[] {
  return fresh.map(c => {
    const old = previous.find(o => o.name === c.name && o.type === c.type && o.kind === c.kind);
    if (!old) return c;
    // A literal edited by hand (or a different project with the same controls)
    // moves the value under the row, which then has to be rebuilt to show it.
    if (old.value.length !== c.value.length || old.value.some((v, i) => v !== c.value[i])) controlsStale = true;
    old.min = c.min; old.max = c.max; old.step = c.step; old.line = c.line;
    old.value = c.value;
    return old;
  });
}

function tryCompile(): void {
  syncActivePassCode();
  const live = livePasses();
  const errors: string[] = [];

  commonPass.error = checkCommon();
  commonPass.controls = commonPass.enabled ? adoptControls(commonPass.controls, rewriteControls(commonPass.code).controls) : [];
  if (commonPass.error != null) {
    // Every pass would fail the same way; leave them on their last good programs.
    errors.push(`Common:\n${commonPass.error}`);
    for (const p of live) p.error = null;
  }

  for (const p of commonPass.error != null ? [] : live) {
    const { prog, log } = buildProgram(p.code, p.channels, p === imagePass);
    if (!prog) {
      p.error = log ?? '';
      errors.push(live.length > 1 ? `${p.name}:\n${p.error}` : p.error);
      continue;
    }
    p.error = null;
    if (p.program) gl.deleteProgram(p.program);
    p.program = prog;
    p.controls = adoptControls(p.controls, rewriteControls(p.code).controls);
    p.uniforms = cacheUniforms(prog, [...commonPass.controls, ...p.controls].map(c => c.name));
  }
  syncControlPanel();

  const anyRunning = livePasses().some(p => p.program);
  errTitle.textContent = anyRunning
    ? 'Compile failed — the last working build keeps running'
    : 'Compile failed — nothing to show yet';
  setError(errors.join('\n\n'));
  refreshDiagnostics();
  saveProject();
  updatePassTabs();
  scheduleRender();
}

/* ── Render loop ─────────────────────────────────────────── */
const STEP_DT = 1 / 60;

function scheduleRender(): void {
  if (!rafRunning) {
    rafRunning = true;
    requestAnimationFrame(render);
  }
}

// Frame pacing is measured from the rAF timestamps themselves, independent of
// shader time, so it stays honest across pause/resume and resets.
let fpsWindowStart = 0, fpsWindowFrames = 0, prevFrameTs: number | null = null;

function render(ts: DOMHighResTimeStamp): void {
  rafRunning = false;
  if (!paused) {
    rafRunning = true;
    requestAnimationFrame(render);
  }

  const advanced = stepFrames > 0 || !paused;
  if (stepFrames > 0) {
    // Paused: advance exactly one frame at a fixed 60 Hz step. As in the
    // live loop, the first frame is 0 at time 0 with no delta.
    stepFrames--;
    iTimeDelta = iFrame < 0 ? 0 : STEP_DT;
    iTime += iTimeDelta;
    iFrame++;
  } else if (!paused) {
    if (lastTimestamp === null) {
      lastTimestamp = ts;
      startTime = ts - iTime * 1000;
    }
    iTimeDelta = (ts - lastTimestamp) / 1000;
    lastTimestamp = ts;
    iTime = (ts - startTime) / 1000;
    iFrame++;
  }

  if (paused) {
    setReadout('fps', `paused · frame ${Math.max(iFrame, 0)}`);
    prevFrameTs = null;
  } else {
    if (prevFrameTs === null || ts - fpsWindowStart > 1000) {
      if (prevFrameTs !== null && fpsWindowFrames > 0) {
        const ms = (ts - fpsWindowStart) / fpsWindowFrames;
        fpsValue = 1000 / ms;
        setReadout('fps', `${fpsValue.toFixed(0)} fps · ${ms.toFixed(1)} ms`);
      }
      fpsWindowStart = ts;
      fpsWindowFrames = 0;
    }
    fpsWindowFrames++;
    prevFrameTs = ts;
  }

  resizeCanvas();
  const W = canvas.width, H = canvas.height;

  // Buffers first, in order, each into its own target; then Image to screen.
  // A pass whose latest edit failed to compile keeps drawing its last build.
  // Buffers only step when time does: a paused repaint (resize, inspector,
  // slider) must not advance a feedback shader. A screenshot at another size
  // is the exception: it runs one frame on copies of the buffers.
  const stepBuffers = advanced || shotOnCopies;
  const buffers = activeBufferPasses();
  if (stepBuffers) {
    // Mip levels only when something samples a buffer with them (32-bit float has none without linear filtering).
    mipWanted.fill(0);
    for (const q of buffers) for (const c of q.channels) if (c.kind === 'buffer' && c.filter === 'mipmap') mipWanted[c.buf] = 1;
    for (const c of imagePass.channels) if (c.kind === 'buffer' && c.filter === 'mipmap') mipWanted[c.buf] = 1;
  }
  for (const p of stepBuffers ? buffers : []) {
    if (!p.program) continue;
    const t = ensureTarget(p.bufIndex, W, H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.write, 0);
    gl.viewport(0, 0, W, H);
    drawPass(p, W, H);
    // Swap so later passes (and the next frame) sample what we just drew.
    const prev = t.read;
    t.read = t.write;
    t.write = prev;
    t.mips = mipWanted[p.bufIndex] === 1 && (t.format !== 'rgba32f' || floatLinear);
    if (t.mips) { gl.bindTexture(gl.TEXTURE_2D, t.read); gl.generateMipmap(gl.TEXTURE_2D); gl.bindTexture(gl.TEXTURE_2D, null); }
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, W, H);

  // Each shown pass is drawn to the screen buffer once, then copied into
  // every window showing it, so two Image windows cost one Image draw. The
  // inspected pixel is read while its pass is on screen.
  viewsByPass.clear();
  for (const v of views) {
    if (isToolView(v.pass) || !v.visible()) continue;
    let list = viewsByPass.get(v.pass);
    if (!list) viewsByPass.set(v.pass, list = []);
    list.push(v);
  }
  for (const [passId, list] of viewsByPass) {
    drawToScreen(passId, W, H);
    for (const v of list) {
      if (inspect.view === v) readInspectedPixel(v, W, H);
      v.blit();
    }
  }
  updateUniformPanel(W, H);
}

/* ── Pixel inspector ─────────────────────────────────────── */
// Reads one pixel of a window's pass each frame: the screen for Image (8-bit,
// what is displayed) or the buffer's own texture (float when available).
const inspect: { view: PassView | null; frozen: boolean; x: number; y: number } = { view: null, frozen: false, x: 0, y: 0 };
const pixelU8 = new Uint8Array(4);
const pixelF32 = new Float32Array(4);

function readInspectedPixel(v: PassView, W: number, H: number): void {
  const x = Math.min(Math.max(inspect.x | 0, 0), W - 1), y = Math.min(Math.max(inspect.y | 0, 0), H - 1);
  let rgba: [number, number, number, number], float = false;
  const p = bufferPasses.find(b => b.id === v.pass);
  const t = p && p.enabled ? targets[p.bufIndex] : null;
  if (t) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.read, 0);
    if (t.format !== 'rgba8') { gl.readPixels(x, y, 1, 1, gl.RGBA, gl.FLOAT, pixelF32); rgba = [...pixelF32] as typeof rgba; float = true; }
    else { gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixelU8); rgba = [...pixelU8].map(v => v / 255) as typeof rgba; }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  } else {
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixelU8);
    rgba = [...pixelU8].map(v => v / 255) as typeof rgba;
  }
  const f = (v: number) => (float ? +v.toPrecision(5) : v.toFixed(3)).toString();
  const hex = '#' + rgba.slice(0, 3).map(v => Math.round(Math.min(Math.max(v, 0), 1) * 255).toString(16).padStart(2, '0')).join('').toUpperCase();
  v.pxSwatch.style.background = hex;
  v.pxText.textContent =
    `fragCoord ${x + 0.5}, ${y + 0.5}\nuv ${((x + 0.5) / W).toFixed(4)}, ${((y + 0.5) / H).toFixed(4)}\n`
    + `rgba ${rgba.map(f).join(' ')}\n${hex}${float ? '  (float)' : ''}${inspect.frozen ? '  · frozen' : ''}`;
}

/** Probe pixels in one window (null: stop). Only one window is probed at a time. */
function setInspecting(v: PassView | null): void {
  inspect.view = v;
  inspect.frozen = false;
  for (const w of views) {
    w.el.classList.toggle('inspecting', w === v);
    w.pxInfo.classList.toggle('open', w === v);
  }
  syncStatusButtons();
  scheduleRender();
}

/** Draw a pass's current output into the (already bound) default framebuffer. */
function drawToScreen(passId: string, W: number, H: number): void {
  if (passId === 'image') {
    if (imagePass.program) { drawPass(imagePass, W, H); return; }
  } else {
    const p = bufferPasses.find(b => b.id === passId);
    const t = p && p.enabled ? targets[p.bufIndex] : null;
    const prog = t && ensureBlit();
    if (t && prog) {
      gl.useProgram(prog);
      gl.uniform2f(blitRes, W, H);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, t.read);
      gl.bindVertexArray(vao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);
      return;
    }
  }
  gl.clearColor(0.05, 0.05, 0.07, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

const channelRes = new Float32Array(NUM_SLOTS * 3);
const mipWanted = new Uint8Array(NUM_BUFFERS);           // per frame: buffers some live pass samples with mipmaps
const viewsByPass = new Map<string, PassView[]>();        // per frame: pass id -> windows showing it

function drawPass(p: Pass, w: number, h: number): void {
  const u = p.uniforms;
  gl.useProgram(p.program);

  if (u.iTime       != null) gl.uniform1f(u.iTime, iTime);
  if (u.iTimeDelta  != null) gl.uniform1f(u.iTimeDelta, iTimeDelta);
  if (u.iFrameRate  != null) gl.uniform1f(u.iFrameRate, fpsValue);
  if (u.iFrame      != null) gl.uniform1i(u.iFrame, iFrame);
  if (u.iResolution != null) gl.uniform3f(u.iResolution, w, h, 1);
  if (u.iSampleRate != null) gl.uniform1f(u.iSampleRate, 44100);
  if (u.iMouse      != null) gl.uniform4fv(u.iMouse, iMouse);

  if (u.iDate != null) {
    const now = new Date();
    gl.uniform4f(u.iDate, now.getFullYear(), now.getMonth(), now.getDate(),
      now.getHours()*3600 + now.getMinutes()*60 + now.getSeconds() + now.getMilliseconds()/1000);
  }

  for (let i = 0; i < NUM_SLOTS; i++) {
    const c = p.channels[i];
    let cw = 0, chh = 0;
    gl.activeTexture(gl.TEXTURE0 + i);
    gl.bindSampler(i, null);
    if (c.kind === 'cube') {
      gl.bindTexture(gl.TEXTURE_CUBE_MAP, c.cubeTex ?? BLACK_CUBE_TEX);
      cw = c.texW; chh = c.texH;
    } else if (c.kind === 'buffer') {
      // A buffer sampling itself sees the previous frame, which is the point.
      const t = targets[c.buf];
      gl.bindTexture(gl.TEXTURE_2D, t ? t.read : WHITE_TEX);
      if (t) { cw = t.w; chh = t.h; bindSampling(i, c, t.format !== 'rgba32f' || floatLinear, t.mips); }
    } else {
      gl.bindTexture(gl.TEXTURE_2D, c.tex ?? WHITE_TEX);
      cw = c.texW; chh = c.texH;
      if (c.tex) bindSampling(i, c, true, true);
    }
    const loc = u[`iChannel${i}`];
    if (loc != null) gl.uniform1i(loc, i);
    channelRes[i * 3] = cw;
    channelRes[i * 3 + 1] = chh;
    channelRes[i * 3 + 2] = 1;
  }
  if (u.iChannelResolution != null) gl.uniform3fv(u.iChannelResolution, channelRes);

  for (const c of commonPass.enabled ? commonPass.controls : []) setControlUniform(u, c);
  for (const c of p.controls) setControlUniform(u, c);

  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
  for (let i = 0; i < NUM_SLOTS; i++) gl.bindSampler(i, null);   // copies and blits use the textures' own parameters
}

function setControlUniform(u: Uniforms, c: Control): void {
  const loc = u[c.name];
  if (loc == null) return;
  const v = c.value;
  switch (c.type) {
    case 'float': gl.uniform1f(loc, v[0]); break;
    case 'int':   gl.uniform1i(loc, Math.round(v[0])); break;
    case 'bool':  gl.uniform1i(loc, v[0] ? 1 : 0); break;
    case 'vec2':  gl.uniform2f(loc, v[0], v[1]); break;
    case 'vec3':  gl.uniform3f(loc, v[0], v[1], v[2]); break;
    case 'vec4':  gl.uniform4f(loc, v[0], v[1], v[2], v[3]); break;
  }
}

/* ── Pointer tracking (mouse + touch) ────────────────────── */
// Any pass window drives iMouse; coordinates are scaled to the GL canvas,
// so a window of a different shape maps its whole area onto the frame.
let pointerDown = false;

function viewCoords(v: PassView, e: PointerEvent): [number, number] {
  const r = v.cv.getBoundingClientRect();
  const x = (e.clientX - r.left) * (canvas.width  / r.width);
  const y = canvas.height - (e.clientY - r.top) * (canvas.height / r.height);
  return [x, y];
}

// As on Shadertoy, iMouse.z goes negative once the button is released.
function pointerRelease(): void {
  pointerDown = false;
  iMouse[2] = -Math.abs(iMouse[2]);
}

function attachPointer(v: PassView): void {
  const cv = v.cv;
  cv.addEventListener('pointermove', (e: PointerEvent) => {
    const [x, y] = viewCoords(v, e);
    if (inspect.view === v) {
      // The inspector takes the pointer so probing never disturbs iMouse.
      if (!inspect.frozen) { inspect.x = x; inspect.y = y; if (paused) scheduleRender(); }
      return;
    }
    // As on Shadertoy: xy follows the pointer only while a button is held,
    // and zw keeps the position of the click (set in pointerdown), so
    // `iMouse.xy - iMouse.zw` is the drag delta.
    if (pointerDown) { iMouse[0] = x; iMouse[1] = y; }
  });
  cv.addEventListener('pointerdown', (e: PointerEvent) => {
    const [x, y] = viewCoords(v, e);
    if (inspect.view === v) {
      inspect.frozen = !inspect.frozen;
      if (!inspect.frozen) { inspect.x = x; inspect.y = y; }
      scheduleRender();
      return;
    }
    pointerDown = true;
    iMouse[0] = x; iMouse[1] = y;
    iMouse[2] = x; iMouse[3] = y;
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener('pointerup',     pointerRelease);
  cv.addEventListener('pointercancel', pointerRelease);
  cv.addEventListener('pointerleave',  pointerRelease);
}

/* ── GLSL language definition ────────────────────────────── */
const GLSL_KEYWORDS = new Set([
  'break','continue','do','for','while','if','else','return','discard',
  'switch','case','default',
]);
const GLSL_QUALIFIERS = new Set([
  'attribute','const','uniform','varying','in','out','inout','centroid',
  'flat','smooth','lowp','mediump','highp','precision','invariant',
  'layout','struct',
]);
const GLSL_TYPES = new Set([
  'float','int','uint','void','bool',
  'vec2','vec3','vec4','ivec2','ivec3','ivec4','bvec2','bvec3','bvec4','uvec2','uvec3','uvec4',
  'mat2','mat3','mat4',
  'mat2x2','mat2x3','mat2x4','mat3x2','mat3x3','mat3x4','mat4x2','mat4x3','mat4x4',
  'sampler2D','sampler3D','samplerCube','sampler2DShadow','samplerCubeShadow',
  'sampler2DArray','sampler2DArrayShadow',
  'isampler2D','isampler3D','isamplerCube','isampler2DArray',
  'usampler2D','usampler3D','usamplerCube','usampler2DArray',
]);
const GLSL_BUILTINS = new Set([
  'radians','degrees','sin','cos','tan','asin','acos','atan','sinh','cosh','tanh',
  'asinh','acosh','atanh','pow','exp','log','exp2','log2','sqrt','inversesqrt',
  'abs','sign','floor','trunc','round','roundEven','ceil','fract','mod','modf',
  'min','max','clamp','mix','step','smoothstep','isnan','isinf',
  'floatBitsToInt','floatBitsToUint','intBitsToFloat','uintBitsToFloat',
  'packSnorm2x16','unpackSnorm2x16','packUnorm2x16','unpackUnorm2x16',
  'packHalf2x16','unpackHalf2x16',
  'length','distance','dot','cross','normalize','faceforward','reflect','refract',
  'matrixCompMult','outerProduct','transpose','determinant','inverse',
  'lessThan','lessThanEqual','greaterThan','greaterThanEqual','equal','notEqual',
  'any','all','not',
  'textureSize','texture','textureProj','textureLod','textureOffset',
  'texelFetch','texelFetchOffset','textureProjOffset','textureLodOffset',
  'textureProjLod','textureProjLodOffset','textureGrad','textureGradOffset',
  'textureProjGrad','textureProjGradOffset','dFdx','dFdy','fwidth',
  'emit','endPrimitive',
  'gl_Position','gl_PointSize','gl_FragCoord','gl_FrontFacing',
  'gl_FragDepth','gl_PointCoord','gl_VertexID','gl_InstanceID',
  'iTime','iTimeDelta','iFrame','iResolution','iMouse','iDate',
  'iChannelResolution',
  'iChannel0','iChannel1','iChannel2','iChannel3','mainImage',
]);
const GLSL_ATOMS = new Set(['true','false']);

interface GlslState { blockComment: boolean; }

const glslParser = {
  startState: (): GlslState => ({ blockComment: false }),
  token(stream: StringStream, state: GlslState): string | null {
    if (state.blockComment) {
      if (stream.match(/^.*?\*\//)) state.blockComment = false;
      else stream.skipToEnd();
      return 'comment';
    }
    if (stream.eatSpace()) return null;
    if (stream.match('/*')) { state.blockComment = true; return 'comment'; }
    if (stream.match('//')) { stream.skipToEnd(); return 'comment'; }
    if (stream.match(/^#\s*\w+/)) { stream.skipToEnd(); return 'meta'; }
    if (stream.match(/^0[xX][0-9a-fA-F]+[uU]?/)) return 'number';
    if (stream.match(/^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?[fFuU]?/)) return 'number';
    if (stream.match(/^"[^"]*"/)) return 'string';
    if (stream.match(/^[+\-*/%=<>!&|^~?:]+/)) return 'operator';
    if (stream.match(/^[{}[\]();,.]/)) return 'bracket';
    if (stream.match(/^\w+/)) {
      const w = stream.current();
      if (GLSL_KEYWORDS.has(w))   return 'keyword';
      if (GLSL_QUALIFIERS.has(w)) return 'keyword';
      if (GLSL_TYPES.has(w))      return 'type';
      if (GLSL_BUILTINS.has(w))   return 'builtin';
      if (GLSL_ATOMS.has(w))      return 'atom';
      return null;
    }
    stream.next();
    return null;
  },
  languageData: {
    commentTokens: { line: '//', block: { open: '/*', close: '*/' } },
  },
};

const glslLanguage = StreamLanguage.define(glslParser);

const glslCompletions = [
  ...[...GLSL_KEYWORDS, ...GLSL_QUALIFIERS].map(k => ({ label: k, type: 'keyword' })),
  ...[...GLSL_TYPES].map(t => ({ label: t, type: 'type' })),
  ...[...GLSL_BUILTINS].map(b => ({ label: b, type: 'function' })),
];

// Names the shader itself defines, across every pass and Common.
const GLSL_TYPE_RE = '(?:void|float|int|uint|bool|[iub]?vec[234]|mat[234](?:x[234])?|sampler\\w+)';
const USER_FN_RE   = new RegExp(`^[ \\t]*(?:(?:highp|mediump|lowp)\\s+)?(${GLSL_TYPE_RE})\\s+(\\w+)\\s*\\(`, 'gm');
const USER_VAR_RE  = new RegExp(`^[ \\t]*(?:const\\s+|uniform\\s+)?(?:(?:highp|mediump|lowp)\\s+)?(${GLSL_TYPE_RE})\\s+(\\w+)\\s*(?:=|;|\\[)`, 'gm');
const USER_DEF_RE  = /^[ \t]*#define\s+(\w+)/gm;
const USER_STRUCT_RE = /^[ \t]*struct\s+(\w+)/gm;

function userCompletions(current: string): Array<{ label: string; type: string; detail?: string }> {
  const seen = new Map<string, { label: string; type: string; detail?: string }>();
  const add = (label: string, type: string, detail?: string) => {
    if (!GLSL_BUILTINS.has(label) && !GLSL_TYPES.has(label) && !seen.has(label)) seen.set(label, { label, type, detail });
  };
  const sources = [current, ...allPasses().filter(p => p !== activePass && p.enabled).map(p => p.code)];
  for (const src of sources) {
    for (const m of src.matchAll(USER_FN_RE)) add(m[2], 'function', m[1]);
    for (const m of src.matchAll(USER_STRUCT_RE)) add(m[1], 'type');
    for (const m of src.matchAll(USER_DEF_RE)) add(m[1], 'constant');
    for (const m of src.matchAll(USER_VAR_RE)) add(m[2], 'variable', m[1]);
  }
  return [...seen.values()];
}

function glslComplete(context: CompletionContext): CompletionResult | null {
  const word = context.matchBefore(/\w+/);
  if (!word || (word.from === word.to && !context.explicit)) return null;
  const q = word.text.toLowerCase();
  const user = userCompletions(context.state.doc.toString()).filter(c => c.label !== word.text);
  return {
    from: word.from,
    options: [...user, ...glslCompletions].filter(c => c.label.toLowerCase().startsWith(q)),
    validFor: /^\w*$/,
  };
}

/* ── Syntax highlight style ──────────────────────────────── */
function makeHighlight(t: Theme) {
  const s = t.syn;
  return syntaxHighlighting(HighlightStyle.define([
    { tag: tags.keyword,                          color: s.keyword },
    { tag: tags.typeName,                         color: s.type },
    { tag: [tags.standard(tags.name), tags.name], color: s.builtin },
    { tag: tags.variableName,                     color: s.variable },
    { tag: tags.comment,    color: s.comment, fontStyle: 'italic' },
    { tag: tags.number,                           color: s.number },
    { tag: tags.string,                           color: s.string },
    { tag: tags.operator,                         color: s.operator },
    { tag: tags.meta,                             color: s.meta },
    { tag: tags.atom,                             color: s.atom },
    { tag: tags.bracket,                          color: s.bracket },
    { tag: tags.modifier,                         color: s.keyword },
  ]));
}

function makeEditorTheme(t: Theme) {
  const u = t.ui;
  return EditorView.theme({
    '&': { backgroundColor: u.bg1, color: u.text, height: '100%' },
    '.cm-scroller': { fontFamily: "'SF Mono','Fira Code','Cascadia Code','Consolas',monospace" },
    '.cm-content': { caretColor: u.accent },
    '&.cm-focused .cm-cursor': { borderLeftColor: u.accent },
    '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': { backgroundColor: u.sel },
    '.cm-panels': { backgroundColor: u.bg0, color: u.text },
    '.cm-panels.cm-panels-top': { borderBottom: `1px solid ${u.border}` },
    '.cm-searchMatch': { backgroundColor: u.sel, outline: `1px solid ${u.accent}` },
    '.cm-searchMatch.cm-searchMatch-selected': { backgroundColor: u.sel, outline: `2px solid ${u.accent}` },
    '.cm-tooltip': { backgroundColor: u.bg2, border: `1px solid ${u.border}`, borderRadius: '4px', color: u.text },
    '.cm-tooltip-autocomplete': { '& > ul > li[aria-selected]': { backgroundColor: u.bg3, color: u.text } },
    '&.cm-focused .cm-matchingBracket': { backgroundColor: u.sel, outline: `1px solid ${u.accent}`, borderRadius: '2px' },
    '&.cm-focused .cm-nonmatchingBracket': { outline: `1px solid ${u.err}`, borderRadius: '2px' },
  }, { dark: t.dark });
}

/* ── GLSL reference tooltips ─────────────────────────────── */
function docTooltipDom(name: string, d: GlslDoc): HTMLElement {
  const dom = document.createElement('div');
  dom.className = 'glsl-doc';
  const head = document.createElement('div');
  head.className = 'gd-head';
  const nameEl = document.createElement('span');
  nameEl.className = 'gd-name';
  nameEl.textContent = name;
  const kindEl = document.createElement('span');
  kindEl.className = 'gd-kind';
  kindEl.textContent = d.kind;
  head.append(nameEl, kindEl);
  dom.appendChild(head);
  for (const sig of d.sig) {
    const pre = document.createElement('pre');
    pre.className = 'gd-sig';
    pre.textContent = sig;
    dom.appendChild(pre);
  }
  const body = document.createElement('div');
  body.className = 'gd-body';
  body.textContent = d.doc;
  dom.appendChild(body);
  return dom;
}

/** The identifier under `pos`, if there's a reference card for it. */
function docTooltipAt(state: EditorState, pos: number): Tooltip | null {
  const line = state.doc.lineAt(pos);
  const text = line.text, col = pos - line.from;
  let from = col, to = col;
  while (from > 0 && /\w/.test(text[from - 1])) from--;
  while (to < text.length && /\w/.test(text[to])) to++;
  if (from === to) return null;
  const name = text.slice(from, to);
  const d = lookupGlslDoc(name);
  if (!d) return null;
  return {
    pos: line.from + from, end: line.from + to, above: true,
    create: () => ({ dom: docTooltipDom(name, d) }),
  };
}

const glslHoverDocs = hoverTooltip((view, pos) => docTooltipAt(view.state, pos), { hoverTime: 400 });

// On demand (K in vim normal mode, F1 otherwise) the card pins to the cursor
// and goes away as soon as the cursor moves or the text changes.
const setDocTooltip = StateEffect.define<Tooltip | null>();
const docTooltipField = StateField.define<Tooltip | null>({
  create: () => null,
  update(value, tr) {
    for (const e of tr.effects) if (e.is(setDocTooltip)) return e.value;
    return tr.docChanged || tr.selection ? null : value;
  },
  provide: f => showTooltip.from(f),
});

function showDocAtCursor(view: EditorView): boolean {
  const tip = docTooltipAt(view.state, view.state.selection.main.head);
  view.dispatch({ effects: setDocTooltip.of(tip) });
  return tip !== null;
}

function closeDocTooltip(view: EditorView): boolean {
  if (!view.state.field(docTooltipField)) return false;
  view.dispatch({ effects: setDocTooltip.of(null) });
  return true;
}

Vim.defineAction('glslDoc', cm => { showDocAtCursor(cm.cm6); });
Vim.mapCommand('K', 'action', 'glslDoc', {}, { context: 'normal' });

/**
 * Format the document (or the lines `from`..`to`, 0-based inclusive). The
 * formatter never joins or splits lines, so only the lines whose text changed
 * are replaced: one undo step, the cursor stays on its line, and a range from
 * `:'<,'>format` maps straight onto the result.
 */
function formatDocument(view: EditorView, from = 0, to = view.state.doc.lines - 1): boolean {
  const doc = view.state.doc;
  const formatted = formatGlsl(doc.toString()).split('\n');
  const changes: { from: number; to: number; insert: string }[] = [];
  for (let i = Math.max(0, from); i <= Math.min(to, doc.lines - 1); i++) {
    const line = doc.line(i + 1);
    if (line.text !== formatted[i]) changes.push({ from: line.from, to: line.to, insert: formatted[i] });
  }
  if (!changes.length) return false;
  const head = view.state.selection.main.head;
  const cursorLine = doc.lineAt(head);
  view.dispatch({ changes, userEvent: 'format' });
  // Keep the cursor on the same line, clamped to its new length.
  const newLine = view.state.doc.line(cursorLine.number);
  const column = Math.min(head - cursorLine.from, newLine.length);
  view.dispatch({ selection: { anchor: newLine.from + column }, scrollIntoView: true });
  return true;
}

// `:format` tidies the whole pass; with a range (`:'<,'>format`, `:10,20format`) just those lines.
Vim.defineEx('format', 'form', (cm, params: { line?: number; lineEnd?: number }) => {
  const all = params.line === undefined;
  formatDocument(cm.cm6, all ? 0 : params.line, all ? undefined : params.lineEnd ?? params.line);
});

const themeCompartment = new Compartment();
const vimCompartment   = new Compartment();

function themeExtensions(t: Theme) { return [makeHighlight(t), makeEditorTheme(t)]; }
function vimExtension(on: boolean) { return on ? [vim({ status: true })] : []; }   // mode + pending keys in a bar under the code

// A plain `y` also lands on the system clipboard (like `clipboard=unnamed`,
// but for yanks only, so deletes stay internal). `"+y` is handled by the
// vim extension itself; pasting from the system still needs `"+p`.
{
  const rc = Vim.getRegisterController() as { pushText: (...a: unknown[]) => void };
  const push = rc.pushText;
  rc.pushText = function (this: unknown, registerName: unknown, operator: unknown, text: unknown, ...rest: unknown[]) {
    push.call(this, registerName, operator, text, ...rest);
    if (operator === 'yank' && (!registerName || registerName === '"') && typeof text === 'string') {
      navigator.clipboard?.writeText(text).catch(() => { /* clipboard unavailable */ });
    }
  };
}

function makeEditorState(doc: string): EditorState {
  const theme = THEME_BY_ID.get(settings.theme) ?? THEMES[0];
  return EditorState.create({
    doc,
    extensions: [
      vimCompartment.of(vimExtension(settings.vimMode)),
      lineNumbers(),
      highlightActiveLineGutter(),
      highlightSpecialChars(),
      history(),
      foldGutter(),
      drawSelection(),
      dropCursor(),
      EditorState.allowMultipleSelections.of(true),
      indentOnInput(),
      bracketMatching(),
      closeBrackets(),
      autocompletion({ override: [glslComplete] }),
      rectangularSelection(),
      crosshairCursor(),
      highlightActiveLine(),
      highlightSelectionMatches(),
      keymap.of([
        { key: 'Escape', run: closeDocTooltip },
        { key: 'F1', run: showDocAtCursor },
        ...closeBracketsKeymap,
        ...defaultKeymap,
        ...searchKeymap,
        ...historyKeymap,
        ...foldKeymap,
        ...completionKeymap,
        ...lintKeymap,
        indentWithTab,
        { key: 'Ctrl-Enter', run: () => { scheduleCompile(true); return true; } },
        { key: 'Ctrl-Alt-r', run: () => { redrawEditor(); return true; } },
        { key: 'Shift-Alt-f', run: view => { formatDocument(view); return true; } },
      ]),
      glslLanguage,
      glslHoverDocs,
      docTooltipField,
      lintGutter(),
      themeCompartment.of(themeExtensions(theme)),
      EditorView.updateListener.of(update => {
        if (update.docChanged) scheduleSave();
        if (update.docChanged && autoCompile) scheduleCompile(false);
      }),
      EditorView.lineWrapping,
    ],
  });
}

let editor: EditorView;
let activePass: Pass = initialPass;

/**
 * Rebuild the editor's DOM from its own state. The document, selection and
 * undo history are untouched; only the rendering is redone. For when what
 * is shown has drifted from what is stored (a browser extension or an
 * input method editing the contenteditable behind CodeMirror's back).
 */
function redrawEditor(): void {
  if (!editor) return;
  const state = editor.state;
  editor.setState(state);
  editor.focus();
}

/** The editor holds the live document for the active pass only. */
function syncActivePassCode(): void {
  if (editor) activePass.code = editor.state.doc.toString();
}

function scheduleCompile(immediate: boolean): void {
  if (compileTimer) clearTimeout(compileTimer);
  const delay = immediate ? 0 : settings.compileDelay;
  compileTimer = setTimeout(tryCompile, delay);
}

/* ── Toolbar wiring ──────────────────────────────────────── */
const btnCompile     = document.getElementById('btn-compile')     as HTMLButtonElement;
const btnAutocompile = document.getElementById('btn-autocompile') as HTMLButtonElement;
const btnExport      = document.getElementById('btn-export')      as HTMLButtonElement;
const btnNewView     = document.getElementById('btn-new-view')    as HTMLButtonElement;
const btnNewPreview  = document.getElementById('btn-new-preview') as HTMLButtonElement;

btnCompile.addEventListener('click', () => scheduleCompile(true));

btnAutocompile.addEventListener('click', () => {
  autoCompile = !autoCompile;
  btnAutocompile.classList.toggle('active', autoCompile);
  btnAutocompile.textContent = autoCompile ? 'Auto' : 'Manual';
});

/* ── Playback (the status bar of every pass window) ──────── */
function setPaused(on: boolean): void {
  paused = on;
  lastTimestamp = null;
  if (!paused) setReadout('fps', 'running');
  scheduleRender();   // paused: repaint so the status bar shows the frame number
  document.body.classList.toggle('paused', paused);
  syncStatusButtons();
}

/** Advance `n` frames while paused, each rendered synchronously. */
function stepFrame(n = 1): void {
  if (!paused) setPaused(true);
  for (let i = 0; i < n; i++) {
    stepFrames = 1;
    render(performance.now());
  }
}

/** Jump to frame `n`: forward runs the frames between; backward restarts and runs `n`. */
function seekFrame(n: number): void {
  n = Math.max(0, Math.min(n | 0, iFrame + 100000));
  if (!paused) setPaused(true);
  if (n < iFrame) {
    iTime = 0; iFrame = -1; iTimeDelta = 0; lastTimestamp = null;
    clearTargets();
  }
  stepFrame(n - iFrame);
  scheduleRender();
}

function resetTime(): void {
  iTime = 0; iFrame = -1; iTimeDelta = 0; lastTimestamp = null;
  clearTargets();   // feedback buffers would otherwise survive the reset
  scheduleRender();
}

function toggleFullscreen(v: PassView): void {
  if (!document.fullscreenElement) v.el.requestFullscreen?.();
  else document.exitFullscreen?.();
}
document.addEventListener('fullscreenchange', () => { syncStatusButtons(); scheduleRender(); });

/** Every pass window's buttons reflect the one global state. */
function syncStatusButtons(): void {
  for (const v of views) {
    if (!v.status) continue;
    const b = (act: string) => v.status!.querySelector<HTMLButtonElement>(`[data-act="${act}"]`)!;
    b('pause').classList.toggle('active', paused);
    b('pause').title = paused ? 'Resume (space)' : 'Pause (space)';
    b('inspect').classList.toggle('active', inspect.view === v);
    b('inspect').title = inspect.view === v ? 'Stop inspecting pixels (i)' : 'Inspect pixels: hover this window, click to freeze (i)';
    b('record').classList.toggle('recording', recording);
    b('record').title = recording ? 'Stop recording' : 'Record video';
    b('fullscreen').classList.toggle('active', document.fullscreenElement === v.el);
  }
}

/* ── Export shader code ──────────────────────────────────── */
btnExport.addEventListener('click', () => {
  syncActivePassCode();
  const live = livePasses();
  // A lone Image with nothing wired exports bare; a wired one (a cubemap,
  // say) keeps the section header so import restores the slots.
  const bare = live.length === 1 && imagePass.channels.every(c => c.kind === '2d' && !samplingNote(c));
  const code = bare
    ? fragSource(imagePass.code, imagePass.channels, false, true)
    : live.map(p => {
        const wiring = p.channels
          .map((c, i) => `//   iChannel${i}: ${channelDescription(c)}${samplingNote(c)}`)
          .join('\n');
        const precision = p.bufIndex >= 0 ? `//   precision: ${BUF_FORMATS.find(([f]) => f === p.format)?.[1]}\n` : '';
        return `// ===== ${p.name} =====\n${wiring}\n${precision}\n`
          + fragSource(p.code, p.channels, false, p === imagePass);
      }).join('\n\n');
  const blob = new Blob([code], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${projectName.replace(/[^\w.-]+/g, '_') || 'shader'}.glsl`;
  a.click();
  URL.revokeObjectURL(url);
  toggleShaders(false);
});

/* ── Import shader code ──────────────────────────────────── */
// Reads a file this editor exported (`// ===== Pass =====` sections, each a
// full fragment source with a wiring header) or any plain mainImage shader,
// which becomes the Image pass. Uploaded images cannot come back; their
// slots stay 2D and empty.
const btnImport  = document.getElementById('btn-import')  as HTMLButtonElement;
const importFile = document.getElementById('import-file') as HTMLInputElement;

const SECTION_RE = /^\/\/ ===== (.+?) =====[ \t]*\r?\n/gm;
const USER_MARK = '#line 1 0\n';
const COMMON_RE = /#line 1 1\n([\s\S]*?)\n(?=uniform sampler(?:2D|Cube) iChannel0;)/;
const MAIN_RE = /\n?void main\(\) \{[^\n]*_fragOut[^\n]*\}\s*$/;   // either suffix, old exports included

/** One exported pass back to what the user wrote, plus the Common code it carried. */
function unwrapExported(text: string): { code: string; common: string | null } {
  const at = text.indexOf(USER_MARK);
  if (at < 0) return { code: text.replace(/\s+$/, ''), common: null };   // not ours: keep as written
  const common = COMMON_RE.exec(text.slice(0, at))?.[1] ?? null;
  return { code: text.slice(at + USER_MARK.length).replace(MAIN_RE, ''), common };
}

/** The wiring header above an exported pass: `//   iChannel0: Buffer A [nearest, clamp]`. */
function parseWiring(header: string): { channels: SavedChannel[]; format?: BufFormat } {
  const channels: SavedChannel[] = Array.from({ length: NUM_SLOTS }, () => ({ kind: '2d' as const, buf: 0 }));
  let format: BufFormat | undefined;
  for (const line of header.split('\n')) {
    const ch = /^\/\/\s+iChannel(\d): (.+?)(?: \[(.*)\])?\s*$/.exec(line);
    if (ch) {
      const i = +ch[1];
      if (i >= NUM_SLOTS) continue;
      const c = channels[i];
      const buf = /^Buffer ([A-D])$/.exec(ch[2]);
      if (buf) { c.kind = 'buffer'; c.buf = BUFFER_NAMES.indexOf(buf[1] as typeof BUFFER_NAMES[number]); c.filter = 'linear'; c.wrap = 'clamp'; }
      else if (ch[2].startsWith('cubemap')) c.kind = 'cube';
      for (const flag of (ch[3] ?? '').split(',').map(f => f.trim())) {
        if (FILTERS.includes(flag as Filter)) c.filter = flag as Filter;
        else if (WRAPS.includes(flag as Wrap)) c.wrap = flag as Wrap;
        else if (flag === 'vflip') c.vflip = true;
      }
      continue;
    }
    const prec = /^\/\/\s+precision: (.+?)\s*$/.exec(line);
    if (prec) format = BUF_FORMATS.find(([, label]) => label === prec[1])?.[0];
  }
  return { channels, ...(format ? { format } : {}) };
}

/** A .glsl file as a project. Throws when nothing in it looks like a pass. */
function parseGlslImport(text: string): SavedProject {
  text = text.replace(/\r\n/g, '\n');
  const passes: SavedProject['passes'] = {};
  let common: string | null = null;
  const sections: Array<{ name: string; body: string }> = [];
  let m: RegExpExecArray | null, last: { name: string; start: number } | null = null;
  SECTION_RE.lastIndex = 0;
  while ((m = SECTION_RE.exec(text))) {
    if (last) sections.push({ name: last.name, body: text.slice(last.start, m.index) });
    last = { name: m[1].trim(), start: m.index + m[0].length };
  }
  if (last) sections.push({ name: last.name, body: text.slice(last.start) });

  if (!sections.length) {
    const { code, common: c } = unwrapExported(text);
    if (!code.trim()) throw new Error('the file is empty');
    passes.image = { code, enabled: true, channels: [] };
    common = c;
  } else {
    for (const { name, body } of sections) {
      const pass = allPasses().find(p => p.name === name && p !== commonPass);
      if (!pass) continue;
      // The wiring header sits between the section line and the compile prefix (`#version`).
      const codeAt = body.search(/^#version /m);
      const header = codeAt >= 0 ? body.slice(0, codeAt) : '';
      const { code, common: c } = unwrapExported(codeAt >= 0 ? body.slice(codeAt) : body);
      if (c !== null) common = c;
      passes[pass.id] = { code, enabled: true, ...parseWiring(header) };
    }
    if (!Object.keys(passes).length) throw new Error('no Image or Buf A–D section found');
    if (!passes.image) passes.image = { code: DEFAULT_SHADER, enabled: true, channels: [] };
  }
  if (common !== null) passes.common = { code: common, enabled: true, channels: [] };
  return { version: 1, active: 'image', passes };
}

function importShaderFile(file: File): void {
  file.text().then(text => {
    let project: SavedProject;
    try { project = parseGlslImport(text); }
    catch (e) { alert(`Could not import ${file.name}: ${(e as Error).message}`); return; }
    if (!confirmDiscard()) return;
    openProject(project, file.name.replace(/\.[^.]+$/, '') || 'Imported', null);
    toggleShaders(false);
  }).catch(e => alert(`Could not read ${file.name}: ${(e as Error).message}`));
}

btnImport.addEventListener('click', () => importFile.click());
importFile.addEventListener('change', () => {
  const f = importFile.files?.[0];
  importFile.value = '';
  if (f) importShaderFile(f);
});

/* ── Screenshot ──────────────────────────────────────────── */
// One frame is drawn synchronously at the chosen size and read back before
// the browser composites (the context has no preserveDrawingBuffer). A size
// other than the preview's runs that frame on stretched copies of the
// buffers, so a feedback shader's picture is its current state at the new
// size and the live buffers carry on untouched.
function captureFrame(): void {
  const [w, h] = parseShotSize(settings.shotSize, canvas.width, canvas.height) ?? [canvas.width, canvas.height];
  const wasPaused = paused;
  // render() clears rafRunning on entry and, seeing paused, does not re-arm;
  // the live loop's callback is still pending though, so the flag has to be
  // put back or scheduleRender() below starts a second loop.
  const wasRaf = rafRunning;
  paused = true;              // hold time still for this frame
  sizeOverride = [w, h];
  const live = targets.slice();
  const resized = w !== canvas.width || h !== canvas.height;
  if (resized) for (let i = 0; i < NUM_BUFFERS; i++) targets[i] = live[i] ? stretchedCopy(live[i]!, w, h) : null;
  shotOnCopies = resized;
  try {
    render(performance.now());
    const name = `${projectName.replace(/[^\w.-]+/g, '_') || 'shader'}-f${iFrame}-${w}x${h}.png`;
    canvas.toBlob(blob => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = name; a.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
  } finally {
    if (resized) for (let i = 0; i < NUM_BUFFERS; i++) { if (targets[i]) disposeCopy(targets[i]!); targets[i] = live[i]; }
    shotOnCopies = false;
    sizeOverride = null;
    paused = wasPaused;
    rafRunning = wasRaf;
    lastTimestamp = null;
    scheduleRender();
  }
  const shots = document.querySelectorAll<HTMLElement>('[data-act="shot"]');
  shots.forEach(b => b.classList.add('active'));
  setTimeout(() => shots.forEach(b => b.classList.remove('active')), 300);
}

/* ── Video recording ─────────────────────────────────────── */
let mediaRecorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];
let recording = false;
let recStart = 0, recTimer: ReturnType<typeof setInterval> | null = null;

function updateRecTime(): void {
  const s = Math.floor((Date.now() - recStart) / 1000);
  setReadout('rec-time', `● ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`);
}

function getRecordingMimeType(): string {
  const types = [
    'video/mp4;codecs=avc1',
    'video/mp4',
    'video/webm;codecs=vp9',
    'video/webm',
  ];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return '';
}

function toggleRecording(): void {
  if (!recording) {
    const stream = canvas.captureStream(60);
    const mimeType = getRecordingMimeType();
    try {
      mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 8_000_000 } : {});
    } catch {
      mediaRecorder = new MediaRecorder(stream);
    }
    recordedChunks = [];
    mediaRecorder.ondataavailable = (e: BlobEvent) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      const type = mediaRecorder!.mimeType;
      const ext = type.includes('mp4') ? 'mp4' : 'webm';
      const blob = new Blob(recordedChunks, { type });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `shader_${Date.now()}.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
    };
    mediaRecorder.start();
    recording = true;
    recStart = Date.now();
    updateRecTime();
    recTimer = setInterval(updateRecTime, 500);
  } else {
    mediaRecorder?.stop();
    recording = false;
    if (recTimer) { clearInterval(recTimer); recTimer = null; }
  }
  document.body.classList.toggle('recording', recording);
  syncStatusButtons();
}

/* ── Pass tabs ───────────────────────────────────────────── */
const passTabs = document.getElementById('pass-tabs') as HTMLElement;
const btnEditorUndock = document.getElementById('btn-editor-undock') as HTMLButtonElement;

/** Tabs as displayed: Image first (Shadertoy's ordering) even though buffers execute before it. */
function shownPasses(): Pass[] {
  return [imagePass, ...(commonPass.enabled ? [commonPass] : []), ...activeBufferPasses()];
}

function updatePassTabs(): void {
  passTabs.replaceChildren();

  for (const p of shownPasses()) {
    const tab = document.createElement('div');
    tab.className = 'pass-tab'
      + (p === activePass ? ' active' : '')
      + (p.error ? ' has-error' : '');
    tab.title = p === commonPass
      ? 'Common — shared code included in every pass'
      : p.bufIndex < 0
      ? 'Image pass — drawn to the canvas, runs last'
      : `Buffer ${BUFFER_NAMES[p.bufIndex]} — offscreen target, runs before Image`;

    const label = document.createElement('span');
    label.textContent = p.name;
    tab.appendChild(label);
    tab.addEventListener('click', () => switchPass(p));
    if (p.bufIndex >= 0) {
      tab.title += ` · ${BUF_FORMATS.find(([f]) => f === p.format)?.[1]} (right-click to change)`;
      tab.addEventListener('contextmenu', (ev: MouseEvent) => { ev.preventDefault(); hideCtxMenu(); showFmtMenu(p, ev.clientX, ev.clientY); });
    }

    if (p !== imagePass) {
      const close = document.createElement('button');
      close.className = 'pass-close';
      close.textContent = '✕';
      close.title = `Remove ${p.name}`;
      close.addEventListener('click', (ev: MouseEvent) => {
        ev.stopPropagation();
        if (p === commonPass) setCommonEnabled(false);
        else removeBuffer(p.bufIndex);
      });
      tab.appendChild(close);
    }
    passTabs.appendChild(tab);
  }

  // One "+" opens a menu of the passes not yet added (buffers A–D and Common), like Shadertoy.
  if ([...bufferPasses, commonPass].some(p => !p.enabled)) {
    const add = document.createElement('button');
    add.className = 'pass-add';
    add.textContent = '+';
    add.title = 'Add a buffer or Common tab';
    add.addEventListener('click', (ev: MouseEvent) => {
      ev.stopPropagation();
      hideCtxMenu(); hideFmtMenu();
      if (addMenu.classList.contains('open')) { hideAddMenu(); return; }
      const r = add.getBoundingClientRect();
      showAddMenu(r.left, r.bottom + 2);
    });
    passTabs.appendChild(add);
  }

  refreshViewSelects();
  passTabs.appendChild(btnEditorUndock);   // always last, pushed to the right edge
}

/* Add-pass menu: every buffer and Common, the ones already open dimmed. */
const addMenu = document.createElement('div');
addMenu.id = 'add-menu';
addMenu.className = 'ctx-menu';
document.body.appendChild(addMenu);
function showAddMenu(x: number, y: number): void {
  addMenu.innerHTML = '';
  for (const p of [...bufferPasses, commonPass]) {
    const item = document.createElement('div');
    item.className = 'ctx-item' + (p.enabled ? ' ctx-disabled' : '');
    item.textContent = p === commonPass ? 'Common' : `Buffer ${BUFFER_NAMES[p.bufIndex]}`;
    item.title = p.enabled ? 'Already added' : p === commonPass ? 'Code shared by every pass' : 'Offscreen target, runs before Image';
    if (!p.enabled) item.addEventListener('click', (e: MouseEvent) => {
      e.stopPropagation();
      hideAddMenu();
      if (p === commonPass) setCommonEnabled(true); else enableBuffer(p.bufIndex);
      switchPass(p);
    });
    addMenu.appendChild(item);
  }
  addMenu.style.left = `${x}px`; addMenu.style.top = `${y}px`;
  addMenu.classList.add('open');
  const r = addMenu.getBoundingClientRect();
  if (x + r.width > window.innerWidth) addMenu.style.left = `${Math.max(4, window.innerWidth - r.width - 4)}px`;
}
function hideAddMenu(): void { addMenu.classList.remove('open'); }
document.addEventListener('click', hideAddMenu);

/** Like buffers, removing Common keeps its code for when it's re-added. */
function setCommonEnabled(on: boolean): void {
  if (commonPass.enabled === on) return;
  commonPass.enabled = on;
  commonPass.error = null;
  if (!on && activePass === commonPass) switchPass(imagePass);
  updatePassTabs();
  scheduleCompile(true);
}

function enableBuffer(i: number): void {
  const p = bufferPasses[i];
  if (p.enabled) return;
  p.enabled = true;
  updatePassTabs();
  rebuildChannelStrip();   // slots pointing here stop reading as "not created"
  scheduleCompile(true);
}

/** Removing a buffer clears it: code, undo history and wiring all go back to stock. */
function removeBuffer(i: number): void {
  const p = bufferPasses[i];
  if (!p.enabled) return;
  const stock = defaultBufferShader(BUFFER_NAMES[i]);
  const code = p === activePass ? editor.state.doc.toString() : p.code;
  if (code !== stock && !confirm(`Remove ${p.name}? Its code will be cleared.`)) return;
  p.enabled = false;
  // Leave the tab first: switchPass() would otherwise write the editor's
  // document back over the stock code.
  if (activePass === p) switchPass(imagePass);
  if (p.program) { gl.deleteProgram(p.program); p.program = null; }
  p.error = null;
  p.code = stock;
  p.state = null;
  p.channels.forEach(resetChannel);
  disposeTarget(i);
  updatePassTabs();
  rebuildChannelStrip();   // channels still wired here now sample black
  scheduleCompile(true);
}

function switchPass(p: Pass): void {
  if (p === activePass) return;
  activePass.code  = editor.state.doc.toString();
  activePass.state = editor.state;

  activePass = p;
  if (!p.state) p.state = makeEditorState(p.code);
  editor.setState(p.state);
  // Saved states carry whatever theme/vim config was live when they were made.
  editor.dispatch({
    effects: [
      themeCompartment.reconfigure(themeExtensions(THEME_BY_ID.get(settings.theme) ?? THEMES[0])),
      vimCompartment.reconfigure(vimExtension(settings.vimMode)),
    ],
  });

  updatePassTabs();
  rebuildChannelStrip();
  refreshDiagnostics();
  editor.focus();
}

/* ── Channel slots UI ────────────────────────────────────── */
const fileInput  = document.getElementById('file-input')    as HTMLInputElement;

/** A channel strip: the active pass's four slots, one strip per preview window. */
interface Strip { label: HTMLElement; slots: HTMLElement[] }
const strips: Strip[] = [];

// Which pass/slot a pending file dialog belongs to — the active pass can
// change while the picker is open.
let uploadTarget: { pass: Pass; slot: number } = { pass: imagePass, slot: 0 };

/* ── Context menu ────────────────────────────────────────── */
const ctxMenu = document.createElement('div');
ctxMenu.id = 'ctx-menu';
ctxMenu.innerHTML = `
  <div class="ctx-item" id="ctx-2d">Image file…</div>
  <div class="ctx-item" id="ctx-cube">Cubemap faces…</div>
  <div class="ctx-sep"></div>
  ${BUFFER_NAMES.map((n, i) => `<div class="ctx-item ctx-buf" data-buf="${i}">Buffer ${n}</div>`).join('\n  ')}
  <div class="ctx-sep"></div>
  <div class="ctx-opts" id="ctx-filter"><span class="ctx-opt-label">Filter</span>${FILTERS.map(f => `<span class="ctx-opt" data-filter="${f}">${f}</span>`).join('')}</div>
  <div class="ctx-opts" id="ctx-wrap"><span class="ctx-opt-label">Wrap</span>${WRAPS.map(w => `<span class="ctx-opt" data-wrap="${w}">${w}</span>`).join('')}</div>
  <div class="ctx-item" id="ctx-vflip">Flip vertically</div>
  <div class="ctx-sep"></div>
  <div class="ctx-item ctx-danger" id="ctx-clear">Clear</div>
`;
document.body.appendChild(ctxMenu);

let ctxSlot = 0;

/** Opens the menu with its top-left at (x, y), or its bottom-left when `above`. */
function showCtxMenu(x: number, y: number, slot: number, above = false): void {
  ctxSlot = slot;
  const ch = activePass.channels[slot];
  ctxMenu.style.left = x + 'px';
  ctxMenu.style.top  = y + 'px';
  ctxMenu.classList.add('open');
  if (above) {
    ctxMenu.style.top = Math.max(4, y - ctxMenu.getBoundingClientRect().height) + 'px';
  }

  ctxMenu.querySelectorAll<HTMLElement>('[data-filter]').forEach(el => el.classList.toggle('ctx-active', el.dataset.filter === ch.filter));
  ctxMenu.querySelectorAll<HTMLElement>('[data-wrap]').forEach(el => el.classList.toggle('ctx-active', el.dataset.wrap === ch.wrap));
  const vflip = document.getElementById('ctx-vflip') as HTMLElement;
  vflip.classList.toggle('ctx-active', ch.vflip);
  vflip.style.display = ch.kind === '2d' ? '' : 'none';
  (document.getElementById('ctx-filter') as HTMLElement).style.display = ch.kind === 'cube' ? 'none' : '';
  (document.getElementById('ctx-wrap') as HTMLElement).style.display = ch.kind === 'cube' ? 'none' : '';
  (document.getElementById('ctx-2d') as HTMLElement).classList.toggle('ctx-active', ch.kind === '2d');
  (document.getElementById('ctx-cube') as HTMLElement).classList.toggle('ctx-active', ch.kind === 'cube');
  ctxMenu.querySelectorAll<HTMLElement>('.ctx-buf').forEach(el => {
    const b = Number(el.dataset.buf);
    el.classList.toggle('ctx-active', ch.kind === 'buffer' && ch.buf === b);
    // Picking a buffer that doesn't exist yet creates it.
    el.textContent = `Buffer ${BUFFER_NAMES[b]}${bufferPasses[b].enabled ? '' : ' (create)'}`;
  });

  // The menu is tall enough to run off-screen from a slot near the bottom.
  const r = ctxMenu.getBoundingClientRect();
  if (y + r.height > window.innerHeight) ctxMenu.style.top  = Math.max(4, y - r.height) + 'px';
  if (x + r.width  > window.innerWidth)  ctxMenu.style.left = Math.max(4, x - r.width) + 'px';
}

function hideCtxMenu(): void { ctxMenu.classList.remove('open'); }

document.addEventListener('click', hideCtxMenu);
document.addEventListener('contextmenu', (e: MouseEvent) => {
  if (!(e.target as HTMLElement).closest('.tex-slot')) hideCtxMenu();
  if (!(e.target as HTMLElement).closest('.pass-tab')) hideFmtMenu();
});

// Sampler type is baked into the shader prefix, so a change recompiles.
function setChannelKind(slot: number, kind: '2d' | 'cube'): void {
  const ch = activePass.channels[slot];
  if (ch.kind === kind) return;
  if (ch.kind === 'buffer') { ch.filter = 'linear'; ch.wrap = 'repeat'; }   // buffer defaults (clamp) do not carry over
  ch.kind = kind;
  refreshSlot(slot);
  scheduleCompile(true);
}

function setChannelBuffer(slot: number, buf: number): void {
  const ch = activePass.channels[slot];
  wireToBuffer(ch, buf);
  if (!bufferPasses[buf].enabled) enableBuffer(buf);
  refreshSlot(slot);
  scheduleCompile(true);
}

/** Back to an empty 2D slot, releasing any GPU textures it held. */
function resetChannel(ch: Channel): void {
  ch.generation++;   // an image still loading for this slot must not land in it
  if (ch.tex)     { gl.deleteTexture(ch.tex);     ch.tex = null; }
  if (ch.cubeTex) { gl.deleteTexture(ch.cubeTex); ch.cubeTex = null; }
  ch.img = null;
  ch.filter = 'linear'; ch.wrap = 'repeat'; ch.vflip = false;
  for (const url of ch.cubeFaceUrls) if (url) URL.revokeObjectURL(url);
  if (ch.imgUrl) URL.revokeObjectURL(ch.imgUrl);   // the cube thumbnail is face +X's URL; revoking twice is harmless
  ch.cubeFaceUrls.fill(null);
  ch.cubeFaceLoaded.fill(false);
  ch.imgUrl = null;
  ch.texW = 0;
  ch.texH = 0;
  ch.kind = '2d';
  ch.buf = 0;
}

function clearSlot(slot: number): void {
  resetChannel(activePass.channels[slot]);
  refreshSlot(slot);
  scheduleCompile(true);
}

document.getElementById('ctx-2d')!.addEventListener('click', (e: MouseEvent) => {
  e.stopPropagation();
  setChannelKind(ctxSlot, '2d');
  uploadTarget = { pass: activePass, slot: ctxSlot };
  hideCtxMenu();
  fileInput.click();
});
document.getElementById('ctx-cube')!.addEventListener('click', (e: MouseEvent) => {
  e.stopPropagation();
  setChannelKind(ctxSlot, 'cube');
  openCubeModal(ctxSlot);
  hideCtxMenu();
});
ctxMenu.querySelectorAll<HTMLElement>('.ctx-buf').forEach(el => {
  el.addEventListener('click', (e: MouseEvent) => {
    e.stopPropagation();
    setChannelBuffer(ctxSlot, Number(el.dataset.buf));
    hideCtxMenu();
  });
});
document.getElementById('ctx-clear')!.addEventListener('click', (e: MouseEvent) => {
  e.stopPropagation();
  clearSlot(ctxSlot);
  hideCtxMenu();
});
// Sampling options stay open after a click, so several can be set at once.
ctxMenu.querySelectorAll<HTMLElement>('.ctx-opt').forEach(el => {
  el.addEventListener('click', (e: MouseEvent) => {
    e.stopPropagation();
    const ch = activePass.channels[ctxSlot];
    if (el.dataset.filter) ch.filter = el.dataset.filter as Filter;
    if (el.dataset.wrap) ch.wrap = el.dataset.wrap as Wrap;
    showCtxMenu(parseFloat(ctxMenu.style.left), parseFloat(ctxMenu.style.top), ctxSlot);
    refreshSlot(ctxSlot);
    scheduleSave();
    scheduleRender();
  });
});
document.getElementById('ctx-vflip')!.addEventListener('click', (e: MouseEvent) => {
  e.stopPropagation();
  const ch = activePass.channels[ctxSlot];
  ch.vflip = !ch.vflip;
  if (ch.img) uploadTexture(ch, ch.img);
  showCtxMenu(parseFloat(ctxMenu.style.left), parseFloat(ctxMenu.style.top), ctxSlot);
  refreshSlot(ctxSlot);
  scheduleSave();
  scheduleRender();
});

/* Buffer precision: right-click a buffer's tab. */
const fmtMenu = document.createElement('div');
fmtMenu.id = 'fmt-menu';
fmtMenu.className = 'ctx-menu';
fmtMenu.innerHTML = `<div class="ctx-opt-label fmt-title"></div>` + BUF_FORMATS.map(([f, label]) => `<div class="ctx-item fmt-item" data-format="${f}">${label}</div>`).join('');
document.body.appendChild(fmtMenu);
let fmtPass: Pass | null = null;
function showFmtMenu(p: Pass, x: number, y: number): void {
  fmtPass = p;
  (fmtMenu.querySelector('.fmt-title') as HTMLElement).textContent = `${p.name} precision${floatRT ? '' : ' (float unsupported here)'}`;
  fmtMenu.querySelectorAll<HTMLElement>('.fmt-item').forEach(el => el.classList.toggle('ctx-active', el.dataset.format === p.format));
  fmtMenu.style.left = `${x}px`; fmtMenu.style.top = `${y}px`;
  fmtMenu.classList.add('open');
  const r = fmtMenu.getBoundingClientRect();
  if (x + r.width > window.innerWidth) fmtMenu.style.left = `${Math.max(4, x - r.width)}px`;
}
function hideFmtMenu(): void { fmtMenu.classList.remove('open'); }
fmtMenu.querySelectorAll<HTMLElement>('.fmt-item').forEach(el => {
  el.addEventListener('click', (e: MouseEvent) => {
    e.stopPropagation();
    if (fmtPass) { fmtPass.format = el.dataset.format as BufFormat; scheduleSave(); scheduleRender(); }   // ensureTarget recreates it
    hideFmtMenu();
  });
});
document.addEventListener('click', hideFmtMenu);

/* ── Cube modal ──────────────────────────────────────────── */
const cubeModal = document.createElement('div');
cubeModal.id = 'cube-modal';
cubeModal.innerHTML = `
  <div class="cube-modal-inner">
    <div class="cube-modal-header">
      <span id="cube-modal-title">Cubemap — ch0</span>
      <button id="cube-modal-close">✕</button>
    </div>
    <p class="cube-modal-hint">Sample in shader: <code id="cube-modal-sample"></code></p>
    <div id="cube-face-grid"></div>
  </div>
`;
document.body.appendChild(cubeModal);

const cubeFaceGrid  = document.getElementById('cube-face-grid')!;
const cubeTitleEl   = document.getElementById('cube-modal-title')!;
const cubeFaceInput = document.createElement('input');
cubeFaceInput.type   = 'file';
cubeFaceInput.accept = 'image/*';
cubeFaceInput.style.display = 'none';
document.body.appendChild(cubeFaceInput);

let cubeModalSlot     = 0;
let cubeModalPass     = imagePass;
let activeCubeFaceIdx = 0;

function openCubeModal(slot: number): void {
  cubeModalSlot = slot;
  cubeModalPass = activePass;
  const ch = activePass.channels[slot];
  cubeTitleEl.textContent = `Cubemap — ${activePass.name} ch${slot}`;
  document.getElementById('cube-modal-sample')!.textContent = `texture(iChannel${slot}, normalize(dir))`;
  cubeFaceGrid.innerHTML = '';
  CUBE_FACES.forEach(({ label }, idx) => {
    const faceDiv = document.createElement('div');
    faceDiv.className = 'cube-face-slot';
    faceDiv.id = `cube-face-${idx}`;
    const img = document.createElement('img');
    img.alt = label;
    const url = ch.cubeFaceUrls[idx];
    if (url) { img.src = url; img.classList.add('loaded'); }
    const plusSpan = document.createElement('span');
    plusSpan.className = 'plus';
    plusSpan.textContent = '+';
    if (url) plusSpan.style.display = 'none';
    const labelSpan = document.createElement('span');
    labelSpan.className = 'face-label';
    labelSpan.textContent = label;
    faceDiv.append(img, plusSpan, labelSpan);
    faceDiv.addEventListener('click', () => {
      activeCubeFaceIdx = idx;
      cubeFaceInput.click();
    });
    cubeFaceGrid.appendChild(faceDiv);
  });
  cubeModal.classList.add('open');
}

document.getElementById('cube-modal-close')!.addEventListener('click', () => {
  cubeModal.classList.remove('open');
});
cubeModal.addEventListener('click', (e: MouseEvent) => {
  if (e.target === cubeModal) cubeModal.classList.remove('open');
});

cubeFaceInput.addEventListener('change', (e: Event) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const ch = cubeModalPass.channels[cubeModalSlot];
  const faceIdx = activeCubeFaceIdx;
  const img = new Image();
  const gen = ch.generation;
  img.onload = () => {
    if (ch.generation !== gen || ch.kind !== 'cube') { URL.revokeObjectURL(url); return; }
    uploadCubeFace(ch, faceIdx, img);
    const old = ch.cubeFaceUrls[faceIdx];
    if (old) URL.revokeObjectURL(old);
    ch.cubeFaceUrls[faceIdx] = url;
    const faceEl = document.getElementById(`cube-face-${faceIdx}`);
    if (faceEl && cubeModal.classList.contains('open')) {
      const imgEl = faceEl.querySelector('img') as HTMLImageElement;
      imgEl.src = url;
      imgEl.classList.add('loaded');
      (faceEl.querySelector('.plus') as HTMLElement).style.display = 'none';
    }
    // Show +X face as the slot thumbnail
    if (faceIdx === 0) {
      ch.imgUrl = url;
      if (cubeModalPass === activePass) refreshSlot(cubeModalSlot);
    }
    scheduleRender();
  };
  img.src = url;
  (e.target as HTMLInputElement).value = '';
});

/* ── Slot rendering ──────────────────────────────────────── */
function refreshSlot(i: number): void {
  for (const s of strips) refreshSlotEl(s.slots[i], i);
}

function refreshSlotEl(el: HTMLElement, i: number): void {
  const ch     = activePass.channels[i];
  const imgEl  = el.querySelector('img')    as HTMLImageElement;
  const plusEl = el.querySelector('.plus')  as HTMLElement;
  const labelEl= el.querySelector('.label') as HTMLElement;

  el.classList.toggle('cube-mode', ch.kind === 'cube');
  el.classList.toggle('buf-mode',  ch.kind === 'buffer');

  if (ch.kind === 'buffer') {
    const name    = BUFFER_NAMES[ch.buf];
    const missing = !bufferPasses[ch.buf].enabled;
    imgEl.removeAttribute('src');
    imgEl.classList.remove('loaded');
    plusEl.style.display = '';
    plusEl.textContent = name;
    labelEl.textContent = missing ? `${name}?` : `buf${name}`;
    el.title = missing
      ? `iChannel${i} → Buffer ${name} (not created — samples black)`
      : `iChannel${i} → Buffer ${name}${samplingNote(ch)} — right-click for sampling`;
    return;
  }

  plusEl.textContent = '+';
  labelEl.textContent = ch.kind === 'cube' ? `cb${i}` : `ch${i}`;
  if (ch.imgUrl) {
    imgEl.src = ch.imgUrl;
    imgEl.classList.add('loaded');
    plusEl.style.display = 'none';
  } else {
    imgEl.removeAttribute('src');
    imgEl.classList.remove('loaded');
    plusEl.style.display = '';
  }
  el.title = ch.kind === 'cube'
    ? `iChannel${i} (samplerCube) — click to upload faces`
    : `iChannel${i} (sampler2D)${samplingNote(ch)} — click to upload, right-click for sampling`;
}

function rebuildChannelStrip(): void {
  const common = activePass === commonPass;
  for (const s of strips) {
    s.label.textContent = common ? 'Common: shared by every pass, no channels' : `${activePass.name}:`;
    s.slots.forEach(el => { el.style.display = common ? 'none' : ''; });
  }
  if (common) { hideCtxMenu(); return; }
  for (let i = 0; i < NUM_SLOTS; i++) refreshSlot(i);
}

/** Build a strip into `container` and keep it in step with the others. */
function makeStrip(container: HTMLElement): Strip {
  const label = document.createElement('span');
  label.className = 'strip-label';
  container.appendChild(label);
  const slots = Array.from({ length: NUM_SLOTS }, (_, i) => makeSlot(container, i));
  const s: Strip = { label, slots };
  strips.push(s);
  rebuildChannelStrip();
  return s;
}

// A click (left or right, or a tap) opens the slot's menu above it; the file
// picker only appears once "Image file…" is chosen from that menu.
function makeSlot(strip: HTMLElement, i: number): HTMLElement {
  const div = document.createElement('div');
  div.className = 'tex-slot' + (i === 0 ? ' active' : '');
  div.innerHTML = `<span class="plus">+</span><img alt=""><span class="label">ch${i}</span>`;

  const openMenu = () => {
    for (const s of strips) s.slots.forEach((el, j) => el.classList.toggle('active', j === i));
    const r = div.getBoundingClientRect();
    showCtxMenu(r.left, r.top - 4, i, true);
  };

  div.addEventListener('click', (e: MouseEvent) => {
    e.stopPropagation();
    if (ctxMenu.classList.contains('open') && ctxSlot === i) { hideCtxMenu(); return; }
    openMenu();
  });
  div.addEventListener('contextmenu', (e: MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    openMenu();
  });
  // Dropping an image file loads it straight into the slot.
  div.addEventListener('dragover', (e: DragEvent) => {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    div.classList.add('over');
  });
  div.addEventListener('dragleave', () => div.classList.remove('over'));
  div.addEventListener('drop', (e: DragEvent) => {
    div.classList.remove('over');
    e.preventDefault();
    const file = Array.from(e.dataTransfer?.files ?? []).find(f => f.type.startsWith('image/'));
    if (!file || activePass === commonPass) return;
    setChannelKind(i, '2d');
    hideCtxMenu();
    loadImageFile(activePass, i, file);
  });

  strip.appendChild(div);
  return div;
}

fileInput.addEventListener('change', (e: Event) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) loadImageFile(uploadTarget.pass, uploadTarget.slot, file);
  (e.target as HTMLInputElement).value = '';
});

/** Decode an image file into a 2D slot (from the picker or a drop). */
function loadImageFile(pass: Pass, slot: number, file: File): void {
  const ch = pass.channels[slot];
  const url = URL.createObjectURL(file);
  const img = new Image();
  const gen = ch.generation;
  img.onload = () => {
    // The slot was cleared (project switched, buffer removed) while the file loaded.
    if (ch.generation !== gen || ch.kind !== '2d') { URL.revokeObjectURL(url); return; }
    uploadTexture(ch, img);
    if (ch.imgUrl) URL.revokeObjectURL(ch.imgUrl);
    ch.imgUrl = url;
    if (pass === activePass) refreshSlot(slot);
    scheduleRender();
  };
  img.src = url;
}

/* ── Pane resize ─────────────────────────────────────────── */
const divider = document.getElementById('divider')     as HTMLElement;
const editorP = document.getElementById('editor-pane') as HTMLElement;
const main    = document.getElementById('main')        as HTMLElement;

// Must match the stacked-layout media query in index.html.
const stackedLayout = window.matchMedia('(max-width: 768px) and (orientation: portrait)');

let dragging = false, dragStart = 0, dragStartSize = 0;

divider.addEventListener('pointerdown', (e: PointerEvent) => {
  if (main.classList.contains('dock-empty') || settings.editorFloating) return;   // divider is hidden; belt and braces
  dragging = true;
  dragStart = stackedLayout.matches ? e.clientY : e.clientX;
  dragStartSize = stackedLayout.matches ? editorP.offsetHeight : editorP.offsetWidth;
  divider.classList.add('dragging');
  document.body.style.cursor = stackedLayout.matches ? 'row-resize' : 'col-resize';
  divider.setPointerCapture(e.pointerId);
  e.preventDefault();
});

divider.addEventListener('pointermove', (e: PointerEvent) => {
  if (!dragging) return;
  const divW = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--divider-w'));
  // When swapped, the editor sits after the divider, so dragging grows it the other way.
  const dir = settings.swapped ? -1 : 1;
  if (stackedLayout.matches) {
    const dy = (e.clientY - dragStart) * dir;
    const totalH = main.offsetHeight;
    const newH = Math.min(Math.max(dragStartSize + dy, 100), totalH - divW - 100);
    editorP.style.height = newH + 'px';
  } else {
    const dx = (e.clientX - dragStart) * dir;
    const totalW = main.offsetWidth;
    const newW = Math.min(Math.max(dragStartSize + dx, 200), totalW - divW - 100);
    editorP.style.width = newW + 'px';
  }
});

function endDividerDrag(): void {
  if (dragging) {
    dragging = false;
    divider.classList.remove('dragging');
    document.body.style.cursor = '';
  }
}
divider.addEventListener('pointerup', endDividerDrag);
divider.addEventListener('pointercancel', endDividerDrag);

// Clear stale inline size when crossing the layout breakpoint
stackedLayout.addEventListener('change', () => {
  editorP.style.width = '';
  editorP.style.height = '';
  if (settings.editorFloating) positionEditorFloat();   // the float rect, not the split size
});

/* ── Floating windows ────────────────────────────────────── */
// Shared move/resize behaviour for the undocked preview and the extra pass
// views: dragged by a bar, resized from any edge, kept inside the viewport.
const FLOAT_BAR_H = 28;     // must match the bar height in index.html
const FLOAT_MARGIN = 8;     // breathing room from viewport edges

interface FloatHost {
  el: HTMLElement;
  bar: HTMLElement;
  minW: number;
  minH: number;
  rect: () => FloatRect;
  setRect: (r: FloatRect) => void;   // store + write styles
  onRelease?: () => void;
}

function toolbarBottom(): number {
  return (document.getElementById('toolbar') as HTMLElement).getBoundingClientRect().bottom;
}

// Fit the viewport, and keep the bar grabbable.
function clampRect(r: FloatRect, minW: number, minH: number): FloatRect {
  const vw = window.innerWidth, vh = window.innerHeight, top = toolbarBottom();
  const w = Math.min(Math.max(r.w, minW), vw - 2 * FLOAT_MARGIN);
  const h = Math.min(Math.max(r.h, minH), vh - top - 2 * FLOAT_MARGIN);
  const x = Math.min(Math.max(r.x, FLOAT_MARGIN), vw - w - FLOAT_MARGIN);
  const y = Math.min(Math.max(r.y, top + FLOAT_MARGIN), vh - h - FLOAT_MARGIN);
  return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
}

function writeRect(el: HTMLElement, r: FloatRect): void {
  el.style.left   = `${r.x}px`;
  el.style.top    = `${r.y}px`;
  el.style.width  = `${r.w}px`;
  el.style.height = `${r.h}px`;
}

const RESIZE_DIRS = ['n', 's', 'w', 'e', 'nw', 'ne', 'sw', 'se'];

function addResizeHandles(el: HTMLElement): HTMLElement[] {
  return RESIZE_DIRS.map(d => {
    const h = document.createElement('div');
    h.className = `fh fh-${d}`;
    h.dataset.dir = d;
    el.appendChild(h);
    return h;
  });
}

type Gesture = { host: FloatHost; dir: string; startX: number; startY: number; start: FloatRect; pointerId: number };
let gesture: Gesture | null = null;

function beginGesture(e: PointerEvent, host: FloatHost, dir: string, target: HTMLElement): void {
  if (e.button !== 0 || document.fullscreenElement) return;
  gesture = { host, dir, startX: e.clientX, startY: e.clientY, start: copyRect(host.rect()), pointerId: e.pointerId };
  target.setPointerCapture(e.pointerId);
  host.el.classList.add('moving');
  e.preventDefault();
}

const SNAP = 8;   // px: edges this close to another edge stick to it (Alt disables)
const snapGuideX = document.createElement('div'), snapGuideY = document.createElement('div');
snapGuideX.className = 'snap-guide snap-x'; snapGuideY.className = 'snap-guide snap-y';
document.body.append(snapGuideX, snapGuideY);

/** Edges worth sticking to: the viewport, the toolbar, the panes and every other window. */
function snapLines(except: HTMLElement): { xs: number[]; ys: number[] } {
  const vw = window.innerWidth, vh = window.innerHeight, top = toolbarBottom();
  const xs = [FLOAT_MARGIN, vw - FLOAT_MARGIN], ys = [top + FLOAT_MARGIN, vh - FLOAT_MARGIN];
  for (const el of document.querySelectorAll<HTMLElement>('.win.floating, #dock, #editor-pane')) {
    if (el === except || !el.isConnected || el.clientWidth === 0) continue;
    const r = el.getBoundingClientRect();
    xs.push(r.left, r.right); ys.push(r.top, r.bottom);
  }
  return { xs, ys };
}
const nearest = (v: number, lines: number[]): number | null => {
  let best: number | null = null, d = SNAP + 1;
  for (const l of lines) { const dd = Math.abs(l - v); if (dd < d) { d = dd; best = l; } }
  return best;
};
function showGuides(x: number | null, y: number | null): void {
  snapGuideX.style.display = x === null ? 'none' : 'block';
  snapGuideY.style.display = y === null ? 'none' : 'block';
  if (x !== null) snapGuideX.style.left = `${x}px`;
  if (y !== null) snapGuideY.style.top = `${y}px`;
}

function moveGesture(e: PointerEvent): void {
  const g = gesture;
  if (!g || e.pointerId !== g.pointerId) return;
  const dx = e.clientX - g.startX, dy = e.clientY - g.startY;
  const vw = window.innerWidth, vh = window.innerHeight, top = toolbarBottom();
  const { minW, minH } = g.host;
  let { x, y, w, h } = g.start;
  const lines = e.altKey ? null : snapLines(g.host.el);
  let guideX: number | null = null, guideY: number | null = null;

  if (g.dir === 'move') {
    // Slide freely while held (re-clamped on release) but never lose the bar.
    x = Math.min(Math.max(g.start.x + dx, -w + 80), vw - 80);
    y = Math.min(Math.max(g.start.y + dy, top), vh - FLOAT_BAR_H);
    if (lines) {
      const l = nearest(x, lines.xs), r = nearest(x + w, lines.xs);
      if (l !== null && (r === null || Math.abs(l - x) <= Math.abs(r - x - w))) { x = l; guideX = l; }
      else if (r !== null) { x = r - w; guideX = r; }
      const t = nearest(y, lines.ys), b = nearest(y + h, lines.ys);
      if (t !== null && (b === null || Math.abs(t - y) <= Math.abs(b - y - h))) { y = t; guideY = t; }
      else if (b !== null) { y = b - h; guideY = b; }
    }
  } else {
    // The edge opposite the one being dragged stays put.
    if (g.dir.includes('e')) w = Math.min(Math.max(g.start.w + dx, minW), vw - FLOAT_MARGIN - x);
    if (g.dir.includes('s')) h = Math.min(Math.max(g.start.h + dy, minH), vh - FLOAT_MARGIN - y);
    if (g.dir.includes('w')) {
      const right = g.start.x + g.start.w;
      x = Math.min(Math.max(g.start.x + dx, FLOAT_MARGIN), right - minW);
      w = right - x;
    }
    if (g.dir.includes('n')) {
      const bottom = g.start.y + g.start.h;
      y = Math.min(Math.max(g.start.y + dy, top + FLOAT_MARGIN), bottom - minH);
      h = bottom - y;
    }
    if (lines) {
      // Only the edge being dragged snaps; the opposite one stays put.
      if (g.dir.includes('e')) { const r = nearest(x + w, lines.xs); if (r !== null && r - x >= minW) { w = r - x; guideX = r; } }
      if (g.dir.includes('w')) { const l = nearest(x, lines.xs); if (l !== null && x + w - l >= minW) { w = x + w - l; x = l; guideX = l; } }
      if (g.dir.includes('s')) { const b = nearest(y + h, lines.ys); if (b !== null && b - y >= minH) { h = b - y; guideY = b; } }
      if (g.dir.includes('n')) { const t = nearest(y, lines.ys); if (t !== null && y + h - t >= minH) { h = y + h - t; y = t; guideY = t; } }
    }
  }
  showGuides(guideX, guideY);
  g.host.setRect({ x, y, w, h });
}

function endGesture(e: PointerEvent): void {
  const g = gesture;
  if (!g || e.pointerId !== g.pointerId) return;
  gesture = null;
  showGuides(null, null);
  g.host.el.classList.remove('moving');
  g.host.setRect(clampRect(g.host.rect(), g.host.minW, g.host.minH));
  g.host.onRelease?.();
  saveSettings();
}

function attachFloatGestures(host: FloatHost, handles: HTMLElement[]): void {
  host.bar.addEventListener('pointerdown', (e: PointerEvent) => {
    if ((e.target as HTMLElement).closest('button, select')) return;
    beginGesture(e, host, 'move', host.bar);
  });
  host.bar.addEventListener('pointermove', moveGesture);
  host.bar.addEventListener('pointerup', endGesture);
  host.bar.addEventListener('pointercancel', endGesture);
  for (const h of handles) {
    h.addEventListener('pointerdown', e => beginGesture(e, host, h.dataset.dir ?? 'se', h));
    h.addEventListener('pointermove', moveGesture);
    h.addEventListener('pointerup', endGesture);
    h.addEventListener('pointercancel', endGesture);
  }
}

// Raise a window above its siblings when touched. z-indices are renumbered
// from a fixed base so they never climb past the popover panels.
const FLOAT_Z = 50;
function raise(el: HTMLElement): void {
  const wins = Array.from(document.querySelectorAll<HTMLElement>('.win.floating, #editor-pane.floating'))
    .filter(w => w !== el)
    .sort((a, b) => (+a.style.zIndex || 0) - (+b.style.zIndex || 0));
  wins.push(el);
  wins.forEach((w, i) => { w.style.zIndex = String(FLOAT_Z + i); });
}

/* ── Undocked editor ─────────────────────────────────────── */
// The same treatment for the editor pane; with both undocked the split is empty.
const editorBar  = document.getElementById('editor-bar')  as HTMLElement;
const editorDock = document.getElementById('editor-dock') as HTMLButtonElement;

const EDITOR_MIN_W = 280, EDITOR_MIN_H = 160;
let dockedEditorWidth = '';   // the divider's split width, restored on dock

// First undock: sit over the left half, where the docked editor was.
function defaultEditorRect(): FloatRect {
  const vw = window.innerWidth, vh = window.innerHeight, top = toolbarBottom();
  const w = Math.round(Math.min(Math.max(vw * 0.45, EDITOR_MIN_W), 820));
  const h = Math.round(Math.min(Math.max((vh - top) * 0.8, EDITOR_MIN_H), 800));
  return { x: 16, y: top + 16, w, h };
}

const editorHost: FloatHost = {
  el: editorP, bar: editorBar, minW: EDITOR_MIN_W, minH: EDITOR_MIN_H,
  rect: () => settings.editorRect ?? defaultEditorRect(),
  setRect: r => { settings.editorRect = r; writeRect(editorP, r); },
};

function positionEditorFloat(): void {
  editorHost.setRect(clampRect(editorHost.rect(), EDITOR_MIN_W, EDITOR_MIN_H));
}

function applyEditorFloating(on: boolean, animate = false): void {
  if (on && !settings.editorFloating) dockedEditorWidth = editorP.style.width;
  settings.editorFloating = on;
  editorP.classList.toggle('floating', on);
  editorP.classList.toggle('pop', on && animate);
  main.classList.toggle('editor-floating', on);
  btnEditorUndock.classList.toggle('active', on);
  btnEditorUndock.title = on ? 'Dock editor back into the split' : 'Undock editor into a floating window';
  if (on) {
    positionEditorFloat();
    raise(editorP);
  } else {
    editorP.style.left = editorP.style.top = editorP.style.height = editorP.style.zIndex = '';
    editorP.style.width = dockedEditorWidth;
  }
  scheduleRender();   // the preview changed size; repaint even while paused
}

btnEditorUndock.addEventListener('click', () => { applyEditorFloating(!settings.editorFloating, true); saveSettings(); });
editorDock.addEventListener('click', () => { applyEditorFloating(false); saveSettings(); });
editorP.addEventListener('animationend', () => editorP.classList.remove('pop'));
editorP.addEventListener('pointerdown', () => { if (settings.editorFloating) raise(editorP); }, true);
editorBar.addEventListener('dblclick', (e: MouseEvent) => {
  if ((e.target as HTMLElement).closest('button')) return;
  applyEditorFloating(false); saveSettings();
});
attachFloatGestures(editorHost, Array.from(editorP.querySelectorAll<HTMLElement>('.fh')));

/* ── Windows ─────────────────────────────────────────────── */
// Every view, the preview included, is one kind of window: a bar with a
// pass/tool selector, a 2D canvas the GL frame is copied into (or a tool
// pad), and for passes a status bar with playback. A window is docked into
// the split beside the editor (stacked with any others there) or floating.
const VIEW_MIN_W = 160, VIEW_MIN_H = 120;
const dock       = document.getElementById('dock')        as HTMLElement;
const uniBody    = document.getElementById('uni-body')    as HTMLElement;
const uniControls = document.getElementById('uni-controls') as HTMLElement;
let controlSig = '';

interface PassView {
  spec: ViewSpec;
  el: HTMLElement;
  select: HTMLSelectElement;
  cv: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  status: HTMLElement | null;     // playback bar; null for tool windows
  pxInfo: HTMLElement; pxSwatch: HTMLElement; pxText: HTMLElement;
  get pass(): string;
  visible(): boolean;
  blit(): void;
  setPass(id: string): void;
  setDocked(on: boolean): void;
  remove(): void;
}
const views: PassView[] = [];

// Phones in portrait have no room for floating windows: everything stacks in the dock.
const phoneMQ = matchMedia('(max-width: 768px) and (orientation: portrait)');
const isDocked = (v: PassView): boolean => v.spec.docked === true || phoneMQ.matches;

function viewablePasses(): Pass[] { return [imagePass, ...activeBufferPasses()]; }

/** Windows in the dock, top to bottom. */
function dockedViews(): PassView[] {
  return Array.from(dock.querySelectorAll<HTMLElement>(':scope > .win'))
    .map(el => views.find(v => v.el === el))
    .filter((v): v is PassView => !!v);
}

/** The pass window whose size sets the render resolution: a preview before a view, docked before floating, oldest first. */
function findPrimaryView(): PassView | null {
  const ordered = [...dockedViews(), ...views].filter(v => !isToolView(v.pass));
  return ordered.find(v => v.spec.kind === 'preview') ?? ordered[0] ?? null;
}
// Cached: resizeCanvas asks every frame, and the answer only changes when a
// window is made, removed, re-docked or switched — all of which syncDock.
let primary: PassView | null = null;
function primaryView(): PassView | null { return primary; }

function syncDock(): void {
  main.classList.toggle('dock-empty', !dock.querySelector(':scope > .win'));
  primary = findPrimaryView();
  for (const v of views) v.el.classList.toggle('primary', v === primary);
  scheduleRender();
}

function fillPassSelect(sel: HTMLSelectElement, current: string, withTools: boolean): void {
  sel.innerHTML = '';
  const addOpt = (parent: HTMLElement, value: string, label: string) => {
    const o = document.createElement('option');
    o.value = value; o.textContent = label;
    parent.appendChild(o);
  };
  if (withTools) {
    const g = document.createElement('optgroup'); g.label = 'Passes';
    for (const p of viewablePasses()) addOpt(g, p.id, p.name);
    sel.appendChild(g);
    const t = document.createElement('optgroup'); t.label = 'Tools';
    for (const [id, name] of TOOL_VIEWS) addOpt(t, id, name);
    sel.appendChild(t);
  } else {
    for (const p of viewablePasses()) addOpt(sel, p.id, p.name);
  }
  const ok = viewablePasses().some(p => p.id === current) || (withTools && isToolView(current));
  sel.value = ok ? current : 'image';
}

// Called whenever the set of live passes changes.
function refreshViewSelects(): void {
  for (const v of views) {
    fillPassSelect(v.select, v.spec.pass, v.spec.kind !== 'preview');
    v.setPass(v.select.value);
  }
}

const ICON_UNDOCK = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M1 4.5V11h6.5M4 1h7v7H4z"/></svg>';
const ICON_DOCK   = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M1 1h10v10H1zM6 1v10"/></svg>';
const STATUS_HTML = `
  <span class="fps">0 fps</span>
  <span class="res">0 × 0</span>
  <span class="mode">WebGL2</span>
  <span class="rec-time">● 0:00</span>
  <span class="spacer"></span>
  <button class="ibtn" data-act="pause" title="Pause (space)">
    <svg class="ic-pause" width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><rect x="2" y="1.5" width="3" height="9" rx="0.5"/><rect x="7" y="1.5" width="3" height="9" rx="0.5"/></svg>
    <svg class="ic-play" width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><polygon points="2.5,1.5 10.5,6 2.5,10.5"/></svg>
  </button>
  <button class="ibtn btn-step" data-act="step" title="Step one frame (Shift: 10, Ctrl: 100, or press .)">
    <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><polygon points="2,1.5 8,6 2,10.5"/><rect x="8.5" y="1.5" width="2" height="9" rx="0.5"/></svg>
  </button>
  <button class="ibtn" data-act="reset" title="Reset time and buffers">
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M2.2 6a3.8 3.8 0 1 0 1.1-2.7"/><path d="M2 1.5v2.5h2.5" stroke-linecap="round"/></svg>
  </button>
  <button class="ibtn" data-act="inspect" title="Inspect pixels: hover this window, click to freeze (i)">
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="6" cy="6" r="3"/><path d="M6 1v2M6 9v2M1 6h2M9 6h2"/></svg>
  </button>
  <button class="ibtn" data-act="shot" title="Save a PNG of this frame (size in Options)">
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M1.5 4h2l1-1.5h3l1 1.5h2v6h-9z"/><circle cx="6" cy="6.8" r="1.8"/></svg>
  </button>
  <button class="ibtn" data-act="record" title="Record video">
    <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><circle cx="6" cy="6" r="3.5"/></svg>
  </button>
  <button class="ibtn" data-act="fullscreen" title="Fullscreen">
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M1 4V1h3M8 1h3v3M11 8v3H8M4 11H1V8"/></svg>
  </button>`;

// Where a window lands the first time it floats: over the right half, like the old undocked preview.
function defaultWinRect(): FloatRect {
  const vw = window.innerWidth, vh = window.innerHeight, top = toolbarBottom();
  const w = Math.round(Math.min(Math.max(vw * 0.45, 240), 720));
  const h = Math.round(Math.min(Math.max((vh - top) * 0.6, 180), 540));
  return { x: vw - w - 16, y: top + 16, w, h };
}

function makeView(spec: ViewSpec, animate: boolean): PassView {
  const preview = spec.kind === 'preview';
  if (!preview) spec.kind = 'view';
  const el = document.createElement('div');
  el.className = `win ${preview ? 'preview' : 'view'}` + (animate ? ' pop' : '');
  el.innerHTML = `
    <div class="win-bar">
      <span class="grip-dots"></span>
      <span class="win-kind">${preview ? 'Preview' : 'View'}</span>
      <select class="pass-select" title="${preview ? 'Pass shown in this preview' : 'Pass or tool shown in this view'}"></select>
      <span class="spacer"></span>
      <button class="float-btn win-dock" title="Dock"></button>
      <button class="float-btn win-close" title="Close window (Esc when floating)">✕</button>
    </div>
    <div class="win-canvas-wrap">
      <canvas></canvas>
      <div class="px-info" title="Click to copy"><span class="px-swatch"></span><pre class="px-text"></pre></div>
    </div>
    ${preview ? `<div class="texture-strip"></div><div class="win-status">${STATUS_HTML}</div>` : ''}
    <div class="view-body"></div>`;
  const bar = el.querySelector('.win-bar') as HTMLElement;
  const select = el.querySelector('select') as HTMLSelectElement;
  const cv = el.querySelector('canvas') as HTMLCanvasElement;
  const body = el.querySelector('.view-body') as HTMLElement;
  const status = el.querySelector('.win-status') as HTMLElement | null;
  const dockBtn = el.querySelector('.win-dock') as HTMLButtonElement;
  const pxInfo = el.querySelector('.px-info') as HTMLElement;
  const ctx = cv.getContext('2d')!;
  const handles = addResizeHandles(el);
  const strip = preview ? makeStrip(el.querySelector('.texture-strip') as HTMLElement) : null;

  // Tool pads persist through spec.data; storage writes are coalesced.
  let pad: ScratchPad | null = null;
  let padSaveTimer: ReturnType<typeof setTimeout> | null = null;
  const padHost = (kind: string) => ({
    data: spec.data?.[kind] ?? '',
    onData: (d: string) => {
      (spec.data ??= {})[kind] = d;
      if (padSaveTimer) clearTimeout(padSaveTimer);
      padSaveTimer = setTimeout(saveSettings, 400);
    },
  });
  function dropPad(): void {
    if (pad) { pad.destroy(); pad = null; }
  }

  const view: PassView = {
    spec, el, select, cv, ctx,
    get status() { return status; },
    pxInfo, pxSwatch: pxInfo.querySelector('.px-swatch') as HTMLElement, pxText: pxInfo.querySelector('.px-text') as HTMLElement,
    get pass() { return spec.pass; },
    visible() { return el.isConnected && el.clientWidth > 0; },
    blit() {
      const dpr = Math.min(devicePixelRatio, MAX_DPR);
      const w = Math.max(1, cv.clientWidth * dpr | 0), h = Math.max(1, cv.clientHeight * dpr | 0);
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
      ctx.drawImage(canvas, 0, 0, w, h);
    },
    // The uniform table is one DOM node; only one window can hold it.
    setPass(id: string) {
      const switching = spec.pass !== id;
      spec.pass = id;
      el.classList.toggle('tool', isToolView(id));
      if (switching) dropPad();
      if (switching && inspect.view === view) setInspecting(null);
      if (id === UNIFORMS_VIEW) {
        for (const o of views) if (o !== view && o.pass === UNIFORMS_VIEW) { o.select.value = 'image'; o.setPass('image'); }
        body.appendChild(uniBody);
        uniBody.hidden = false;
      } else if (uniBody.parentElement === body) {
        uniBody.hidden = true;
        document.body.appendChild(uniBody);
      }
      if (isToolView(id) && id !== UNIFORMS_VIEW && !pad) {
        pad = id === 'math' ? makeMathPad(padHost(id))
          : id === 'texcalc' ? makeMathPad(padHost(id), { evaluate: true, noSource: true })
          : makeScratch(id as ScratchKind, padHost(id));
        body.appendChild(pad.el);
      }
      if (select.value !== id) select.value = id;
      if (switching) { syncDock(); syncStatusButtons(); }
    },
    setDocked(on: boolean) {
      spec.docked = on;
      placeView(view);
      settings.views = views.map(v => v.spec);
    },
    remove() {
      dropPad();
      if (padSaveTimer) clearTimeout(padSaveTimer);
      if (strip) strips.splice(strips.indexOf(strip), 1);
      if (inspect.view === view) setInspecting(null);
      if (uniBody.parentElement === body) { uniBody.hidden = true; document.body.appendChild(uniBody); }
      el.remove();
      const i = views.indexOf(view);
      if (i >= 0) views.splice(i, 1);
      settings.views = views.map(v => v.spec);
      syncDock();
      saveSettings();
    },
  };

  const host: FloatHost = {
    el, bar, minW: VIEW_MIN_W, minH: VIEW_MIN_H,
    rect: () => spec.rect,
    setRect: r => { spec.rect = r; writeRect(el, r); },
    onRelease: () => { settings.views = views.map(v => v.spec); },
  };
  attachFloatGestures(host, handles);
  attachPointer(view);

  fillPassSelect(select, spec.pass, !preview);
  const initial = select.value;
  spec.pass = '';          // so the first setPass builds the pad
  view.setPass(initial);
  select.addEventListener('change', () => {
    view.setPass(select.value);
    // Picking a tool that needs more room than the window has grows it (the saved layout keeps whatever size the user settles on).
    const need = TOOL_MIN_SIZE[select.value];
    if (need && !isDocked(view) && (spec.rect.w < need[0] || spec.rect.h < need[1])) {
      host.setRect(clampRect({ ...spec.rect, w: Math.max(spec.rect.w, need[0]), h: Math.max(spec.rect.h, need[1]) }, VIEW_MIN_W, VIEW_MIN_H));
      settings.views = views.map(v => v.spec);
    }
    saveSettings(); scheduleRender();
  });
  (el.querySelector('.win-close') as HTMLButtonElement).addEventListener('click', () => view.remove());
  dockBtn.addEventListener('click', () => { view.setDocked(!spec.docked); saveSettings(); });
  status?.addEventListener('click', (e: MouseEvent) => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>('[data-act]');
    if (!b) return;
    switch (b.dataset.act) {
      case 'pause':      setPaused(!paused); break;
      case 'step':       stepFrame(e.ctrlKey || e.metaKey ? 100 : e.shiftKey ? 10 : 1); break;
      case 'reset':      resetTime(); break;
      case 'inspect':    setInspecting(inspect.view === view ? null : view); break;
      case 'shot':       captureFrame(); break;
      case 'record':     toggleRecording(); break;
      case 'fullscreen': toggleFullscreen(view); break;
    }
  });
  pxInfo.addEventListener('click', () => {
    navigator.clipboard?.writeText(view.pxText.textContent ?? '').catch(() => {});
    pxInfo.classList.add('copied');
    setTimeout(() => pxInfo.classList.remove('copied'), 500);
  });
  el.addEventListener('pointerdown', () => { if (!isDocked(view)) raise(el); }, true);
  // Double-click the bar: cycle through halves and quarters of the free area.
  bar.addEventListener('dblclick', (e: MouseEvent) => {
    if ((e.target as HTMLElement).closest('button, select') || isDocked(view)) return;
    const vw = window.innerWidth, vh = window.innerHeight, top = toolbarBottom() + FLOAT_MARGIN;
    const m = FLOAT_MARGIN, W = vw - 2 * m, H = vh - top - m;
    const zones: FloatRect[] = [
      { x: m, y: top, w: W / 2, h: H }, { x: m + W / 2, y: top, w: W / 2, h: H },
      { x: m, y: top, w: W / 2, h: H / 2 }, { x: m + W / 2, y: top, w: W / 2, h: H / 2 },
      { x: m, y: top + H / 2, w: W / 2, h: H / 2 }, { x: m + W / 2, y: top + H / 2, w: W / 2, h: H / 2 },
    ];
    const i = (Number(el.dataset.tile ?? -1) + 1) % zones.length;
    el.dataset.tile = String(i);
    host.setRect(clampRect(zones[i], VIEW_MIN_W, VIEW_MIN_H));
    settings.views = views.map(v => v.spec);
    saveSettings();
  });
  el.addEventListener('animationend', () => el.classList.remove('pop'));
  new ResizeObserver(entries => {
    const w = entries[0].contentRect.width;
    for (const step of WIDTH_STEPS) el.classList.toggle(`w-lt-${step}`, w < step);
  }).observe(el);

  views.push(view);
  settings.views = views.map(v => v.spec);
  placeView(view);
  setReadout('res', resText);
  syncStatusButtons();
  return view;
}

/** Put a window where its state says: in the dock, or floating at its rect. */
function placeView(v: PassView): void {
  const docked = isDocked(v);
  const dockBtn = v.el.querySelector('.win-dock') as HTMLButtonElement;
  v.el.classList.toggle('docked', docked);
  v.el.classList.toggle('floating', !docked);
  dockBtn.innerHTML = docked ? ICON_UNDOCK : ICON_DOCK;
  dockBtn.title = docked ? 'Undock into a floating window' : 'Dock beside the editor';
  dockBtn.disabled = phoneMQ.matches;
  if (docked) {
    if (v.el.parentElement !== dock) dock.appendChild(v.el);
    v.el.style.left = v.el.style.top = v.el.style.width = v.el.style.height = v.el.style.zIndex = '';
  } else {
    if (v.el.parentElement !== document.body) document.body.appendChild(v.el);
    if (!v.spec.rect.w) v.spec.rect = defaultWinRect();
    v.spec.rect = clampRect(v.spec.rect, VIEW_MIN_W, VIEW_MIN_H);
    writeRect(v.el, v.spec.rect);
    raise(v.el);
  }
  syncDock();
}

phoneMQ.addEventListener('change', () => { for (const v of views) placeView(v); scheduleRender(); });

// Width classes drive which status-bar readouts a window can afford.
const WIDTH_STEPS = [560, 460, 360, 320, 300];

function currentLayout(): SavedLayout {
  return JSON.parse(JSON.stringify({
    editorFloating: settings.editorFloating, editorRect: settings.editorRect,
    views: views.map(v => v.spec),
  })) as SavedLayout;
}

/** Replace the windows with a shader's saved layout. */
function applySavedLayout(l: SavedLayout): void {
  for (const v of [...views]) v.remove();
  settings.editorRect = l.editorRect ?? settings.editorRect;
  applyEditorFloating(l.editorFloating);
  for (const spec of l.views) makeView(spec, false);
  refreshViewSelects();
  saveSettings();
  scheduleRender();
}

// New floating windows cascade down from the top-right so they never stack exactly.
function nextViewRect(w: number, h: number): FloatRect {
  const vw = window.innerWidth, top = toolbarBottom();
  const n = views.filter(v => !isDocked(v)).length;
  return { x: vw - w - 24 - n * 28, y: top + 24 + n * 28, w, h };
}

/** A new window: docked when the dock is empty (or on a phone), floating otherwise. */
function openView(pass: string, kind: WinKind): PassView {
  const docked = phoneMQ.matches || !dock.querySelector(':scope > .win');
  const [w, h] = isToolView(pass) ? TOOL_MIN_SIZE[pass] ?? [320, 240] : kind === 'preview' ? [520, 420] : [400, 300];
  const v = makeView({ pass, kind, docked, rect: nextViewRect(w, h) }, !docked);
  saveSettings();
  return v;
}

// Default to the pass being edited, when it has output of its own.
const editedPass = (): string => (viewablePasses().includes(activePass) ? activePass.id : 'image');
btnNewPreview.addEventListener('click', () => openView(editedPass(), 'preview'));
btnNewView.addEventListener('click', () => openView(editedPass(), 'view'));

// Keep windows reachable when the viewport shrinks (or rotates).
window.addEventListener('resize', () => {
  if (gesture || document.fullscreenElement) return;
  if (settings.editorFloating) positionEditorFloat();
  for (const v of views) {
    if (isDocked(v)) continue;
    v.spec.rect = clampRect(v.spec.rect, VIEW_MIN_W, VIEW_MIN_H);
    writeRect(v.el, v.spec.rect);
  }
  scheduleRender();
});

/* ── Options panel ───────────────────────────────────────── */
const btnOptions   = document.getElementById('btn-options')        as HTMLButtonElement;
const optPanel     = document.getElementById('options-panel')      as HTMLElement;
const optClose     = document.getElementById('options-close')      as HTMLButtonElement;
const optRes       = document.getElementById('opt-res')            as HTMLInputElement;
const optResVal    = document.getElementById('opt-res-val')        as HTMLElement;
const optLayoutNorm= document.getElementById('opt-layout-normal')  as HTMLButtonElement;
const optLayoutSwap= document.getElementById('opt-layout-swap')    as HTMLButtonElement;
const optVim       = document.getElementById('opt-vim')            as HTMLButtonElement;
const optRedraw    = document.getElementById('opt-redraw')         as HTMLButtonElement;
optRedraw.addEventListener('click', () => { redrawEditor(); toggleOptions(false); });
const optDelay     = document.getElementById('opt-delay')          as HTMLInputElement;
const optShot      = document.getElementById('opt-shot')           as HTMLInputElement;
const optShotVal   = document.getElementById('opt-shot-val')       as HTMLElement;
const optDelayVal  = document.getElementById('opt-delay-val')      as HTMLElement;
const optThemeGrid = document.getElementById('opt-theme-grid')     as HTMLElement;
const optThemeVal  = document.getElementById('opt-theme-val')      as HTMLElement;
const optThemeDark = document.getElementById('opt-theme-dark')     as HTMLElement;
const optThemeLight= document.getElementById('opt-theme-light')    as HTMLElement;
const optReset     = document.getElementById('opt-reset')          as HTMLButtonElement;
const optRestore   = document.getElementById('opt-restore')        as HTMLButtonElement;

const CSS_VARS: Array<[keyof Theme['ui'], string]> = [
  ['bg0', '--bg0'], ['bg1', '--bg1'], ['bg2', '--bg2'], ['bg3', '--bg3'],
  ['border', '--border'], ['accent', '--accent'], ['accent2', '--accent2'],
  ['err', '--err'], ['warn', '--warn'], ['ok', '--ok'],
  ['text', '--text'], ['textDim', '--text-dim'], ['preview', '--preview'],
  ['hl', '--hl'], ['sel', '--sel'],
];

function applyTheme(id: string): void {
  const t = THEME_BY_ID.get(id) ?? THEMES[0];
  settings.theme = t.id;
  const root = document.documentElement;
  for (const [key, cssVar] of CSS_VARS) root.style.setProperty(cssVar, t.ui[key]);
  root.style.setProperty('color-scheme', t.dark ? 'dark' : 'light');
  root.style.setProperty('--text-hi', t.dark ? '#ffffff' : '#000000');
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', t.ui.bg1);
  if (editor) editor.dispatch({ effects: themeCompartment.reconfigure(themeExtensions(t)) });
  optThemeVal.textContent = t.name;
  optThemeGrid.querySelectorAll('.theme-swatch').forEach(el => {
    el.classList.toggle('active', (el as HTMLElement).dataset.theme === t.id);
  });
}

// syncInput is off while typing, so the field isn't rewritten under the caret.
function applyResScale(scale: number, syncInput = true): void {
  settings.resScale = clampRes(scale);
  optResVal.textContent = `${+(settings.resScale * 100).toFixed(2)}%`;
  if (syncInput) optRes.value = String(+settings.resScale.toFixed(4));
  scheduleRender();   // repaint at the new framebuffer size even while paused
}

function refreshShotReadout(): void {
  const size = parseShotSize(settings.shotSize, canvas.width, canvas.height);
  optShotVal.textContent = size ? `${size[0]} × ${size[1]}` : '';
}

function applyShotSize(spec: string, syncInput = true): void {
  const ok = parseShotSize(spec, 1, 1);
  if (ok) settings.shotSize = spec.trim();
  refreshShotReadout();
  optShot.classList.toggle('bad', !ok);
  if (syncInput) optShot.value = settings.shotSize;
}

function applyCompileDelay(ms: number, syncInput = true): void {
  settings.compileDelay = clampDelay(ms);
  optDelayVal.textContent = settings.compileDelay === 0 ? 'instant' : `${settings.compileDelay} ms`;
  if (syncInput) optDelay.value = String(settings.compileDelay);
}

function applyLayout(swapped: boolean): void {
  settings.swapped = swapped;
  main.classList.toggle('swapped', swapped);
  optLayoutNorm.classList.toggle('active', !swapped);
  optLayoutSwap.classList.toggle('active', swapped);
}

/* ── Editor glass ────────────────────────────────────────── */
const optGlassModes = document.getElementById('opt-glass-modes')     as HTMLElement;
const optGlassVal   = document.getElementById('opt-glass-val')       as HTMLElement;
const optGlassTint  = document.getElementById('opt-glass-tint')      as HTMLInputElement;
const optGlassTintV = document.getElementById('opt-glass-tint-val')  as HTMLElement;
const optGlassFrost = document.getElementById('opt-glass-frost')     as HTMLInputElement;
const optGlassFrostV= document.getElementById('opt-glass-frost-val') as HTMLElement;
const GLASS_NAMES: Record<GlassMode, string> = { solid: 'Solid', tint: 'Tint', clear: 'Clear', glass: 'Glass' };

// Only the backgrounds change (CSS in index.html keys off these classes and
// variables); the pane is still opaque while docked.
function applyEditorGlass(mode: GlassMode, tint = settings.glassTint, frost = settings.glassFrost): void {
  settings.editorGlass = mode;
  settings.glassTint = clamp01(tint);
  settings.glassFrost = Math.min(Math.max(Math.round(frost), 0), 40);
  const a = mode === 'clear' ? 0 : settings.glassTint;
  const root = document.documentElement;
  root.style.setProperty('--glass-a', String(a));
  root.style.setProperty('--glass-blur', `${settings.glassFrost}px`);
  editorP.classList.toggle('glass', mode !== 'solid');
  editorP.classList.toggle('glass-frost', mode === 'glass');
  optGlassVal.textContent = GLASS_NAMES[mode];
  optGlassModes.querySelectorAll<HTMLButtonElement>('button').forEach(b => b.classList.toggle('active', b.dataset.glass === mode));
  (document.getElementById('opt-glass-tint-row') as HTMLElement).hidden = !(mode === 'tint' || mode === 'glass');
  (document.getElementById('opt-glass-frost-row') as HTMLElement).hidden = mode !== 'glass';
  optGlassTint.value = String(settings.glassTint);
  optGlassTintV.textContent = `${Math.round(settings.glassTint * 100)}%`;
  optGlassFrost.value = String(settings.glassFrost);
  optGlassFrostV.textContent = `${settings.glassFrost}px`;
}

optGlassModes.addEventListener('click', (e: MouseEvent) => {
  const b = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-glass]');
  if (!b || !isGlassMode(b.dataset.glass)) return;
  applyEditorGlass(b.dataset.glass); saveSettings();
});
optGlassTint.addEventListener('input', () => applyEditorGlass(settings.editorGlass, parseFloat(optGlassTint.value)));
optGlassTint.addEventListener('change', saveSettings);
optGlassFrost.addEventListener('input', () => applyEditorGlass(settings.editorGlass, settings.glassTint, parseFloat(optGlassFrost.value)));
optGlassFrost.addEventListener('change', saveSettings);

function applyVimMode(on: boolean): void {
  settings.vimMode = on;
  optVim.classList.toggle('active', on);
  optVim.textContent = on ? 'Vim mode: on' : 'Vim mode: off';
  if (editor) editor.dispatch({ effects: vimCompartment.reconfigure(vimExtension(on)) });
}

// Build theme swatches
for (const t of THEMES) {
  const btn = document.createElement('button');
  btn.className = 'theme-swatch';
  btn.dataset.theme = t.id;
  btn.title = t.name;
  const dots = document.createElement('span');
  dots.className = 'dots';
  for (const c of [t.ui.bg1, t.ui.accent, t.syn.keyword, t.syn.string, t.syn.number]) {
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = c;
    dot.style.border = `1px solid ${t.ui.border}`;
    dots.appendChild(dot);
  }
  const name = document.createElement('span');
  name.className = 'tname';
  name.textContent = t.name;
  btn.append(dots, name);
  btn.addEventListener('click', () => { applyTheme(t.id); saveSettings(); });
  (t.dark ? optThemeDark : optThemeLight).appendChild(btn);
}

if (!hasFinePointer) {
  const hint = document.createElement('div');
  hint.className = 'opt-hint';
  hint.textContent = 'Vim mode needs a hardware Esc key — off by default on touch devices.';
  optVim.parentElement!.after(hint);
}

optRes.addEventListener('input', () => {
  const v = parseFloat(optRes.value);
  if (!isFinite(v)) return;   // empty or mid-typing — leave the last good value
  applyResScale(v, false);
});
optRes.addEventListener('change', () => {
  const v = parseFloat(optRes.value);
  applyResScale(isFinite(v) ? v : settings.resScale);
  saveSettings();
});

optDelay.addEventListener('input', () => {
  const v = parseFloat(optDelay.value);
  if (!isFinite(v)) return;
  applyCompileDelay(v, false);
});
optDelay.addEventListener('change', () => {
  const v = parseFloat(optDelay.value);
  applyCompileDelay(isFinite(v) ? v : settings.compileDelay);
  saveSettings();
});

optShot.addEventListener('input', () => applyShotSize(optShot.value, false));
optShot.addEventListener('change', () => { applyShotSize(optShot.value); saveSettings(); });

optLayoutNorm.addEventListener('click', () => { applyLayout(false); saveSettings(); });
optLayoutSwap.addEventListener('click', () => { applyLayout(true);  saveSettings(); });
optVim.addEventListener('click', () => { applyVimMode(!settings.vimMode); saveSettings(); });

optRestore.addEventListener('click', () => {
  if (!confirmDiscard()) return;
  openProject(structuredClone(EXAMPLE_PROJECT), 'Reaction–diffusion', null);
  toggleOptions(false);
});

// Catch edits still waiting on the save debounce.
window.addEventListener('pagehide', saveProject);
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') saveProject();
});

optReset.addEventListener('click', () => {
  const d = defaultSettings();
  applyTheme(d.theme);
  applyResScale(d.resScale);
  applyLayout(d.swapped);
  applyVimMode(d.vimMode);
  applyCompileDelay(d.compileDelay);
  applyShotSize(d.shotSize);
  settings.editorRect = d.editorRect;
  applyEditorFloating(d.editorFloating);
  applyEditorGlass(d.editorGlass, d.glassTint, d.glassFrost);
  for (const v of [...views]) v.remove();
  for (const spec of d.views) makeView(spec, false);
  refreshViewSelects();
  saveSettings();
});

function toggleOptions(open: boolean): void {
  optPanel.classList.toggle('open', open);
  btnOptions.classList.toggle('active', open);
  if (open) toggleShaders(false);
}

btnOptions.addEventListener('click', (e: MouseEvent) => {
  e.stopPropagation();
  toggleOptions(!optPanel.classList.contains('open'));
});
optClose.addEventListener('click', () => toggleOptions(false));
optPanel.addEventListener('click', (e: MouseEvent) => e.stopPropagation());
document.addEventListener('click', () => toggleOptions(false));

/* ── Uniform table ───────────────────────────────────────── */
// Lists every uniform the active pass receives, refreshed each frame while a
// view window is showing it.
const uniPass   = document.getElementById('uni-pass')    as HTMLElement;

const UNI_ROWS: Array<[name: string, type: string]> = [
  ['iTime', 'float'], ['iTimeDelta', 'float'], ['iFrameRate', 'float'], ['iFrame', 'int'], ['iResolution', 'vec3'],
  ['iMouse', 'vec4'], ['iDate', 'vec4'],
  ['iChannel0', ''], ['iChannel1', ''], ['iChannel2', ''], ['iChannel3', ''],
  ['iChannelResolution', 'vec3[4]'],
];
const uniVals = new Map<string, HTMLElement>();
const uniTypes = new Map<string, HTMLElement>();
for (const [name, type] of UNI_ROWS) {
  const row = document.createElement('div');
  row.className = 'uni-row';
  const n = document.createElement('span'); n.className = 'uni-name'; n.textContent = name;
  const t = document.createElement('span'); t.className = 'uni-type'; t.textContent = type;
  const v = document.createElement('span'); v.className = 'uni-val';
  row.append(n, t, v);
  uniBody.insertBefore(row, uniBody.querySelector('.uni-foot'));
  uniVals.set(name, v);
  uniTypes.set(name, t);
}

// iFrame can be typed into while paused: Enter seeks to that frame.
{
  const v = uniVals.get('iFrame')!;
  v.classList.add('uni-edit');
  v.title = 'Pause, then click to jump to a frame';
  v.addEventListener('click', () => {
    if (!paused || v.querySelector('input')) return;
    const input = document.createElement('input');
    input.type = 'number'; input.min = '0'; input.step = '1'; input.value = String(Math.max(iFrame, 0));
    v.replaceChildren(input);
    input.focus(); input.select();
    const done = (apply: boolean) => {
      const n = Number(input.value);
      v.textContent = String(Math.max(iFrame, 0));
      if (apply && Number.isFinite(n)) seekFrame(n);
    };
    input.addEventListener('keydown', e => {
      e.stopPropagation();
      if (e.key === 'Enter') done(true);
      else if (e.key === 'Escape') done(false);
    });
    input.addEventListener('blur', () => { if (input.isConnected) done(false); });
  });
}

function setUni(name: string, text: string): void {
  const el = uniVals.get(name)!;
  if (el.firstElementChild) return;   // being edited
  if (el.textContent !== text) el.textContent = text;
}

const f3 = (x: number) => x.toFixed(3);

function updateUniformPanel(w: number, h: number): void {
  const shown = views.find(v => v.pass === UNIFORMS_VIEW);
  if (!shown || !shown.visible()) return;
  setUni('iTime', f3(iTime));
  setUni('iFrame', String(Math.max(iFrame, 0)));

  // Channels belong to a render pass; Common shows Image's, since that's what runs.
  const p = activePass === commonPass ? imagePass : activePass;
  uniPass.textContent = p.name;

  setUni('iTimeDelta', `${f3(iTimeDelta)}  (${(iTimeDelta * 1000).toFixed(1)} ms)`);
  setUni('iFrameRate', paused ? '—' : fpsValue.toFixed(1));
  setUni('iResolution', `${w}, ${h}, 1`);
  setUni('iMouse', `${iMouse[0].toFixed(0)}, ${iMouse[1].toFixed(0)}, ${iMouse[2].toFixed(0)}, ${iMouse[3].toFixed(0)}`);
  const d = new Date();
  const secs = d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds() + d.getMilliseconds() / 1000;
  setUni('iDate', `${d.getFullYear()}, ${d.getMonth()}, ${d.getDate()}, ${secs.toFixed(2)}`);

  const res: string[] = [];
  for (let i = 0; i < NUM_SLOTS; i++) {
    const c = p.channels[i];
    let cw = 0, chh = 0;
    if (c.kind === 'buffer') { const t = targets[c.buf]; if (t) { cw = t.w; chh = t.h; } }
    else { cw = c.texW; chh = c.texH; }
    uniTypes.get(`iChannel${i}`)!.textContent = c.kind === 'cube' ? 'samplerCube' : 'sampler2D';
    setUni(`iChannel${i}`, channelDescription(c));
    res.push(`${cw}×${chh}`);
  }
  setUni('iChannelResolution', res.join('  '));
}

/* ── Shader library panel ────────────────────────────────── */
const btnShaders   = document.getElementById('btn-shaders')    as HTMLButtonElement;
const btnShaderNm  = document.getElementById('btn-shader-name') as HTMLElement;
const shPanel      = document.getElementById('shaders-panel')  as HTMLElement;
const shClose      = document.getElementById('shaders-close')  as HTMLButtonElement;
const shName       = document.getElementById('shader-name')    as HTMLInputElement;
const shState      = document.getElementById('shaders-state')  as HTMLElement;
const shSave       = document.getElementById('shader-save')    as HTMLButtonElement;
const shCopy       = document.getElementById('shader-copy')    as HTMLButtonElement;
const shNew        = document.getElementById('shader-new')     as HTMLButtonElement;
const shList       = document.getElementById('shader-list')    as HTMLElement;
const shCount      = document.getElementById('shaders-count')  as HTMLElement;

let flashTimer: ReturnType<typeof setTimeout> | null = null;

/** Toolbar button: the shader's name, with a dot while it has unsaved changes. */
function updateShaderButton(): void {
  if (flashTimer) return;
  const dirty = projectDirty();
  btnShaderNm.textContent = projectName;
  btnShaders.classList.toggle('dirty', dirty);
  btnShaders.title = dirty
    ? (libraryEntry() ? 'Unsaved changes — Ctrl+S to save' : 'Not saved yet — Ctrl+S to save')
    : 'Saved';
  shState.textContent = libraryEntry() ? (dirty ? 'modified' : 'saved') : 'unsaved';
  if (document.activeElement !== shName) shName.value = projectName;
}

function flashShaderButton(text: string): void {
  if (flashTimer) clearTimeout(flashTimer);
  btnShaderNm.textContent = text;
  btnShaders.classList.remove('dirty');
  btnShaders.classList.add('flash');
  flashTimer = setTimeout(() => {
    flashTimer = null;
    btnShaders.classList.remove('flash');
    updateShaderButton();
  }, 900);
}

function passSummary(project: SavedProject): string {
  const n = Object.values(project.passes).filter(p => p.enabled).length;
  return n === 1 ? 'Image only' : `${n} passes`;
}

function fmtDate(ms: number): string {
  const d = new Date(ms), now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  return sameDay
    ? d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function renderShaderList(): void {
  shList.innerHTML = '';
  shCount.textContent = library.length ? String(library.length) : '';
  if (!library.length) {
    const empty = document.createElement('div');
    empty.className = 'opt-hint';
    empty.textContent = 'Nothing saved yet. Name the shader above and press Save.';
    shList.appendChild(empty);
    return;
  }
  const sorted = [...library].sort((a, b) => b.updated - a.updated);
  for (const entry of sorted) {
    const row = document.createElement('div');
    row.className = 'shader-row' + (entry.id === projectLibId ? ' current' : '');
    row.title = `Open "${entry.name}"`;

    const main = document.createElement('div');
    main.className = 'sr-main';
    const name = document.createElement('div'); name.className = 'sr-name'; name.textContent = entry.name;
    const meta = document.createElement('div'); meta.className = 'sr-meta';
    meta.textContent = `${passSummary(entry.project)} · ${fmtDate(entry.updated)}`;
    main.append(name, meta);

    const actions = document.createElement('div');
    actions.className = 'sr-actions';
    const rename = document.createElement('button');
    rename.className = 'sr-btn'; rename.textContent = '✎'; rename.title = 'Rename';
    rename.addEventListener('click', (e: MouseEvent) => {
      e.stopPropagation();
      const typed = prompt('Rename shader', entry.name);
      if (typed === null || !typed.trim()) return;
      entry.name = typed.trim();
      if (entry.id === projectLibId) projectName = entry.name;
      persistLibrary();
      saveProject();
      renderShaderList();
    });
    const del = document.createElement('button');
    del.className = 'sr-btn sr-danger'; del.textContent = '✕'; del.title = 'Delete';
    del.addEventListener('click', (e: MouseEvent) => {
      e.stopPropagation();
      if (!confirm(`Delete "${entry.name}" from this browser?`)) return;
      library = library.filter(x => x !== entry);
      if (entry.id === projectLibId) projectLibId = null;   // the working copy lives on, unsaved
      persistLibrary();
      saveProject();
      renderShaderList();
    });
    actions.append(rename, del);

    row.append(main, actions);
    row.addEventListener('click', () => {
      if (entry.id === projectLibId && !projectDirty()) { toggleShaders(false); return; }
      if (!confirmDiscard()) return;
      openProject(structuredClone(entry.project), entry.name, entry.id);
      toggleShaders(false);
    });
    shList.appendChild(row);
  }
}

function toggleShaders(open: boolean): void {
  shPanel.classList.toggle('open', open);
  btnShaders.classList.toggle('active', open);
  if (open) { toggleOptions(false); renderShaderList(); updateShaderButton(); }
}

btnShaders.addEventListener('click', (e: MouseEvent) => {
  e.stopPropagation();
  toggleShaders(!shPanel.classList.contains('open'));
});
shClose.addEventListener('click', () => toggleShaders(false));
shPanel.addEventListener('click', (e: MouseEvent) => e.stopPropagation());
document.addEventListener('click', () => toggleShaders(false));

shName.addEventListener('input', () => {
  projectName = shName.value.trim() || 'Untitled';
  btnShaderNm.textContent = projectName;
  scheduleSave();
});
shName.addEventListener('keydown', (e: KeyboardEvent) => {
  if (e.key === 'Enter') { e.preventDefault(); saveToLibrary(); }
});
shSave.addEventListener('click', () => saveToLibrary());
shCopy.addEventListener('click', () => saveToLibrary(true));
shNew.addEventListener('click', () => {
  if (!confirmDiscard()) return;
  openProject(blankProject(), 'Untitled', null);
  toggleShaders(false);
});

/* ── Init ────────────────────────────────────────────────── */
initQuad();

applyTheme(settings.theme);
applyResScale(settings.resScale);
applyLayout(settings.swapped);
applyVimMode(settings.vimMode);
applyCompileDelay(settings.compileDelay);
applyShotSize(settings.shotSize);
applyEditorFloating(settings.editorFloating);
applyEditorGlass(settings.editorGlass);
for (const spec of settings.views.splice(0)) makeView(spec, false);

editor = new EditorView({
  state: makeEditorState(activePass.code),
  parent: document.getElementById('editor-wrap') as HTMLElement,
});
editor.focus();

updatePassTabs();
rebuildChannelStrip();
updateShaderButton();

tryCompile();
scheduleRender();

// The worker is generated at deploy time (pwa/build.mjs); dev servers have none.
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register('./sw.js').catch(() => { /* offline support is optional */ });
}

/* ── Control panel: sliders for annotated consts ─────────── */

/** Writes a control's value back into its declaration line, in the editor or the stored code. */
function commitControl(pass: Pass, c: Control): void {
  if (pass === activePass && editor) {
    const doc = editor.state.doc;
    if (c.line >= doc.lines) return;
    const l = doc.line(c.line + 1);
    const text = replaceLiteral(l.text, c);
    if (text !== null && text !== l.text) editor.dispatch({ changes: { from: l.from, to: l.to, insert: text } });
  } else if (pass.state) {
    // Patch the kept state through a transaction so its undo history survives.
    const doc = pass.state.doc;
    if (c.line >= doc.lines) return;
    const l = doc.line(c.line + 1);
    const text = replaceLiteral(l.text, c);
    if (text !== null && text !== l.text) {
      pass.state = pass.state.update({ changes: { from: l.from, to: l.to, insert: text } }).state;
      pass.code = pass.state.doc.toString();
      scheduleSave();
    }
  } else {
    const lines = pass.code.split('\n');
    const text = replaceLiteral(lines[c.line] ?? '', c);
    if (text !== null && text !== lines[c.line]) { lines[c.line] = text; pass.code = lines.join('\n'); scheduleSave(); }
  }
}

const fmtLane = (c: Control, v: number) => (c.type === 'int' ? String(Math.round(v)) : v.toFixed(c.max - c.min > 10 ? 2 : 3));

function controlRow(pass: Pass, c: Control): HTMLElement {
  const row = document.createElement('div');
  row.className = `ctl-row ctl-${c.kind}`;
  const name = document.createElement('span'); name.className = 'ctl-name'; name.textContent = c.name; name.title = `${c.type} ${c.name}`;
  row.appendChild(name);
  const body = document.createElement('div'); body.className = 'ctl-body';
  row.appendChild(body);
  const live = () => scheduleRender();

  if (c.kind === 'toggle') {
    const box = document.createElement('input'); box.type = 'checkbox'; box.checked = !!c.value[0];
    box.addEventListener('change', () => { c.value[0] = box.checked ? 1 : 0; live(); commitControl(pass, c); });
    body.appendChild(box);
    return row;
  }
  if (c.kind === 'color') {
    const pick = document.createElement('input'); pick.type = 'color';
    const toHex = () => '#' + c.value.slice(0, 3).map(v => Math.round(Math.min(Math.max(v, 0), 1) * 255).toString(16).padStart(2, '0')).join('');
    pick.value = toHex();
    pick.addEventListener('input', () => {
      const h = pick.value;
      for (let i = 0; i < 3; i++) c.value[i] = parseInt(h.slice(1 + i * 2, 3 + i * 2), 16) / 255;
      live();
    });
    pick.addEventListener('change', () => commitControl(pass, c));
    body.appendChild(pick);
    if (c.type === 'vec4') body.appendChild(laneSlider(pass, c, 3, 0, 1, 0, live));
    return row;
  }
  for (let i = 0; i < c.value.length; i++) body.appendChild(laneSlider(pass, c, i, c.min, c.max, c.step, live));
  return row;
}

function laneSlider(pass: Pass, c: Control, i: number, min: number, max: number, step: number, live: () => void): HTMLElement {
  const lane = document.createElement('label'); lane.className = 'ctl-lane';
  const range = document.createElement('input'); range.type = 'range';
  range.min = String(min); range.max = String(max);
  range.step = String(step || (c.type === 'int' ? 1 : (max - min) / 1000));
  range.value = String(c.value[i]);
  const val = document.createElement('span'); val.className = 'ctl-val'; val.textContent = fmtLane(c, c.value[i]);
  range.addEventListener('input', () => { c.value[i] = parseFloat(range.value); val.textContent = fmtLane(c, c.value[i]); live(); });
  range.addEventListener('change', () => commitControl(pass, c));
  // Typing a value: click the readout.
  val.addEventListener('click', e => {
    e.preventDefault();
    const edit = document.createElement('input'); edit.type = 'number'; edit.value = String(c.value[i]); edit.step = range.step;
    val.replaceWith(edit); edit.focus(); edit.select();
    const done = (apply: boolean) => {
      const n = parseFloat(edit.value);
      if (apply && Number.isFinite(n)) { c.value[i] = n; range.value = String(n); live(); commitControl(pass, c); }
      val.textContent = fmtLane(c, c.value[i]);
      edit.replaceWith(val);
    };
    edit.addEventListener('keydown', ev => { ev.stopPropagation(); if (ev.key === 'Enter') done(true); else if (ev.key === 'Escape') done(false); });
    edit.addEventListener('blur', () => { if (edit.isConnected) done(true); });
  });
  lane.append(range, val);
  return lane;
}

/** Rebuilds the control rows when the set of controls changes (not on value changes). */
function syncControlPanel(): void {
  const groups = [commonPass, imagePass, ...activeBufferPasses()].filter(p => p.enabled && p.controls.length);
  const sig = groups.map(p => `${p.id}=${controlSignature(p.controls)}`).join(';');
  if (sig === controlSig && !controlsStale) return;
  controlSig = sig;
  controlsStale = false;
  uniControls.replaceChildren();
  uniControls.hidden = groups.length === 0;
  for (const p of groups) {
    if (groups.length > 1) {
      const h = document.createElement('div'); h.className = 'ctl-group'; h.textContent = p.name;
      uniControls.appendChild(h);
    }
    for (const c of p.controls) uniControls.appendChild(controlRow(p, c));
  }
  const sep = document.createElement('div'); sep.className = 'ctl-sep';
  uniControls.appendChild(sep);
}

/** True when a key press belongs to a text field or the code editor. */
function typingTarget(e: KeyboardEvent): boolean {
  const t = e.target;
  return t instanceof HTMLElement && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || !!t.closest('.cm-editor'));
}
document.addEventListener('keydown', (e: KeyboardEvent) => {
  if (e.ctrlKey || e.metaKey || e.altKey || typingTarget(e)) return;
  const onButton = (e.target as HTMLElement).tagName === 'BUTTON';   // space activates it already
  if (e.key === ' ' && !onButton) { e.preventDefault(); setPaused(!paused); }
  else if (e.key === 'i') setInspecting(inspect.view ? null : primaryView());
  else if (e.code === 'Period' && paused) { e.preventDefault(); stepFrame(e.shiftKey ? 10 : 1); }   // Shift+. reads as '>'
});

// Escape closes the topmost thing, one layer per press: menus and modals,
// then the popover panels, then the front view window. Text fields (and the
// editor, which has its own Escape uses) are left alone unless a menu is up.
document.addEventListener('keydown', (e: KeyboardEvent) => {
  if (e.key !== 'Escape') return;
  const open = (el: HTMLElement) => el.classList.contains('open');
  if (open(helpModal)) toggleHelp(false);
  else if (open(cubeModal)) cubeModal.classList.remove('open');
  else if (open(ctxMenu) || open(fmtMenu) || open(addMenu)) { hideCtxMenu(); hideFmtMenu(); hideAddMenu(); }
  else if (open(optPanel)) toggleOptions(false);
  else if (open(shPanel)) toggleShaders(false);
  else if (typingTarget(e)) return;
  else if (frontFloatingView()) frontFloatingView()!.remove();
  else return;
  e.preventDefault();
});

/** The floating view window on top (the editor pane may still be above it). */
function frontFloatingView(): PassView | undefined {
  let best: PassView | undefined, z = -Infinity;
  for (const v of views) {
    if (!v.el.classList.contains('floating')) continue;
    const vz = +v.el.style.zIndex || 0;
    if (vz >= z) { best = v; z = vz; }
  }
  return best;
}

document.addEventListener('keydown', (e: KeyboardEvent) => {
  // S8: inside the editor (or any field) Ctrl+Shift+V pastes as plain text.
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'v' && !typingTarget(e)) { e.preventDefault(); btnNewView.click(); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
    e.preventDefault();
    scheduleCompile(true);
    saveToLibrary(e.shiftKey);   // Ctrl+Shift+S saves a copy
    return;
  }
  // Alt+1…: the pass tabs in the order they're shown (Chromium keeps Ctrl+digit for its own tabs).
  if (e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey && /^Digit[1-8]$/.test(e.code)) {
    const p = shownPasses()[Number(e.code.slice(5)) - 1];
    if (p) { e.preventDefault(); switchPass(p); }
  }
});

/* ── Shortcuts sheet ─────────────────────────────────────── */
const helpModal = document.getElementById('help-modal') as HTMLElement;
function toggleHelp(open: boolean): void {
  helpModal.classList.toggle('open', open);
  if (open) { toggleOptions(false); toggleShaders(false); }
}
document.getElementById('help-close')!.addEventListener('click', () => toggleHelp(false));
helpModal.addEventListener('click', (e: MouseEvent) => { if (e.target === helpModal) toggleHelp(false); });
document.getElementById('opt-keys')!.addEventListener('click', () => toggleHelp(true));
document.addEventListener('keydown', (e: KeyboardEvent) => {
  if (e.key === '?' && !e.ctrlKey && !e.metaKey && !e.altKey && !typingTarget(e)) { e.preventDefault(); toggleHelp(!helpModal.classList.contains('open')); }
});

// A file dropped anywhere but a target would open in the tab, taking the editor with it.
// (The code editor takes drops itself: CodeMirror inserts a text file's contents.)
document.addEventListener('dragover', (e: DragEvent) => {
  if (!(e.target as HTMLElement).closest?.('.tex-slot, .pad-image, .cm-editor') && e.dataTransfer) { e.preventDefault(); e.dataTransfer.dropEffect = 'none'; }
});
document.addEventListener('drop', (e: DragEvent) => e.preventDefault());
