/**
 * A GLSL formatter for the `:format` ex command (and Shift+Alt+F).
 *
 * Deliberately conservative: it never joins or splits lines, so line numbers
 * (and so error markers, the cursor, and a `:'<,'>format` range) survive. What
 * it does to each line: re-indents it from the brace depth, continuation and
 * `case` state, and normalises the spacing between tokens — one space around
 * binary operators and after commas, none inside brackets or before a call's
 * `(`. Comments, preprocessor lines and the inside of block comments are left
 * exactly as written, and the whitespace before a trailing `//` is kept so
 * aligned comments stay aligned.
 */

export interface FormatOptions {
  /** The indentation unit; four spaces by default. */
  indent?: string;
}

type Kind = 'id' | 'num' | 'op' | 'lcomment' | 'bcomment' | 'ws';
interface Tok { kind: Kind; text: string }

const KEYWORDS_BEFORE_PAREN = new Set(['if', 'for', 'while', 'switch', 'return', 'else', 'case']);
// Longest first, so `<<=` wins over `<<` over `<`.
const OPERATORS = [
  '<<=', '>>=', '...', '->',
  '++', '--', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||', '^^', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=',
  '+', '-', '*', '/', '%', '<', '>', '=', '!', '~', '&', '|', '^', '?', ':', ',', ';', '.', '(', ')', '[', ']', '{', '}', '#',
];
const BINARY = new Set(['+', '-', '*', '/', '%', '<', '>', '<=', '>=', '==', '!=', '&&', '||', '^^', '&', '|', '^', '<<', '>>',
  '=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '?']);

/** Tokens of one line of code (no newlines inside, except in a block comment's text). */
function tokenize(line: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  const n = line.length;
  while (i < n) {
    const c = line[i];
    if (c === ' ' || c === '\t') {
      let j = i; while (j < n && (line[j] === ' ' || line[j] === '\t')) j++;
      out.push({ kind: 'ws', text: line.slice(i, j) }); i = j; continue;
    }
    if (c === '/' && line[i + 1] === '/') { out.push({ kind: 'lcomment', text: line.slice(i) }); break; }
    if (c === '/' && line[i + 1] === '*') {
      const end = line.indexOf('*/', i + 2);
      const j = end < 0 ? n : end + 2;
      out.push({ kind: 'bcomment', text: line.slice(i, j) }); i = j; continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i; while (j < n && /[A-Za-z0-9_]/.test(line[j])) j++;
      out.push({ kind: 'id', text: line.slice(i, j) }); i = j; continue;
    }
    if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(line[i + 1] ?? ''))) {
      // 1.0, .5, 1., 1e-3, 1.5e+2f, 0x1F, 3u, 2lf
      const m = /^(0[xX][0-9a-fA-F]+|[0-9]*\.?[0-9]*(?:[eE][+-]?[0-9]+)?)(?:[uUfF]|lf|LF)?/.exec(line.slice(i));
      const text = m && m[0] ? m[0] : c;
      out.push({ kind: 'num', text }); i += text.length; continue;
    }
    const op = OPERATORS.find(o => line.startsWith(o, i));
    if (op) { out.push({ kind: 'op', text: op }); i += op.length; continue; }
    out.push({ kind: 'op', text: c }); i++;
  }
  return out;
}

/** Does a space belong between these two neighbouring code tokens? */
function spaceBetween(a: Tok, b: Tok, prev: Tok | null, continuing: boolean): boolean {
  const at = a.text, bt = b.text;
  // Nothing to the left that could be an operand: a `-`, `+`, `!` or `~` here is unary.
  const unaryContext = (t: Tok | null): boolean =>
    t === null ? !continuing : (t.kind === 'op' && t.text !== ')' && t.text !== ']' && t.text !== '++' && t.text !== '--') || (t.kind === 'id' && (t.text === 'return' || t.text === 'case'));
  if (a.kind === 'bcomment' || b.kind === 'bcomment') return true;
  // Brackets.
  if (at === '(' || at === '[' || bt === ')' || bt === ']') return false;
  if (bt === '[') return false;
  if (bt === '(') {
    if (a.kind === 'id') return KEYWORDS_BEFORE_PAREN.has(at);
    if (at === ')' || at === ']' || at === '(' || at === '[' || at === '!' || at === '~' || at === '++' || at === '--') return false;
    if (at === '-' || at === '+') return !unaryContext(prev);
    return true;
  }
  if (bt === '{') return at !== '(' && at !== '[';
  if (at === '{') return bt !== '}';
  if (bt === '}') return at !== '{';
  if (at === '}') return bt !== ';' && bt !== ',' && bt !== ')';
  // Member access and separators.
  if (at === '.' || bt === '.') return false;
  if (bt === ',' || bt === ';') return false;
  if (at === ',' || at === ';') return true;
  // `case 1:`, `default:` — but `a ? b : c` spaces the colon.
  if (bt === ':') return !(prev && (prev.text === 'case' || prev.text === 'default')) && !(a.kind === 'id' && at === 'default');
  // Increment and decrement stick to their operand: `i++`, `++i`, `a++ + b`.
  const operand = (t: Tok | null): boolean => t !== null && (t.kind === 'id' || t.kind === 'num' || t.text === ')' || t.text === ']');
  if (at === '++' || at === '--') return operand(prev);
  if (bt === '++' || bt === '--') return !operand(a);
  if (at === '!' || at === '~') return false;
  if ((at === '-' || at === '+') && unaryContext(prev)) return false;
  // Binary operators get a space each side.
  if (BINARY.has(at) || BINARY.has(bt) || at === ':' || bt === ':') return true;
  // Two words (`const float`), a word and a number, `)` then a word (`) x`).
  return true;
}

export function formatGlsl(src: string, opts: FormatOptions = {}): string {
  const unit = opts.indent ?? '    ';
  const lines = src.split('\n');
  const out: string[] = [];

  let depth = 0;                  // brace depth at the start of the line
  let parenDepth = 0;             // unclosed ( and [ carried over from earlier lines
  const caseStack: boolean[] = [];   // per open brace: are we inside a case body?
  let inBlockComment = false;     // a /* that an earlier line opened
  let continuation = false;       // the previous line ended mid-expression
  let hangingHead = 0;            // lines owed an extra indent by a braceless if/for/while/else
  let directive = false;          // the previous line was a preprocessor line ending in `\`

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');
    if (inBlockComment) {
      // Inside a block comment: leave the text alone, watch for the end.
      out.push(line);
      const end = line.indexOf('*/');
      if (end >= 0) {
        inBlockComment = false;
        // Code after a closing `*/` on the same line is rare; keep it as written.
      }
      continue;
    }
    if (directive) { out.push(line); directive = line.endsWith('\\'); continue; }   // `#define` continuation: verbatim
    if (line.trim() === '') { out.push(''); continue; }

    const toks = tokenize(line.trimStart());
    const first = toks[0];
    if (first && first.kind === 'op' && first.text === '#') { out.push(line.trim()); directive = line.endsWith('\\'); continue; }   // preprocessor: column 0, verbatim

    // Trailing block comment left open on this line.
    const lastTok = toks[toks.length - 1];
    if (lastTok && lastTok.kind === 'bcomment' && !lastTok.text.endsWith('*/')) inBlockComment = true;

    const code = toks.filter(t => t.kind !== 'ws' && t.kind !== 'lcomment');
    const commentOnly = code.length === 0 || code.every(t => t.kind === 'bcomment');

    // Indentation for this line.
    let level = depth;
    const leadsWithClose = code[0]?.text === '}';
    const leadsWithCase = code[0]?.kind === 'id' && (code[0].text === 'case' || code[0].text === 'default') && code.some(t => t.text === ':');
    if (leadsWithClose) level = Math.max(0, depth - 1);
    else if (leadsWithCase) level = depth;   // at the switch body's level; statements beneath get +1
    else if (depth > 0 && caseStack[depth - 1]) level = depth + 1;
    const leadsWithOpen = code[0]?.text === '{';
    if (!leadsWithClose && !leadsWithOpen && (continuation || parenDepth > 0)) level += 1;
    if (!leadsWithClose && !leadsWithOpen && !commentOnly) level += hangingHead;

    // Rebuild the line's text.
    let text = '';
    let prev: Tok | null = null;     // previous code token
    let prevPrev: Tok | null = null;
    for (let i = 0; i < toks.length; i++) {
      const t = toks[i];
      if (t.kind === 'ws') continue;
      if (t.kind === 'lcomment') {
        // Keep whatever gap preceded a trailing comment, so columns of comments line up.
        const gap = i > 0 && toks[i - 1].kind === 'ws' ? toks[i - 1].text : (prev ? ' ' : '');
        text += (prev ? gap : '') + t.text;
        break;
      }
      if (prev && spaceBetween(prev, t, prevPrev, continuation || parenDepth > 0)) text += ' ';
      text += t.text;
      prevPrev = prev; prev = t;
    }
    out.push(unit.repeat(level) + text);

    if (commentOnly) continue;

    // State for the next line. A `case N: {` body belongs to the switch's brace, not the one it opens.
    const switchDepth = depth;
    for (const t of code) {
      if (t.kind !== 'op') continue;
      if (t.text === '{') { depth++; caseStack[depth - 1] = false; }
      else if (t.text === '}') { depth = Math.max(0, depth - 1); caseStack.length = depth; }
      else if (t.text === '(' || t.text === '[') parenDepth++;
      else if (t.text === ')' || t.text === ']') parenDepth = Math.max(0, parenDepth - 1);
    }
    if (leadsWithCase && switchDepth > 0) caseStack[switchDepth - 1] = true;

    const last = code[code.length - 1];
    const endsStatement = last.text === ';' || last.text === '{' || last.text === '}';
    const head = code[0].kind === 'id' ? code[0].text : code[0].text;
    // `if (…)`, `for (…)`, `while (…)`, `else`, `} else`, `} else if (…)` with no brace: the next statement hangs under it.
    const braceless = parenDepth === 0 && !endsStatement && (
      ((head === 'if' || head === 'for' || head === 'while') && last.text === ')') ||
      (head === 'else' && (code.length === 1 || (code[1]?.text === 'if' && last.text === ')'))) ||
      (head === '}' && code[1]?.text === 'else' && (code.length === 2 || (code[2]?.text === 'if' && last.text === ')'))));
    if (last.text === '{' || last.text === '}') hangingHead = 0;
    else if (braceless) hangingHead++;
    else if (endsStatement) hangingHead = 0;
    // Anything else left unterminated (`x = a +`, `f(a,`, `vec3 c = mix(a, b`) continues on the next line.
    continuation = !braceless && !endsStatement && !leadsWithCase && parenDepth === 0;
  }

  // Keep (or not) the trailing newline exactly as the source had it.
  return out.join('\n');
}
