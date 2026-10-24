#!/usr/bin/env node
// Checks the GLSL formatter: fixed cases, and that formatting is idempotent
// and keeps the line count on the bundled example shaders.
//
//   node test/format.mjs

import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const out = mkdtempSync(join(tmpdir(), 'shader-format-test-'));
try {
  execFileSync('npx', ['esbuild', join(root, 'src/glslFormat.ts'), '--bundle', '--platform=node', '--format=esm', '--log-level=error', `--outdir=${out}`], { cwd: root, stdio: 'inherit' });
  var { formatGlsl } = await import(pathToFileURL(join(out, 'glslFormat.js')));
} finally {
  process.on('exit', () => rmSync(out, { recursive: true, force: true }));
}

const cases = [
  // spacing
  ['vec3 c=a*b+c/2.0-x;', 'vec3 c = a * b + c / 2.0 - x;'],
  ['float x = -a + (-b) * !c;', 'float x = -a + (-b) * !c;'],
  ['x=-1.;y=.5*-z;', 'x = -1.; y = .5 * -z;'],
  ['foo( a,b ,c );', 'foo(a, b, c);'],
  ['if(x>0.0){y=1.0;}', 'if (x > 0.0) { y = 1.0; }'],
  ['for(int i=0;i<4;i++){', 'for (int i = 0; i < 4; i++) {'],
  ['i++; ++i; a = b++ + ++c;', 'i++; ++i; a = b++ + ++c;'],
  ['return a>b?a:b;', 'return a > b ? a : b;'],
  ['v.xy=v .yx;', 'v.xy = v.yx;'],
  ['a[i]=b[ j ];', 'a[i] = b[j];'],
  ['x<<=2;y>>=1;z|=w&&q;', 'x <<= 2; y >>= 1; z |= w && q;'],
  ['vec3 p = vec3( uv , 0.0 ) ;', 'vec3 p = vec3(uv, 0.0);'],
  ['float a[3] = float[]( 1., 2., 3. );', 'float a[3] = float[](1., 2., 3.);'],
  ['layout(location=0) out vec4 fragColor;', 'layout(location = 0) out vec4 fragColor;'],
  ['const float DU   = 1.0;     // diffusion', 'const float DU = 1.0;     // diffusion'],
  ['x = y; /* mid */ z = w;', 'x = y; /* mid */ z = w;'],
  ['   #define  FOO(x)  (x*2)', '#define  FOO(x)  (x*2)'],
  ['uint n = 3u ^ 0x1Fu; float f = 1.5e-3f;', 'uint n = 3u ^ 0x1Fu; float f = 1.5e-3f;'],
  // indentation
  ['void main(){\nx=1;\n}', 'void main() {\n    x = 1;\n}'],
  ['void main()\n{\nx=1;\n}', 'void main()\n{\n    x = 1;\n}'],
  ['if (a)\nx = 1;\nelse\ny = 2;\nz = 3;', 'if (a)\n    x = 1;\nelse\n    y = 2;\nz = 3;'],
  ['if (a) {\nx = 1;\n} else if (b)\ny = 2;\nelse {\nz = 3;\n}', 'if (a) {\n    x = 1;\n} else if (b)\n    y = 2;\nelse {\n    z = 3;\n}'],
  ['for (int i = 0; i < 4; i++)\nfor (int j = 0; j < 4; j++)\ns += f(i, j);\nt = 0;', 'for (int i = 0; i < 4; i++)\n    for (int j = 0; j < 4; j++)\n        s += f(i, j);\nt = 0;'],
  ['vec3 c = mix(a,\nb,\nt);\nx = 1;', 'vec3 c = mix(a,\n    b,\n    t);\nx = 1;'],
  ['float x = a +\nb +\nc;\ny = 2;', 'float x = a +\n    b +\n    c;\ny = 2;'],
  ['vec2 lap = -c\n+ 0.2 * (a\n+ b)\n- d;', 'vec2 lap = -c\n    + 0.2 * (a\n    + b)\n    - d;'],
  ['x = 1;\n-y;', 'x = 1;\n-y;'],
  ['switch (n) {\ncase 0:\nx = 1;\nbreak;\ndefault:\nx = 2;\n}', 'switch (n) {\n    case 0:\n        x = 1;\n        break;\n    default:\n        x = 2;\n}'],
  ['struct S {\nfloat a;\n};\nS s;', 'struct S {\n    float a;\n};\nS s;'],
  ['void f() {\n/* a\n   b */\n  x = 1;\n}', 'void f() {\n    /* a\n   b */\n    x = 1;\n}'],
  ['void f() {\n#ifdef X\n  x = 1;\n#endif\n}', 'void f() {\n#ifdef X\n    x = 1;\n#endif\n}'],
  ['void f() {\n\n\n  x = 1;   \n}', 'void f() {\n\n\n    x = 1;\n}'],
  ['void f() {\n  // comment\n  x = 1; // trailing\n}', 'void f() {\n    // comment\n    x = 1; // trailing\n}'],
  ['\tvoid f() {\n\t\tx = 1;\n\t}\n', 'void f() {\n    x = 1;\n}\n'],
  // G25: #define continuation lines are preprocessor text
  ['#define F(x) \\\n    (x*2) + \\\n    1\nx=1;', '#define F(x) \\\n    (x*2) + \\\n    1\nx = 1;'],
  // G26: case with a braced body
  ['switch (n) {\ncase 0: {\nx = 1;\nbreak;\n}\ndefault:\nx = 2;\n}', 'switch (n) {\n    case 0: {\n        x = 1;\n        break;\n    }\n    default:\n        x = 2;\n}'],
  // G27: postfix ++ followed by a binary operator
  ['a=b++ -c;d=e-- +f;g=h++ -(i);', 'a = b++ - c; d = e-- + f; g = h++ - (i);'],
];

let pass = 0;
const failures = [];
for (const [src, want] of cases) {
  const got = formatGlsl(src);
  if (got === want) pass++; else failures.push({ src, want, got });
  const again = formatGlsl(got);
  if (again !== got) failures.push({ src: `(idempotence) ${src}`, want: got, got: again });
}

// The example shaders in main.ts: every `const … = \`…\`` GLSL block must survive a format with the same line count, and format to a fixed point.
const main = readFileSync(join(root, 'src/main.ts'), 'utf8');
const examples = ['DEFAULT_SHADER', 'DEFAULT_BUFFER_A', 'DEFAULT_COMMON', 'BLANK_SHADER'];
for (const name of examples) {
  const m = new RegExp(`const ${name} = \`([^\`]*)\``).exec(main);
  if (!m) { failures.push({ src: `(example ${name})`, want: 'found', got: 'missing' }); continue; }
  const src = m[1];
  const got = formatGlsl(src);
  if (got.split('\n').length !== src.split('\n').length) failures.push({ src: '(example line count)', want: src.split('\n').length, got: got.split('\n').length });
  if (formatGlsl(got) !== got) failures.push({ src: '(example idempotence)', want: got, got: formatGlsl(got) });
  pass++;
}

for (const f of failures) {
  console.log(`FAIL\n--- source\n${f.src}\n--- expected\n${JSON.stringify(f.want)}\n--- got\n${JSON.stringify(f.got)}\n`);
}
console.log(`${pass} passed, ${failures.length} failed`);
process.exit(failures.length ? 1 : 0);
