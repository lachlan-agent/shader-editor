// Unit checks for the control annotations (bundled by run.mjs alongside the calculators).
export function checkControls({ rewriteControls, replaceLiteral, formatLiteral }) {
  const src = [
    'const float speed = 1.0;   // @slider(0.0, 5.0)',
    'const int   steps = 64; // @slider(1, 256, 1)',
    'const vec3  tint = vec3(1.0, 0.5, 0.2);  // @color',
    'const vec2  c = vec2(0.5); // @slider',
    'const bool  wire = false; // @toggle',
    'const float plain = 2.0;',
    'const float bad = foo; // @slider(0, 1)',
    'void mainImage(out vec4 o, in vec2 p) { o = vec4(speed); }',
    'const float prod = 2.0 * 3.0; // @slider(0, 10)',
    'const int   mask = 0x0F; // @slider(0, 255)',
    'const float a = 1.0; const float b = 2.0; // @slider(0, 1)',
    'void f() {',
    '  const float local = 1.0; // @slider(0, 1)',
    '}',
    'const float flat = 1.0; // @slider(1, 1)',
    'const float shifted = 0.5; // @slider(foo, 0, 1)',
    'const vec2  call = vec2(f(1.0), 2.0); // @slider(0, 1)',
    'const float sci = 1e-3f; // @slider(0, 1)',
  ].join('\n');
  const { code, controls, problems } = rewriteControls(src);
  const failures = [];
  const eq = (what, got, want) => { if (JSON.stringify(got) !== JSON.stringify(want)) failures.push(`${what}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); };
  eq('line count kept', code.split('\n').length, src.split('\n').length);
  eq('names', controls.map(c => c.name), ['speed', 'steps', 'tint', 'c', 'wire', 'mask', 'flat', 'sci']);
  eq('uniform rewrite', code.split('\n')[0].startsWith('uniform float speed;'), true);
  eq('comment kept', code.split('\n')[0].includes('// @slider(0.0, 5.0)'), true);
  eq('plain const untouched', code.split('\n')[5], 'const float plain = 2.0;');
  eq('unparsable literal untouched', code.split('\n')[6], 'const float bad = foo; // @slider(0, 1)');
  eq('int range', [controls[1].min, controls[1].max, controls[1].step], [1, 256, 1]);
  eq('vec2 splat', controls[3].value, [0.5, 0.5]);
  eq('default range from value', [controls[3].min, controls[3].max], [0, 2]);
  eq('toggle value', controls[4].value, [0]);
  controls[0].value = [2.5];
  eq('replace float', replaceLiteral(src.split('\n')[0], controls[0]), 'const float speed = 2.5;   // @slider(0.0, 5.0)');
  controls[2].value = [0, 1, 0.333333];
  eq('replace vec3', replaceLiteral(src.split('\n')[2], controls[2]), 'const vec3  tint = vec3(0.0, 1.0, 0.333333);  // @color');
  controls[1].value = [7.6];
  eq('int literal', formatLiteral(controls[1]), '8');
  eq('float format 1', formatLiteral({ ...controls[0], value: [3] }), '3.0');
  eq('float format 2', formatLiteral({ ...controls[0], value: [0.0001] }), '0.0001');
  eq('float format 3', formatLiteral({ ...controls[0], value: [1e-7] }), '1.00000e-7');
  const at = n => code.split('\n')[n], srcAt = n => src.split('\n')[n];
  eq('G6 expression untouched', at(8), srcAt(8));
  eq('G6 hex int', controls[5].value, [15]);
  eq('G6 float suffix', controls[7].value, [0.001]);
  eq('G6 call in vec untouched', at(16), srcAt(16));
  eq('G7 two consts untouched', at(10), srcAt(10));
  eq('G8 local const untouched', at(12), srcAt(12));
  eq('G35 empty range widened', controls[6].max > controls[6].min, true);
  eq('G35 bad arg untouched', at(15), srcAt(15));
  eq('G35 bad arg reported', problems.length, 1);
  eq('mismatched line', replaceLiteral('float other = 1.0;', controls[0]), null);
  return failures;
}
