/*
 * Reference cards for GLSL ES 3.00 built-ins and the Shadertoy-style
 * uniforms, shown as a tooltip on hover or on `K` / F1 at the cursor.
 *
 * `genType` stands for float/vec2/vec3/vec4, `genIType` for the int vectors,
 * `genBType` for the bool vectors and `mat` for any square matrix type, as
 * in the spec. Each entry is terse on purpose: it is a reminder, not a manual.
 */

export interface GlslDoc {
  /** One or more signatures, as they would appear in a header. */
  sig: string[];
  /** What it does, in a sentence or two. */
  doc: string;
  /** Where it comes from, for the tooltip's corner label. */
  kind: 'function' | 'uniform' | 'variable' | 'type' | 'keyword';
}

const F = 'function', U = 'uniform', V = 'variable', T = 'type', K = 'keyword';

type Entry = [name: string, sig: string | string[], doc: string, kind?: GlslDoc['kind']];

const ENTRIES: Entry[] = [
  // ── Shadertoy uniforms ─────────────────────────────────────────────
  ['iTime', 'uniform float iTime', 'Seconds since the shader started (or was reset). Paused time stands still.', U],
  ['iTimeDelta', 'uniform float iTimeDelta', 'Seconds elapsed since the previous frame. Use it for frame-rate-independent animation.', U],
  ['iFrame', 'uniform int iFrame', 'Number of frames rendered so far, starting at 0. Handy for one-off initialisation in buffer passes: `if (iFrame == 0) …`.', U],
  ['iFrameRate', 'uniform float iFrameRate', 'Measured frames per second over the last second.', U],
  ['iResolution', 'uniform vec3 iResolution', 'Render target size in pixels, z = 1 (pixel aspect). fragCoord / iResolution.xy gives 0–1 UVs; divide by iResolution.y to keep the aspect ratio.', U],
  ['iSampleRate', 'uniform float iSampleRate', 'Audio sample rate, 44100. Declared so Shadertoy code compiles; there is no sound pass here.', U],
  ['iChannelTime', 'uniform float iChannelTime[4]', 'Playback time of each channel. Always 0 here (no video or audio inputs); declared so Shadertoy code compiles.', U],
  ['iMouse', 'uniform vec4 iMouse', 'xy: current pointer position in pixels (origin bottom-left). zw: position where the button was pressed; z is negative while the button is up, so `iMouse.z > 0.0` means "dragging".', U],
  ['iDate', 'uniform vec4 iDate', 'x: year, y: month (0–11), z: day of month, w: seconds since midnight, with a fractional part.', U],
  ['iChannelResolution', 'uniform vec3 iChannelResolution[4]', 'Size of each channel\'s texture in pixels, as (width, height, 1). Zero for an empty channel.', U],
  ['iChannel0', 'uniform sampler2D iChannel0  //  or samplerCube', 'Channel 0: a 2D texture, a cubemap or another buffer\'s previous frame, as wired in the strip below the preview. Read it with texture() or texelFetch().', U],
  ['iChannel1', 'uniform sampler2D iChannel1  //  or samplerCube', 'Channel 1: a 2D texture, a cubemap or another buffer\'s previous frame. Read it with texture() or texelFetch().', U],
  ['iChannel2', 'uniform sampler2D iChannel2  //  or samplerCube', 'Channel 2: a 2D texture, a cubemap or another buffer\'s previous frame. Read it with texture() or texelFetch().', U],
  ['iChannel3', 'uniform sampler2D iChannel3  //  or samplerCube', 'Channel 3: a 2D texture, a cubemap or another buffer\'s previous frame. Read it with texture() or texelFetch().', U],
  ['mainImage', 'void mainImage(out vec4 fragColor, in vec2 fragCoord)', 'The entry point every pass must define. fragCoord is the pixel centre (so 0.5, 1.5, …) with the origin bottom-left; write the pixel\'s colour to fragColor.', F],

  // ── Built-in variables ─────────────────────────────────────────────
  ['gl_FragCoord', 'in vec4 gl_FragCoord', 'Window-relative fragment position: xy in pixels (origin bottom-left, pixel centres at .5), z depth, w = 1/clip w. Same as mainImage\'s fragCoord.', V],
  ['gl_FragDepth', 'out float gl_FragDepth', 'Overrides the fragment\'s depth value. Rarely useful in a full-screen pass.', V],
  ['gl_FrontFacing', 'in bool gl_FrontFacing', 'True when the fragment belongs to a front-facing primitive. Always true for the full-screen quad here.', V],
  ['gl_PointCoord', 'in vec2 gl_PointCoord', 'Position within a point sprite. Undefined here, since the editor draws triangles, not points.', V],
  ['gl_Position', 'out vec4 gl_Position', 'Vertex shader output: clip-space position. Not available in a fragment shader.', V],
  ['gl_PointSize', 'out float gl_PointSize', 'Vertex shader output: point sprite size in pixels. Not available in a fragment shader.', V],
  ['gl_VertexID', 'in int gl_VertexID', 'Vertex shader input: index of the vertex being processed. Not available in a fragment shader.', V],
  ['gl_InstanceID', 'in int gl_InstanceID', 'Vertex shader input: instance index. Not available in a fragment shader.', V],

  // ── Trigonometry ───────────────────────────────────────────────────
  ['radians', 'genType radians(genType degrees)', 'Converts degrees to radians: degrees · π / 180.'],
  ['degrees', 'genType degrees(genType radians)', 'Converts radians to degrees: radians · 180 / π.'],
  ['sin', 'genType sin(genType angle)', 'Sine of an angle in radians.'],
  ['cos', 'genType cos(genType angle)', 'Cosine of an angle in radians.'],
  ['tan', 'genType tan(genType angle)', 'Tangent of an angle in radians.'],
  ['asin', 'genType asin(genType x)', 'Arc sine, in [−π/2, π/2]. Undefined for |x| > 1.'],
  ['acos', 'genType acos(genType x)', 'Arc cosine, in [0, π]. Undefined for |x| > 1.'],
  ['atan', ['genType atan(genType y, genType x)', 'genType atan(genType y_over_x)'], 'Two-argument form: angle of the vector (x, y), in [−π, π], with the signs of both used to pick the quadrant (like atan2). One-argument form: arc tangent in [−π/2, π/2].'],
  ['sinh', 'genType sinh(genType x)', 'Hyperbolic sine: (eˣ − e⁻ˣ) / 2.'],
  ['cosh', 'genType cosh(genType x)', 'Hyperbolic cosine: (eˣ + e⁻ˣ) / 2.'],
  ['tanh', 'genType tanh(genType x)', 'Hyperbolic tangent: sinh(x) / cosh(x). A smooth squash of ℝ onto (−1, 1).'],
  ['asinh', 'genType asinh(genType x)', 'Inverse hyperbolic sine.'],
  ['acosh', 'genType acosh(genType x)', 'Inverse hyperbolic cosine. Undefined for x < 1.'],
  ['atanh', 'genType atanh(genType x)', 'Inverse hyperbolic tangent. Undefined for |x| ≥ 1.'],

  // ── Exponential ────────────────────────────────────────────────────
  ['pow', 'genType pow(genType x, genType y)', 'x raised to the power y. Undefined for x < 0, and for x = 0 with y ≤ 0: clamp or abs() the base first.'],
  ['exp', 'genType exp(genType x)', 'The natural exponential eˣ.'],
  ['log', 'genType log(genType x)', 'Natural logarithm. Undefined for x ≤ 0.'],
  ['exp2', 'genType exp2(genType x)', '2 raised to the power x.'],
  ['log2', 'genType log2(genType x)', 'Base-2 logarithm. Undefined for x ≤ 0.'],
  ['sqrt', 'genType sqrt(genType x)', 'Square root. Undefined for x < 0.'],
  ['inversesqrt', 'genType inversesqrt(genType x)', '1 / sqrt(x), usually faster than dividing. Undefined for x ≤ 0.'],

  // ── Common ─────────────────────────────────────────────────────────
  ['abs', ['genType abs(genType x)', 'genIType abs(genIType x)'], 'Absolute value, per component.'],
  ['sign', ['genType sign(genType x)', 'genIType sign(genIType x)'], '1.0 for positive, −1.0 for negative, 0.0 for zero, per component.'],
  ['floor', 'genType floor(genType x)', 'Largest integer ≤ x, per component.'],
  ['trunc', 'genType trunc(genType x)', 'Rounds toward zero, per component.'],
  ['round', 'genType round(genType x)', 'Nearest integer; halves may round either way (implementation-defined). Use roundEven() for a defined rule.'],
  ['roundEven', 'genType roundEven(genType x)', 'Nearest integer, with halves rounded to the even neighbour.'],
  ['ceil', 'genType ceil(genType x)', 'Smallest integer ≥ x, per component.'],
  ['fract', 'genType fract(genType x)', 'Fractional part: x − floor(x), always in [0, 1). The workhorse of tiling and hashes.'],
  ['mod', ['genType mod(genType x, float y)', 'genType mod(genType x, genType y)'], 'Modulus: x − y · floor(x / y). Unlike C\'s %, the result takes the sign of y, so mod(-1.0, 3.0) is 2.0.'],
  ['modf', 'genType modf(genType x, out genType i)', 'Splits x into an integer part (written to i) and a fractional part (returned), both with the sign of x.'],
  ['min', ['genType min(genType x, genType y)', 'genType min(genType x, float y)', 'genIType min(genIType x, genIType y)', 'genUType min(genUType x, genUType y)'], 'The smaller of x and y, per component.'],
  ['max', ['genType max(genType x, genType y)', 'genType max(genType x, float y)', 'genIType max(genIType x, genIType y)', 'genUType max(genUType x, genUType y)'], 'The larger of x and y, per component.'],
  ['clamp', ['genType clamp(genType x, genType lo, genType hi)', 'genType clamp(genType x, float lo, float hi)', 'genIType clamp(genIType x, genIType lo, genIType hi)'], 'min(max(x, lo), hi), per component. Undefined if lo > hi.'],
  ['mix', ['genType mix(genType x, genType y, genType a)', 'genType mix(genType x, genType y, float a)', 'genType mix(genType x, genType y, genBType a)'], 'Linear blend: x · (1 − a) + y · a. With a bool vector, picks y where a is true (a per-component select).'],
  ['step', ['genType step(genType edge, genType x)', 'genType step(float edge, genType x)'], '0.0 where x < edge, else 1.0. A hard threshold; see smoothstep() for a soft one.'],
  ['smoothstep', ['genType smoothstep(genType e0, genType e1, genType x)', 'genType smoothstep(float e0, float e1, genType x)'], '0.0 below e0, 1.0 above e1, and a smooth Hermite curve (3t² − 2t³) between. Undefined if e0 ≥ e1. Swap the edges to get a falling curve.'],
  ['isnan', 'genBType isnan(genType x)', 'True where x is NaN.'],
  ['isinf', 'genBType isinf(genType x)', 'True where x is positive or negative infinity.'],
  ['floatBitsToInt', 'genIType floatBitsToInt(genType value)', 'Reinterprets the float\'s IEEE 754 bits as a signed int, with no conversion.'],
  ['floatBitsToUint', 'genUType floatBitsToUint(genType value)', 'Reinterprets the float\'s IEEE 754 bits as an unsigned int, with no conversion.'],
  ['intBitsToFloat', 'genType intBitsToFloat(genIType value)', 'Reinterprets the int\'s bits as an IEEE 754 float.'],
  ['uintBitsToFloat', 'genType uintBitsToFloat(genUType value)', 'Reinterprets the uint\'s bits as an IEEE 754 float.'],
  ['packSnorm2x16', 'uint packSnorm2x16(vec2 v)', 'Clamps each component to [−1, 1], scales to a signed 16-bit int and packs both into one uint (x in the low bits).'],
  ['unpackSnorm2x16', 'vec2 unpackSnorm2x16(uint p)', 'Inverse of packSnorm2x16(): two signed 16-bit ints back to floats in [−1, 1].'],
  ['packUnorm2x16', 'uint packUnorm2x16(vec2 v)', 'Clamps each component to [0, 1], scales to an unsigned 16-bit int and packs both into one uint (x in the low bits).'],
  ['unpackUnorm2x16', 'vec2 unpackUnorm2x16(uint p)', 'Inverse of packUnorm2x16(): two unsigned 16-bit ints back to floats in [0, 1].'],
  ['packHalf2x16', 'uint packHalf2x16(vec2 v)', 'Converts both components to 16-bit floats and packs them into one uint (x in the low bits).'],
  ['unpackHalf2x16', 'vec2 unpackHalf2x16(uint p)', 'Inverse of packHalf2x16(): two 16-bit floats back to a vec2.'],

  // ── Geometric ──────────────────────────────────────────────────────
  ['length', 'float length(genType x)', 'Euclidean length of a vector: sqrt(dot(x, x)).'],
  ['distance', 'float distance(genType p0, genType p1)', 'Distance between two points: length(p0 − p1).'],
  ['dot', 'float dot(genType x, genType y)', 'Dot product: the sum of the component-wise products.'],
  ['cross', 'vec3 cross(vec3 x, vec3 y)', 'Cross product of two 3D vectors, perpendicular to both.'],
  ['normalize', 'genType normalize(genType x)', 'The same direction with length 1. Undefined for a zero vector.'],
  ['faceforward', 'genType faceforward(genType N, genType I, genType Nref)', 'Returns N if dot(Nref, I) < 0, else −N: flips a normal to face the viewer.'],
  ['reflect', 'genType reflect(genType I, genType N)', 'Reflects incident direction I about the normal N: I − 2 · dot(N, I) · N. N must be normalised.'],
  ['refract', 'genType refract(genType I, genType N, float eta)', 'Refracts I through the surface with normal N for the index-of-refraction ratio eta. Returns a zero vector on total internal reflection. Both inputs must be normalised.'],

  // ── Matrix ─────────────────────────────────────────────────────────
  ['matrixCompMult', 'mat matrixCompMult(mat x, mat y)', 'Multiplies the matrices component by component. Use the * operator for a proper linear-algebra product.'],
  ['outerProduct', 'mat outerProduct(vec c, vec r)', 'Treats c as a column and r as a row and multiplies them, giving a matrix.'],
  ['transpose', 'mat transpose(mat m)', 'Transpose of a matrix.'],
  ['determinant', 'float determinant(mat m)', 'Determinant of a square matrix.'],
  ['inverse', 'mat inverse(mat m)', 'Inverse of a square matrix. Undefined for a singular or near-singular one.'],

  // ── Vector relational ──────────────────────────────────────────────
  ['lessThan', 'bvec lessThan(vec x, vec y)', 'Component-wise x < y. The comparison operators only work on scalars.'],
  ['lessThanEqual', 'bvec lessThanEqual(vec x, vec y)', 'Component-wise x ≤ y.'],
  ['greaterThan', 'bvec greaterThan(vec x, vec y)', 'Component-wise x > y.'],
  ['greaterThanEqual', 'bvec greaterThanEqual(vec x, vec y)', 'Component-wise x ≥ y.'],
  ['equal', 'bvec equal(vec x, vec y)', 'Component-wise x == y. (The == operator compares whole vectors.)'],
  ['notEqual', 'bvec notEqual(vec x, vec y)', 'Component-wise x != y.'],
  ['any', 'bool any(bvec x)', 'True if any component is true.'],
  ['all', 'bool all(bvec x)', 'True only if every component is true.'],
  ['not', 'bvec not(bvec x)', 'Component-wise logical negation.'],

  // ── Texture ────────────────────────────────────────────────────────
  ['textureSize', 'ivec2 textureSize(sampler2D s, int lod)', 'Size of a mip level in texels. iChannelResolution gives the same for level 0.'],
  ['texture', ['vec4 texture(sampler2D s, vec2 uv [, float bias])', 'vec4 texture(samplerCube s, vec3 dir [, float bias])'], 'Samples with filtering and (for 2D) wrapping, using 0–1 UVs or a direction for a cubemap. A buffer sampling itself reads the previous frame.'],
  ['textureProj', 'vec4 textureProj(sampler2D s, vec3 P [, float bias])', 'Projective sample: divides P.xy by P.z before looking up.'],
  ['textureLod', ['vec4 textureLod(sampler2D s, vec2 uv, float lod)', 'vec4 textureLod(samplerCube s, vec3 dir, float lod)'], 'Samples an explicit mip level, with no derivative-based selection. Buffers have no mips, so lod is 0.0 there.'],
  ['textureOffset', 'vec4 textureOffset(sampler2D s, vec2 uv, ivec2 offset [, float bias])', 'Like texture(), with the lookup shifted by a constant number of texels.'],
  ['texelFetch', 'vec4 texelFetch(sampler2D s, ivec2 p, int lod)', 'Reads one texel by integer coordinate, unfiltered and unwrapped: exact for buffer-to-buffer reads. Out-of-range coordinates are undefined (wrap them yourself).'],
  ['texelFetchOffset', 'vec4 texelFetchOffset(sampler2D s, ivec2 p, int lod, ivec2 offset)', 'texelFetch() shifted by a constant texel offset.'],
  ['textureProjOffset', 'vec4 textureProjOffset(sampler2D s, vec3 P, ivec2 offset [, float bias])', 'Projective sample shifted by a constant texel offset.'],
  ['textureLodOffset', 'vec4 textureLodOffset(sampler2D s, vec2 uv, float lod, ivec2 offset)', 'Explicit-level sample shifted by a constant texel offset.'],
  ['textureProjLod', 'vec4 textureProjLod(sampler2D s, vec3 P, float lod)', 'Projective sample from an explicit mip level.'],
  ['textureProjLodOffset', 'vec4 textureProjLodOffset(sampler2D s, vec3 P, float lod, ivec2 offset)', 'Projective, explicit-level sample shifted by a constant texel offset.'],
  ['textureGrad', ['vec4 textureGrad(sampler2D s, vec2 uv, vec2 dPdx, vec2 dPdy)', 'vec4 textureGrad(samplerCube s, vec3 dir, vec3 dPdx, vec3 dPdy)'], 'Samples with explicit derivatives for mip selection, e.g. inside a branch where dFdx() would be undefined.'],
  ['textureGradOffset', 'vec4 textureGradOffset(sampler2D s, vec2 uv, vec2 dPdx, vec2 dPdy, ivec2 offset)', 'textureGrad() shifted by a constant texel offset.'],
  ['textureProjGrad', 'vec4 textureProjGrad(sampler2D s, vec3 P, vec2 dPdx, vec2 dPdy)', 'Projective sample with explicit derivatives.'],
  ['textureProjGradOffset', 'vec4 textureProjGradOffset(sampler2D s, vec3 P, vec2 dPdx, vec2 dPdy, ivec2 offset)', 'Projective sample with explicit derivatives, shifted by a constant texel offset.'],

  // ── Derivatives ────────────────────────────────────────────────────
  ['dFdx', 'genType dFdx(genType p)', 'Rate of change of p across one pixel horizontally, from the 2×2 pixel quad. Undefined inside non-uniform branches.'],
  ['dFdy', 'genType dFdy(genType p)', 'Rate of change of p across one pixel vertically, from the 2×2 pixel quad. Undefined inside non-uniform branches.'],
  ['fwidth', 'genType fwidth(genType p)', 'abs(dFdx(p)) + abs(dFdy(p)): how much p changes per pixel. Feed it to smoothstep() for resolution-independent anti-aliasing.'],

  // ── Types (constructors) ───────────────────────────────────────────
  ['float', 'float(x)', 'A 32-bit floating-point scalar. Literals need a decimal point or exponent (1.0, 1e3); 1 is an int.', T],
  ['int', 'int(x)', 'A 32-bit signed integer. Converting from float truncates toward zero.', T],
  ['uint', 'uint(x)', 'A 32-bit unsigned integer; literals take a u suffix (3u).', T],
  ['bool', 'bool(x)', 'true or false. Numbers convert as non-zero → true.', T],
  ['vec2', ['vec2(float)', 'vec2(float x, float y)'], 'Two floats. A single argument fills every component. Swizzle with .xy, .st or .rg in any order (v.yx).', T],
  ['vec3', ['vec3(float)', 'vec3(float x, float y, float z)', 'vec3(vec2 xy, float z)'], 'Three floats. Constructors flatten their arguments, so vec3(uv, 0.0) works. Swizzle with .xyz, .rgb or .stp.', T],
  ['vec4', ['vec4(float)', 'vec4(float x, float y, float z, float w)', 'vec4(vec3 xyz, float w)', 'vec4(vec2 xy, vec2 zw)'], 'Four floats, e.g. an RGBA colour. Constructors flatten their arguments. Swizzle with .xyzw or .rgba.', T],
  ['ivec2', 'ivec2(int x, int y)', 'Two ints, e.g. a texel coordinate for texelFetch(). Converting from vec2 truncates toward zero.', T],
  ['ivec3', 'ivec3(int x, int y, int z)', 'Three ints.', T],
  ['ivec4', 'ivec4(int x, int y, int z, int w)', 'Four ints.', T],
  ['uvec2', 'uvec2(uint x, uint y)', 'Two unsigned ints.', T],
  ['uvec3', 'uvec3(uint x, uint y, uint z)', 'Three unsigned ints.', T],
  ['uvec4', 'uvec4(uint x, uint y, uint z, uint w)', 'Four unsigned ints.', T],
  ['bvec2', 'bvec2(bool x, bool y)', 'Two bools, as returned by lessThan() and friends. Collapse with any() or all().', T],
  ['bvec3', 'bvec3(bool x, bool y, bool z)', 'Three bools. Collapse with any() or all().', T],
  ['bvec4', 'bvec4(bool x, bool y, bool z, bool w)', 'Four bools. Collapse with any() or all().', T],
  ['mat2', ['mat2(float diagonal)', 'mat2(vec2 col0, vec2 col1)', 'mat2(float, float, float, float)'], '2×2 float matrix, stored column-major: the first two values are the first column. m * v treats v as a column vector.', T],
  ['mat3', ['mat3(float diagonal)', 'mat3(vec3 col0, vec3 col1, vec3 col2)'], '3×3 float matrix, column-major. mat3(1.0) is the identity. Index as m[column][row].', T],
  ['mat4', ['mat4(float diagonal)', 'mat4(vec4 c0, vec4 c1, vec4 c2, vec4 c3)'], '4×4 float matrix, column-major. mat4(1.0) is the identity.', T],
  ['sampler2D', 'uniform sampler2D name', 'An opaque handle to a 2D texture. Only usable as a texture-function argument; cannot be assigned or constructed.', T],
  ['samplerCube', 'uniform samplerCube name', 'An opaque handle to a cubemap. Sample it with a direction: texture(s, normalize(dir)).', T],
  ['void', 'void', 'No value: the return type of a function with nothing to return, such as mainImage.', T],

  // ── Keywords and qualifiers ────────────────────────────────────────
  ['const', 'const float NAME = 1.0;', 'A compile-time constant. Also marks a function parameter as read-only.', K],
  ['uniform', 'uniform float name;', 'A value set from outside the shader, the same for every pixel. Only the editor\'s built-in uniforms receive values here.', K],
  ['in', 'in vec2 p', 'Function parameter passed by value (the default). At file scope, an input from the previous stage.', K],
  ['out', 'out vec4 result', 'Function parameter written back to the caller, like mainImage\'s fragColor. Its value on entry is undefined.', K],
  ['inout', 'inout vec3 p', 'Function parameter that is both read on entry and written back on return.', K],
  ['discard', 'discard;', 'Abandons the fragment: nothing is written. In a feedback buffer the old texel survives.', K],
  ['precision', 'precision highp float;', 'Default precision for a type. The editor already declares highp for floats, ints and samplers.', K],
  ['highp', 'highp float x', 'Highest precision: 32-bit float. The editor\'s default for everything.', K],
  ['mediump', 'mediump float x', 'At least 16-bit float. Can be faster on mobile GPUs, with visible banding in big values.', K],
  ['lowp', 'lowp float x', 'At least 9-bit fixed point: enough for colours in [0, 1], not for coordinates.', K],
  ['struct', 'struct Name { float a; vec3 b; };', 'A user-defined aggregate type, constructed as Name(1.0, vec3(0.0)).', K],
  ['return', 'return value;', 'Leaves the current function with an optional value.', K],
  ['if', 'if (cond) { … } else { … }', 'Conditional. Branching on values that differ between nearby pixels can cost performance and breaks derivatives; mix() or step() often does instead.', K],
  ['else', 'if (cond) { … } else { … }', 'The other branch of an if.', K],
  ['for', 'for (int i = 0; i < N; i++) { … }', 'Counted loop. Keep the bound a constant or uniform where possible; the compiler may unroll it.', K],
  ['while', 'while (cond) { … }', 'Loop while the condition holds. Make sure it terminates: a stuck shader hangs the GPU.', K],
  ['do', 'do { … } while (cond);', 'Loop that runs its body at least once.', K],
  ['break', 'break;', 'Leaves the innermost loop or switch.', K],
  ['continue', 'continue;', 'Skips to the next iteration of the innermost loop.', K],
  ['switch', 'switch (i) { case 0: …; break; default: …; }', 'Multi-way branch on an integer.', K],
  ['true', 'bool', 'The boolean true literal.', K],
  ['false', 'bool', 'The boolean false literal.', K],
];

const DOCS = new Map<string, GlslDoc>(
  ENTRIES.map(([name, sig, doc, kind]) => [name, { sig: Array.isArray(sig) ? sig : [sig], doc, kind: kind ?? F }]),
);

/** The reference card for a name, or null if there is none. */
export function lookupGlslDoc(name: string): GlslDoc | null {
  return DOCS.get(name) ?? null;
}
