#!/usr/bin/env python3
"""Reference values for the scratch-pad calculators, computed with SymPy.

Prints JSON: a list of {kind, src, expect} where kind is "calc" (the REPL
calculator) or "tex" (the LaTeX evaluator), src is what the user would type,
and expect is a number, a list of rows (matrix), a string, or {"error": true}.

Each case pairs the source with the equivalent SymPy expression (the LaTeX
parser's antlr runtime is not installed); SymPy evaluates to 30 digits.
Consecutive cases of one kind share variable bindings, in order, like lines
in a pad.
"""
import json
import struct
import sys

import numpy as np
import sympy as sp
from sympy import (Matrix, Rational, cos, det, factorial, log, pi, sin, sqrt, tan, E, S,
                   floor, ceiling, atan2, atan, asin, acos, sinh, cosh, tanh, GoldenRatio)

PREC = 30
cases = []


def num(x):
    """A SymPy scalar to a Python float at 30 digits; ±inf/nan as strings for JSON."""
    f = float(sp.N(x, PREC))
    if f != f:
        return "nan"
    if f in (float("inf"), float("-inf")):
        return "inf" if f > 0 else "-inf"
    return f


def calc(src, expr):
    cases.append({"kind": "calc", "src": src, "expect": expr if isinstance(expr, (str, dict)) else num(expr)})


def tex(src, expr):
    if isinstance(expr, Matrix):
        expect = [[num(expr[i, j]) for j in range(expr.cols)] for i in range(expr.rows)]
    elif isinstance(expr, (dict, str)):
        expect = expr
    else:
        expect = num(expr)
    cases.append({"kind": "tex", "src": src, "expect": expect})


ERR = {"error": True}

# ── REPL calculator: arithmetic & precedence ─────────────────────────────────
calc("1 + 2 * 3", 7)
calc("(1 + 2) * 3", 9)
calc("2 ^ 3 ^ 2", 2 ** 9)                       # right-assoc
calc("-2 ^ 2", -4)                              # ^ binds tighter than unary minus
calc("2 ^ -1", Rational(1, 2))
calc("7 % 3", 1)
calc("-7 % 3", 2)                               # math.js: % is mod
calc("mod(-7, 3)", 2)                           # GLSL mod
calc("5!", factorial(5))
calc("3! ^ 2", factorial(3) ** 2)
calc("1e-3 + 2.5e2", Rational(1, 1000) + 250)
calc(".5 * 4", 2)
calc("10 / 4", Rational(5, 2))

# ── literals & bit operators (32-bit unsigned) ───────────────────────────────
calc("0xFF", 255)
calc("0b1010", 10)
calc("0o17", 15)
calc("0xFF & ~0x0F", 0xF0)
calc("0xF0 | 0x0F", 0xFF)
calc("1 << 20", 1 << 20)
calc("0x80000000 >> 31", 1)
calc("~0", -1)                                 # arbitrary width, like Python
calc("u32(~0)", 0xFFFFFFFF)
calc("bxor(0xFF, 0x0F)", 0xF0)
calc("0xFF ^| 0x0F", 0xF0)
calc("1 + 2 << 1", (1 + 2) << 1)                # shift is looser than +
calc("0xF & 0x3 | 0x8", (0xF & 0x3) | 0x8)      # & tighter than |
calc("popcnt(0xF0F0)", 8)
calc("clz(1)", 31)
calc("ctz(8)", 3)
calc("ctz(0)", 32)
calc("rol(0x80000001, 1)", 0x00000003)
calc("ror(0x80000001, 1)", 0xC0000000)
calc("bit(5)", 32)
calc("mask(4)", 15)
calc("mask(32)", 0xFFFFFFFF)
calc("ispow2(64)", 1)
calc("ispow2(96)", 0)
calc("nextpow2(100)", 128)
calc("nextpow2(128)", 128)
calc("ilog2(1000)", floor(log(1000, 2)))
calc("align(13, 8)", 16)
calc("gcd(48, 18)", 6)
calc("u8(300)", 300 & 0xFF)
calc("i32(0xFFFFFFFF)", -1)
calc("u32(-1)", 0xFFFFFFFF)
calc("hex(255)", "0xFF")
calc("hex(255, 8)", "0x000000FF")
calc("bin(5, 8)", "0b00000101")
calc("oct(8)", "0o10")

# ── floats ───────────────────────────────────────────────────────────────────
def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]

def f16(x):
    # numpy rounds to nearest-even and overflows to inf, as a GPU store does;
    # struct's "e" format raises on overflow instead.
    with np.errstate(over="ignore"):
        return num(float(np.float16(x)))

def f16_bits(x):
    with np.errstate(over="ignore"):
        return int(np.float16(x).view(np.uint16))

for v in [0.1, 1.0, 3.14159, -2.5, 65504.0, 1e-5, 123456.789]:
    calc(f"f32({v!r})", f32(v))
    calc(f"bits({v!r})", "0x%08X" % f32_bits(v))
    calc(f"fromBits(0x{f32_bits(v):08X})", f32(v))
for v in [0.1, 1.0, 3.14159, -2.5, 65504.0, 1e-5, 0.333, 1e-8, 70000.0, 0.5, 2049.0, 2047.5, 6.1e-5]:
    calc(f"f16({v!r})", f16(v))
    calc(f"bits16({v!r})", "0x%04X" % f16_bits(v))
    calc(f"fromBits16(0x{f16_bits(v):04X})", f16(v))
calc("ulp(1)", 2.0 ** -23)
calc("ulp(1024)", 2.0 ** -13)
calc("eps(1)", 2.0 ** -23)
calc("epsilon", 2.0 ** -23)
calc("fltmax", 3.4028234663852886e38)
calc("fltmin", 1.1754943508222875e-38)

# ── functions ────────────────────────────────────────────────────────────────
calc("sin(pi/4)", sin(pi / 4))
calc("cos(pi)", cos(pi))
calc("tan(pi/3)", tan(pi / 3))
calc("asin(1)", asin(1))
calc("acos(0.5)", acos(S(1) / 2))
calc("atan(1)", atan(1))
calc("atan2(1, -1)", atan2(1, -1))
calc("sinh(1)", sinh(1))
calc("cosh(1)", cosh(1))
calc("tanh(0.5)", tanh(S(1) / 2))
calc("sqrt(2)", sqrt(2))
calc("cbrt(27)", 3)
calc("inversesqrt(4)", Rational(1, 2))
calc("exp(1)", E)
calc("exp2(10)", 1024)
calc("ln(e)", 1)
calc("log(100)", log(100))
calc("log(100, 10)", 2)
calc("log2(1024)", 10)
calc("log10(1000)", 3)
calc("pow(2, 0.5)", sqrt(2))
calc("hypot(3, 4)", 5)
calc("length(1, 2, 2)", 3)
calc("abs(-3.5)", Rational(7, 2))
calc("sign(-2)", -1)
calc("floor(-1.5)", floor(-S(3) / 2))
calc("ceil(-1.5)", ceiling(-S(3) / 2))
calc("round(2.5)", 3)
calc("trunc(-2.7)", -2)
calc("fract(-1.25)", -S(5) / 4 - floor(-S(5) / 4))
calc("min(3, 1, 2)", 1)
calc("max(3, 1, 2)", 3)
calc("clamp(5, 0, 1)", 1)
calc("sat(-0.5)", 0)
calc("radians(180)", pi)
calc("degrees(pi/2)", 90)
calc("mix(0, 10, 0.25)", Rational(5, 2))
calc("lerp(2, 4, 0.5)", 3)
calc("unlerp(2, 4, 3)", Rational(1, 2))
calc("remap(5, 0, 10, 100, 200)", 150)
calc("step(0.5, 0.4)", 0)
calc("step(0.5, 0.6)", 1)
calc("smoothstep(0, 1, 0.5)", Rational(1, 2))
calc("smoothstep(0, 1, 0.25)", S(1) / 16 * (3 - 2 * S(1) / 4))
calc("smoothstep(0, 1, 2)", 1)
calc("pi", pi)
calc("tau", 2 * pi)
calc("e", E)
calc("phi", GoldenRatio)

# ── colour ───────────────────────────────────────────────────────────────────
def srgb_enc(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

def srgb_dec(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

calc("byte(0.5)", 128)
calc("byte(1.2)", 255)
calc("norm(51)", Rational(1, 5))
calc("srgb(0.5)", srgb_enc(0.5))
calc("srgb(0.002)", srgb_enc(0.002))
calc("linear(0.5)", srgb_dec(0.5))
calc("linear(0.03)", srgb_dec(0.03))
calc("luma(1, 1, 1)", 1)
calc("luma(1, 0, 0)", Rational(2126, 10000))
calc("rgb(0.2, 0.4, 0.8)", "#3366CC")
calc("rgb8(255, 128, 0)", "#FF8000")
calc("pack(1, 0.5, 0)", "0xFF0080FF")
calc("pack(1, 0.5, 0, 0.5)", "0x800080FF")
calc("unpack(0xFF0080FF)", "1, 0.501960784314, 0, 1")

# ── geometry ─────────────────────────────────────────────────────────────────
calc("dist(0, 0, 3, 4)", 5)
calc("dot([1, 2], [3, 4])", 11)
calc("cross([1, 0, 0], [0, 1, 0])", "[[0], [0], [1]]")
calc("aspect(16, 9)", Rational(16, 9))
calc("texel(512)", Rational(1, 512))
calc("mip(1024, 512)", 11)
calc("mip(1, 1)", 1)

# ── variables & errors ───────────────────────────────────────────────────────
calc("x = 2 ^ 10", 1024)
calc("x * 3", 3072)
calc("ans + 1", 3073)
calc("y = x / 4 + ans", 256 + 3073)
calc("x = x + 1", 1025)            # rebinding sees the old value
calc("y - x", 3329 - 1025)
calc("foo(2)", ERR)
calc("2 +", ERR)
calc("1 2", ERR)
calc("hex(1) + 1", 2)             # math.js coerces the string
calc("sin(2 pi)", 0)              # noise below 1e-54 is chopped
calc("sin(2*pi) + 1e-40", 1e-40)  # but f32-scale values survive
calc("0.1 + 0.2", Rational(3, 10))
calc("[1, 2; 3, 4] * [1; 2]", "[[5], [11]]")
calc("det([1, 2; 3, 4])", -2)
calc("1e400", "inf")
calc("sin = 3", ERR)              # can't shadow a function
calc("pi = 3", ERR)
calc("z", ERR)                    # unbound
calc("@", ERR)
calc("w = 7;", 7)                 # G9: a trailing `;` is not a ResultSet
calc("w + 1", 8)
calc("w = 1; w = 2", ERR)         # one statement per line
calc("fract([1.5, 2.25])", "[[0.5], [0.25]]")   # G28: extensions map over vectors
calc("sat([2, -1])", "[[1], [0]]")
calc("fract([1, 2], [3, 4])", ERR)
calc("version", ERR)              # G29: housekeeping exports are not values
calc("config", ERR)
calc("E", E)

# ── LaTeX evaluator: scalars ─────────────────────────────────────────────────
tex(r"\frac{1}{2}", Rational(1, 2))
tex(r"\frac13+\frac27", Rational(1, 3) + Rational(2, 7))   # eeltex drops braces round one digit
tex(r"2^23", 12)
tex(r"\sqrt23", 3 * sqrt(2))
tex(r"\dfrac{3}{4} + \tfrac{1}{4}", 1)
tex(r"\frac{1}{2}\left|\nabla u\right|^{2}", ERR)      # symbolic: not evaluable
tex(r"2x", ERR)                                          # x unbound at this point
tex(r"x = \frac{3}{4}", Rational(3, 4))
tex(r"2x", Rational(3, 2))
tex(r"2x^2 + \sqrt{16} - \sin^2\pi", 2 * Rational(9, 16) + 4)
tex(r"\sqrt[3]{27}", 3)
tex(r"\sqrt{2}\sin\left(\frac{\pi}{4}\right)", 1)
tex(r"\cos^2 \theta + \sin^2 \theta", ERR)             # theta unbound
tex(r"\sin^{-1} 1", asin(1))                            # G31: inverse, not reciprocal
tex(r"\cos^{-1}(0)", acos(0))
tex(r"\tan^{-1} 1", atan(1))
tex(r"\sin^{2} 1", sin(1) ** 2)
tex(r"\sin^{-2} 1", sin(1) ** -2)
tex(r"1.2.3", ERR)                                      # G30: not 1.2 · .3
tex(r"1.5 + 2", Rational(7, 2))
tex(r"\min(3, 2)", 2)                                  # G10: comma-separated arguments
tex(r"\max{1, 5}", 5)
tex(r"\gcd(48, 18)", 6)
tex(r"\min\left(2, 7\right)", 2)
tex(r"\max(1, 2, 3) + 1", 4)
tex(r"\min 4", 4)
tex(r"\theta = \frac{\pi}{6}", pi / 6)
tex(r"\cos^2 \theta + \sin^2 \theta", 1)
tex(r"\sin\theta", Rational(1, 2))
tex(r"e^{2}", E ** 2)
tex(r"e^{i \pi}", -1)                                   # complex arithmetic, chopped to real
tex(r"\ln e", 1)
tex(r"\log 1000", 3)
tex(r"\lg 8", 3)
tex(r"\exp 0", 1)
tex(r"\pi^2", pi ** 2)
tex(r"\tau", 2 * pi)
tex(r"\phi", GoldenRatio)
tex(r"|{-3}|", 3)
tex(r"\left| -3 \right|", 3)
tex(r"(1+2)(3+4)", 21)
tex(r"\left( \frac{1}{3} \right)^{-2}", 9)
tex(r"2 \cdot 3 \times 4", 24)
tex(r"10 \div 4", Rational(5, 2))
tex(r"\frac{\pi}{4} =", pi / 4)
tex(r"5!", factorial(5))
tex(r"-2^2", -4)
tex(r"2^{3^2}", 2 ** 9)
tex(r"\arctan 1", pi / 4)
tex(r"\log_2 8", 3)
tex(r"\log_{10} 1000", 3)
tex(r"\arcsin 1 + \arccos 0", pi)
tex(r"\tanh 0", 0)
tex(r"\sin 2\pi", 0)
tex(r"\sin 2\pi \cos 0", 0)
tex(r"\cos 2\pi x", cos(2 * pi * Rational(3, 4)))
tex(r"\sin(\pi) 2", 0)
tex(r"\sin\frac{\pi}{2} + 1", 2)
tex(r"\ln e^{2}", 2)
tex(r"x\,x\;x", Rational(27, 64))
tex(r"\frac{x}{x}", 1)

# ── LaTeX evaluator: variable binding ────────────────────────────────────────
tex(r"\alpha_1 = 3", 3)
tex(r"2\alpha_1 e^{1}", 6 * E)
tex(r"\alpha_{12} = 5", 5)
tex(r"\alpha_1 \alpha_{12}", 15)
tex(r"\alpha_1 = \alpha_1 + 1", 4)                      # rebinding sees the old value
tex(r"\alpha_1 \alpha_{12}", 20)
tex(r"y = 2x", Rational(3, 2))
tex(r"y^2 + x", Rational(9, 4) + Rational(3, 4))
tex(r"\mathrm{x}", Rational(3, 4))                      # \mathrm{x} is x
tex(r"\mathrm{rate} = 0.25", Rational(1, 4))             # multi-letter names
tex(r"\text{rate} \cdot 8", 2)
tex(r"\mathrm{rate}_2 = 1", 1)
tex(r"\mathrm{rate} + \mathrm{rate}_2", Rational(5, 4))
tex(r"\mathrm{nope}", ERR)
tex(r"e = 10", ERR)                                     # constants cannot be rebound
tex(r"\pi = 3", ERR)
tex(r"\sin = 3", ERR)
tex(r"x_1 = 1", 1)
tex(r"x_2 = 2", 2)
tex(r"x_1 + x_2 + x", Rational(15, 4))
tex(r"\Delta t = 0.5", ERR)                             # two names, not an assignment
tex(r"\Delta = 0.5", Rational(1, 2))
tex(r"\Delta t", ERR)                                   # t unbound
tex(r"t = 4", 4)
tex(r"\Delta t", 2)
tex(r"a_{n-1} = 5", 5)                                  # G11: a subscript is part of the name
tex(r"a_{n-1} + 1", 6)
tex(r"a_{n+1}", ERR)
tex(r"a_n", ERR)
tex(r"\hat{x}_1 + 1", 2)                               # G32: accent with a subscript
tex(r"\vec{x}", Rational(3, 4))

# ── LaTeX evaluator: matrices ────────────────────────────────────────────────
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[0, 1], [1, 0]])
C = Matrix([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
v = Matrix([[1], [2]])
tex(r"A = \begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}", A)
tex(r"B = \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}", B)
tex(r"C = \begin{pmatrix}2&0&0\\0&3&0\\0&0&4\end{pmatrix}", C)
tex(r"v = \begin{pmatrix}1 \\ 2\end{pmatrix}", v)
tex(r"A^{-1}", A.inv())
tex(r"\det A", det(A))
tex(r"|A|", det(A))
tex(r"\begin{vmatrix}2&0\\0&3\end{vmatrix}", 6)
tex(r"A A^T", A * A.T)
tex(r"A^{T}", A.T)
tex(r"A^\top", A.T)
tex(r"A'", A.T)
tex(r"A^{\top} B", A.T * B)
tex(r"A + B", A + B)
tex(r"A - B", A - B)
tex(r"2A", 2 * A)
tex(r"A \cdot 2", 2 * A)
tex(r"\frac{A}{2}", A / 2)
tex(r"AB", A * B)
tex(r"BA", B * A)
tex(r"A v", A * v)
tex(r"v^T v", v.T * v)
tex(r"A^2", A ** 2)
tex(r"A^{3}", A ** 3)
tex(r"A^{-2}", A.inv() ** 2)
tex(r"A A^{-1}", Matrix.eye(2))
tex(r"C^{-1}", C.inv())
tex(r"\det C", det(C))
tex(r"\tr A", A.trace())
tex(r"\operatorname{tr} C", C.trace())
tex(r"\det (A B)", det(A * B))
tex(r"\det A \det B", det(A) * det(B))
tex(r"(A + B)^{-1}", (A + B).inv())
tex(r"\frac{1}{\det A} A", A / det(A))
tex(r"A + 1", A + Matrix([[1, 1], [1, 1]]))             # scalar broadcasts
tex(r"A v^T", ERR)                                      # 2×2 by 1×2
tex(r"v^{-1}", ERR)                                     # not square
tex(r"\begin{pmatrix}1 & 1 \\ 2 & 2\end{pmatrix}^{-1}", ERR)   # singular
tex(r"\begin{pmatrix}1 & 2 \\ 3\end{pmatrix}", ERR)    # ragged
tex(r"\sin A", ERR)
tex(r"\begin{pmatrix}\sqrt{-1} & 0 \\ 0 & 1\end{pmatrix}", '[["0+1i",0],[0,1]]')   # G12: complex entries pass through
tex(r"\frac{A}{0}", ERR)                                # G12: non-finite entry (not "singular")
tex(r"\begin{pmatrix}\frac{1}{2} & \sqrt{4} \\ x & \pi\end{pmatrix}", Matrix([[Rational(1, 2), 2], [Rational(3, 4), pi]]))
M = Matrix([[4, 7, 2], [3, 6, 1], [2, 5, 3]])
tex(r"M = \begin{bmatrix}4&7&2\\3&6&1\\2&5&3\end{bmatrix}", M)
tex(r"M^{-1}", M.inv())
tex(r"\det M", det(M))
tex(r"M M^{-1}", Matrix.eye(3))
tex(r"M^{-1} M v", ERR)                                 # 3×3 by 2×1
tex(r"\left( M^T \right)^{-1}", M.T.inv())
tex(r"A = 2A", 2 * A)                                   # rebinding a matrix
tex(r"\det A", det(2 * A))

json.dump(cases, sys.stdout)
