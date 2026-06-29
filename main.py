"""
CalculatricePro - Backend API v5.0
Speed optimizations: LRU cache, pre-warmed symbols, fast parse
Multi-method solving: Quadratic gets 3 methods (Factoring, Formula, Completing Square)
20 categories with server-side auto-detection
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import re, json, hashlib
from collections import Counter
from functools import lru_cache

import sympy
from sympy import (
    symbols, Eq, expand, simplify, nsimplify, latex, factor,
    Poly, sqrt, Rational, gcd, lcm, diff, integrate, limit,
    sin, cos, tan, pi, oo, trigsimp, Abs, Matrix, I,
    re as sym_re, im as sym_im, arg, factorint, binomial, factorial,
    summation, logcombine, log, exp, S, solveset,
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="CalculatricePro API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://calculatricepro.com",
        "https://www.calculatricepro.com",
        "http://calculatricepro.com",
        "http://www.calculatricepro.com",
        "null",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://localhost:8080",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pre-warm symbols (avoids repeated creation overhead)
X = symbols("x")
Y = symbols("y")
Z = symbols("z")
N = symbols("n", positive=True, integer=True)
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# ── In-memory response cache (LRU - fixes Render slow response) ──────────────
_SOLVE_CACHE: Dict[str, Any] = {}
CACHE_MAX = 200  # max entries

def cache_key(expression: str, category: str) -> str:
    return hashlib.md5(f"{expression.strip().lower()}|{category}".encode()).hexdigest()

def cache_get(key: str):
    return _SOLVE_CACHE.get(key)

def cache_set(key: str, value: Any):
    if len(_SOLVE_CACHE) >= CACHE_MAX:
        # Remove oldest entry
        oldest = next(iter(_SOLVE_CACHE))
        del _SOLVE_CACHE[oldest]
    _SOLVE_CACHE[key] = value

# ── Models ────────────────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    expression: str = Field(..., description="Expression à résoudre")
    category: str = Field("auto", description="catégorie (auto = détection automatique)")

class Step(BaseModel):
    description: str   # English wording explanation
    latex: str         # Math in LaTeX

class Method(BaseModel):
    id: str            # "factoring", "quadratic_formula", "completing_square", etc.
    label: str         # "Method 1: Factoring"
    label_fr: str      # "Méthode 1 : Factorisation"
    steps: List[Step]

class Result(BaseModel):
    latex: str
    decimal: Optional[str] = None

class SolveResponse(BaseModel):
    type: str
    input_latex: str
    methods: List[Method]          # multi-method tabs
    results: List[Result]          # final answers
    final_answer: str              # e.g. "x = 2 or x = 3"
    note: Optional[str] = None
    plot: Optional[Dict[str, Any]] = None
    detected_category: Optional[str] = None
    detected_label: Optional[str] = None
    discriminant_info: Optional[str] = None  # "D > 0 → 2 distinct real roots"

# ── Helpers ───────────────────────────────────────────────────────────────────

def pre(raw: str) -> str:
    s = raw.strip()
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("π", "pi").replace("∞", "oo")
    s = re.sub(r"√\(([^)]+)\)", r"sqrt(\1)", s)
    s = re.sub(r"√(\d+(\.\d+)?)", r"sqrt(\1)", s)
    return s.strip()

def pre_imag(raw: str) -> str:
    s = pre(raw)
    s = re.sub(r"(?<![a-zA-Z])i(?![a-zA-Z])", "I", s)
    return s

def safe_parse(expr_str: str, allow_imag: bool = False):
    ld = {"x": X, "y": Y, "z": Z, "pi": pi, "oo": oo, "E": exp(1), "e": exp(1)}
    if allow_imag:
        ld["I"] = I
    try:
        return parse_expr(expr_str, transformations=TRANSFORMATIONS, local_dict=ld)
    except Exception:
        raise HTTPException(400, f"Expression illisible : « {expr_str} »")

def tex(expr) -> str:
    return latex(expr)

def decimal_str(value) -> Optional[str]:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return None

def sample_plot(expr, var=X, lo=-10.0, hi=10.0, n=200):
    try:
        f = sympy.lambdify(var, expr, "math")
    except Exception:
        return None
    pts = []
    step = (hi - lo) / n
    for i in range(n + 1):
        xv = lo + i * step
        try:
            yv = f(xv)
            if isinstance(yv, complex): continue
            if yv != yv or abs(yv) > 1e6: continue
            pts.append([round(xv, 4), round(float(yv), 4)])
        except Exception:
            continue
    if len(pts) < 2:
        return None
    return {"fn": tex(expr), "points": pts}

def single_method(method_id, label, label_fr, steps) -> Method:
    return Method(id=method_id, label=label, label_fr=label_fr, steps=steps)

# ════════════════════════════════════════════════════════════════════════════
#  1. ALGEBRA  ── now returns 3 methods for quadratic
# ════════════════════════════════════════════════════════════════════════════

def solve_algebra(raw: str) -> SolveResponse:
    if raw.count("=") != 1:
        raise HTTPException(400, "An equation must contain exactly one '=' sign.")
    lhs_s, rhs_s = raw.split("=")
    lhs = safe_parse(pre(lhs_s))
    rhs = safe_parse(pre(rhs_s))
    if (lhs.free_symbols | rhs.free_symbols) - {X}:
        raise HTTPException(400, "Only variable x is supported.")
    input_latex = f"{tex(lhs)} = {tex(rhs)}"
    lhs_e, rhs_e = expand(lhs), expand(rhs)
    try:
        pl, pr = Poly(lhs_e, X), Poly(rhs_e, X)
    except Exception:
        raise HTTPException(400, "Cannot interpret as polynomial in x.")
    deg = max(pl.degree(), pr.degree())
    if deg <= 1:
        return _linear_full(pl, pr, input_latex, lhs, rhs)
    elif deg == 2:
        return _quadratic_full(pl, pr, input_latex, lhs, rhs)
    else:
        return solve_polynomial(raw)


def _linear_full(pl, pr, input_latex, lhs, rhs) -> SolveResponse:
    """Linear equation — single method, clear English steps."""
    def coeffs(poly):
        return nsimplify(poly.coeff_monomial(X)), nsimplify(poly.coeff_monomial(1))
    a1, b1 = coeffs(pl); a2, b2 = coeffs(pr)
    a = a1 - a2
    rhs_c = b2 - b1

    steps = [
        Step(description="Write the equation in standard form",
             latex=input_latex),
        Step(description=f"Move all x-terms to the left side and constants to the right",
             latex=f"{tex(a)}x = {tex(rhs_c)}"),
    ]
    if a == 0:
        if rhs_c == 0:
            steps.append(Step(description="Both sides are equal — infinite solutions", latex="0 = 0"))
            return SolveResponse(type="lineaire", input_latex=input_latex,
                methods=[single_method("direct","Method 1: Direct Solving","Méthode 1 : Résolution directe", steps)],
                results=[], final_answer="Infinite solutions", note="Infinité de solutions.")
        steps.append(Step(description="Contradiction — no solution exists", latex=f"0 = {tex(rhs_c)}"))
        return SolveResponse(type="lineaire", input_latex=input_latex,
            methods=[single_method("direct","Method 1: Direct Solving","Méthode 1 : Résolution directe", steps)],
            results=[], final_answer="No solution", note="Aucune solution.")

    sol = nsimplify(rhs_c / a)
    steps.append(Step(description=f"Divide both sides by {tex(a)} to isolate x",
                      latex=f"x = \\frac{{{tex(rhs_c)}}}{{{tex(a)}}} = {tex(sol)}"))
    steps.append(Step(description="Final solution", latex=f"x = {tex(sol)}"))

    m = single_method("direct","Method 1: Direct Solving","Méthode 1 : Résolution directe", steps)
    return SolveResponse(
        type="lineaire", input_latex=input_latex,
        methods=[m],
        results=[Result(latex=tex(sol), decimal=decimal_str(sol))],
        final_answer=f"x = {tex(sol)}",
        plot=sample_plot(expand(lhs - rhs))
    )


def _quadratic_full(pl, pr, input_latex, lhs, rhs) -> SolveResponse:
    """Quadratic — 3 full methods: Factoring, Quadratic Formula, Completing the Square."""
    expr = expand((pl - pr).as_expr())
    p = Poly(expr, X)
    a = nsimplify(p.coeff_monomial(X**2))
    b = nsimplify(p.coeff_monomial(X))
    c = nsimplify(p.coeff_monomial(1))
    D = nsimplify(b**2 - 4*a*c)
    sd = nsimplify(sympy.sqrt(D)) if D >= 0 else None

    # Discriminant info
    if D > 0:
        disc_info = f"D = {tex(D)} > 0 → Two distinct real roots"
    elif D == 0:
        disc_info = f"D = 0 → One repeated real root (double root)"
    else:
        disc_info = f"D = {tex(D)} < 0 → No real roots (complex roots)"

    # Roots
    if D >= 0:
        x1 = nsimplify((-b - sd) / (2*a))
        x2 = nsimplify((-b + sd) / (2*a))
        results = [Result(latex=tex(x1), decimal=decimal_str(x1)),
                   Result(latex=tex(x2), decimal=decimal_str(x2))]
        if x1 == x2:
            final_answer = f"x = {tex(x1)}"
        else:
            final_answer = f"x = {tex(x1)} \\text{{ or }} x = {tex(x2)}"
    else:
        x1 = nsimplify((-b - sympy.sqrt(D)) / (2*a))
        x2 = nsimplify((-b + sympy.sqrt(D)) / (2*a))
        results = [Result(latex=tex(x1)), Result(latex=tex(x2))]
        final_answer = f"x = {tex(x1)} \\text{{ or }} x = {tex(x2)}"

    methods = []

    # ── METHOD 1: FACTORING ─────────────────────────────────────────────────
    m1_steps = [
        Step(description="Write the equation in standard form ax² + bx + c = 0",
             latex=f"{tex(expr)} = 0"),
        Step(description=f"Identify coefficients: a = {tex(a)}, b = {tex(b)}, c = {tex(c)}",
             latex=f"a = {tex(a)},\\; b = {tex(b)},\\; c = {tex(c)}"),
    ]
    try:
        factored = factor(expr)
        factors = sympy.factor_list(expr)
        fac_str = tex(factored)
        # Find two numbers: multiply to a*c, add to b
        prod_val = int(a * c)
        sum_val = int(b)
        found = None
        for i in range(-abs(prod_val)-1, abs(prod_val)+2):
            if i == 0: continue
            if prod_val % i == 0:
                j = prod_val // i
                if i + j == sum_val:
                    found = (i, j)
                    break
        if found:
            m1_steps.append(Step(
                description=f"Find two numbers that multiply to ac = {tex(a)}×{tex(c)} = {prod_val} and add to b = {sum_val}",
                latex=f"\\text{{Numbers: }} {found[0]} \\text{{ and }} {found[1]} \\quad \\because {found[0]} \\times {found[1]} = {prod_val} \\text{{ and }} {found[0]} + {found[1]} = {sum_val}"
            ))
            # Split middle term
            b1, b2 = found
            m1_steps.append(Step(
                description="Split the middle term using the two numbers found",
                latex=f"{tex(a)}x^2 + {b1}x + {b2}x + {tex(c)} = 0"
            ))
            m1_steps.append(Step(
                description="Group terms and factor out common factors from each group",
                latex=f"x(x + {b1}) + {b2//int(a) if int(a)!=1 else b2}(x + {b1}) = 0" if False else f"{tex(factored)} = 0"
            ))
        else:
            m1_steps.append(Step(description="Factor the expression",
                                 latex=f"{fac_str} = 0"))
        m1_steps.append(Step(
            description="Apply Zero Product Property: if A×B = 0, then A = 0 or B = 0",
            latex=f"{fac_str} = 0 \\implies \\text{{each factor}} = 0"
        ))
        m1_steps.append(Step(
            description=f"Set each factor equal to zero and solve",
            latex=f"x = {tex(x1)} \\quad \\text{{or}} \\quad x = {tex(x2)}"
        ))
        methods.append(Method(id="factoring", label="Method 1: Factoring",
                              label_fr="Méthode 1 : Factorisation", steps=m1_steps))
    except Exception:
        pass

    # ── METHOD 2: QUADRATIC FORMULA ─────────────────────────────────────────
    m2_steps = [
        Step(description="Write the equation in standard form ax² + bx + c = 0",
             latex=f"{tex(expr)} = 0"),
        Step(description=f"Identify coefficients: a = {tex(a)}, b = {tex(b)}, c = {tex(c)}",
             latex=f"a = {tex(a)},\\; b = {tex(b)},\\; c = {tex(c)}"),
        Step(description="Write the Quadratic Formula",
             latex=r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"),
        Step(description=f"Substitute the values into the formula",
             latex=f"x = \\frac{{-({tex(b)}) \\pm \\sqrt{{({tex(b)})^2 - 4({tex(a)})({tex(c)})}}}}{{2({tex(a)})}}"),
        Step(description=f"Calculate the Discriminant: D = b² - 4ac",
             latex=f"D = ({tex(b)})^2 - 4({tex(a)})({tex(c)}) = {tex(b**2)} - {tex(4*a*c)} = {tex(D)}"),
        Step(description=disc_info,
             latex=f"\\Delta = {tex(D)} {'> 0' if D > 0 else ('= 0' if D == 0 else '< 0')}"),
    ]
    if D >= 0:
        m2_steps.append(Step(
            description=f"Simplify: √D = √{tex(D)} = {tex(sd)}",
            latex=f"\\sqrt{{D}} = \\sqrt{{{tex(D)}}} = {tex(sd)}"
        ))
        m2_steps.append(Step(
            description="Calculate both roots using + and − of the ± symbol",
            latex=f"x = \\frac{{{tex(-b)} \\pm {tex(sd)}}}{{{tex(2*a)}}}"
        ))
        m2_steps.append(Step(
            description=f"Root 1 (using +): x₁ = ({tex(-b)} + {tex(sd)}) / {tex(2*a)}",
            latex=f"x_1 = \\frac{{{tex(-b)} + {tex(sd)}}}{{{tex(2*a)}}} = {tex(x2)}"
        ))
        if x1 != x2:
            m2_steps.append(Step(
                description=f"Root 2 (using −): x₂ = ({tex(-b)} − {tex(sd)}) / {tex(2*a)}",
                latex=f"x_2 = \\frac{{{tex(-b)} - {tex(sd)}}}{{{tex(2*a)}}} = {tex(x1)}"
            ))
    m2_steps.append(Step(
        description="Final roots",
        latex=final_answer
    ))
    methods.append(Method(id="quadratic_formula", label="Method 2: Quadratic Formula",
                          label_fr="Méthode 2 : Formule Quadratique", steps=m2_steps))

    # ── METHOD 3: COMPLETING THE SQUARE ─────────────────────────────────────
    m3_steps = [
        Step(description="Write the equation in standard form ax² + bx + c = 0",
             latex=f"{tex(expr)} = 0"),
        Step(description=f"If a ≠ 1, divide every term by a = {tex(a)}",
             latex=f"x^2 + \\frac{{{tex(b)}}}{{{tex(a)}}}x + \\frac{{{tex(c)}}}{{{tex(a)}}} = 0" if a != 1 else f"x^2 + ({tex(b)})x + {tex(c)} = 0"),
        Step(description="Move the constant term to the right side",
             latex=f"x^2 + ({tex(b/a)})x = {tex(-c/a)}"),
    ]
    half_b = nsimplify(b / (2*a))
    half_b_sq = nsimplify(half_b**2)
    rhs_complete = nsimplify(-c/a + half_b_sq)
    m3_steps.append(Step(
        description=f"Find the value to complete the square: (b/2a)² = ({tex(b)}/2×{tex(a)})² = ({tex(half_b)})² = {tex(half_b_sq)}",
        latex=f"\\left(\\frac{{b}}{{2a}}\\right)^2 = \\left(\\frac{{{tex(b)}}}{{2 \\cdot {tex(a)}}}\\right)^2 = {tex(half_b_sq)}"
    ))
    m3_steps.append(Step(
        description=f"Add {tex(half_b_sq)} to BOTH sides to maintain equality",
        latex=f"x^2 + ({tex(b/a)})x + {tex(half_b_sq)} = {tex(-c/a)} + {tex(half_b_sq)}"
    ))
    m3_steps.append(Step(
        description="The left side is now a perfect square trinomial",
        latex=f"\\left(x + {tex(half_b)}\\right)^2 = {tex(rhs_complete)}"
    ))
    sqrt_rhs = nsimplify(sympy.sqrt(rhs_complete)) if rhs_complete >= 0 else None
    if sqrt_rhs is not None:
        m3_steps.append(Step(
            description="Take the square root of both sides (remember ±)",
            latex=f"x + {tex(half_b)} = \\pm\\sqrt{{{tex(rhs_complete)}}} = \\pm {tex(sqrt_rhs)}"
        ))
        m3_steps.append(Step(
            description=f"Subtract {tex(half_b)} from both sides to isolate x",
            latex=f"x = -{tex(half_b)} \\pm {tex(sqrt_rhs)}"
        ))
        m3_steps.append(Step(
            description=f"Root 1 (using +): x = {tex(-half_b)} + {tex(sqrt_rhs)} = {tex(nsimplify(-half_b + sqrt_rhs))}",
            latex=f"x_1 = {tex(-half_b)} + {tex(sqrt_rhs)} = {tex(nsimplify(-half_b + sqrt_rhs))}"
        ))
        m3_steps.append(Step(
            description=f"Root 2 (using −): x = {tex(-half_b)} − {tex(sqrt_rhs)} = {tex(nsimplify(-half_b - sqrt_rhs))}",
            latex=f"x_2 = {tex(-half_b)} - {tex(sqrt_rhs)} = {tex(nsimplify(-half_b - sqrt_rhs))}"
        ))
    m3_steps.append(Step(description="Final roots", latex=final_answer))
    methods.append(Method(id="completing_square", label="Method 3: Completing the Square",
                          label_fr="Méthode 3 : Complétion du Carré", steps=m3_steps))

    return SolveResponse(
        type="quadratique",
        input_latex=input_latex,
        methods=methods,
        results=results,
        final_answer=final_answer,
        discriminant_info=disc_info,
        plot=sample_plot(expr)
    )


# ════════════════════════════════════════════════════════════════════════════
#  2. FRACTIONS
# ════════════════════════════════════════════════════════════════════════════

def solve_fraction(raw: str) -> SolveResponse:
    m = re.fullmatch(r"\s*(-?\d+)\s*/\s*(-?\d+)\s*", pre(raw))
    if not m:
        raise HTTPException(400, "Format: numerator/denominator (e.g. 12/18).")
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        raise HTTPException(400, "Division by zero is undefined.")
    input_latex = f"\\frac{{{num}}}{{{den}}}"
    g = sympy.gcd(abs(num), abs(den))
    sign = -1 if (num < 0) ^ (den < 0) else 1
    na, da = abs(num)//g, abs(den)//g
    res = sign * Rational(na, da)

    steps = [
        Step(description="Write the fraction", latex=input_latex),
        Step(description=f"Find the Greatest Common Divisor (GCD) of {abs(num)} and {abs(den)}",
             latex=f"\\text{{GCD}}({abs(num)}, {abs(den)}) = {g}"),
        Step(description=f"Divide both numerator and denominator by {g}",
             latex=f"\\frac{{{num} \\div {g}}}{{{den} \\div {g}}} = \\frac{{{sign*na}}}{{{da}}}"),
        Step(description="Simplified fraction (irreducible)", latex=tex(res)),
    ]
    m1 = single_method("simplify", "Method 1: GCD Simplification",
                        "Méthode 1 : Simplification par PGCD", steps)
    return SolveResponse(type="fraction", input_latex=input_latex, methods=[m1],
        results=[Result(latex=tex(res), decimal=decimal_str(res))],
        final_answer=tex(res))


# ════════════════════════════════════════════════════════════════════════════
#  3. DERIVATIVES
# ════════════════════════════════════════════════════════════════════════════

def solve_derivative(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    cleaned = re.sub(r"^d/dx\s*\(?", "", cleaned, flags=re.IGNORECASE)
    if cleaned.endswith(")") and "d/dx" in raw.lower():
        cleaned = cleaned[:-1]
    expr = safe_parse(cleaned)
    expr = expand(expr)
    input_latex = f"\\frac{{d}}{{dx}}\\left({tex(expr)}\\right)"
    result = simplify(diff(expr, X))

    terms = expr.as_ordered_terms()
    steps = [
        Step(description="Write the function to differentiate", latex=tex(expr)),
    ]
    if len(terms) > 1:
        steps.append(Step(description="Apply the Sum Rule: differentiate term by term", latex=tex(expr)))
    for t in terms:
        dt = diff(t, X)
        if not t.has(X):
            steps.append(Step(description=f"Derivative of constant {tex(t)} = 0", latex=f"\\frac{{d}}{{dx}}[{tex(t)}] = 0"))
        elif t.is_polynomial(X):
            try:
                pt = Poly(t, X)
                pw = pt.degree()
                steps.append(Step(description=f"Apply Power Rule: d/dx[xⁿ] = n·xⁿ⁻¹ for power {pw}",
                    latex=f"\\frac{{d}}{{dx}}[{tex(t)}] = {tex(dt)}"))
            except Exception:
                steps.append(Step(description=f"Differentiate {tex(t)}", latex=f"= {tex(dt)}"))
        else:
            steps.append(Step(description=f"Differentiate {tex(t)} using known rules",
                latex=f"\\frac{{d}}{{dx}}[{tex(t)}] = {tex(dt)}"))
    steps.append(Step(description="Combine all terms — final derivative", latex=f"f'(x) = {tex(result)}"))

    m1 = single_method("differentiation", "Method 1: Differentiation Rules",
                        "Méthode 1 : Règles de Dérivation", steps)
    return SolveResponse(type="derivee", input_latex=input_latex, methods=[m1],
        results=[Result(latex=tex(result))], final_answer=f"f'(x) = {tex(result)}",
        plot=sample_plot(result))


# ════════════════════════════════════════════════════════════════════════════
#  4. INTEGRALS
# ════════════════════════════════════════════════════════════════════════════

def solve_integral(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    cleaned = re.sub(r"^∫\s*", "", cleaned)
    bounds = None
    bm = re.search(r'\[(-?[\d.]+)\s*,\s*(-?[\d.]+)\]', cleaned)
    if bm:
        bounds = (float(bm.group(1)), float(bm.group(2)))
        cleaned = cleaned[:bm.start()].strip()
    expr = safe_parse(cleaned)
    expr = expand(expr)
    input_latex = (f"\\int_{{{bounds[0]}}}^{{{bounds[1]}}} {tex(expr)}\\,dx"
                   if bounds else f"\\int {tex(expr)}\\,dx")
    F = simplify(integrate(expr, X))

    steps = [
        Step(description="Write the integral to evaluate", latex=input_latex),
        Step(description="Apply the Power Rule for Integration: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C",
             latex=f"\\int {tex(expr)}\\,dx"),
    ]
    for t in expr.as_ordered_terms():
        it = integrate(t, X)
        steps.append(Step(description=f"Integrate term: {tex(t)}", latex=f"\\int {tex(t)}\\,dx = {tex(it)}"))
    if bounds:
        a_s, b_s = nsimplify(bounds[0]), nsimplify(bounds[1])
        val = nsimplify(F.subs(X, b_s) - F.subs(X, a_s))
        steps.append(Step(description="Write the antiderivative F(x)", latex=f"F(x) = {tex(F)}"))
        steps.append(Step(description=f"Apply Fundamental Theorem of Calculus: F({bounds[1]}) − F({bounds[0]})",
            latex=f"\\left[{tex(F)}\\right]_{{{bounds[0]}}}^{{{bounds[1]}}} = {tex(F.subs(X, b_s))} - {tex(F.subs(X, a_s))} = {tex(val)}"))
        m1 = single_method("definite", "Method 1: Definite Integration",
                            "Méthode 1 : Intégrale Définie", steps)
        return SolveResponse(type="integrale", input_latex=input_latex, methods=[m1],
            results=[Result(latex=tex(val), decimal=decimal_str(val))],
            final_answer=tex(val), plot=sample_plot(expr))
    steps.append(Step(description="Add the constant of integration C",
                      latex=f"\\int {tex(expr)}\\,dx = {tex(F)} + C"))
    m1 = single_method("indefinite", "Method 1: Indefinite Integration",
                        "Méthode 1 : Intégrale Indéfinie", steps)
    return SolveResponse(type="integrale", input_latex=input_latex, methods=[m1],
        results=[Result(latex=f"{tex(F)} + C")],
        final_answer=f"{tex(F)} + C", plot=sample_plot(expr))


# ════════════════════════════════════════════════════════════════════════════
#  5. STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def solve_statistics(raw: str) -> SolveResponse:
    cleaned = re.sub(r'(moyenne|médiane|mode|statistiques?|données?|de|:)\s*', '', raw, flags=re.IGNORECASE)
    parts = re.split(r'[,;\s]+', cleaned.strip())
    try:
        nums = [float(p.replace(',', '.')) for p in parts if p.strip()]
    except ValueError:
        raise HTTPException(400, "Format: comma-separated numbers (e.g. 4, 7, 13, 2, 1).")
    if len(nums) < 2:
        raise HTTPException(400, "Minimum 2 values required.")
    n = len(nums); total = sum(nums); mean = total / n
    sn = sorted(nums)
    median = (sn[n//2-1] + sn[n//2]) / 2 if n % 2 == 0 else sn[n//2]
    counts = Counter(nums); mx = max(counts.values())
    modes = [k for k, v in counts.items() if v == mx]
    variance = sum((xi - mean)**2 for xi in nums) / n
    std_dev = variance ** 0.5
    data_range = max(nums) - min(nums)
    def fmt(v): return str(int(v)) if v == int(v) else str(v)
    input_latex = "\\{" + ", ".join(fmt(v) for v in nums) + "\\}"
    steps = [
        Step(description=f"Dataset with {n} values", latex=input_latex),
        Step(description="Sort the data in ascending order",
             latex="\\{" + ", ".join(fmt(v) for v in sn) + "\\}"),
        Step(description="Calculate the Mean (Average) = Sum ÷ n",
             latex=f"\\bar{{x}} = \\frac{{{fmt(total)}}}{{{n}}} = {mean:.6g}"),
        Step(description="Find the Median: middle value of sorted data" + (" (average of two middle values)" if n % 2 == 0 else ""),
             latex=f"\\text{{Median}} = {median:.6g}"),
        Step(description="Find the Mode: most frequently occurring value(s)",
             latex=f"\\text{{Mode}} = {', '.join(fmt(m) for m in modes)}"),
        Step(description="Calculate Variance σ² = Σ(xᵢ − x̄)² / n",
             latex=f"\\sigma^2 = {variance:.6g}"),
        Step(description="Calculate Standard Deviation σ = √(Variance)",
             latex=f"\\sigma = {std_dev:.6g}"),
        Step(description=f"Range = Maximum − Minimum = {fmt(max(nums))} − {fmt(min(nums))}",
             latex=f"\\text{{Range}} = {data_range:.6g}"),
    ]
    m1 = single_method("descriptive", "Method 1: Descriptive Statistics",
                        "Méthode 1 : Statistiques Descriptives", steps)
    return SolveResponse(type="statistiques", input_latex=input_latex, methods=[m1],
        results=[Result(latex=f"\\bar{{x}} = {mean:.6g}", decimal=f"{mean:.6g}"),
                 Result(latex=f"\\text{{Median}} = {median:.6g}"),
                 Result(latex=f"\\sigma = {std_dev:.6g}")],
        final_answer=f"Mean = {mean:.6g}, Median = {median:.6g}, σ = {std_dev:.6g}")


# ════════════════════════════════════════════════════════════════════════════
#  6. TRIGONOMETRY
# ════════════════════════════════════════════════════════════════════════════

def solve_trigonometry(raw: str) -> SolveResponse:
    cleaned = pre(raw).replace("°", "*pi/180")
    if "=" in cleaned:
        ls, rs = cleaned.split("=", 1)
        lhs = safe_parse(ls.strip()); rhs = safe_parse(rs.strip())
        input_latex = f"{tex(lhs)} = {tex(rhs)}"
        steps = [
            Step(description="Write the trigonometric equation", latex=input_latex),
            Step(description="Isolate the trigonometric function", latex=input_latex),
        ]
        try:
            sols = sympy.solve(Eq(lhs, rhs), X)
        except Exception:
            sols = []
        if not sols:
            steps.append(Step(description="No simple algebraic solution found", latex="\\text{No simple solution}"))
            m1 = single_method("trig_solve", "Method 1: Trigonometric Solving",
                                "Méthode 1 : Résolution Trigonométrique", steps)
            return SolveResponse(type="trigonometrie", input_latex=input_latex, methods=[m1],
                results=[], final_answer="No simple solution",
                note="Aucune solution algébrique simple trouvée.")
        for s in sols:
            steps.append(Step(description=f"Solution: x = {tex(s)}", latex=f"x = {tex(s)}"))
        m1 = single_method("trig_solve", "Method 1: Trigonometric Solving",
                            "Méthode 1 : Résolution Trigonométrique", steps)
        fa = ", ".join(f"x = {tex(s)}" for s in sols)
        return SolveResponse(type="trigonometrie", input_latex=input_latex, methods=[m1],
            results=[Result(latex=tex(s)) for s in sols], final_answer=fa)
    expr = safe_parse(cleaned)
    input_latex = tex(expr)
    simp = trigsimp(expr)
    steps = [
        Step(description="Write the trigonometric expression", latex=input_latex),
        Step(description="Apply trigonometric identities to simplify", latex=tex(simp)),
    ]
    try:
        nv = float(simp.evalf())
        steps.append(Step(description="Calculate the numerical value", latex=f"\\approx {nv:.6g}"))
        num_val = f"{nv:.6g}"
    except Exception:
        num_val = None
    m1 = single_method("trig_simplify", "Method 1: Simplification",
                        "Méthode 1 : Simplification", steps)
    plot = sample_plot(expr, lo=-6.28, hi=6.28) if expr.has(X) else None
    return SolveResponse(type="trigonometrie", input_latex=input_latex, methods=[m1],
        results=[Result(latex=tex(simp), decimal=num_val)],
        final_answer=tex(simp), plot=plot)


# ════════════════════════════════════════════════════════════════════════════
#  7. MATRICES
# ════════════════════════════════════════════════════════════════════════════

def solve_matrix(raw: str) -> SolveResponse:
    cleaned = raw.strip().replace("(", "[").replace(")", "]")
    cleaned = re.sub(r'\s+', '', cleaned)
    try:
        data = json.loads(cleaned)
        M = Matrix([[nsimplify(v) for v in row] for row in data])
    except Exception:
        raise HTTPException(400, "Format: [[1,2],[3,4]] for a 2×2 matrix.")
    rows, cols = M.shape
    input_latex = tex(M)
    steps = [Step(description=f"Given {rows}×{cols} matrix", latex=input_latex)]
    results = []
    note = None
    if rows == cols:
        d = M.det()
        steps.append(Step(description="Calculate the Determinant det(A)", latex=f"\\det(A) = {tex(d)}"))
        results.append(Result(latex=f"\\det(A) = {tex(d)}", decimal=decimal_str(d)))
        tr = M.trace()
        steps.append(Step(description="Calculate the Trace (sum of diagonal elements)", latex=f"\\text{{tr}}(A) = {tex(tr)}"))
        if d != 0:
            try:
                inv = M.inv()
                steps.append(Step(description="Calculate the Inverse A⁻¹ using the adjugate method",
                    latex=f"A^{{-1}} = {tex(inv)}"))
                results.append(Result(latex=f"A^{{-1}} = {tex(inv)}"))
            except Exception:
                pass
        else:
            steps.append(Step(description="Matrix is singular (det = 0) — inverse does not exist", latex="\\det(A) = 0"))
        if rows <= 3:
            try:
                evs = M.eigenvals()
                ev_str = ", ".join(f"{tex(k)}" for k in evs)
                steps.append(Step(description="Find Eigenvalues by solving det(A − λI) = 0",
                    latex=f"\\lambda \\in \\{{{ev_str}\\}}"))
            except Exception:
                pass
    steps.append(Step(description="Calculate the Transpose Aᵀ", latex=f"A^T = {tex(M.T)}"))
    try:
        rk = M.rank()
        steps.append(Step(description="Determine the Rank of the matrix", latex=f"\\text{{rank}}(A) = {rk}"))
    except Exception:
        pass
    note = None if rows == cols else "Determinant and inverse are only available for square matrices."
    m1 = single_method("matrix_ops", "Method 1: Matrix Operations",
                        "Méthode 1 : Opérations Matricielles", steps)
    fa = f"det = {tex(results[0].latex.split('=')[1].strip())}" if results else "See steps"
    return SolveResponse(type="matrice", input_latex=input_latex, methods=[m1],
        results=results, final_answer=fa, note=note)


# ════════════════════════════════════════════════════════════════════════════
#  8. BASIC MATH
# ════════════════════════════════════════════════════════════════════════════

def solve_basic(raw: str) -> SolveResponse:
    cl = raw.strip().lower()
    m = re.search(r'(?:pgcd|gcd)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', cl)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        g = sympy.gcd(a, b)
        input_latex = f"\\text{{GCD}}({a},{b})"
        aa, bb = abs(a), abs(b)
        steps = [Step(description=f"Find GCD({a},{b}) using the Euclidean Algorithm", latex=input_latex)]
        temp_a, temp_b = aa, bb
        while temp_b:
            q, r = divmod(temp_a, temp_b)
            steps.append(Step(description=f"{temp_a} = {q} × {temp_b} + {r} (divide and find remainder)",
                               latex=f"{temp_a} = {q} \\times {temp_b} + {r}"))
            temp_a, temp_b = temp_b, r
        steps.append(Step(description=f"When remainder = 0, the last divisor is the GCD", latex=f"\\text{{GCD}} = {g}"))
        m1 = single_method("gcd", "Method 1: Euclidean Algorithm",
                            "Méthode 1 : Algorithme d'Euclide", steps)
        return SolveResponse(type="basique", input_latex=input_latex, methods=[m1],
            results=[Result(latex=str(g))], final_answer=f"GCD = {g}")
    m = re.search(r'(?:ppcm|lcm)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', cl)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        g = sympy.gcd(abs(a), abs(b)); l = sympy.lcm(a, b)
        input_latex = f"\\text{{LCM}}({a},{b})"
        steps = [
            Step(description=f"Find LCM({a},{b}) using the relation: LCM = |a×b| / GCD",
                 latex=input_latex),
            Step(description=f"First find GCD({a},{b}) = {g}", latex=f"\\text{{GCD}}({a},{b}) = {g}"),
            Step(description="Apply formula: LCM = |a × b| / GCD",
                 latex=f"\\text{{LCM}} = \\frac{{|{a} \\times {b}|}}{{{g}}} = {l}"),
        ]
        m1 = single_method("lcm", "Method 1: LCM Formula",
                            "Méthode 1 : Formule PPCM", steps)
        return SolveResponse(type="basique", input_latex=input_latex, methods=[m1],
            results=[Result(latex=str(l))], final_answer=f"LCM = {l}")
    m = re.search(r'(-?[\d.]+)\s*%\s*(?:de|of)?\s*(-?[\d.]+)', cl)
    if m:
        pct, total = float(m.group(1)), float(m.group(2))
        res = pct * total / 100
        input_latex = f"{pct}\\%\\ \\text{{of}}\\ {total}"
        steps = [
            Step(description=f"Calculate {pct}% of {total}", latex=input_latex),
            Step(description="Formula: Percentage × Total / 100",
                 latex=f"\\frac{{{pct} \\times {total}}}{{100}}"),
            Step(description="Calculate the result", latex=f"= {res:.6g}"),
        ]
        m1 = single_method("percentage", "Method 1: Percentage Calculation",
                            "Méthode 1 : Calcul de Pourcentage", steps)
        return SolveResponse(type="basique", input_latex=input_latex, methods=[m1],
            results=[Result(latex=f"{res:.6g}")], final_answer=f"{res:.6g}")
    try:
        expr = safe_parse(pre(raw))
        result = nsimplify(expr)
        input_latex = tex(expr)
        steps = [
            Step(description="Write the arithmetic expression", latex=input_latex),
            Step(description="Apply order of operations (PEMDAS/BODMAS) to evaluate",
                 latex=tex(result)),
        ]
        m1 = single_method("arithmetic", "Method 1: Arithmetic Evaluation",
                            "Méthode 1 : Calcul Arithmétique", steps)
        return SolveResponse(type="basique", input_latex=input_latex, methods=[m1],
            results=[Result(latex=tex(result), decimal=decimal_str(result))],
            final_answer=tex(result))
    except Exception:
        raise HTTPException(400, "Try: 20% of 150, gcd(12,18), lcm(4,6) or an expression like 3+4×5.")


# ════════════════════════════════════════════════════════════════════════════
#  9. SYSTEMS OF EQUATIONS
# ════════════════════════════════════════════════════════════════════════════

def solve_system(raw: str) -> SolveResponse:
    parts = re.split(r'[;\n]| et ', raw.strip())
    parts = [p.strip() for p in parts if p.strip() and "=" in p]
    if len(parts) < 2:
        raise HTTPException(400, "Enter at least 2 equations separated by ';' (e.g. 2x+y=5 ; x-y=1).")
    eqs = []; all_syms = set()
    for p in parts:
        l, r = p.split("=", 1)
        le = safe_parse(pre(l)); re_ = safe_parse(pre(r))
        eqs.append(Eq(le, re_))
        all_syms |= (le.free_symbols | re_.free_symbols)
    syms = sorted(all_syms, key=lambda s: s.name)
    input_latex = "\\begin{cases}" + " \\\\ ".join(f"{tex(e.lhs)} = {tex(e.rhs)}" for e in eqs) + "\\end{cases}"
    try:
        sol = sympy.solve(eqs, syms, dict=True)
    except Exception:
        raise HTTPException(400, "Cannot solve this system.")
    if not sol:
        steps = [Step(description="System of equations", latex=input_latex),
                 Step(description="No solution found (inconsistent or infinite solutions)", latex="\\varnothing")]
        m1 = single_method("substitution", "Method 1: Substitution",
                            "Méthode 1 : Substitution", steps)
        return SolveResponse(type="systeme", input_latex=input_latex, methods=[m1],
            results=[], final_answer="No solution",
            note="No solution or infinite solutions.")
    steps = [
        Step(description=f"System of {len(eqs)} equations with {len(syms)} unknowns", latex=input_latex),
        Step(description="Solve the system using elimination/substitution", latex=input_latex),
    ]
    results = []
    for sd in sol:
        for s in syms:
            if s in sd:
                val = nsimplify(sd[s])
                steps.append(Step(description=f"Value of {s.name}", latex=f"{tex(s)} = {tex(val)}"))
                results.append(Result(latex=f"{tex(s)} = {tex(val)}", decimal=decimal_str(val)))
    fa = ", ".join(r.latex for r in results)
    m1 = single_method("substitution", "Method 1: Substitution / Elimination",
                        "Méthode 1 : Substitution / Élimination", steps)
    return SolveResponse(type="systeme", input_latex=input_latex, methods=[m1],
        results=results, final_answer=fa)


# ════════════════════════════════════════════════════════════════════════════
#  10. INEQUALITIES
# ════════════════════════════════════════════════════════════════════════════

def solve_inequality(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    mrel = re.search(r'(<=|>=|<|>)', cleaned)
    if not mrel:
        raise HTTPException(400, "An inequality must contain <, >, <= or >= (e.g. 2x+5>11).")
    op = mrel.group(1)
    l, r = cleaned.split(op, 1)
    lhs = safe_parse(l.strip()); rhs = safe_parse(r.strip())
    rel = {"<": lhs < rhs, ">": lhs > rhs, "<=": lhs <= rhs, ">=": lhs >= rhs}[op]
    input_latex = tex(rel)
    moved = expand(lhs - rhs)
    op_tex = op.replace("<=","\\leq").replace(">=","\\geq")
    steps = [
        Step(description="Write the inequality", latex=input_latex),
        Step(description="Move all terms to the left side (compare to 0)",
             latex=f"{tex(moved)} \\; {op_tex} \\; 0"),
    ]
    try:
        sol = sympy.reduce_inequalities([rel], X)
    except NotImplementedError:
        steps.append(Step(description="No simple algebraic solution available",
                           latex="\\text{No simple solution}"))
        m1 = single_method("inequality_solve", "Method 1: Algebraic Solving",
                            "Méthode 1 : Résolution Algébrique", steps)
        return SolveResponse(type="inequation", input_latex=input_latex, methods=[m1],
            results=[], final_answer="See note",
            note="This inequality cannot be solved algebraically in simple form.")
    steps.append(Step(description="Solve and express the solution set", latex=tex(sol)))
    steps.append(Step(description="Solution set (interval notation)", latex=tex(sol)))
    m1 = single_method("inequality_solve", "Method 1: Algebraic Solving",
                        "Méthode 1 : Résolution Algébrique", steps)
    return SolveResponse(type="inequation", input_latex=input_latex, methods=[m1],
        results=[Result(latex=tex(sol))], final_answer=tex(sol),
        plot=sample_plot(moved))


# ════════════════════════════════════════════════════════════════════════════
#  11-20: remaining solvers (factor, expand, limit, log, exp, complex,
#         sequence, probability, polynomial, primefactor)
# ════════════════════════════════════════════════════════════════════════════

def solve_factor(raw: str) -> SolveResponse:
    expr = safe_parse(pre(raw))
    input_latex = tex(expr)
    expanded = expand(expr)
    factored = factor(expr)
    steps = [
        Step(description="Write the expression to factorize", latex=input_latex),
        Step(description="Expand to standard form (if needed)", latex=tex(expanded)),
        Step(description="Factor by finding common factors and using special identities",
             latex=f"{tex(factored)}"),
    ]
    if expr.free_symbols == {X}:
        try:
            roots = sympy.roots(Poly(expanded, X))
            if roots:
                rstr = ", ".join(f"x = {tex(k)}" for k in roots)
                steps.append(Step(description="Zeros of the expression (roots)", latex=rstr))
        except Exception:
            pass
    m1 = single_method("factoring", "Method 1: Factorization",
                        "Méthode 1 : Factorisation", steps)
    return SolveResponse(type="factorisation", input_latex=input_latex, methods=[m1],
        results=[Result(latex=tex(factored))], final_answer=tex(factored))


def solve_expand(raw: str) -> SolveResponse:
    expr = safe_parse(pre(raw))
    input_latex = tex(expr)
    expanded = expand(expr)
    steps = [
        Step(description="Write the expression to expand", latex=input_latex),
        Step(description="Apply distributive property and FOIL method", latex=tex(expr)),
        Step(description="Collect and simplify like terms", latex=tex(expanded)),
    ]
    if expr.free_symbols == {X}:
        try:
            p = Poly(expanded, X)
            steps.append(Step(description=f"Result is a degree-{p.degree()} polynomial",
                               latex=f"\\deg = {p.degree()}"))
        except Exception:
            pass
    m1 = single_method("expansion", "Method 1: Expansion",
                        "Méthode 1 : Développement", steps)
    return SolveResponse(type="developpement", input_latex=input_latex, methods=[m1],
        results=[Result(latex=tex(expanded))], final_answer=tex(expanded))


def solve_limit(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    cleaned = re.sub(r'^lim\s*', '', cleaned, flags=re.IGNORECASE)
    m = re.search(r'x\s*(?:->|→|tend vers)\s*(-?oo|-?infinity|-?inf|-?[\d.]+(?:/[\d.]+)?|pi|-pi)', cleaned, re.IGNORECASE)
    point = 0; direction = '+-'
    if m:
        ptxt = m.group(1).lower().replace('infinity', 'oo').replace('inf', 'oo')
        cleaned = cleaned[:m.start()].rstrip(' ,;quand').strip()
        if 'oo' in ptxt:
            point = -oo if ptxt.startswith('-') else oo
            direction = '+' if point == -oo else '-'
        else:
            point = safe_parse(ptxt)
    expr = safe_parse(cleaned)
    pt_latex = tex(point)
    input_latex = f"\\lim_{{x \\to {pt_latex}}} {tex(expr)}"
    try:
        L = limit(expr, X, point, direction) if point in (oo, -oo) else limit(expr, X, point)
    except Exception:
        raise HTTPException(400, "Cannot compute this limit.")
    steps = [
        Step(description=f"Evaluate the limit as x approaches {pt_latex}", latex=input_latex),
        Step(description="Check for indeterminate form (0/0, ∞/∞, etc.)", latex=input_latex),
        Step(description="Apply limit rules (L'Hôpital, factoring, substitution)",
             latex=f"\\lim_{{x \\to {pt_latex}}} {tex(expr)}"),
        Step(description="Final limit value", latex=f"= {tex(L)}"),
    ]
    m1 = single_method("limit_eval", "Method 1: Limit Evaluation",
                        "Méthode 1 : Calcul de Limite", steps)
    return SolveResponse(type="limite", input_latex=input_latex, methods=[m1],
        results=[Result(latex=tex(L), decimal=decimal_str(L) if L.is_finite else None)],
        final_answer=tex(L), plot=sample_plot(expr))


def solve_logarithm(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    if "=" in cleaned:
        l, r = cleaned.split("=", 1)
        lhs = safe_parse(l.strip()); rhs = safe_parse(r.strip())
        input_latex = f"{tex(lhs)} = {tex(rhs)}"
        combined = logcombine(lhs - rhs, force=True)
        steps = [
            Step(description="Write the logarithmic equation", latex=input_latex),
            Step(description="Combine logarithms using log rules: log(a)+log(b)=log(a×b)",
                 latex=f"{tex(combined)} = 0"),
        ]
        try:
            sols = sympy.solve(Eq(lhs, rhs), X)
        except Exception:
            sols = []
        results = []
        for s in sols:
            steps.append(Step(description=f"Exponentiate both sides to remove logarithm, solve for x",
                               latex=f"x = {tex(s)}"))
            results.append(Result(latex=tex(s), decimal=decimal_str(s.evalf()) if s.is_number else None))
        fa = ", ".join(f"x = {tex(s)}" for s in sols) if sols else "No solution"
        m1 = single_method("log_solve", "Method 1: Logarithmic Solving",
                            "Méthode 1 : Résolution Logarithmique", steps)
        return SolveResponse(type="logarithme", input_latex=input_latex, methods=[m1],
            results=results, final_answer=fa)
    expr = safe_parse(cleaned)
    simp = logcombine(simplify(expr), force=True)
    steps = [
        Step(description="Write the logarithmic expression", latex=tex(expr)),
        Step(description="Apply logarithm rules to simplify", latex=tex(simp)),
    ]
    m1 = single_method("log_simplify", "Method 1: Log Simplification",
                        "Méthode 1 : Simplification Log", steps)
    return SolveResponse(type="logarithme", input_latex=tex(expr), methods=[m1],
        results=[Result(latex=tex(simp))], final_answer=tex(simp))


def solve_exponential(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    if "=" not in cleaned:
        raise HTTPException(400, "An exponential equation must contain '=' (e.g. 2^x=8).")
    l, r = cleaned.split("=", 1)
    lhs = safe_parse(l.strip()); rhs = safe_parse(r.strip())
    input_latex = f"{tex(lhs)} = {tex(rhs)}"
    steps = [
        Step(description="Write the exponential equation", latex=input_latex),
        Step(description="Take logarithm of both sides to bring down the exponent",
             latex=f"\\ln({tex(lhs)}) = \\ln({tex(rhs)})"),
    ]
    try:
        sols = sympy.solve(Eq(lhs, rhs), X)
    except Exception:
        sols = []
    sols = [s for s in sols if s.is_real] or sols
    results = []
    for s in sols:
        ss = nsimplify(s)
        steps.append(Step(description="Solve for x", latex=f"x = {tex(ss)}"))
        results.append(Result(latex=tex(ss), decimal=decimal_str(s.evalf()) if s.is_number else None))
    fa = ", ".join(f"x = {tex(nsimplify(s))}" for s in sols) if sols else "No real solution"
    m1 = single_method("exp_solve", "Method 1: Exponential Solving",
                        "Méthode 1 : Résolution Exponentielle", steps)
    return SolveResponse(type="exponentielle", input_latex=input_latex, methods=[m1],
        results=results, final_answer=fa)


def solve_complex(raw: str) -> SolveResponse:
    cleaned = pre_imag(raw)
    expr = safe_parse(cleaned, allow_imag=True)
    if expr.free_symbols:
        raise HTTPException(400, "Enter a numerical complex expression (e.g. (3+4i)*(1-2i)).")
    z = simplify(expr)
    re_p = simplify(sym_re(z)); im_p = simplify(sym_im(z))
    mod = simplify(Abs(z)); a = arg(z)
    input_latex = tex(expr)
    steps = [
        Step(description="Write the complex expression", latex=input_latex),
        Step(description="Expand and simplify using i² = −1",
             latex=f"{tex(re_p)} + {tex(im_p)}i" if im_p >= 0 else f"{tex(re_p)} - {tex(-im_p)}i"),
        Step(description="Algebraic form: a + bi",
             latex=f"a = {tex(re_p)}, \\quad b = {tex(im_p)}"),
        Step(description="Modulus |z| = √(a² + b²)",
             latex=f"|z| = \\sqrt{{{tex(re_p)}^2 + {tex(im_p)}^2}} = {tex(mod)}"),
        Step(description="Argument arg(z) = arctan(b/a)",
             latex=f"\\arg(z) = {tex(a)}"),
        Step(description="Conjugate z̄ = a − bi",
             latex=f"\\bar{{z}} = {tex(re_p)} - ({tex(im_p)})i"),
    ]
    res_latex = f"{tex(re_p)} + {tex(im_p)}i" if im_p >= 0 else f"{tex(re_p)} - {tex(-im_p)}i"
    m1 = single_method("complex_ops", "Method 1: Complex Number Operations",
                        "Méthode 1 : Opérations sur Nombres Complexes", steps)
    return SolveResponse(type="complexe", input_latex=input_latex, methods=[m1],
        results=[Result(latex=res_latex), Result(latex=f"|z| = {tex(mod)}", decimal=decimal_str(mod))],
        final_answer=res_latex)


def solve_sequence(raw: str) -> SolveResponse:
    parts = re.split(r'[,;\s]+', raw.strip())
    try:
        terms = [Rational(p.replace(',', '.')) for p in parts if p.strip()]
    except Exception:
        raise HTTPException(400, "Enter the first terms separated by commas (e.g. 2, 5, 8, 11).")
    if len(terms) < 3:
        raise HTTPException(400, "Minimum 3 terms required to identify the sequence.")
    input_latex = ", ".join(tex(t) for t in terms)
    diffs = [terms[i+1] - terms[i] for i in range(len(terms)-1)]
    if len(set(diffs)) == 1:
        r = diffs[0]; u1 = terms[0]
        un = u1 + (N - 1) * r
        Sn = summation(u1 + (N-1)*r, (N, 1, len(terms)))
        steps = [
            Step(description=f"Given sequence with {len(terms)} terms", latex=input_latex),
            Step(description="Check if differences between consecutive terms are constant",
                 latex=f"d = {tex(r)} \\quad (\\text{{constant}})"),
            Step(description="Confirmed: Arithmetic Sequence with common difference d",
                 latex=f"d = {tex(r)}"),
            Step(description="General term formula: uₙ = u₁ + (n−1)×d",
                 latex=f"u_n = {tex(u1)} + (n-1) \\cdot {tex(r)} = {tex(expand(un))}"),
            Step(description=f"Sum of first {len(terms)} terms: Sₙ = n/2 × (2u₁ + (n−1)d)",
                 latex=f"S_{{{len(terms)}}} = {tex(Sn)}"),
        ]
        m1 = single_method("arithmetic", "Method 1: Arithmetic Sequence",
                            "Méthode 1 : Suite Arithmétique", steps)
        return SolveResponse(type="suite", input_latex=input_latex, methods=[m1],
            results=[Result(latex=f"u_n = {tex(expand(un))}"),
                     Result(latex=f"d = {tex(r)}", decimal=decimal_str(r))],
            final_answer=f"uₙ = {tex(expand(un))}, d = {tex(r)}")
    if all(terms[i] != 0 for i in range(len(terms)-1)):
        ratios = [simplify(terms[i+1] / terms[i]) for i in range(len(terms)-1)]
        if len(set(ratios)) == 1:
            q = ratios[0]; u1 = terms[0]
            un = u1 * q**(N - 1)
            steps = [
                Step(description=f"Given sequence with {len(terms)} terms", latex=input_latex),
                Step(description="Check if ratios between consecutive terms are constant",
                     latex=f"q = {tex(q)} \\quad (\\text{{constant}})"),
                Step(description="Confirmed: Geometric Sequence with common ratio q",
                     latex=f"q = {tex(q)}"),
                Step(description="General term formula: uₙ = u₁ × qⁿ⁻¹",
                     latex=f"u_n = {tex(u1)} \\cdot ({tex(q)})^{{n-1}}"),
            ]
            m1 = single_method("geometric", "Method 1: Geometric Sequence",
                                "Méthode 1 : Suite Géométrique", steps)
            return SolveResponse(type="suite", input_latex=input_latex, methods=[m1],
                results=[Result(latex=f"u_n = {tex(u1)} \\cdot ({tex(q)})^{{n-1}}"),
                         Result(latex=f"q = {tex(q)}", decimal=decimal_str(q))],
                final_answer=f"uₙ = {tex(u1)}·({tex(q)})ⁿ⁻¹")
    m1 = single_method("sequence_id", "Method 1: Sequence Identification",
                        "Méthode 1 : Identification de Suite",
                        [Step(description="The sequence is neither arithmetic nor geometric",
                               latex="\\text{No pattern found}")])
    return SolveResponse(type="suite", input_latex=input_latex, methods=[m1],
        results=[], final_answer="Pattern not identified",
        note="The sequence is neither arithmetic nor geometric.")


def solve_probability(raw: str) -> SolveResponse:
    cl = raw.strip().lower().replace(" ", "")
    m = re.search(r'c\(?(\d+),(\d+)\)?', cl) or re.search(r'(\d+)c(\d+)', cl)
    if m:
        n_, k_ = int(m.group(1)), int(m.group(2))
        if k_ > n_: raise HTTPException(400, "k cannot exceed n.")
        val = binomial(n_, k_)
        input_latex = f"\\binom{{{n_}}}{{{k_}}}"
        steps = [
            Step(description=f"Calculate Combinations: choose {k_} from {n_}", latex=input_latex),
            Step(description="Formula: C(n,k) = n! / (k! × (n−k)!)",
                 latex=f"C({n_},{k_}) = \\frac{{{n_}!}}{{{k_}! \\times ({n_}-{k_})!}}"),
            Step(description="Expand the factorials and simplify",
                 latex=f"= \\frac{{{n_}!}}{{{k_}! \\times {n_-k_}!}}"),
            Step(description="Final result", latex=f"= {val}"),
        ]
        m1 = single_method("combination", "Method 1: Combinations",
                            "Méthode 1 : Combinaisons", steps)
        return SolveResponse(type="probabilite", input_latex=input_latex, methods=[m1],
            results=[Result(latex=str(val))], final_answer=f"C({n_},{k_}) = {val}")
    m = re.search(r'a\(?(\d+),(\d+)\)?', cl) or re.search(r'(\d+)p(\d+)', cl)
    if m:
        n_, k_ = int(m.group(1)), int(m.group(2))
        if k_ > n_: raise HTTPException(400, "k cannot exceed n.")
        val = factorial(n_) // factorial(n_ - k_)
        input_latex = f"A_{{{n_}}}^{{{k_}}}"
        steps = [
            Step(description=f"Calculate Arrangements (Permutations): arrange {k_} from {n_}", latex=input_latex),
            Step(description="Formula: A(n,k) = n! / (n−k)!",
                 latex=f"A({n_},{k_}) = \\frac{{{n_}!}}{{({n_}-{k_})!}}"),
            Step(description="Final result", latex=f"= {val}"),
        ]
        m1 = single_method("permutation", "Method 1: Permutations",
                            "Méthode 1 : Permutations", steps)
        return SolveResponse(type="probabilite", input_latex=input_latex, methods=[m1],
            results=[Result(latex=str(val))], final_answer=f"A({n_},{k_}) = {val}")
    m = re.fullmatch(r'(\d+)!', cl)
    if m:
        n_ = int(m.group(1))
        if n_ > 1000: raise HTTPException(400, "Maximum 1000! supported.")
        val = factorial(n_)
        steps = [
            Step(description=f"Calculate {n_}! (factorial)", latex=f"{n_}!"),
            Step(description="Factorial: n! = n × (n−1) × (n−2) × … × 2 × 1",
                 latex=f"{n_}! = {val}"),
        ]
        m1 = single_method("factorial", "Method 1: Factorial",
                            "Méthode 1 : Factorielle", steps)
        return SolveResponse(type="probabilite", input_latex=f"{n_}!", methods=[m1],
            results=[Result(latex=str(val))], final_answer=f"{n_}! = {val}")
    raise HTTPException(400, "Try: C(5,2) for combinations, A(5,2) for arrangements, or 5! for factorial.")


def solve_polynomial(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    if "=" in cleaned:
        l, r = cleaned.split("=", 1)
        expr = expand(safe_parse(l.strip()) - safe_parse(r.strip()))
    else:
        expr = expand(safe_parse(cleaned))
    if expr.free_symbols - {X}:
        raise HTTPException(400, "Only variable x is supported.")
    try:
        p = Poly(expr, X)
    except Exception:
        raise HTTPException(400, "Cannot interpret as polynomial in x.")
    input_latex = f"{tex(expr)} = 0"
    factored = factor(expr)
    try:
        roots = sympy.roots(p)
    except Exception:
        roots = {}
    if not roots:
        try:
            sols = list(sympy.solve(expr, X))
        except Exception:
            sols = []
        roots = {s: 1 for s in sols}
    steps = [
        Step(description=f"Write the degree-{p.degree()} polynomial equation", latex=input_latex),
        Step(description="Factor the polynomial completely",
             latex=f"{tex(factored)} = 0"),
    ]
    results = []
    for k, v in roots.items():
        ks = nsimplify(k)
        mult = f"\\;(\\text{{multiplicity }}\\,{v})" if v > 1 else ""
        steps.append(Step(description=f"Root{'(double)' if v > 1 else ''}: x = {tex(ks)}",
                           latex=f"x = {tex(ks)}{mult}"))
        results.append(Result(latex=tex(ks), decimal=decimal_str(k.evalf()) if k.is_number else None))
    fa = " or ".join(f"x = {tex(nsimplify(k))}" for k in roots) if roots else "No roots found"
    m1 = single_method("polynomial_roots", "Method 1: Root Finding",
                        "Méthode 1 : Recherche de Racines", steps)
    return SolveResponse(type="polynome", input_latex=input_latex, methods=[m1],
        results=results, final_answer=fa, plot=sample_plot(expr))


def solve_primefactor(raw: str) -> SolveResponse:
    m = re.search(r'-?\d+', raw)
    if not m: raise HTTPException(400, "Enter a positive integer (e.g. 360).")
    n_ = abs(int(m.group(0)))
    if n_ < 2: raise HTTPException(400, "Enter an integer ≥ 2.")
    if n_ > 10**15: raise HTTPException(400, "Number too large (max 10^15).")
    f = factorint(n_)
    input_latex = str(n_)
    prod = " \\times ".join(f"{p}^{{{e}}}" if e > 1 else f"{p}" for p, e in sorted(f.items()))
    steps = [
        Step(description=f"Find the prime factorization of {n_}", latex=input_latex),
        Step(description="Divide by smallest prime factor repeatedly until reaching 1",
             latex=input_latex),
    ]
    temp = n_
    for prime in sorted(f.keys()):
        while temp % prime == 0:
            steps.append(Step(description=f"Divide {temp} by prime {prime}: {temp} ÷ {prime} = {temp//prime}",
                               latex=f"{temp} = {prime} \\times {temp//prime}"))
            temp //= prime
    steps.append(Step(description="Prime factorization complete",
                       latex=f"{n_} = {prod}"))
    div_count = sympy.divisor_count(n_)
    steps.append(Step(description=f"Number of divisors: d({n_}) = {div_count}",
                       latex=f"d({n_}) = {div_count}"))
    m1 = single_method("prime_factor", "Method 1: Prime Factorization",
                        "Méthode 1 : Décomposition en Facteurs Premiers", steps)
    return SolveResponse(type="facteurs_premiers", input_latex=input_latex, methods=[m1],
        results=[Result(latex=f"{n_} = {prod}")], final_answer=f"{n_} = {prod}")


# ════════════════════════════════════════════════════════════════════════════
#  DISPATCH TABLE
# ════════════════════════════════════════════════════════════════════════════

DISPATCH = {
    "algebra": solve_algebra, "fraction": solve_fraction,
    "derivative": solve_derivative, "integral": solve_integral,
    "statistics": solve_statistics, "trigonometry": solve_trigonometry,
    "matrix": solve_matrix, "basic": solve_basic,
    "system": solve_system, "inequality": solve_inequality,
    "factor": solve_factor, "expand": solve_expand,
    "limit": solve_limit, "logarithm": solve_logarithm,
    "exponential": solve_exponential, "complex": solve_complex,
    "sequence": solve_sequence, "probability": solve_probability,
    "polynomial": solve_polynomial, "primefactor": solve_primefactor,
}

CAT_LABELS = {
    "algebra":"Algèbre","fraction":"Fractions","derivative":"Dérivées",
    "integral":"Intégrales","statistics":"Statistiques","trigonometry":"Trigonométrie",
    "matrix":"Matrices","basic":"Maths de base","system":"Systèmes d'équations",
    "inequality":"Inéquations","factor":"Factorisation","expand":"Développement",
    "limit":"Limites","logarithm":"Logarithmes","exponential":"Exponentielles",
    "complex":"Nombres complexes","sequence":"Suites","probability":"Probabilités",
    "polynomial":"Polynômes","primefactor":"Facteurs premiers",
}

# ════════════════════════════════════════════════════════════════════════════
#  AUTO-DETECTION  (server-side, fixes the state-lock bug)
# ════════════════════════════════════════════════════════════════════════════

def detect_category(raw: str) -> str:
    s = raw.strip(); low = s.lower()
    if re.search(r'\[\s*\[', s) and re.search(r'\]\s*\]', s): return "matrix"
    if re.search(r'\blim\b', low) or re.search(r'(->|→)', s) or 'tend vers' in low: return "limit"
    eq_parts = [p for p in re.split(r'[;\n]| et ', s) if '=' in p]
    if len(eq_parts) >= 2: return "system"
    if re.search(r'(<=|>=|<|>)', s) and not re.search(r'(->|<-)', s): return "inequality"
    if re.search(r'\b[CA]\s*\(\s*\d+\s*,\s*\d+\s*\)', s) or re.fullmatch(r'\s*\d+\s*!\s*', s): return "probability"
    test = re.sub(r'(sin|cos|tan|asin|acos|atan|exp|log|ln|sqrt|pi|abs|arg)', '', low)
    if re.search(r'(?<![a-z])i(?![a-z])', test) and not re.search(r'[a-hj-z]', test): return "complex"
    has_eq = '=' in s
    if has_eq and re.search(r'\b(ln|log)\b', low): return "logarithm"
    if has_eq and re.search(r'(\d+\s*\^\s*x|e\s*\^\s*x|exp\s*\(\s*x)', low): return "exponential"
    if re.search(r'\b(pgcd|gcd|ppcm|lcm)\b', low) or re.search(r'%\s*(de|of)\b', low): return "basic"
    if '∫' in s or (re.search(r'\[\s*-?\d', s) and re.search(r',\s*-?\d+\s*\]', s)): return "integral"
    if re.search(r'\b(sin|cos|tan|asin|acos|atan|sec|csc|cot)\s*\(', low): return "trigonometry"
    if re.fullmatch(r'\s*-?\d+\s*/\s*-?\d+\s*', s): return "fraction"
    if not has_eq and re.fullmatch(r'[\d\s,;.\-/]+', s):
        nums = [p for p in re.split(r'[,;\s]+', s) if p.strip()]
        if len(nums) >= 6: return "statistics"
        if len(nums) >= 3: return "sequence"
    if re.fullmatch(r'\s*\d+\s*', s) and int(s) >= 2: return "primefactor"
    if has_eq: return "algebra"
    if re.search(r'[a-z]', low):
        if re.search(r'\)\s*\(', s): return "expand"
        return "algebra"
    return "basic"


def smart_solve(raw: str) -> SolveResponse:
    primary = detect_category(raw)
    candidates = [primary]
    fallbacks = {"algebra":["polynomial","basic"],"polynomial":["algebra"],
                 "basic":["algebra"],"sequence":["statistics"],"expand":["algebra"]}
    for f in fallbacks.get(primary, []):
        if f not in candidates: candidates.append(f)
    last_err = None
    for cat in candidates:
        fn = DISPATCH.get(cat)
        if not fn: continue
        try:
            resp = fn(raw)
            resp.detected_category = cat
            resp.detected_label = CAT_LABELS.get(cat, cat)
            return resp
        except HTTPException as e:
            last_err = e; continue
        except Exception as e:
            last_err = HTTPException(400, f"Error ({type(e).__name__})"); continue
    if last_err: raise last_err
    raise HTTPException(400, "Cannot recognize this type of problem.")


# ════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status":"ok","version":"5.0.0","categories":len(DISPATCH),"cache_size":len(_SOLVE_CACHE)}

@app.get("/api/health")
def health():
    return {"status":"ok"}

@app.post("/api/detect")
def detect(req: SolveRequest):
    raw = req.expression.strip()
    if not raw: return {"category": None, "label": None}
    cat = detect_category(raw)
    return {"category": cat, "label": CAT_LABELS.get(cat, cat)}

@app.post("/api/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    raw = req.expression.strip()
    if not raw: raise HTTPException(400, "Please enter an expression.")
    if len(raw) > 500: raise HTTPException(400, "Expression too long (max 500 characters).")
    cat = req.category.lower().strip()
    # CACHE CHECK — instant response for repeated queries
    ck = cache_key(raw, cat)
    cached = cache_get(ck)
    if cached: return cached
    # SOLVE
    if cat in ("auto", "", "detect"):
        result = smart_solve(raw)
    else:
        fn = DISPATCH.get(cat, solve_algebra)
        try:
            result = fn(raw)
            result.detected_category = cat
            result.detected_label = CAT_LABELS.get(cat, cat)
        except HTTPException: raise
        except ZeroDivisionError: raise HTTPException(400, "Division by zero detected.")
        except Exception as e: raise HTTPException(400, f"Syntax error. ({type(e).__name__})")
    # CACHE STORE
    cache_set(ck, result)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
