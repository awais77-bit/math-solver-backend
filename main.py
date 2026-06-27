"""
CalculatricePro - Backend API v3.0
20 catégories : algèbre, fractions, dérivées, intégrales, statistiques,
trigonométrie, matrices, maths de base, systèmes, inéquations, factorisation,
développement, limites, logarithmes, exponentielles, nombres complexes,
suites, probabilités, polynômes, facteurs premiers.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import re, json
from collections import Counter

import sympy
from sympy import (
    symbols, Eq, expand, simplify, nsimplify, latex, factor,
    Poly, sqrt, Rational, gcd, lcm, diff, integrate, limit,
    sin, cos, tan, pi, oo, trigsimp, Abs, Matrix, I,
    re as sym_re, im as sym_im, arg, factorint, binomial, factorial,
    summation, logcombine, log, exp, S, solveset, Interval, FiniteSet,
)
from sympy.calculus.util import continuous_domain
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="CalculatricePro API", version="3.0.0")

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

X = symbols("x")
N = symbols("n", positive=True, integer=True)
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# ── Models ─────────────────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    expression: str = Field(..., description="Expression à résoudre")
    category: str = Field("algebra", description="catégorie")

class Step(BaseModel):
    description: str
    latex: str

class Result(BaseModel):
    latex: str
    decimal: Optional[str] = None

class SolveResponse(BaseModel):
    type: str
    input_latex: str
    steps: List[Step]
    results: List[Result]
    note: Optional[str] = None
    plot: Optional[Dict[str, Any]] = None   # { "fn": "x^2", "points": [[x,y]...] } pour le graphique

# ── Helpers ────────────────────────────────────────────────────────────────────

def pre(raw: str) -> str:
    s = raw.strip()
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("π", "pi").replace("∞", "oo")
    s = re.sub(r"√\(([^)]+)\)", r"sqrt(\1)", s)
    s = re.sub(r"√(\d+(\.\d+)?)", r"sqrt(\1)", s)
    # virgule décimale entre chiffres -> point (mais pas dans les listes)
    return s.strip()

def pre_imag(raw: str) -> str:
    """Préparation pour nombres complexes : i -> I."""
    s = pre(raw)
    s = re.sub(r"(?<![a-zA-Z])i(?![a-zA-Z])", "I", s)
    return s

def safe_parse(expr_str: str, allow_imag: bool = False):
    ld = {"x": X, "pi": pi, "oo": oo, "E": exp(1), "e": exp(1)}
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

def phrase_side(coeff, label: str) -> str:
    c = nsimplify(coeff)
    if c == 0:
        return ""
    return (f"Soustraire {tex(c)}{label} des deux membres"
            if c > 0 else
            f"Ajouter {tex(-c)}{label} aux deux membres")

def sample_plot(expr, var=X, lo=-10.0, hi=10.0, n=240):
    """Échantillonne une fonction f(var) pour le graphique frontend."""
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
            if isinstance(yv, complex):
                continue
            if yv != yv or abs(yv) > 1e6:   # NaN ou explosion
                continue
            pts.append([round(xv, 4), round(float(yv), 4)])
        except Exception:
            continue
    if len(pts) < 2:
        return None
    return {"fn": tex(expr), "points": pts}

# ════════════════════════════════════════════════════════════════════════════════
#  1. ALGÈBRE
# ════════════════════════════════════════════════════════════════════════════════

def solve_algebra(raw: str) -> SolveResponse:
    if raw.count("=") != 1:
        raise HTTPException(400, "Une équation doit contenir exactement un signe « = ».")
    lhs_s, rhs_s = raw.split("=")
    lhs = safe_parse(pre(lhs_s))
    rhs = safe_parse(pre(rhs_s))
    if (lhs.free_symbols | rhs.free_symbols) - {X}:
        raise HTTPException(400, "Seule la variable x est prise en charge.")
    input_latex = f"{tex(lhs)} = {tex(rhs)}"
    steps = [Step(description="Équation de départ", latex=input_latex)]
    lhs_e, rhs_e = expand(lhs), expand(rhs)
    if lhs_e != lhs or rhs_e != rhs:
        steps.append(Step(description="Développer les deux membres", latex=f"{tex(lhs_e)} = {tex(rhs_e)}"))
    try:
        pl, pr = Poly(lhs_e, X), Poly(rhs_e, X)
    except Exception:
        raise HTTPException(400, "Impossible d'interpréter comme polynôme en x.")
    deg = max(pl.degree(), pr.degree())
    if deg <= 1:
        resp = _linear(pl, pr, steps, input_latex)
    elif deg == 2:
        resp = _quadratic(pl, pr, steps, input_latex)
    else:
        return solve_polynomial(raw)
    # graphique : f(x) = lhs - rhs
    resp.plot = sample_plot(expand(lhs - rhs))
    return resp

def _coeffs(poly):
    return nsimplify(poly.coeff_monomial(X)), nsimplify(poly.coeff_monomial(1))

def _linear(pl, pr, steps, input_latex):
    a1, b1 = _coeffs(pl); a2, b2 = _coeffs(pr)
    a = a1 - a2
    if a2 != 0:
        p = phrase_side(a2, "x")
        if p:
            steps.append(Step(description=p,
                latex=f"{tex(a)}x {'+' if b1 >= 0 else '-'} {tex(abs(b1))} = {tex(b2)}" if b1 != 0 else f"{tex(a)}x = {tex(b2)}"))
    rhs_c = b2 - b1
    if b1 != 0:
        p = phrase_side(b1, "")
        if p:
            steps.append(Step(description=p, latex=f"{tex(a)}x = {tex(rhs_c)}"))
    if a == 0:
        if rhs_c == 0:
            steps.append(Step(description="Simplification", latex="0 = 0"))
            return SolveResponse(type="lineaire", input_latex=input_latex, steps=steps, results=[], note="Infinité de solutions.")
        steps.append(Step(description="Simplification", latex=f"0 = {tex(rhs_c)}"))
        return SolveResponse(type="lineaire", input_latex=input_latex, steps=steps, results=[], note="Aucune solution.")
    if a != 1:
        steps.append(Step(description=f"Diviser les deux membres par {tex(a)}", latex=f"x = \\frac{{{tex(rhs_c)}}}{{{tex(a)}}}"))
    sol = nsimplify(rhs_c / a)
    steps.append(Step(description="Solution", latex=f"x = {tex(sol)}"))
    return SolveResponse(type="lineaire", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(sol), decimal=decimal_str(sol))])

def _quadratic(pl, pr, steps, input_latex):
    expr = expand((pl - pr).as_expr())
    if pr.as_expr() != 0:
        steps.append(Step(description="Forme canonique ax² + bx + c = 0", latex=f"{tex(expr)} = 0"))
    p = Poly(expr, X)
    a = nsimplify(p.coeff_monomial(X**2))
    b = nsimplify(p.coeff_monomial(X))
    c = nsimplify(p.coeff_monomial(1))
    steps.append(Step(description=f"Coefficients : a={tex(a)}, b={tex(b)}, c={tex(c)}", latex=f"a={tex(a)},\\; b={tex(b)},\\; c={tex(c)}"))
    D = nsimplify(b**2 - 4*a*c)
    steps.append(Step(description="Discriminant Δ = b² − 4ac", latex=f"\\Delta = ({tex(b)})^2 - 4({tex(a)})({tex(c)}) = {tex(D)}"))
    if D > 0:
        sd = nsimplify(sympy.sqrt(D))
        steps.append(Step(description="Δ > 0 → deux solutions réelles", latex=f"\\sqrt{{\\Delta}} = {tex(sd)}"))
        x1 = nsimplify((-b - sd) / (2*a)); x2 = nsimplify((-b + sd) / (2*a))
        steps.append(Step(description="Formule quadratique x = (−b ± √Δ) / 2a",
            latex=f"x_1 = {tex(x1)}, \\quad x_2 = {tex(x2)}"))
        return SolveResponse(type="quadratique", input_latex=input_latex, steps=steps,
            results=[Result(latex=tex(x1), decimal=decimal_str(x1)), Result(latex=tex(x2), decimal=decimal_str(x2))])
    elif D == 0:
        x0 = nsimplify(-b / (2*a))
        steps.append(Step(description="Δ = 0 → solution double", latex=f"x = \\frac{{-b}}{{2a}} = {tex(x0)}"))
        return SolveResponse(type="quadratique", input_latex=input_latex, steps=steps,
            results=[Result(latex=tex(x0), decimal=decimal_str(x0))])
    steps.append(Step(description="Δ < 0 → aucune solution réelle", latex=f"\\Delta = {tex(D)} < 0"))
    # solutions complexes
    sd = sympy.sqrt(D)
    x1 = nsimplify((-b - sd) / (2*a)); x2 = nsimplify((-b + sd) / (2*a))
    steps.append(Step(description="Solutions complexes", latex=f"x_1 = {tex(x1)}, \\quad x_2 = {tex(x2)}"))
    return SolveResponse(type="quadratique", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(x1)), Result(latex=tex(x2))],
        note="Aucune solution réelle (solutions complexes affichées).")

# ════════════════════════════════════════════════════════════════════════════════
#  2. FRACTIONS
# ════════════════════════════════════════════════════════════════════════════════

def solve_fraction(raw: str) -> SolveResponse:
    m = re.fullmatch(r"\s*(-?\d+)\s*/\s*(-?\d+)\s*", pre(raw))
    if not m:
        raise HTTPException(400, "Format : numérateur/dénominateur (ex: 12/18).")
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        raise HTTPException(400, "Division par zéro impossible.")
    input_latex = f"\\frac{{{num}}}{{{den}}}"
    steps = [Step(description="Fraction de départ", latex=input_latex)]
    sign = -1 if (num < 0) ^ (den < 0) else 1
    na, da = abs(num), abs(den)
    g = sympy.gcd(na, da)
    steps.append(Step(description=f"PGCD({na}, {da}) = {g}", latex=f"\\text{{PGCD}}({na},{da}) = {g}"))
    if g == 1:
        steps.append(Step(description="Fraction déjà irréductible", latex=input_latex))
        res = Rational(num, den)
    else:
        nn, nd = na // g, da // g
        steps.append(Step(description=f"Diviser num. et dén. par {g}",
            latex=f"\\frac{{{na}\\div{g}}}{{{da}\\div{g}}} = \\frac{{{nn}}}{{{nd}}}"))
        res = sign * Rational(nn, nd)
    return SolveResponse(type="fraction", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(res), decimal=decimal_str(res))])

# ════════════════════════════════════════════════════════════════════════════════
#  3. DÉRIVÉES
# ════════════════════════════════════════════════════════════════════════════════

def solve_derivative(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    cleaned = re.sub(r"^d/dx\s*\(?", "", cleaned, flags=re.IGNORECASE)
    if cleaned.endswith(")") and "d/dx" in raw.lower():
        cleaned = cleaned[:-1]
    expr = safe_parse(cleaned)
    if expr.free_symbols - {X}:
        raise HTTPException(400, "Seule la dérivation par rapport à x est supportée.")
    expr = expand(expr)
    input_latex = f"\\frac{{d}}{{dx}}\\left({tex(expr)}\\right)"
    steps = [Step(description="Expression de départ", latex=tex(expr))]
    terms = expr.as_ordered_terms()
    if len(terms) > 1:
        steps.append(Step(description="Règle de la somme : dériver terme par terme", latex=tex(expr)))
    d_terms = []
    for t in terms:
        dt = diff(t, X); d_terms.append(dt)
        if t.has(X):
            pt = Poly(t, X) if t.is_polynomial(X) else None
            if pt and pt.degree() >= 1:
                pw = pt.degree(); co = pt.LC()
                cd = "" if co == 1 else ("-" if co == -1 else tex(co))
                desc = (f"Règle de puissance : d/dx[{cd}x^{{{pw}}}] = {tex(dt)}"
                        if pw > 1 else f"d/dx[{cd}x] = {tex(dt)}")
            else:
                desc = f"d/dx[{tex(t)}] = {tex(dt)}"
        else:
            desc = f"Constante → dérivée nulle : d/dx[{tex(t)}] = 0"
        steps.append(Step(description=desc, latex=tex(dt)))
    result = simplify(sum(d_terms))
    steps.append(Step(description="Résultat final", latex=f"f'(x) = {tex(result)}"))
    return SolveResponse(type="derivee", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(result))],
        plot=sample_plot(result))

# ════════════════════════════════════════════════════════════════════════════════
#  4. INTÉGRALES
# ════════════════════════════════════════════════════════════════════════════════

def solve_integral(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    cleaned = re.sub(r"^∫\s*", "", cleaned)
    bounds = None
    bm = re.search(r'\[(-?[\d.]+)\s*,\s*(-?[\d.]+)\]', cleaned)
    if bm:
        bounds = (float(bm.group(1)), float(bm.group(2)))
        cleaned = cleaned[:bm.start()].strip()
    expr = safe_parse(cleaned)
    if expr.free_symbols - {X}:
        raise HTTPException(400, "Seule l'intégration par rapport à x est supportée.")
    expr = expand(expr)
    input_latex = (f"\\int_{{{bounds[0]}}}^{{{bounds[1]}}} {tex(expr)}\\,dx"
                   if bounds else f"\\int {tex(expr)}\\,dx")
    steps = [Step(description="Expression à intégrer", latex=tex(expr))]
    terms = expr.as_ordered_terms()
    if len(terms) > 1:
        steps.append(Step(description="Règle de la somme : intégrer terme par terme", latex=tex(expr)))
    int_terms = []
    for t in terms:
        it = integrate(t, X); int_terms.append(it)
        if t.has(X):
            pt = Poly(t, X) if t.is_polynomial(X) else None
            if pt and pt.degree() >= 1:
                pw = pt.degree(); co = pt.LC()
                nc = nsimplify(Rational(co, pw + 1))
                desc = f"Règle inverse : ∫{tex(co)}x^{{{pw}}}dx = {tex(nc)}x^{{{pw+1}}}"
            else:
                desc = f"∫{tex(t)}dx = {tex(it)}"
        else:
            desc = f"Intégrale d'une constante : ∫{tex(t)}dx = {tex(t)}·x"
        steps.append(Step(description=desc, latex=tex(it)))
    F = simplify(sum(int_terms))
    if bounds:
        a_s, b_s = nsimplify(bounds[0]), nsimplify(bounds[1])
        val = nsimplify(F.subs(X, b_s) - F.subs(X, a_s))
        steps.append(Step(description="Primitive F(x)", latex=f"F(x) = {tex(F)} + C"))
        steps.append(Step(description=f"Intégrale définie : F({bounds[1]}) − F({bounds[0]})",
            latex=f"\\left[{tex(F)}\\right]_{{{bounds[0]}}}^{{{bounds[1]}}} = {tex(val)}"))
        return SolveResponse(type="integrale", input_latex=input_latex, steps=steps,
            results=[Result(latex=tex(val), decimal=decimal_str(val))],
            plot=sample_plot(expr))
    steps.append(Step(description="Primitive (+ constante C)", latex=f"{tex(F)} + C"))
    return SolveResponse(type="integrale", input_latex=input_latex, steps=steps,
        results=[Result(latex=f"{tex(F)} + C")],
        plot=sample_plot(expr))

# ════════════════════════════════════════════════════════════════════════════════
#  5. STATISTIQUES
# ════════════════════════════════════════════════════════════════════════════════

def solve_statistics(raw: str) -> SolveResponse:
    cleaned = re.sub(r'(moyenne|médiane|mode|statistiques?|données?|de|:)\s*', '', raw, flags=re.IGNORECASE)
    parts = re.split(r'[,;\s]+', cleaned.strip())
    try:
        nums = [float(p.replace(',', '.')) for p in parts if p.strip()]
    except ValueError:
        raise HTTPException(400, "Format : liste de nombres séparés par des virgules (ex: 4, 7, 13, 2, 1).")
    if len(nums) < 2:
        raise HTTPException(400, "Minimum 2 valeurs requises.")
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
        Step(description=f"Données ({n} valeurs)", latex=input_latex),
        Step(description="Tri croissant", latex="\\{" + ", ".join(fmt(v) for v in sn) + "\\}"),
        Step(description="Moyenne = somme ÷ n",
            latex=f"\\bar{{x}} = \\frac{{{fmt(total)}}}{{{n}}} = {mean:.6g}"),
        Step(description="Médiane : valeur centrale" + (" (moy. des 2 centrales)" if n % 2 == 0 else ""),
            latex=f"\\text{{Médiane}} = {median:.6g}"),
        Step(description="Mode : valeur(s) la(les) plus fréquente(s)",
            latex=f"\\text{{Mode}} = {', '.join(fmt(m) for m in modes)}"),
        Step(description="Variance σ²", latex=f"\\sigma^2 = {variance:.6g}"),
        Step(description="Écart-type σ = √(variance)", latex=f"\\sigma = {std_dev:.6g}"),
        Step(description=f"Étendue = max − min = {fmt(max(nums))} − {fmt(min(nums))}",
            latex=f"\\text{{Étendue}} = {data_range:.6g}"),
    ]
    return SolveResponse(type="statistiques", input_latex=input_latex, steps=steps, results=[
        Result(latex=f"\\bar{{x}} = {mean:.6g}", decimal=f"{mean:.6g}"),
        Result(latex=f"\\text{{Médiane}} = {median:.6g}", decimal=f"{median:.6g}"),
        Result(latex=f"\\sigma = {std_dev:.6g}", decimal=f"{std_dev:.6g}"),
    ])

# ════════════════════════════════════════════════════════════════════════════════
#  6. TRIGONOMÉTRIE
# ════════════════════════════════════════════════════════════════════════════════

def solve_trigonometry(raw: str) -> SolveResponse:
    cleaned = pre(raw).replace("°", "*pi/180")
    if "=" in cleaned:
        ls, rs = cleaned.split("=", 1)
        lhs = safe_parse(ls.strip()); rhs = safe_parse(rs.strip())
        input_latex = f"{tex(lhs)} = {tex(rhs)}"
        steps = [Step(description="Équation trigonométrique", latex=input_latex)]
        try:
            sols = sympy.solve(Eq(lhs, rhs), X)
        except Exception:
            sols = []
        if not sols:
            steps.append(Step(description="Résolution", latex="\\text{Pas de solution algébrique simple}"))
            return SolveResponse(type="trigonometrie", input_latex=input_latex, steps=steps, results=[],
                note="Aucune solution algébrique simple trouvée.")
        steps.append(Step(description=f"{len(sols)} solution(s) trouvée(s)",
            latex=", ".join(tex(s) for s in sols)))
        return SolveResponse(type="trigonometrie", input_latex=input_latex, steps=steps,
            results=[Result(latex=tex(s), decimal=decimal_str(float(s.evalf())) if s.is_number else None) for s in sols])
    expr = safe_parse(cleaned)
    input_latex = tex(expr)
    steps = [Step(description="Expression trigonométrique", latex=input_latex)]
    simp = trigsimp(expr)
    exp_trig = sympy.expand_trig(expr)
    if exp_trig != expr:
        steps.append(Step(description="Développement", latex=tex(exp_trig)))
    steps.append(Step(description="Simplification trigonométrique", latex=tex(simp)))
    num_val = None
    try:
        nv = float(simp.evalf())
        num_val = f"{nv:.6g}"
        steps.append(Step(description="Valeur numérique", latex=f"\\approx {num_val}"))
    except Exception:
        pass
    plot = sample_plot(expr, lo=-6.28, hi=6.28) if expr.has(X) else None
    return SolveResponse(type="trigonometrie", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(simp), decimal=num_val)], plot=plot)

# ════════════════════════════════════════════════════════════════════════════════
#  7. MATRICES
# ════════════════════════════════════════════════════════════════════════════════

def solve_matrix(raw: str) -> SolveResponse:
    cleaned = raw.strip().replace("(", "[").replace(")", "]")
    cleaned = re.sub(r'\s+', '', cleaned)
    try:
        data = json.loads(cleaned)
        M = Matrix([[nsimplify(v) for v in row] for row in data])
    except Exception:
        raise HTTPException(400, "Format : [[1,2],[3,4]] pour une matrice 2×2.")
    rows, cols = M.shape
    input_latex = tex(M)
    steps = [Step(description=f"Matrice {rows}×{cols}", latex=input_latex)]
    results = []
    if rows == cols:
        d = M.det()
        steps.append(Step(description="Déterminant det(A)", latex=f"\\det(A) = {tex(d)}"))
        results.append(Result(latex=f"\\det(A) = {tex(d)}", decimal=decimal_str(d)))
        tr = M.trace()
        steps.append(Step(description="Trace (somme diagonale)", latex=f"\\text{{tr}}(A) = {tex(tr)}"))
        if d != 0:
            try:
                inv = M.inv()
                steps.append(Step(description="Inverse A⁻¹", latex=f"A^{{-1}} = {tex(inv)}"))
                results.append(Result(latex=f"A^{{-1}} = {tex(inv)}"))
            except Exception:
                steps.append(Step(description="Inverse", latex="\\text{Calcul impossible}"))
        else:
            steps.append(Step(description="Matrice singulière (det = 0) — pas d'inverse", latex="\\det(A) = 0"))
        if rows <= 3:
            try:
                evs = M.eigenvals()
                ev_str = ", ".join(f"{tex(k)}" + (f"\\,(\\times{v})" if v > 1 else "") for k, v in evs.items())
                steps.append(Step(description="Valeurs propres (eigenvalues)", latex=f"\\lambda \\in \\{{{ev_str}\\}}"))
            except Exception:
                pass
    steps.append(Step(description="Transposée Aᵀ", latex=f"A^T = {tex(M.T)}"))
    try:
        rk = M.rank()
        steps.append(Step(description="Rang", latex=f"\\text{{rang}}(A) = {rk}"))
    except Exception:
        pass
    note = None if rows == cols else "Déterminant et inverse disponibles pour les matrices carrées uniquement."
    return SolveResponse(type="matrice", input_latex=input_latex, steps=steps, results=results, note=note)

# ════════════════════════════════════════════════════════════════════════════════
#  8. MATHS DE BASE
# ════════════════════════════════════════════════════════════════════════════════

def solve_basic(raw: str) -> SolveResponse:
    cl = raw.strip().lower()
    m = re.search(r'(?:pgcd|gcd)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', cl)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        g = sympy.gcd(a, b)
        input_latex = f"\\text{{PGCD}}({a},{b})"
        steps = [Step(description=f"PGCD({a},{b}) — algorithme d'Euclide", latex=input_latex)]
        aa, bb = abs(a), abs(b)
        while bb:
            steps.append(Step(description=f"{aa} = {aa//bb}×{bb} + {aa%bb}", latex=f"{aa} = {aa//bb} \\times {bb} + {aa%bb}"))
            aa, bb = bb, aa % bb
        steps.append(Step(description="Résultat", latex=f"\\text{{PGCD}} = {g}"))
        return SolveResponse(type="basique", input_latex=input_latex, steps=steps,
            results=[Result(latex=str(g), decimal=str(g))])
    m = re.search(r'(?:ppcm|lcm)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', cl)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        g = sympy.gcd(abs(a), abs(b)); l = sympy.lcm(a, b)
        input_latex = f"\\text{{PPCM}}({a},{b})"
        steps = [
            Step(description=f"PGCD({a},{b}) = {g}", latex=f"\\text{{PGCD}}({a},{b}) = {g}"),
            Step(description="PPCM = |a×b| ÷ PGCD",
                latex=f"\\text{{PPCM}} = \\frac{{|{a} \\times {b}|}}{{{g}}} = {l}"),
        ]
        return SolveResponse(type="basique", input_latex=input_latex, steps=steps,
            results=[Result(latex=str(l), decimal=str(l))])
    m = re.search(r'(-?[\d.]+)\s*%\s*(?:de|of)?\s*(-?[\d.]+)', cl)
    if m:
        pct, total = float(m.group(1)), float(m.group(2))
        res = pct * total / 100
        input_latex = f"{pct}\\%\\ \\text{{de}}\\ {total}"
        steps = [
            Step(description=f"{pct}% de {total}", latex=input_latex),
            Step(description="Formule : % × total ÷ 100", latex=f"\\frac{{{pct} \\times {total}}}{{100}}"),
            Step(description="Résultat", latex=f"= {res:.6g}"),
        ]
        return SolveResponse(type="basique", input_latex=input_latex, steps=steps,
            results=[Result(latex=f"{res:.6g}", decimal=f"{res:.6g}")])
    try:
        expr = safe_parse(pre(raw))
        result = nsimplify(expr)
        input_latex = tex(expr)
        steps = [
            Step(description="Expression arithmétique", latex=input_latex),
            Step(description="Résultat", latex=tex(result)),
        ]
        return SolveResponse(type="basique", input_latex=input_latex, steps=steps,
            results=[Result(latex=tex(result), decimal=decimal_str(result))])
    except Exception:
        raise HTTPException(400, "Essayez : 20% de 150, pgcd(12,18), ppcm(4,6) ou une expression comme 3+4×5.")

# ════════════════════════════════════════════════════════════════════════════════
#  9. SYSTÈMES D'ÉQUATIONS
# ════════════════════════════════════════════════════════════════════════════════

def solve_system(raw: str) -> SolveResponse:
    # séparateurs possibles : ; ou retour ligne ou " et "
    parts = re.split(r'[;\n]| et ', raw.strip())
    parts = [p.strip() for p in parts if p.strip() and "=" in p]
    if len(parts) < 2:
        raise HTTPException(400, "Entrez au moins 2 équations séparées par « ; » (ex: 2x+y=5 ; x-y=1).")
    if len(parts) > 4:
        raise HTTPException(400, "Maximum 4 équations supportées.")
    # collecter variables
    eqs = []
    all_syms = set()
    for p in parts:
        l, r = p.split("=", 1)
        le = safe_parse(pre(l)); re_ = safe_parse(pre(r))
        eqs.append(Eq(le, re_))
        all_syms |= (le.free_symbols | re_.free_symbols)
    syms = sorted(all_syms, key=lambda s: s.name)
    if not syms:
        raise HTTPException(400, "Aucune variable détectée.")
    if len(syms) > 4:
        raise HTTPException(400, "Maximum 4 variables supportées.")
    input_latex = "\\begin{cases}" + " \\\\ ".join(f"{tex(e.lhs)} = {tex(e.rhs)}" for e in eqs) + "\\end{cases}"
    steps = [Step(description=f"Système de {len(eqs)} équations à {len(syms)} inconnues", latex=input_latex)]
    steps.append(Step(description="Variables : " + ", ".join(s.name for s in syms),
        latex=", ".join(tex(s) for s in syms)))
    try:
        sol = sympy.solve(eqs, syms, dict=True)
    except Exception:
        raise HTTPException(400, "Impossible de résoudre ce système.")
    if not sol:
        steps.append(Step(description="Résolution", latex="\\varnothing"))
        return SolveResponse(type="systeme", input_latex=input_latex, steps=steps, results=[],
            note="Aucune solution (système incompatible) ou infinité de solutions.")
    results = []
    for sd in sol:
        for s in syms:
            if s in sd:
                val = nsimplify(sd[s])
                steps.append(Step(description=f"Valeur de {s.name}", latex=f"{tex(s)} = {tex(val)}"))
                results.append(Result(latex=f"{tex(s)} = {tex(val)}", decimal=decimal_str(val)))
    return SolveResponse(type="systeme", input_latex=input_latex, steps=steps, results=results)

# ════════════════════════════════════════════════════════════════════════════════
#  10. INÉQUATIONS
# ════════════════════════════════════════════════════════════════════════════════

def solve_inequality(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    mrel = re.search(r'(<=|>=|<|>)', cleaned)
    if not mrel:
        raise HTTPException(400, "Une inéquation doit contenir <, >, <= ou >= (ex: 2x+5>11).")
    op = mrel.group(1)
    l, r = cleaned.split(op, 1)
    lhs = safe_parse(l.strip()); rhs = safe_parse(r.strip())
    if (lhs.free_symbols | rhs.free_symbols) - {X}:
        raise HTTPException(400, "Seule la variable x est prise en charge.")
    rel = {"<": lhs < rhs, ">": lhs > rhs, "<=": lhs <= rhs, ">=": lhs >= rhs}[op]
    input_latex = tex(rel)
    steps = [Step(description="Inéquation de départ", latex=input_latex)]
    moved = expand(lhs - rhs)
    steps.append(Step(description="Tout ramener à gauche (comparer à 0)",
        latex=f"{tex(moved)} \\;{op.replace('<=','\\leq').replace('>=','\\geq').replace('<','<').replace('>','>')}\\; 0"))
    try:
        sol = sympy.reduce_inequalities([rel], X)
    except NotImplementedError:
        steps.append(Step(description="Résolution", latex="\\text{Méthode algébrique non disponible}"))
        return SolveResponse(type="inequation", input_latex=input_latex, steps=steps, results=[],
            note="Cette inéquation ne peut pas être résolue algébriquement de façon simple.")
    except Exception:
        raise HTTPException(400, "Impossible de résoudre cette inéquation.")
    steps.append(Step(description="Ensemble solution", latex=tex(sol)))
    return SolveResponse(type="inequation", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(sol))],
        plot=sample_plot(moved))

# ════════════════════════════════════════════════════════════════════════════════
#  11. FACTORISATION
# ════════════════════════════════════════════════════════════════════════════════

def solve_factor(raw: str) -> SolveResponse:
    expr = safe_parse(pre(raw))
    input_latex = tex(expr)
    steps = [Step(description="Expression de départ", latex=input_latex)]
    expanded = expand(expr)
    if expanded != expr:
        steps.append(Step(description="Forme développée", latex=tex(expanded)))
    factored = factor(expr)
    if factored == expr or factored == expanded:
        steps.append(Step(description="L'expression est déjà factorisée ou irréductible", latex=tex(factored)))
    else:
        steps.append(Step(description="Forme factorisée", latex=tex(factored)))
    # racines si polynôme en x
    note = None
    if expr.free_symbols == {X}:
        try:
            roots = sympy.roots(Poly(expanded, X))
            if roots:
                rstr = ", ".join(f"{tex(k)}" + (f"\\,(\\times{v})" if v > 1 else "") for k, v in roots.items())
                steps.append(Step(description="Racines (zéros)", latex=f"x \\in \\{{{rstr}\\}}"))
        except Exception:
            pass
    return SolveResponse(type="factorisation", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(factored))], note=note)

# ════════════════════════════════════════════════════════════════════════════════
#  12. DÉVELOPPEMENT
# ════════════════════════════════════════════════════════════════════════════════

def solve_expand(raw: str) -> SolveResponse:
    expr = safe_parse(pre(raw))
    input_latex = tex(expr)
    steps = [Step(description="Expression de départ", latex=input_latex)]
    expanded = expand(expr)
    steps.append(Step(description="Forme développée", latex=tex(expanded)))
    if expr.free_symbols == {X}:
        try:
            p = Poly(expanded, X)
            steps.append(Step(description=f"Polynôme de degré {p.degree()}",
                latex=f"\\deg = {p.degree()}"))
        except Exception:
            pass
    return SolveResponse(type="developpement", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(expanded))])

# ════════════════════════════════════════════════════════════════════════════════
#  13. LIMITES
# ════════════════════════════════════════════════════════════════════════════════

def solve_limit(raw: str) -> SolveResponse:
    # formats : "sin(x)/x, x->0"  ou  "sin(x)/x quand x->0"  ou  "lim sin(x)/x x->0"
    cleaned = pre(raw)
    cleaned = re.sub(r'^lim\s*', '', cleaned, flags=re.IGNORECASE)
    m = re.search(r'x\s*(?:->|→|tend vers)\s*(-?oo|-?infinity|-?inf|-?[\d.]+(?:/[\d.]+)?|pi|-pi)', cleaned, re.IGNORECASE)
    point = 0
    direction = '+-'
    if m:
        ptxt = m.group(1).lower().replace('infinity', 'oo').replace('inf', 'oo')
        cleaned = cleaned[:m.start()].rstrip(' ,;quand').strip()
        if 'oo' in ptxt:
            point = -oo if ptxt.startswith('-') else oo
            direction = '+' if point == -oo else '-'
        else:
            point = safe_parse(ptxt)
    expr = safe_parse(cleaned)
    if expr.free_symbols - {X}:
        raise HTTPException(400, "Seule la variable x est prise en charge.")
    pt_latex = tex(point)
    input_latex = f"\\lim_{{x \\to {pt_latex}}} {tex(expr)}"
    steps = [Step(description=f"Limite de la fonction quand x → {pt_latex}", latex=input_latex)]
    # tentative substitution directe
    try:
        direct = expr.subs(X, point) if point not in (oo, -oo) else None
        if direct is not None and direct.is_finite and not direct.has(sympy.zoo, sympy.nan):
            steps.append(Step(description="Substitution directe", latex=f"{tex(expr)}\\Big|_{{x={pt_latex}}} = {tex(direct)}"))
    except Exception:
        pass
    try:
        L = limit(expr, X, point, direction) if point in (oo, -oo) else limit(expr, X, point)
    except Exception:
        raise HTTPException(400, "Impossible de calculer cette limite.")
    steps.append(Step(description="Résultat de la limite", latex=f"= {tex(L)}"))
    return SolveResponse(type="limite", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(L), decimal=decimal_str(L) if L.is_finite else None)],
        plot=sample_plot(expr))

# ════════════════════════════════════════════════════════════════════════════════
#  14. LOGARITHMES
# ════════════════════════════════════════════════════════════════════════════════

def solve_logarithm(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    if "=" in cleaned:
        l, r = cleaned.split("=", 1)
        lhs = safe_parse(l.strip()); rhs = safe_parse(r.strip())
        input_latex = f"{tex(lhs)} = {tex(rhs)}"
        steps = [Step(description="Équation logarithmique", latex=input_latex)]
        combined = logcombine(lhs - rhs, force=True)
        if combined != (lhs - rhs):
            steps.append(Step(description="Regrouper les logarithmes", latex=f"{tex(combined)} = 0"))
        try:
            sols = sympy.solve(Eq(lhs, rhs), X)
        except Exception:
            sols = []
        if not sols:
            return SolveResponse(type="logarithme", input_latex=input_latex, steps=steps, results=[],
                note="Aucune solution trouvée.")
        results = []
        for s in sols:
            steps.append(Step(description="Solution", latex=f"x = {tex(s)}"))
            results.append(Result(latex=tex(s), decimal=decimal_str(s.evalf()) if s.is_number else None))
        return SolveResponse(type="logarithme", input_latex=input_latex, steps=steps, results=results)
    expr = safe_parse(cleaned)
    input_latex = tex(expr)
    steps = [Step(description="Expression logarithmique", latex=input_latex)]
    simp = logcombine(simplify(expr), force=True)
    steps.append(Step(description="Simplification", latex=tex(simp)))
    val = None
    try:
        val = f"{float(simp.evalf()):.6g}"
        steps.append(Step(description="Valeur numérique", latex=f"\\approx {val}"))
    except Exception:
        pass
    return SolveResponse(type="logarithme", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(simp), decimal=val)])

# ════════════════════════════════════════════════════════════════════════════════
#  15. ÉQUATIONS EXPONENTIELLES
# ════════════════════════════════════════════════════════════════════════════════

def solve_exponential(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    if "=" not in cleaned:
        raise HTTPException(400, "Une équation exponentielle doit contenir « = » (ex: 2^x=8).")
    l, r = cleaned.split("=", 1)
    lhs = safe_parse(l.strip()); rhs = safe_parse(r.strip())
    if (lhs.free_symbols | rhs.free_symbols) - {X}:
        raise HTTPException(400, "Seule la variable x est prise en charge.")
    input_latex = f"{tex(lhs)} = {tex(rhs)}"
    steps = [Step(description="Équation exponentielle", latex=input_latex)]
    try:
        sols = sympy.solve(Eq(lhs, rhs), X)
    except Exception:
        sols = []
    sols = [s for s in sols if s.is_real] or sols
    if not sols:
        steps.append(Step(description="Résolution", latex="\\varnothing"))
        return SolveResponse(type="exponentielle", input_latex=input_latex, steps=steps, results=[],
            note="Aucune solution réelle trouvée.")
    results = []
    for s in sols:
        ss = nsimplify(s)
        steps.append(Step(description="Solution (logarithme des deux membres)", latex=f"x = {tex(ss)}"))
        results.append(Result(latex=tex(ss), decimal=decimal_str(s.evalf()) if s.is_number else None))
    return SolveResponse(type="exponentielle", input_latex=input_latex, steps=steps, results=results)

# ════════════════════════════════════════════════════════════════════════════════
#  16. NOMBRES COMPLEXES
# ════════════════════════════════════════════════════════════════════════════════

def solve_complex(raw: str) -> SolveResponse:
    cleaned = pre_imag(raw)
    expr = safe_parse(cleaned, allow_imag=True)
    if expr.free_symbols:
        raise HTTPException(400, "Entrez un nombre complexe numérique (ex: (3+4i)*(1-2i)).")
    z = simplify(expr)
    input_latex = tex(expr)
    steps = [Step(description="Expression complexe", latex=input_latex)]
    re_p = simplify(sym_re(z)); im_p = simplify(sym_im(z))
    steps.append(Step(description="Forme algébrique a + bi", latex=f"{tex(re_p)} + {tex(im_p)}i" if im_p >= 0 else f"{tex(re_p)} - {tex(-im_p)}i"))
    mod = simplify(Abs(z))
    steps.append(Step(description="Module |z| = √(a² + b²)", latex=f"|z| = {tex(mod)}"))
    a = arg(z)
    steps.append(Step(description="Argument arg(z)", latex=f"\\arg(z) = {tex(a)}"))
    conj = sympy.conjugate(z)
    steps.append(Step(description="Conjugué z̄", latex=f"\\bar{{z}} = {tex(conj)}"))
    res_latex = f"{tex(re_p)} + {tex(im_p)}i" if im_p >= 0 else f"{tex(re_p)} - {tex(-im_p)}i"
    return SolveResponse(type="complexe", input_latex=input_latex, steps=steps,
        results=[
            Result(latex=res_latex),
            Result(latex=f"|z| = {tex(mod)}", decimal=decimal_str(mod)),
        ])

# ════════════════════════════════════════════════════════════════════════════════
#  17. SUITES (arithmétiques / géométriques)
# ════════════════════════════════════════════════════════════════════════════════

def solve_sequence(raw: str) -> SolveResponse:
    parts = re.split(r'[,;\s]+', raw.strip())
    try:
        terms = [Rational(p.replace(',', '.')) for p in parts if p.strip()]
    except Exception:
        raise HTTPException(400, "Entrez les premiers termes séparés par des virgules (ex: 2, 5, 8, 11).")
    if len(terms) < 3:
        raise HTTPException(400, "Minimum 3 termes requis pour identifier la suite.")
    input_latex = ", ".join(tex(t) for t in terms)
    steps = [Step(description=f"Suite donnée ({len(terms)} termes)", latex=input_latex)]
    # arithmétique ?
    diffs = [terms[i+1] - terms[i] for i in range(len(terms)-1)]
    if len(set(diffs)) == 1:
        r = diffs[0]; u1 = terms[0]
        steps.append(Step(description=f"Différence constante r = {tex(r)} → suite arithmétique",
            latex=f"r = u_{{n+1}} - u_n = {tex(r)}"))
        un = u1 + (N - 1) * r
        steps.append(Step(description="Terme général uₙ = u₁ + (n−1)·r",
            latex=f"u_n = {tex(u1)} + (n-1)\\cdot {tex(r)} = {tex(expand(un))}"))
        n_terms = len(terms)
        Sn = summation(u1 + (N-1)*r, (N, 1, n_terms))
        steps.append(Step(description=f"Somme des {n_terms} premiers termes",
            latex=f"S_{{{n_terms}}} = {tex(Sn)}"))
        return SolveResponse(type="suite", input_latex=input_latex, steps=steps,
            results=[
                Result(latex=f"u_n = {tex(expand(un))}"),
                Result(latex=f"r = {tex(r)}", decimal=decimal_str(r)),
            ])
    # géométrique ?
    if all(terms[i] != 0 for i in range(len(terms)-1)):
        ratios = [simplify(terms[i+1] / terms[i]) for i in range(len(terms)-1)]
        if len(set(ratios)) == 1:
            q = ratios[0]; u1 = terms[0]
            steps.append(Step(description=f"Raison constante q = {tex(q)} → suite géométrique",
                latex=f"q = \\frac{{u_{{n+1}}}}{{u_n}} = {tex(q)}"))
            un = u1 * q**(N - 1)
            steps.append(Step(description="Terme général uₙ = u₁ · qⁿ⁻¹",
                latex=f"u_n = {tex(u1)} \\cdot ({tex(q)})^{{n-1}}"))
            return SolveResponse(type="suite", input_latex=input_latex, steps=steps,
                results=[
                    Result(latex=f"u_n = {tex(u1)} \\cdot ({tex(q)})^{{n-1}}"),
                    Result(latex=f"q = {tex(q)}", decimal=decimal_str(q)),
                ])
    steps.append(Step(description="Analyse", latex="\\text{Ni arithmétique ni géométrique}"))
    return SolveResponse(type="suite", input_latex=input_latex, steps=steps, results=[],
        note="La suite n'est ni arithmétique ni géométrique (différence et raison non constantes).")

# ════════════════════════════════════════════════════════════════════════════════
#  18. PROBABILITÉS / COMBINATOIRE
# ════════════════════════════════════════════════════════════════════════════════

def solve_probability(raw: str) -> SolveResponse:
    cl = raw.strip().lower().replace(" ", "")
    # combinaisons C(n,k) ou nCk
    m = re.search(r'c\(?(\d+),(\d+)\)?', cl) or re.search(r'(\d+)c(\d+)', cl)
    if m:
        n_, k_ = int(m.group(1)), int(m.group(2))
        if k_ > n_:
            raise HTTPException(400, "k ne peut pas dépasser n.")
        val = binomial(n_, k_)
        input_latex = f"\\binom{{{n_}}}{{{k_}}}"
        steps = [
            Step(description=f"Combinaison de {k_} parmi {n_}", latex=input_latex),
            Step(description="Formule : n! / (k!·(n−k)!)",
                latex=f"\\frac{{{n_}!}}{{{k_}!\\,({n_}-{k_})!}}"),
            Step(description="Résultat", latex=f"= {val}"),
        ]
        return SolveResponse(type="probabilite", input_latex=input_latex, steps=steps,
            results=[Result(latex=str(val), decimal=str(val))])
    # arrangements A(n,k) ou nPk
    m = re.search(r'a\(?(\d+),(\d+)\)?', cl) or re.search(r'(\d+)p(\d+)', cl)
    if m:
        n_, k_ = int(m.group(1)), int(m.group(2))
        if k_ > n_:
            raise HTTPException(400, "k ne peut pas dépasser n.")
        val = factorial(n_) // factorial(n_ - k_)
        input_latex = f"A_{{{n_}}}^{{{k_}}}"
        steps = [
            Step(description=f"Arrangement de {k_} parmi {n_}", latex=input_latex),
            Step(description="Formule : n! / (n−k)!",
                latex=f"\\frac{{{n_}!}}{{({n_}-{k_})!}}"),
            Step(description="Résultat", latex=f"= {val}"),
        ]
        return SolveResponse(type="probabilite", input_latex=input_latex, steps=steps,
            results=[Result(latex=str(val), decimal=str(val))])
    # factorielle n!
    m = re.fullmatch(r'(\d+)!', cl)
    if m:
        n_ = int(m.group(1))
        if n_ > 1000:
            raise HTTPException(400, "Maximum 1000! supporté.")
        val = factorial(n_)
        input_latex = f"{n_}!"
        steps = [
            Step(description=f"Factorielle de {n_}", latex=input_latex),
            Step(description="n! = n × (n−1) × … × 2 × 1", latex=f"{n_}! = {val}"),
        ]
        return SolveResponse(type="probabilite", input_latex=input_latex, steps=steps,
            results=[Result(latex=str(val), decimal=decimal_str(val) if n_ < 20 else None)])
    raise HTTPException(400, "Essayez : C(5,2) pour combinaisons, A(5,2) pour arrangements, ou 5! pour factorielle.")

# ════════════════════════════════════════════════════════════════════════════════
#  19. POLYNÔMES (racines, degré 3+)
# ════════════════════════════════════════════════════════════════════════════════

def solve_polynomial(raw: str) -> SolveResponse:
    cleaned = pre(raw)
    if "=" in cleaned:
        l, r = cleaned.split("=", 1)
        lhs = safe_parse(l.strip()); rhs = safe_parse(r.strip())
        expr = expand(lhs - rhs)
    else:
        expr = expand(safe_parse(cleaned))
    if expr.free_symbols - {X}:
        raise HTTPException(400, "Seule la variable x est prise en charge.")
    try:
        p = Poly(expr, X)
    except Exception:
        raise HTTPException(400, "Impossible d'interpréter comme polynôme en x.")
    input_latex = f"{tex(expr)} = 0"
    steps = [
        Step(description=f"Polynôme de degré {p.degree()}", latex=input_latex),
    ]
    factored = factor(expr)
    if factored != expr:
        steps.append(Step(description="Factorisation", latex=f"{tex(factored)} = 0"))
    try:
        roots = sympy.roots(p)
    except Exception:
        roots = {}
    if not roots:
        # fallback solveset
        try:
            sols = list(sympy.solve(expr, X))
        except Exception:
            sols = []
        roots = {s: 1 for s in sols}
    if not roots:
        steps.append(Step(description="Racines", latex="\\text{Aucune racine exacte trouvée}"))
        return SolveResponse(type="polynome", input_latex=input_latex, steps=steps, results=[],
            note="Aucune racine exacte trouvée.")
    results = []
    for k, v in roots.items():
        ks = nsimplify(k)
        mult = f"\\;(\\text{{mult.}}\\,{v})" if v > 1 else ""
        steps.append(Step(description=f"Racine{' (multiple)' if v > 1 else ''}", latex=f"x = {tex(ks)}{mult}"))
        results.append(Result(latex=tex(ks), decimal=decimal_str(k.evalf()) if k.is_number else None))
    return SolveResponse(type="polynome", input_latex=input_latex, steps=steps, results=results,
        plot=sample_plot(expr))

# ════════════════════════════════════════════════════════════════════════════════
#  20. DÉCOMPOSITION EN FACTEURS PREMIERS
# ════════════════════════════════════════════════════════════════════════════════

def solve_primefactor(raw: str) -> SolveResponse:
    m = re.search(r'-?\d+', raw)
    if not m:
        raise HTTPException(400, "Entrez un nombre entier (ex: 360).")
    n_ = abs(int(m.group(0)))
    if n_ < 2:
        raise HTTPException(400, "Entrez un entier ≥ 2.")
    if n_ > 10**15:
        raise HTTPException(400, "Nombre trop grand (max 10^15).")
    f = factorint(n_)
    input_latex = str(n_)
    steps = [Step(description=f"Décomposition de {n_} en facteurs premiers", latex=input_latex)]
    # division successive
    temp = n_
    for prime in sorted(f.keys()):
        while temp % prime == 0:
            steps.append(Step(description=f"{temp} ÷ {prime} = {temp//prime}",
                latex=f"{temp} = {prime} \\times {temp//prime}"))
            temp //= prime
    prod = " \\times ".join(f"{p}^{{{e}}}" if e > 1 else f"{p}" for p, e in sorted(f.items()))
    steps.append(Step(description="Produit de facteurs premiers", latex=f"{n_} = {prod}"))
    divisors = sympy.divisor_count(n_)
    steps.append(Step(description="Nombre de diviseurs", latex=f"d({n_}) = {divisors}"))
    return SolveResponse(type="facteurs_premiers", input_latex=input_latex, steps=steps,
        results=[Result(latex=f"{n_} = {prod}")])

# ════════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════════

DISPATCH = {
    "algebra": solve_algebra,
    "fraction": solve_fraction,
    "derivative": solve_derivative,
    "integral": solve_integral,
    "statistics": solve_statistics,
    "trigonometry": solve_trigonometry,
    "matrix": solve_matrix,
    "basic": solve_basic,
    "system": solve_system,
    "inequality": solve_inequality,
    "factor": solve_factor,
    "expand": solve_expand,
    "limit": solve_limit,
    "logarithm": solve_logarithm,
    "exponential": solve_exponential,
    "complex": solve_complex,
    "sequence": solve_sequence,
    "probability": solve_probability,
    "polynomial": solve_polynomial,
    "primefactor": solve_primefactor,
}

@app.get("/")
def root():
    return {"status": "ok", "service": "CalculatricePro API", "version": "3.0.0", "categories": len(DISPATCH)}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    raw = req.expression.strip()
    if not raw:
        raise HTTPException(400, "Veuillez saisir une expression.")
    if len(raw) > 500:
        raise HTTPException(400, "Expression trop longue (max 500 caractères).")
    cat = req.category.lower().strip()
    fn = DISPATCH.get(cat, solve_algebra)
    try:
        return fn(raw)
    except HTTPException:
        raise
    except ZeroDivisionError:
        raise HTTPException(400, "Division par zéro détectée.")
    except Exception as e:
        raise HTTPException(400, f"Erreur de syntaxe. ({type(e).__name__})")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
