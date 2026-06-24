"""
CalculatricePro - Backend API v2.0
8 categories: algèbre, fractions, dérivées, intégrales,
              statistiques, trigonométrie, matrices, maths de base
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import re, json
from collections import Counter

import sympy
from sympy import (
    symbols, Eq, expand, simplify, nsimplify, latex,
    Poly, sqrt, Rational, gcd, lcm, diff, integrate,
    sin, cos, tan, pi, trigsimp, Abs, Matrix,
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="CalculatricePro API", version="2.0.0")

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
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# ── Models ─────────────────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    expression: str = Field(..., description="Expression à résoudre")
    category: str = Field("algebra", description="algebra|fraction|derivative|integral|statistics|trigonometry|matrix|basic")

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

# ── Helpers ────────────────────────────────────────────────────────────────────

def pre(raw: str) -> str:
    s = raw.strip()
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = re.sub(r"√\(([^)]+)\)", r"sqrt(\1)", s)
    s = re.sub(r"√(\d+(\.\d+)?)", r"sqrt(\1)", s)
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    return s.strip()

def safe_parse(expr_str: str):
    try:
        return parse_expr(expr_str, transformations=TRANSFORMATIONS, local_dict={"x": X, "pi": pi})
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

# ── 1. ALGÈBRE ─────────────────────────────────────────────────────────────────

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
        return _linear(pl, pr, steps, input_latex)
    elif deg == 2:
        return _quadratic(pl, pr, steps, input_latex)
    raise HTTPException(400, f"Degré {deg} non supporté (max degré 2).")

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
    return SolveResponse(type="quadratique", input_latex=input_latex, steps=steps, results=[], note="Aucune solution réelle.")

# ── 2. FRACTIONS ───────────────────────────────────────────────────────────────

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

# ── 3. DÉRIVÉES ────────────────────────────────────────────────────────────────

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
        results=[Result(latex=tex(result))])

# ── 4. INTÉGRALES ──────────────────────────────────────────────────────────────

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
            results=[Result(latex=tex(val), decimal=decimal_str(val))])
    steps.append(Step(description="Primitive (+ constante C)", latex=f"{tex(F)} + C"))
    return SolveResponse(type="integrale", input_latex=input_latex, steps=steps,
        results=[Result(latex=f"{tex(F)} + C")])

# ── 5. STATISTIQUES ────────────────────────────────────────────────────────────

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
        Step(description="Variance σ²",
            latex=f"\\sigma^2 = {variance:.6g}"),
        Step(description="Écart-type σ = √(variance)",
            latex=f"\\sigma = {std_dev:.6g}"),
        Step(description=f"Étendue = max − min = {fmt(max(nums))} − {fmt(min(nums))}",
            latex=f"\\text{{Étendue}} = {data_range:.6g}"),
    ]
    return SolveResponse(type="statistiques", input_latex=input_latex, steps=steps, results=[
        Result(latex=f"\\bar{{x}} = {mean:.6g}", decimal=f"{mean:.6g}"),
        Result(latex=f"\\text{{Médiane}} = {median:.6g}", decimal=f"{median:.6g}"),
        Result(latex=f"\\sigma = {std_dev:.6g}", decimal=f"{std_dev:.6g}"),
    ])

# ── 6. TRIGONOMÉTRIE ───────────────────────────────────────────────────────────

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
    return SolveResponse(type="trigonometrie", input_latex=input_latex, steps=steps,
        results=[Result(latex=tex(simp), decimal=num_val)])

# ── 7. MATRICES ────────────────────────────────────────────────────────────────

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

# ── 8. MATHS DE BASE ───────────────────────────────────────────────────────────

def solve_basic(raw: str) -> SolveResponse:
    cl = raw.strip().lower()
    # PGCD
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
    # PPCM
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
    # Pourcentage
    m = re.search(r'(-?[\d.]+)\s*%\s*(?:de|of)?\s*(-?[\d.]+)', cl)
    if m:
        pct, total = float(m.group(1)), float(m.group(2))
        res = pct * total / 100
        input_latex = f"{pct}\\%\\ \\text{{de}}\\ {total}"
        steps = [
            Step(description=f"{pct}% de {total}", latex=input_latex),
            Step(description="Formule : % × total ÷ 100",
                latex=f"\\frac{{{pct} \\times {total}}}{{100}}"),
            Step(description="Résultat", latex=f"= {res:.6g}"),
        ]
        return SolveResponse(type="basique", input_latex=input_latex, steps=steps,
            results=[Result(latex=f"{res:.6g}", decimal=f"{res:.6g}")])
    # Expression arithmétique générale
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

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "CalculatricePro API", "version": "2.0.0"}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    raw = req.expression.strip()
    if not raw:
        raise HTTPException(400, "Veuillez saisir une expression.")
    cat = req.category.lower().strip()
    try:
        if cat == "statistics":   return solve_statistics(raw)
        if cat == "integral":     return solve_integral(raw)
        if cat == "basic":        return solve_basic(raw)
        if cat == "trigonometry": return solve_trigonometry(raw)
        if cat == "matrix":       return solve_matrix(raw)
        if cat == "fraction":     return solve_fraction(raw)
        if cat == "derivative":   return solve_derivative(raw)
        return solve_algebra(raw)
    except HTTPException:
        raise
    except ZeroDivisionError:
        raise HTTPException(400, "Division par zéro détectée.")
    except Exception as e:
        raise HTTPException(400, f"Erreur de syntaxe. ({type(e).__name__})")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
