"""
================================================================================
 CalculatricePro - Moteur de Résolution Mathématique (Backend API)
================================================================================
Phase 1 : API Python (FastAPI + SymPy) avec génération d'étapes en français.

SymPy résout les équations mais ne fournit aucune explication textuelle.
Ce fichier contient donc une couche personnalisée qui intercepte le
processus de résolution (coefficients, discriminant, dérivées terme à
terme, PGCD) et reformule chaque étape intermédiaire en français,
au format attendu par le frontend (KaTeX).

Déploiement prévu : Render.com / Railway.app (tier gratuit)
Démarrage : uvicorn main:app --host 0.0.0.0 --port $PORT
================================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import re

import sympy
from sympy import (
    symbols, Eq, sympify, expand, simplify, nsimplify, latex,
    Poly, sqrt, Rational, gcd, diff, S, oo, I
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor,
)

# ------------------------------------------------------------------------------
# Configuration FastAPI
# ------------------------------------------------------------------------------

app = FastAPI(
    title="CalculatricePro - API de Résolution Mathématique",
    description="Moteur SymPy avec génération d'étapes pédagogiques en français.",
    version="1.0.0",
)

# IMPORTANT : restreindre strictement aux domaines autorisés en production.
# Le tier gratuit Render/Railway peut changer d'URL : ajoutez l'URL du
# backend lui-même n'est pas nécessaire, seulement les origines FRONTEND.
ALLOWED_ORIGINS = [
    "https://calculatricepro.com",
    "https://www.calculatricepro.com",
    "http://calculatricepro.com",
    "http://www.calculatricepro.com",
    "null",  # local file:/// testing
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

X = symbols("x")

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,  # permet "2x" au lieu de "2*x"
    convert_xor,                          # permet "x^2" au lieu de "x**2"
)


# ------------------------------------------------------------------------------
# Modèles Pydantic (contrat API)
# ------------------------------------------------------------------------------

class SolveRequest(BaseModel):
    expression: str = Field(..., description="Expression ou équation saisie par l'utilisateur")
    operation: Literal["auto", "equation", "fraction", "derivative"] = "auto"


class Step(BaseModel):
    description: str   # explication en français, ex : "Soustraire 5 des deux côtés"
    latex: str          # représentation LaTeX de l'état de l'équation/expression à cette étape


class Result(BaseModel):
    latex: str
    decimal: Optional[str] = None   # valeur décimale approximative si pertinente


class SolveResponse(BaseModel):
    type: str                 # "lineaire" | "quadratique" | "fraction" | "derivee"
    input_latex: str          # l'entrée originale, rendue en LaTeX, pour affichage
    steps: List[Step]
    results: List[Result]     # une ou plusieurs solutions / résultat final
    note: Optional[str] = None


# ------------------------------------------------------------------------------
# Pré-traitement : notation française -> notation SymPy
# ------------------------------------------------------------------------------

def preprocess_input(raw: str) -> str:
    """
    Convertit les symboles mathématiques français/courants vers une syntaxe
    que parse_expr comprend.
    """
    s = raw.strip()

    # Symboles de multiplication / division
    s = s.replace("×", "*").replace("÷", "/")
    s = s.replace("−", "-")  # tiret moins typographique -> signe moins ASCII

    # Racine carrée écrite "√(...)" ou "√25"
    s = re.sub(r"√\(([^)]+)\)", r"sqrt(\1)", s)
    s = re.sub(r"√(\d+(\.\d+)?)", r"sqrt(\1)", s)

    # Nombres décimaux à la française : "3,5" -> "3.5"
    # (uniquement entre deux chiffres, pour ne pas casser d'éventuels séparateurs)
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)

    # Notation dérivée "d/dx(...)" -> on la retire, l'opération est gérée
    # séparément via le champ `operation`.
    s = re.sub(r"^d/dx\s*\(?", "", s, flags=re.IGNORECASE)
    if s.endswith(")") and "d/dx" in raw.lower():
        s = s[:-1]

    return s.strip()


def safe_parse(expr_str: str):
    try:
        return parse_expr(expr_str, transformations=TRANSFORMATIONS, local_dict={"x": X})
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Expression illisible : « {expr_str} ». Vérifiez la syntaxe."
        )


def to_latex(expr) -> str:
    return latex(expr)


def fmt_decimal(value) -> Optional[str]:
    try:
        f = float(value)
        return f"{f:.6g}"
    except Exception:
        return None


# ------------------------------------------------------------------------------
# Détection automatique du type de problème
# ------------------------------------------------------------------------------

def detect_operation(raw: str, requested: str) -> str:
    if requested != "auto":
        return requested

    if raw.lower().strip().startswith("d/dx") or "dérivée" in raw.lower():
        return "derivative"

    if "=" in raw:
        return "equation"

    # Une fraction pure : uniquement chiffres, espaces, et un seul "/"
    if re.fullmatch(r"\s*-?\d+\s*/\s*-?\d+\s*", raw):
        return "fraction"

    # Par défaut, si aucune variable n'est présente et qu'il y a un "/",
    # on tente la simplification de fraction. Sinon on tente une dérivée.
    if "x" not in raw.lower():
        return "fraction"

    return "derivative"


# ------------------------------------------------------------------------------
# Mots-clés / phrases françaises réutilisables
# ------------------------------------------------------------------------------

def phrase_add_sub(coeff, term_label: str) -> str:
    """
    Construit une phrase du type "Soustraire 5x des deux côtés" ou
    "Ajouter 3 aux deux côtés" selon le signe de `coeff`.
    """
    coeff_s = sympy.nsimplify(coeff)
    if coeff_s == 0:
        return ""
    if coeff_s > 0:
        return f"Soustraire {latex(coeff_s)}{term_label} des deux côtés"
    else:
        return f"Ajouter {latex(-coeff_s)}{term_label} aux deux côtés"


# ------------------------------------------------------------------------------
# 1. ÉQUATIONS (linéaires et quadratiques)
# ------------------------------------------------------------------------------

def solve_equation(raw: str) -> SolveResponse:
    if raw.count("=") != 1:
        raise HTTPException(
            status_code=400,
            detail="Une équation doit contenir exactement un signe « = »."
        )

    lhs_str, rhs_str = raw.split("=")
    lhs = safe_parse(preprocess_input(lhs_str))
    rhs = safe_parse(preprocess_input(rhs_str))

    free_syms = (lhs.free_symbols | rhs.free_symbols)
    if free_syms - {X}:
        raise HTTPException(
            status_code=400,
            detail="Seule la variable « x » est prise en charge pour le moment."
        )

    input_latex = f"{to_latex(lhs)} = {to_latex(rhs)}"
    steps: List[Step] = [Step(
        description="Équation de départ",
        latex=input_latex,
    )]

    lhs_exp, rhs_exp = expand(lhs), expand(rhs)
    if lhs_exp != lhs or rhs_exp != rhs:
        steps.append(Step(
            description="Développer les deux côtés",
            latex=f"{to_latex(lhs_exp)} = {to_latex(rhs_exp)}",
        ))

    try:
        poly_lhs = Poly(lhs_exp, X)
        poly_rhs = Poly(rhs_exp, X)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Impossible d'interpréter cette équation comme un polynôme en x."
        )

    degree = max(poly_lhs.degree(), poly_rhs.degree())

    if degree <= 1:
        return _solve_linear(poly_lhs, poly_rhs, steps, input_latex)
    elif degree == 2:
        return _solve_quadratic(poly_lhs, poly_rhs, steps, input_latex)
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Seules les équations linéaires (degré 1) et quadratiques "
                "(degré 2) sont prises en charge pour le moment."
            ),
        )


def _coeffs(poly: Poly):
    """Retourne (a, b) tels que poly == a*x + b, pour un polynôme de degré <= 1."""
    a = poly.coeff_monomial(X)
    b = poly.coeff_monomial(1)
    return sympy.nsimplify(a), sympy.nsimplify(b)


def _solve_linear(poly_lhs: Poly, poly_rhs: Poly, steps: List[Step], input_latex: str) -> SolveResponse:
    a1, b1 = _coeffs(poly_lhs)
    a2, b2 = _coeffs(poly_rhs)

    # Étape : regrouper les termes en x à gauche
    a = a1 - a2
    if a2 != 0:
        phrase = phrase_add_sub(a2, "x")
        if phrase:
            steps.append(Step(
                description=phrase,
                latex=f"{latex(a)}x {'+' if b1 >= 0 else '-'} {latex(abs(b1))} = {latex(b2)}"
                       if b1 != 0 else f"{latex(a)}x = {latex(b2)}",
            ))

    # Étape : isoler la constante à droite
    rhs_const = b2 - b1
    if b1 != 0:
        phrase = phrase_add_sub(b1, "")
        if phrase:
            steps.append(Step(
                description=phrase,
                latex=f"{latex(a)}x = {latex(rhs_const)}",
            ))

    if a == 0:
        if rhs_const == 0:
            note = "Cette équation est vraie pour tout x (infinité de solutions)."
            steps.append(Step(description="Simplification", latex="0 = 0"))
            return SolveResponse(
                type="lineaire", input_latex=input_latex, steps=steps,
                results=[], note=note,
            )
        else:
            note = "Cette équation n'a aucune solution (impossible)."
            steps.append(Step(description="Simplification", latex=f"0 = {latex(rhs_const)}"))
            return SolveResponse(
                type="lineaire", input_latex=input_latex, steps=steps,
                results=[], note=note,
            )

    # Étape : diviser des deux côtés par a
    if a != 1:
        steps.append(Step(
            description=f"Diviser les deux côtés par {latex(a)}",
            latex=f"x = \\frac{{{latex(rhs_const)}}}{{{latex(a)}}}",
        ))

    solution = sympy.nsimplify(rhs_const / a)
    steps.append(Step(
        description="Solution",
        latex=f"x = {latex(solution)}",
    ))

    return SolveResponse(
        type="lineaire",
        input_latex=input_latex,
        steps=steps,
        results=[Result(latex=latex(solution), decimal=fmt_decimal(solution))],
    )


def _solve_quadratic(poly_lhs: Poly, poly_rhs: Poly, steps: List[Step], input_latex: str) -> SolveResponse:
    diff_poly = (poly_lhs - poly_rhs)
    diff_expr = expand(diff_poly.as_expr())

    if poly_rhs.as_expr() != 0:
        steps.append(Step(
            description="Regrouper tous les termes du même côté (forme canonique)",
            latex=f"{to_latex(diff_expr)} = 0",
        ))

    poly = Poly(diff_expr, X)
    a = sympy.nsimplify(poly.coeff_monomial(X**2))
    b = sympy.nsimplify(poly.coeff_monomial(X))
    c = sympy.nsimplify(poly.coeff_monomial(1))

    a_term = "x^2" if a == 1 else ("-x^2" if a == -1 else f"{latex(a)}x^2")
    b_term = "x" if abs(b) == 1 else f"{latex(abs(b))}x"
    steps.append(Step(
        description=f"Identifier les coefficients : a = {latex(a)}, b = {latex(b)}, c = {latex(c)}",
        latex=f"{a_term} {'+' if b >= 0 else '-'} {b_term} {'+' if c >= 0 else '-'} {latex(abs(c))} = 0",
    ))

    discriminant = sympy.nsimplify(b**2 - 4*a*c)
    steps.append(Step(
        description="Calculer le discriminant Δ = b² − 4ac",
        latex=f"\\Delta = ({latex(b)})^2 - 4({latex(a)})({latex(c)}) = {latex(discriminant)}",
    ))

    if discriminant > 0:
        sqrt_d = sympy.sqrt(discriminant)
        sqrt_d_simpl = sympy.nsimplify(sqrt_d)
        steps.append(Step(
            description="Le discriminant est positif : il y a deux solutions réelles",
            latex=f"\\sqrt{{\\Delta}} = {latex(sqrt_d_simpl)}",
        ))
        x1 = sympy.nsimplify((-b - sqrt_d) / (2*a))
        x2 = sympy.nsimplify((-b + sqrt_d) / (2*a))
        steps.append(Step(
            description="Appliquer la formule x = (−b ± √Δ) / (2a)",
            latex=(
                f"x_1 = \\frac{{-({latex(b)}) - \\sqrt{{{latex(discriminant)}}}}}{{2({latex(a)})}}, \\quad "
                f"x_2 = \\frac{{-({latex(b)}) + \\sqrt{{{latex(discriminant)}}}}}{{2({latex(a)})}}"
            ),
        ))
        results = [
            Result(latex=latex(x1), decimal=fmt_decimal(x1)),
            Result(latex=latex(x2), decimal=fmt_decimal(x2)),
        ]
        note = None

    elif discriminant == 0:
        x0 = sympy.nsimplify(-b / (2*a))
        steps.append(Step(
            description="Le discriminant est nul : il y a une solution unique (racine double)",
            latex=f"x_0 = \\frac{{-b}}{{2a}} = \\frac{{-({latex(b)})}}{{2({latex(a)})}} = {latex(x0)}",
        ))
        results = [Result(latex=latex(x0), decimal=fmt_decimal(x0))]
        note = None

    else:
        steps.append(Step(
            description="Le discriminant est négatif : il n'y a pas de solution réelle",
            latex=f"\\Delta = {latex(discriminant)} < 0",
        ))
        results = []
        note = "Aucune solution réelle (le discriminant est négatif)."

    return SolveResponse(
        type="quadratique",
        input_latex=input_latex,
        steps=steps,
        results=results,
        note=note,
    )


# ------------------------------------------------------------------------------
# 2. SIMPLIFICATION DE FRACTIONS
# ------------------------------------------------------------------------------

def solve_fraction(raw: str) -> SolveResponse:
    cleaned = preprocess_input(raw)
    m = re.fullmatch(r"\s*(-?\d+)\s*/\s*(-?\d+)\s*", cleaned)
    if not m:
        raise HTTPException(
            status_code=400,
            detail="Format de fraction attendu : « numérateur/dénominateur », ex. 12/18.",
        )

    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        raise HTTPException(status_code=400, detail="Division par zéro impossible (dénominateur nul).")

    input_latex = f"\\frac{{{num}}}{{{den}}}"
    steps: List[Step] = [Step(description="Fraction de départ", latex=input_latex)]

    sign = -1 if (num < 0) ^ (den < 0) else 1
    num_abs, den_abs = abs(num), abs(den)

    g = sympy.gcd(num_abs, den_abs)
    steps.append(Step(
        description=f"Calculer le PGCD (plus grand commun diviseur) de {num_abs} et {den_abs}",
        latex=f"\\text{{PGCD}}({num_abs}, {den_abs}) = {g}",
    ))

    if g == 1:
        steps.append(Step(
            description="La fraction est déjà irréductible (PGCD = 1)",
            latex=input_latex,
        ))
        result_expr = Rational(num, den)
    else:
        new_num, new_den = num_abs // g, den_abs // g
        steps.append(Step(
            description=f"Diviser le numérateur et le dénominateur par {g}",
            latex=f"\\frac{{{num_abs} \\div {g}}}{{{den_abs} \\div {g}}} = \\frac{{{new_num}}}{{{new_den}}}",
        ))
        result_expr = sign * Rational(new_num, new_den)
        if sign < 0:
            steps.append(Step(
                description="Résultat simplifié",
                latex=f"-\\frac{{{new_num}}}{{{new_den}}}",
            ))

    return SolveResponse(
        type="fraction",
        input_latex=input_latex,
        steps=steps,
        results=[Result(latex=latex(result_expr), decimal=fmt_decimal(result_expr))],
    )


# ------------------------------------------------------------------------------
# 3. DÉRIVÉES (polynômes simples, terme par terme)
# ------------------------------------------------------------------------------

def solve_derivative(raw: str) -> SolveResponse:
    cleaned = preprocess_input(raw)
    expr = safe_parse(cleaned)

    if expr.free_symbols - {X}:
        raise HTTPException(
            status_code=400,
            detail="Seule la dérivation par rapport à « x » est prise en charge pour le moment.",
        )

    expr = expand(expr)
    input_latex = f"\\frac{{d}}{{dx}}\\left({to_latex(expr)}\\right)"
    steps: List[Step] = [Step(description="Expression de départ", latex=to_latex(expr))]

    terms = expr.as_ordered_terms()
    if len(terms) > 1:
        term_strs = []
        for i, t in enumerate(terms):
            t_latex = to_latex(t)
            if i > 0 and not t_latex.startswith("-"):
                t_latex = "+ " + t_latex
            elif i > 0:
                t_latex = "- " + t_latex[1:].lstrip()
            term_strs.append(t_latex)
        steps.append(Step(
            description="Dériver chaque terme séparément (règle de la somme)",
            latex=" ".join(term_strs),
        ))

    term_derivative_steps = []
    derivative_terms = []
    for term in terms:
        d_term = diff(term, X)
        derivative_terms.append(d_term)

        if term.has(X):
            poly_t = Poly(term, X) if term.is_polynomial(X) else None
            if poly_t is not None and poly_t.degree() >= 1:
                power = poly_t.degree()
                coeff = poly_t.LC()
                coeff_disp = "" if coeff == 1 else ("-" if coeff == -1 else latex(coeff))
                rule_desc = (
                    f"Règle de la puissance : d/dx[{coeff_disp}x^{{{power}}}] "
                    f"= {latex(coeff)} \\times {power} \\times x^{{{power-1}}} = {latex(d_term)}"
                    if power > 1 else
                    f"Règle de la puissance : d/dx[{coeff_disp}x] = {latex(coeff)} = {latex(d_term)}"
                )
            else:
                rule_desc = f"d/dx[{latex(term)}] = {latex(d_term)}"
        else:
            rule_desc = f"La dérivée d'une constante est 0 : d/dx[{latex(term)}] = 0"

        term_derivative_steps.append(Step(description=rule_desc, latex=latex(d_term)))

    steps.extend(term_derivative_steps)

    result = simplify(sum(derivative_terms))
    steps.append(Step(
        description="Additionner les dérivées de chaque terme",
        latex=f"f'(x) = {latex(result)}",
    ))

    return SolveResponse(
        type="derivee",
        input_latex=input_latex,
        steps=steps,
        results=[Result(latex=latex(result))],
    )


# ------------------------------------------------------------------------------
# Routes API
# ------------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "CalculatricePro Math API", "version": "1.0.0"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    raw = req.expression.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Veuillez saisir une expression ou une équation.")

    op = detect_operation(raw, req.operation)

    try:
        if op == "equation":
            return solve_equation(raw)
        elif op == "fraction":
            return solve_fraction(raw)
        elif op == "derivative":
            return solve_derivative(raw)
        else:
            raise HTTPException(status_code=400, detail="Opération non reconnue.")
    except HTTPException:
        raise
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Division par zéro détectée dans le calcul.")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de résoudre cette expression. Vérifiez la syntaxe. ({type(e).__name__})",
        )


# ------------------------------------------------------------------------------
# Démarrage local (pour tests) :
#   uvicorn main:app --reload
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
