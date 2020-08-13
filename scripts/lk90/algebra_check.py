import sympy as sp

def check_Iedot_scaling():
    # track j, e separately so we can set e = 1 without setting j = 0
    G, m12, mu, msl, m3, a, a3, c, e, j, I = sp.symbols(
        'G m12 mu msl m3 a a3 c e j I', positive=True)
    n = sp.sqrt(G) * sp.sqrt(m12) / a**(sp.Rational(3, 2))
    t_lk = (m12 / m3) * (a3 / a)**3 / n

    dadt_gw = (
        -sp.Rational(64, 5) * G**3 * mu * m12**2 / (c**5 * a**3 * j**7)
            * (1 + sp.Rational(73, 24) * e**2 + sp.Rational(37, 96) * e**4))
    dedt_gw = (
        -sp.Rational(304, 15) * G**3 * mu * m12**2 / (c**5 * a**4 * j**5)
            * (1 + sp.Rational(121, 304) * e**2))

    W_SL = 3 * G * n * msl / (2 * c**2 * a * j**2)
    W_L = 3 / (4 * t_lk) * (sp.cos(I) * (sp.Rational(5, 2) * e**2 - 4 * e**2 - 1)) / j
    A = -W_SL / W_L
    Adot_over_A = -4 * (dadt_gw) / a + e / j**2 * dedt_gw

    # cast into e -> 1 limit
    W_L_expr = W_L.subs(e, 1)
    Adot_lim = Adot_over_A.subs(e, 1)
    A_lim = A.subs(e, 1)

    sols = sp.solve(A_lim - 1, a) # all four sols are the same
    astar = -sols[0] # hack
    print(sp.latex(astar))

    Adot_star = Adot_lim.subs(a, astar) # Adot / A at A = 1

    # A = 1
    goal_expr = sp.simplify(Adot_star / W_L_expr.subs(a, astar) * sp.sin(I) / (
        2 + 2 * sp.cos(I))**(sp.Rational(3, 2)))
    print(sp.latex(goal_expr))

    # we already have tho goal expression, multiply out a few useless terms to
    # simplify
    factor = (595 * sp.sin(I)) / (
        (1 / sp.cos(I))**sp.Rational(3, 8) *
            2**(sp.Rational(3, 2)) * (1 + sp.cos(I))**(sp.Rational(3, 2)) *
            36 * j**(sp.Rational(37, 8))
    )
    sub_expr = goal_expr / factor
    print()
    print(sp.latex(sub_expr**8))
    print(sp.latex(factor))

if __name__ == '__main__':
    check_Iedot_scaling()
