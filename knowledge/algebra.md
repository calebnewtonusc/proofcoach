# Algebra — ProofCoach Knowledge Base

## Vieta's Formulas

For $p(x) = x^n + a_{n-1}x^{n-1} + \cdots + a_0$ with roots $r_1, \ldots, r_n$:
- $\sum r_i = -a_{n-1}$
- $\sum_{i<j} r_i r_j = a_{n-2}$
- $r_1 r_2 \cdots r_n = (-1)^n a_0$

---

## AM-GM Inequality

For non-negative reals $a_1, \ldots, a_n$:
$$\frac{a_1 + \cdots + a_n}{n} \geq \sqrt[n]{a_1 \cdots a_n}$$

Equality iff $a_1 = a_2 = \cdots = a_n$.

**Competition trick**: To minimize a sum $f(a) + g(a)$, apply AM-GM.

---

## Cauchy-Schwarz Inequality

$$(a_1^2 + \cdots + a_n^2)(b_1^2 + \cdots + b_n^2) \geq (a_1 b_1 + \cdots + a_n b_n)^2$$

Equality iff $a_i/b_i$ is constant.

---

## Sequences and Series

**Arithmetic series**: $\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$

**Geometric series**: $\sum_{k=0}^{n-1} r^k = \frac{r^n - 1}{r - 1}$ for $r \neq 1$

**Telescoping**: $\sum_{k=1}^{n} (f(k) - f(k-1)) = f(n) - f(0)$

---

## Key Identities

- Difference of squares: $a^2 - b^2 = (a-b)(a+b)$
- Sophie Germain: $a^4 + 4b^4 = (a^2+2b^2+2ab)(a^2+2b^2-2ab)$
- Sum of cubes: $a^3 + b^3 = (a+b)(a^2-ab+b^2)$
