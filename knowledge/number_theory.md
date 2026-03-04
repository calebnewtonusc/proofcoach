# Number Theory — ProofCoach Knowledge Base

## Divisibility

**Definition**: $a \mid b$ (a divides b) if there exists integer $k$ such that $b = ka$.

**Key properties**:
- If $a \mid b$ and $a \mid c$, then $a \mid (bx + cy)$ for all integers $x, y$.
- If $a \mid b$ and $b \mid c$, then $a \mid c$ (transitivity).

**GCD and LCM**:
- $\gcd(a, b) \cdot \text{lcm}(a, b) = ab$
- Bezout's lemma: $\gcd(a, b) = ax + by$ for some integers $x, y$.

---

## Modular Arithmetic

**Definition**: $a \equiv b \pmod{n}$ iff $n \mid (a - b)$.

**Operations preserved mod n**:
- Addition: $(a + b) \equiv (a \bmod n + b \bmod n) \pmod{n}$
- Multiplication: $(ab) \equiv (a \bmod n)(b \bmod n) \pmod{n}$

**NOT preserved**: Division (only when $\gcd(\text{divisor}, n) = 1$).

---

## Fermat's Little Theorem

If $p$ is prime and $\gcd(a, p) = 1$:
$$a^{p-1} \equiv 1 \pmod{p}$$

**Application**: Computing large powers mod prime.
$$a^{100} \pmod{7}: \text{ since } a^6 \equiv 1, \text{ use } 100 = 6 \cdot 16 + 4, \text{ so } a^{100} \equiv a^4 \pmod{7}$$

---

## Chinese Remainder Theorem (CRT)

If $\gcd(m_1, m_2) = 1$, then for any $a_1, a_2$:
$$x \equiv a_1 \pmod{m_1}, \quad x \equiv a_2 \pmod{m_2}$$
has a unique solution mod $m_1 m_2$.

**Constructive solution**: $x = a_1 M_1 y_1 + a_2 M_2 y_2 \pmod{M}$ where $M = m_1 m_2$, $M_i = M/m_i$, $y_i = M_i^{-1} \pmod{m_i}$.

---

## Key Competition Patterns

1. **Last digit problems**: Work mod 10, note period of last digits of powers.
2. **Divisibility conditions**: Often reduce to checking mod small primes.
3. **Sum of divisors**: $\sigma(p^k) = 1 + p + p^2 + \cdots + p^k = \frac{p^{k+1}-1}{p-1}$.
4. **Number of divisors**: $\tau(p_1^{a_1} \cdots p_r^{a_r}) = (a_1+1)(a_2+1)\cdots(a_r+1)$.
