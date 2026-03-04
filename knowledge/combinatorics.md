# Combinatorics — ProofCoach Knowledge Base

## Counting Fundamentals

**Multiplication principle**: If event A has $m$ outcomes and B has $n$ outcomes, together they have $mn$.

**Permutations**: $P(n, k) = \frac{n!}{(n-k)!}$ — ordered selections.

**Combinations**: $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ — unordered selections.

---

## Binomial Theorem

$$(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^k b^{n-k}$$

**Key identities**:
- $\sum_{k=0}^{n} \binom{n}{k} = 2^n$ (set $a=b=1$)
- $\sum_{k=0}^{n} (-1)^k \binom{n}{k} = 0$ (set $a=1, b=-1$)
- $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$ (Pascal's identity)

---

## Pigeonhole Principle

**Basic**: If $n+1$ objects are placed in $n$ boxes, at least one box contains $\geq 2$ objects.

**Generalized**: If $kn+1$ objects are in $n$ boxes, some box has $\geq k+1$ objects.

**Competition application**: Often the hard part is identifying the "boxes" and "objects."

---

## Inclusion-Exclusion

$$|A_1 \cup A_2 \cup \cdots \cup A_n| = \sum|A_i| - \sum|A_i \cap A_j| + \cdots + (-1)^{n+1}|A_1 \cap \cdots \cap A_n|$$

**Derangements**: $D_n = n!\sum_{k=0}^{n} \frac{(-1)^k}{k!} \approx n!/e$

---

## Generating Functions

**Ordinary GF**: $F(x) = \sum_{n \geq 0} a_n x^n$ where $a_n$ counts something.

**Partition**: Number of ways to write $n$ as ordered sum of positive integers.

**Stars and bars**: $\binom{n+k-1}{k-1}$ ways to put $n$ indistinct balls in $k$ distinct bins.
