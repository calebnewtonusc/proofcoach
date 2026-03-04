"""
Problem Taxonomy — Topic taxonomy with prerequisite graph

Defines the mathematical topic hierarchy used for:
  - Student knowledge tracking (skill_model.py)
  - Adaptive practice sequencing (practice_sequencer_agent.py)
  - Problem tagging and search

The prerequisite graph ensures we never teach generating functions
before counting basics — the system enforces correct pedagogical order.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TopicNode:
    """A node in the mathematical topic taxonomy."""
    id: str
    name: str
    description: str
    category: str                    # "number_theory", "combinatorics", "algebra", "geometry"
    prerequisites: list[str]         # IDs of prerequisite nodes
    difficulty_range: tuple[int, int]  # (min, max) difficulty 1-10
    example_problems: list[str]      # Example problem IDs


# ---------------------------------------------------------------------------
# Full Taxonomy
# ---------------------------------------------------------------------------

TAXONOMY: dict[str, TopicNode] = {
    # -----------------------------------------------------------------------
    # Number Theory
    # -----------------------------------------------------------------------
    "divisibility_basics": TopicNode(
        id="divisibility_basics",
        name="Divisibility and Factors",
        description="Division algorithm, factors, multiples, GCD, LCM",
        category="number_theory",
        prerequisites=[],
        difficulty_range=(1, 4),
        example_problems=["AMC_8-2015-05", "AMC_10A-2010-03"],
    ),
    "prime_numbers": TopicNode(
        id="prime_numbers",
        name="Prime Numbers",
        description="Primes, composite numbers, fundamental theorem of arithmetic, prime factorization",
        category="number_theory",
        prerequisites=["divisibility_basics"],
        difficulty_range=(2, 5),
        example_problems=["AMC_10A-2012-08", "AIME_I-2015-04"],
    ),
    "modular_arithmetic": TopicNode(
        id="modular_arithmetic",
        name="Modular Arithmetic",
        description="Congruences, residues, arithmetic mod n, properties of modular operations",
        category="number_theory",
        prerequisites=["divisibility_basics", "prime_numbers"],
        difficulty_range=(3, 7),
        example_problems=["AMC_12A-2018-14", "AIME_I-2019-07"],
    ),
    "fermat_euler": TopicNode(
        id="fermat_euler",
        name="Fermat's Little Theorem and Euler's Theorem",
        description="a^(p-1) ≡ 1 (mod p) for prime p, Euler's totient function",
        category="number_theory",
        prerequisites=["modular_arithmetic", "prime_numbers"],
        difficulty_range=(5, 8),
        example_problems=["AIME_I-2000-09", "USAMO-2003-01"],
    ),
    "chinese_remainder": TopicNode(
        id="chinese_remainder",
        name="Chinese Remainder Theorem",
        description="Simultaneous congruences, constructive CRT",
        category="number_theory",
        prerequisites=["modular_arithmetic", "divisibility_basics"],
        difficulty_range=(5, 8),
        example_problems=["AIME_II-2011-04", "USAMO-2010-02"],
    ),
    "number_theory_functions": TopicNode(
        id="number_theory_functions",
        name="Multiplicative Functions",
        description="Euler's totient, Mobius function, sum of divisors, number of divisors",
        category="number_theory",
        prerequisites=["prime_numbers", "fermat_euler"],
        difficulty_range=(6, 9),
        example_problems=["USAMO-2014-01", "IMO-2005-06"],
    ),
    "diophantine_equations": TopicNode(
        id="diophantine_equations",
        name="Diophantine Equations",
        description="Integer solutions to equations, Bezout's lemma, Pell equations",
        category="number_theory",
        prerequisites=["divisibility_basics", "modular_arithmetic"],
        difficulty_range=(5, 9),
        example_problems=["AIME_I-2006-05", "IMO-2007-02"],
    ),
    "quadratic_residues": TopicNode(
        id="quadratic_residues",
        name="Quadratic Residues",
        description="Legendre symbol, quadratic reciprocity, squares mod p",
        category="number_theory",
        prerequisites=["modular_arithmetic", "fermat_euler"],
        difficulty_range=(7, 10),
        example_problems=["USAMO-2015-06", "IMO-2013-02"],
    ),

    # -----------------------------------------------------------------------
    # Combinatorics
    # -----------------------------------------------------------------------
    "counting_basics": TopicNode(
        id="counting_basics",
        name="Basic Counting",
        description="Multiplication principle, permutations, combinations",
        category="combinatorics",
        prerequisites=[],
        difficulty_range=(1, 4),
        example_problems=["AMC_8-2018-10", "AMC_10A-2014-06"],
    ),
    "binomial_theorem": TopicNode(
        id="binomial_theorem",
        name="Binomial Theorem",
        description="(a+b)^n expansion, binomial coefficients, Pascal's triangle",
        category="combinatorics",
        prerequisites=["counting_basics"],
        difficulty_range=(3, 6),
        example_problems=["AMC_12A-2019-12", "AIME_I-2014-03"],
    ),
    "pigeonhole": TopicNode(
        id="pigeonhole",
        name="Pigeonhole Principle",
        description="Basic pigeonhole, generalized pigeonhole, applications",
        category="combinatorics",
        prerequisites=["counting_basics"],
        difficulty_range=(3, 7),
        example_problems=["AMC_10A-2016-15", "USAMO-2001-01"],
    ),
    "inclusion_exclusion": TopicNode(
        id="inclusion_exclusion",
        name="Inclusion-Exclusion Principle",
        description="|A ∪ B| = |A| + |B| - |A ∩ B|, generalized to n sets",
        category="combinatorics",
        prerequisites=["counting_basics"],
        difficulty_range=(4, 7),
        example_problems=["AIME_I-2011-05", "USAMO-2011-02"],
    ),
    "generating_functions": TopicNode(
        id="generating_functions",
        name="Generating Functions",
        description="Ordinary and exponential generating functions, counting with GFs",
        category="combinatorics",
        prerequisites=["counting_basics", "binomial_theorem", "inclusion_exclusion"],
        difficulty_range=(6, 10),
        example_problems=["USAMO-2013-02", "IMO-2006-06"],
    ),
    "graph_theory": TopicNode(
        id="graph_theory",
        name="Graph Theory",
        description="Paths, cycles, trees, colorings, Euler characteristic",
        category="combinatorics",
        prerequisites=["counting_basics"],
        difficulty_range=(5, 9),
        example_problems=["USAMO-2007-01", "IMO-2011-03"],
    ),
    "combinatorial_games": TopicNode(
        id="combinatorial_games",
        name="Combinatorial Game Theory",
        description="Nim, Sprague-Grundy theorem, winning strategies",
        category="combinatorics",
        prerequisites=["counting_basics", "modular_arithmetic"],
        difficulty_range=(5, 9),
        example_problems=["USAMO-2005-02", "IMO-2014-03"],
    ),

    # -----------------------------------------------------------------------
    # Algebra
    # -----------------------------------------------------------------------
    "polynomial_basics": TopicNode(
        id="polynomial_basics",
        name="Polynomials",
        description="Roots, factoring, Vieta's formulas, remainder theorem",
        category="algebra",
        prerequisites=[],
        difficulty_range=(2, 6),
        example_problems=["AMC_12A-2020-15", "AIME_I-2018-06"],
    ),
    "inequalities": TopicNode(
        id="inequalities",
        name="Algebraic Inequalities",
        description="AM-GM, Cauchy-Schwarz, power mean, rearrangement",
        category="algebra",
        prerequisites=["polynomial_basics"],
        difficulty_range=(4, 8),
        example_problems=["USAMO-2008-01", "IMO-2000-02"],
    ),
    "sequences_series": TopicNode(
        id="sequences_series",
        name="Sequences and Series",
        description="Arithmetic/geometric series, telescoping, recurrences",
        category="algebra",
        prerequisites=["polynomial_basics"],
        difficulty_range=(3, 7),
        example_problems=["AIME_I-2016-08", "USAMO-2016-02"],
    ),
    "complex_numbers": TopicNode(
        id="complex_numbers",
        name="Complex Numbers",
        description="Complex plane, roots of unity, de Moivre's theorem",
        category="algebra",
        prerequisites=["polynomial_basics"],
        difficulty_range=(5, 9),
        example_problems=["USAMO-2012-03", "IMO-2009-01"],
    ),
    "functional_equations": TopicNode(
        id="functional_equations",
        name="Functional Equations",
        description="Cauchy's equation, finding functions satisfying f(f(x))=x, etc.",
        category="algebra",
        prerequisites=["polynomial_basics", "inequalities"],
        difficulty_range=(6, 10),
        example_problems=["IMO-2010-01", "USAMO-2010-04"],
    ),

    # -----------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------
    "triangle_basics": TopicNode(
        id="triangle_basics",
        name="Triangle Geometry",
        description="Area formulas, similar triangles, basic angle chasing",
        category="geometry",
        prerequisites=[],
        difficulty_range=(1, 4),
        example_problems=["AMC_8-2019-15", "AMC_10A-2015-08"],
    ),
    "circle_theorems": TopicNode(
        id="circle_theorems",
        name="Circle Theorems",
        description="Inscribed angle, power of a point, tangent lines",
        category="geometry",
        prerequisites=["triangle_basics"],
        difficulty_range=(4, 7),
        example_problems=["AMC_12A-2017-14", "AIME_I-2013-07"],
    ),
    "trigonometry": TopicNode(
        id="trigonometry",
        name="Trigonometry",
        description="Law of sines/cosines, angle formulas, identities",
        category="geometry",
        prerequisites=["triangle_basics"],
        difficulty_range=(4, 7),
        example_problems=["AIME_II-2019-12", "USAMO-2006-04"],
    ),
    "coordinates": TopicNode(
        id="coordinates",
        name="Coordinate Geometry",
        description="Cartesian coordinates, distance formula, conic sections",
        category="geometry",
        prerequisites=["triangle_basics"],
        difficulty_range=(3, 7),
        example_problems=["AMC_12A-2021-17", "AIME_I-2021-10"],
    ),
    "advanced_euclidean": TopicNode(
        id="advanced_euclidean",
        name="Advanced Euclidean Geometry",
        description="Ptolemy, Menelaus, Ceva, projective techniques",
        category="geometry",
        prerequisites=["circle_theorems", "trigonometry"],
        difficulty_range=(7, 10),
        example_problems=["USAMO-2014-03", "IMO-2003-04"],
    ),
}


# ---------------------------------------------------------------------------
# Prerequisite Graph Operations
# ---------------------------------------------------------------------------

def get_prerequisites(topic_id: str, transitive: bool = True) -> set[str]:
    """
    Get all prerequisites for a topic (transitively if requested).

    Args:
        topic_id: The topic to get prerequisites for
        transitive: If True, include prerequisites of prerequisites

    Returns:
        Set of prerequisite topic IDs
    """
    if topic_id not in TAXONOMY:
        return set()

    node = TAXONOMY[topic_id]
    direct_prereqs = set(node.prerequisites)

    if not transitive:
        return direct_prereqs

    all_prereqs = set()
    to_process = list(direct_prereqs)
    while to_process:
        current = to_process.pop()
        if current not in all_prereqs and current in TAXONOMY:
            all_prereqs.add(current)
            to_process.extend(TAXONOMY[current].prerequisites)

    return all_prereqs


def get_unlockable_topics(mastered: set[str]) -> set[str]:
    """
    Get topics that are now unlockable given the set of mastered topics.

    A topic is unlockable if all its prerequisites are mastered.
    """
    unlockable = set()
    for topic_id, node in TAXONOMY.items():
        if topic_id in mastered:
            continue
        if all(prereq in mastered for prereq in node.prerequisites):
            unlockable.add(topic_id)
    return unlockable


def get_topics_by_category(category: str) -> list[TopicNode]:
    """Get all topics in a category, sorted by difficulty."""
    nodes = [n for n in TAXONOMY.values() if n.category == category]
    return sorted(nodes, key=lambda n: n.difficulty_range[0])


def get_topic_by_id(topic_id: str) -> Optional[TopicNode]:
    """Get a topic node by ID."""
    return TAXONOMY.get(topic_id)


def get_all_topics() -> list[TopicNode]:
    """Get all topic nodes."""
    return list(TAXONOMY.values())


def get_learning_order() -> list[str]:
    """
    Topological sort of topics respecting prerequisite order.

    Returns a valid order to learn all topics.
    """
    result = []
    visited = set()
    temp_mark = set()

    def visit(node_id: str) -> None:
        if node_id in temp_mark:
            raise ValueError(f"Cycle detected at {node_id}")
        if node_id in visited:
            return
        temp_mark.add(node_id)
        node = TAXONOMY.get(node_id)
        if node:
            for prereq in node.prerequisites:
                visit(prereq)
        temp_mark.discard(node_id)
        visited.add(node_id)
        result.append(node_id)

    for topic_id in TAXONOMY:
        visit(topic_id)

    return result


LEARNING_ORDER = get_learning_order()
