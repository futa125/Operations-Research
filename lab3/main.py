from typing import Dict, List, Tuple

from pulp import LpMaximize, LpProblem, LpVariable, LpInteger


def construct_LP_equivalent() -> Tuple[LpProblem, Dict[str, LpVariable]]:
    problem: LpProblem = LpProblem(name="robust-linear-program", sense=LpMaximize)

    x1: LpVariable = LpVariable(name="x1", lowBound=0, upBound=4, cat=LpInteger)
    x2: LpVariable = LpVariable(name="x2", lowBound=0, upBound=6, cat=LpInteger)

    variables: Dict[str, LpVariable] = {
        x1.name: x1,
        x2.name: x2,
    }

    problem += 3 * x1 + 5 * x2

    max_alpha: int = 6
    max_beta: int = 6
    points: List[Tuple[int, int]] = [
        (alpha, beta) for alpha in range(max_alpha + 1) for beta in range(max_beta + 1)
        if 0 <= alpha - beta <= 2 and 4 <= alpha + beta <= 6
    ]

    for alpha, beta in points:
        problem += alpha * x1 + beta * x2 <= 18

    return problem, variables
