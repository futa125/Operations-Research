from typing import Dict, Tuple

import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum


def construct_LP_equivalent() -> Tuple[LpProblem, Dict[str, LpVariable]]:
    problem: LpProblem = LpProblem(name="robust-linear-program", sense=LpMaximize)

    x1: LpVariable = LpVariable(name="x1", lowBound=0, upBound=4)
    x2: LpVariable = LpVariable(name="x2", lowBound=0, upBound=6)

    variables: Dict[str, LpVariable] = {
        "x1": x1,
        "x2": x2,
    }

    ud = 4
    p1 = LpVariable.dicts(name="p1", indexs=range(ud), lowBound=0)
    d = np.array([0, 2, -4, 6]).T
    d1 = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])

    problem += 3 * x1 + 5 * x2

    problem += lpSum([p1[i] * d[i] for i in range(ud)]) <= 18
    problem += lpSum([p1[i] * d1[i, 0] for i in range(ud)]) == x1
    problem += lpSum([p1[i] * d1[i, 1] for i in range(ud)]) == x2

    return problem, variables
