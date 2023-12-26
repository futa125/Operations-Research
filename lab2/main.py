from typing import Dict

import numpy as np
import numpy.typing as npt
from pulp import LpBinary, LpMinimize, LpProblem, LpVariable, PULP_CBC_CMD, PULP_CHOCO_CMD, lpSum


def construct_problem(costs: npt.NDArray[np.int64]) -> LpProblem:
    rows: int
    columns: int
    rows, columns = costs.shape

    x: Dict[int, Dict[int, LpVariable]] = LpVariable.dicts(name="x", indexs=(range(rows), range(columns)), cat=LpBinary)

    problem: LpProblem = LpProblem(name="assignment_problem", sense=LpMinimize)
    problem += lpSum(costs[i][j] * x[i][j] for i in range(rows) for j in range(columns))

    for i in range(rows):
        problem += lpSum(x[i][j] for j in range(columns)) == 1

    for j in range(columns):
        problem += lpSum(x[i][j] for i in range(rows)) == 1

    return problem


def solve_LP(problem: LpProblem) -> LpProblem:
    problem.solve(solver=PULP_CBC_CMD(msg=False))

    return problem


def solve_CP(problem: LpProblem) -> LpProblem:
    problem.solve(solver=PULP_CHOCO_CMD(msg=False))

    return problem


np.random.seed(3652249321)

mali_costs: npt.NDArray[np.int64] = np.random.randint(low=1, high=20 + 1, size=(8, 8), dtype=np.int64)
veliki_costs: npt.NDArray[np.int64] = np.random.randint(low=1, high=20 + 1, size=(16, 16), dtype=np.int64)
