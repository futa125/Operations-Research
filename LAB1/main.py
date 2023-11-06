"""
Neko poduzeće CBA proizvodi tri tipa istovrsnog proizvoda A, B i C.
Ugovoreno je da se trgovačkoj mreži isporuči točno 300 tona tog proizvoda bez obzira na tip.
Za potrebe proizvodnje treba koristiti određene kemikalije u iznosu od 3, 4 i 2 litra po jednoj toni proizvoda A, B i C,
pri čemu su odobrena sredstva za uvoz 1000 litara te kemikalije.
Također je odlučeno da u procesu proizvodnje radnike treba uposliti na najmanje 120 radnih sati,
a zna se da je za proizvodnju nužno uložiti 2, 1 odnosno 3 radna sata po kilogramu proizvoda A, B i C.
Odrediti optimalni plan proizvodnje ukoliko se zna da prodajne cijene ovih proizvoda iznose 1000, 2000 odnosno 800 eura
po toni za proizvode A, B i C.
"""

import numpy as np
import matplotlib.pyplot as plt

from pulp import LpProblem, LpMaximize, LpVariable, LpInteger, LpMinimize, PULP_CBC_CMD, LpSolutionOptimal, LpSenses


def print_solution(problem: LpProblem) -> None:
    print(f"# {problem.name}")
    print(f"{LpSenses[problem.sense]}: {problem.objective}")
    for k in problem.constraints:
        print(problem.constraints[k])
    print()

    status = problem.solve(solver=PULP_CBC_CMD(msg=False))

    print(f"Objective function value: {problem.objective.value()}")
    print(f"Is optimal solution? {status == LpSolutionOptimal}")
    for x in problem.variables():
        print(x, x.value())
    print()


def main() -> None:
    problem = LpProblem("maximise_profit", LpMaximize)

    x1 = LpVariable("x1", 0, None, LpInteger)
    x2 = LpVariable("x2", 0, None, LpInteger)
    x3 = LpVariable("x3", 0, None, LpInteger)

    # Odrediti optimalni plan proizvodnje ukoliko se zna da prodajne cijene ovih proizvoda
    # iznose 1000, 2000 odnosno 800 eura po toni za proizvode A, B i C.
    problem += 1000 * x1 + 2000 * x2 + 800 * x3

    # Ugovoreno je da se trgovačkoj mreži isporuči točno 300 tona tog proizvoda bez obzira na tip.
    problem += x1 + x2 + x3 == 300

    # Za potrebe proizvodnje treba koristiti određene kemikalije u iznosu od 3, 4 i 2 litra
    # po jednoj toni proizvoda A, B i C, pri čemu su odobrena sredstva za uvoz 1000 litara te kemikalije.
    problem += 3 * x1 + 4 * x2 + 2 * x3 <= 1000

    # Također je odlučeno da u procesu proizvodnje radnike treba uposliti na najmanje 120 radnih sati,
    # a zna se da je za proizvodnju nužno uložiti 2, 1 odnosno 3 radna sata po kilogramu proizvoda A, B i C.
    problem += 2 * 1000 * x1 + 1 * 1000 * x2 + 3 * 1000 * x3 >= 120

    print_solution(problem)

    problem_simplex = LpProblem("maximise_profit_simplex", LpMaximize)

    x1 = LpVariable("x1", 0, None, LpInteger)
    x2 = LpVariable("x2", 0, None, LpInteger)
    x3 = LpVariable("x3", 0, None, LpInteger)

    problem_simplex += 1000 * x1 + 2000 * x2 + 800 * x3

    problem_simplex += -x1 - x2 - x3 <= -300
    problem_simplex += x1 + x2 + x3 <= 300
    problem_simplex += 3 * x1 + 4 * x2 + 2 * x3 <= 1000
    problem_simplex += -2 * 1000 * x1 - 1 * 1000 * x2 - 3 * 1000 * x3 <= -120

    print_solution(problem_simplex)

    problem_dual_simplex = LpProblem("maximise_profit_dual_simplex", LpMinimize)

    y1 = LpVariable("y1", 0, None, LpInteger)
    y2 = LpVariable("y2", 0, None, LpInteger)
    y3 = LpVariable("y3", 0, None, LpInteger)
    y4 = LpVariable("y4", 0, None, LpInteger)

    problem_dual_simplex += -300 * y1 + 300 * y2 + 1000 * y3 - 120 * y4

    problem_dual_simplex += -1 * y1 + y2 + 3 * y3 - 2000 * y4 >= 1000
    problem_dual_simplex += -1 * y1 + y2 + 4 * y3 - 1000 * y4 >= 2000
    problem_dual_simplex += -1 * y1 + y2 + 2 * y3 - 3000 * y4 >= 800

    problem_dual_simplex.solve(solver=PULP_CBC_CMD(msg=False))

    print_solution(problem_dual_simplex)

    # x + y + z = 300
    # z = 300 - x - y

    # 3x + 4y + 2z = 1000
    # 2z = 1000 - 3x - 4y
    # z = (1000 - 3x - 4y) / 2

    # 2000x + 1000y + 3000z = 120
    # 3000z = 120 - 2000x - 1000y
    # z = (120 - 2000x - 1000y) / 3000

    x, y = np.meshgrid(np.linspace(0, 300), np.linspace(0, 300))

    c1 = 300 - x - y
    c2 = (1000 - 3 * x - 4 * y) / 2
    c3 = (120 - 2000 * x - 1000 * y) / 3000

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot_surface(x, y, c1, label="x1 + x2 + x3 = 300")
    ax.plot_surface(x, y, c2, label="3x1 + 4x2 - 2x3 = 1000")
    ax.plot_surface(x, y, c3, label="-2000x1 - 1000x2 + 3000x3 = -120")

    ax.scatter(0, 200, 100, label='Optimal solution (x1=0, x2=200, x3=100)')

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
