"""Numerical methods with explicit feasibility and convergence evidence."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess
from scipy.linalg import hilbert
from utils import OUT, save


def conjugate_gradient(A, b, tol=1e-10, max_iter=None):
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if A.shape != (len(b), len(b)) or not np.allclose(A, A.T):
        raise ValueError("symmetric square matrix required")
    np.linalg.cholesky(A)
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rr = r @ r
    history = [float(np.linalg.norm(r))]
    limit = max_iter or len(b) * 10
    for _ in range(limit):
        if np.sqrt(rr) <= tol * max(1, np.linalg.norm(b)):
            break
        curvature = p @ A @ p
        if curvature <= 0:
            break
        alpha = rr / curvature
        x = x + alpha * p
        r = r - alpha * (A @ p)
        new = r @ r
        history.append(float(np.sqrt(new)))
        p = r + (new / rr) * p
        rr = new
    return x, history


def himmelblau(x):
    a, b = x
    return (a * a + b - 11) ** 2 + (a + b * b - 7) ** 2


def himmel_gradient(x):
    a, b = x
    return np.array(
        [
            4 * a * (a * a + b - 11) + 2 * (a + b * b - 7),
            2 * (a * a + b - 11) + 4 * b * (a + b * b - 7),
        ]
    )


def nonlinear_cg(start, tol=1e-7, max_iter=2000):
    x = np.asarray(start, float)
    g = himmel_gradient(x)
    d = -g
    values = [float(himmelblau(x))]
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        if d @ g >= -1e-12 * np.dot(g, g):
            d = -g
        step = 1.0
        slope = g @ d
        fx = himmelblau(x)
        for _ in range(60):
            if himmelblau(x + step * d) <= fx + 1e-4 * step * slope:
                break
            step *= 0.5
        else:
            break
        x = x + step * d
        new = himmel_gradient(x)
        beta = (new @ new) / (g @ g)
        d = -new + beta * d
        g = new
        values.append(float(himmelblau(x)))
    return x, values, float(np.linalg.norm(g))


def run():
    # Enumerate all derivative roots and endpoints for the scalar global certificate.
    f = lambda x: 2 * x**4 - 10 * x * x + x - 5
    roots = np.roots([8, 0, -20, 1])
    candidates = [-3.0, 3.0] + [
        float(z.real) for z in roots if abs(z.imag) < 1e-10 and -3 <= z.real <= 3
    ]
    winner = min(candidates, key=f)
    # This separable box problem also admits exact stationary/boundary candidates.
    xr = float(np.cbrt(-1 / 8))
    xs = [-3.0, 3.0, xr]
    ys = [-3.0, 3.0, -0.05]
    box = lambda x, y: 2 * x**4 - 10 * y * y + x - y - 25
    pairs = [(x, y, box(x, y)) for x in xs for y in ys]
    minimum = min(pairs, key=lambda a: a[2])
    maximum = max(pairs, key=lambda a: a[2])
    constrained = minimize(
        lambda z: (z[0] - 2) ** 2 + (z[1] - 1) ** 2,
        [0.0, 2.0],
        constraints=[
            {"type": "ineq", "fun": lambda z: z[1] - (z[0] + 1) ** 2},
            {"type": "ineq", "fun": lambda z: z[1] - z[0] - 1},
        ],
        method="SLSQP",
        options={"ftol": 1e-12},
    )
    assert constrained.success
    # Rosenbrock trust-region runs retain local status and gradient evidence.
    trusts = []
    for start in [[0, -1], [0, 0.5], [-1.2, 1]]:
        q = minimize(
            rosen,
            start,
            jac=rosen_der,
            hess=rosen_hess,
            method="trust-ncg",
            options={"gtol": 1e-8, "maxiter": 2000},
        )
        trusts.append(
            dict(
                start=start,
                point=q.x.tolist(),
                objective=float(q.fun),
                gradient_norm=float(np.linalg.norm(rosen_der(q.x))),
                success=bool(q.success),
            )
        )
    systems = {}
    for n in [5, 8, 12]:
        A = hilbert(n)
        b = np.ones(n)
        x, history = conjugate_gradient(A, b, max_iter=2000)
        systems[str(n)] = dict(
            iterations=len(history) - 1,
            condition_number=float(np.linalg.cond(A)),
            relative_residual=float(np.linalg.norm(A @ x - b) / np.linalg.norm(b)),
            residual_history=history,
        )
    nonlinear = []
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for start in [[0, 0], [-4, 0], [4, 0], [0, -4]]:
        x, values, gn = nonlinear_cg(start)
        nonlinear.append(
            dict(
                start=start,
                point=x.tolist(),
                objective=values[-1],
                gradient_norm=gn,
                iterations=len(values) - 1,
            )
        )
        ax[0].semilogy(np.maximum(values, 1e-16), label=str(start))
    # Weighted sums give candidate Pareto points, not a completeness certificate.
    pareto = []
    constraints = [
        {"type": "ineq", "fun": lambda z: z.sum() - 1},
        {"type": "ineq", "fun": lambda z: 4 - z @ z},
    ]
    for w in np.linspace(0, 1, 21):
        obj = lambda z: w * ((z[0] - 1) ** 2 + (z[1] + 2) ** 2) + (1 - w) * z[0] ** 2
        fits = [
            minimize(obj, s, method="SLSQP", constraints=constraints)
            for s in [[0, 1], [1.5, 0], [0.5, 0.5]]
        ]
        feasible = [
            q for q in fits if q.success and min(q.x.sum() - 1, 4 - q.x @ q.x) >= -1e-6
        ]
        if feasible:
            q = min(feasible, key=lambda v: v.fun)
            pareto.append(
                [
                    float(w),
                    *q.x.tolist(),
                    float((q.x[0] - 1) ** 2 + (q.x[1] + 2) ** 2),
                    float(q.x[0] ** 2),
                ]
            )
    ar = np.asarray(pareto)
    ax[1].scatter(ar[:, 3], ar[:, 4], c=ar[:, 0])
    ax[0].legend(fontsize=7)
    ax[0].set(xlabel="Iteration", ylabel="Himmelblau objective")
    ax[1].set(xlabel="Distance objective", ylabel="Squared x objective")
    fig.tight_layout()
    fig.savefig(OUT / "convergence-tradeoffs.png")
    plt.close(fig)
    return save(
        dict(
            scalar_global=dict(
                x=winner, objective=float(f(winner)), candidates=candidates
            ),
            box_global=dict(minimum=list(minimum), maximum=list(maximum)),
            constrained_projection=dict(
                point=constrained.x.tolist(),
                objective=float(constrained.fun),
                max_violation=float(
                    max(
                        0,
                        (constrained.x[0] + 1) ** 2 - constrained.x[1],
                        constrained.x[0] + 1 - constrained.x[1],
                    )
                ),
            ),
            trust_regions=trusts,
            hilbert_systems=systems,
            nonlinear_cg=nonlinear,
            pareto_candidates=pareto,
            scope="Ill-conditioned residual convergence does not guarantee accurate solutions. Weighted sums do not recover every Pareto point in general.",
        )
    )


if __name__ == "__main__":
    run()
