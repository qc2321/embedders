import torch


def hyperbolic_midpoint(u, v, assert_hyperbolic=False):
    w = torch.sin(2.0 * u - 2.0 * v) / (torch.sin(u + v) * torch.sin(v - u))
    coef = -1.0 if u + v < torch.pi else 1.0
    sol = (-w + coef * torch.sqrt(w**2 - 4.0)) / 2.0
    m = torch.arctan2(torch.tensor(1.0), sol) % torch.pi
    if assert_hyperbolic:
        assert is_hyperbolic_midpoint(u, v, m)
    return m


def is_hyperbolic_midpoint(u, v, m):
    a = lambda x: torch.sqrt(-1.0 / torch.cos(2.0 * x))  # Alpha coefficient to reach manifold
    d = lambda x, y: a(x) * a(y) * torch.cos(x - y)  # Hyperbolic distance function (angular)
    return torch.isclose(d(u, m), d(m, v))


def spherical_midpoint(u, v):
    return (u + v) / 2.0


def euclidean_midpoint(u, v):
    return torch.arctan2(torch.tensor(2.0), (1.0 / torch.tan(u) + 1.0 / torch.tan(v)))


def midpoint(u, v, manifold, special_first=False):
    if torch.isclose(u, v):
        return u
    elif manifold.type == "H" and special_first:
        return hyperbolic_midpoint(u, v)
    elif manifold.type == "H":
        return spherical_midpoint(u, v)  # Naive bisection
    elif manifold.type == "S":
        return spherical_midpoint(u, v)
    elif manifold.type == "E":
        return euclidean_midpoint(u, v)
    else:
        raise ValueError(f"No midpoint formula for manifold type '{manifold.type}'")
