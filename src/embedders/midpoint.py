import torch


def hyperbolic_midpoint(u, v, assert_hyperbolic=False):
    w = torch.sin(2 * u - 2 * v) / (torch.sin(u + v) * torch.sin(v - u))
    coef = -1 if u + v < torch.pi else 1
    sol = (-w + coef * torch.sqrt(w**2 - 4)) / 2
    m = torch.arctan2(torch.tensor(1), sol) % torch.pi
    if assert_hyperbolic:
        assert is_hyperbolic_midpoint(u, v, m)
    return m


def is_hyperbolic_midpoint(u, v, m):
    a = lambda x: torch.sqrt(-1 / torch.cos(2 * x))  # Alpha coefficient to reach manifold
    d = lambda x, y: a(x) * a(y) * torch.cos(x - y)  # Hyperbolic distance function (angular)
    return torch.isclose(d(u, m), d(m, v))


def spherical_midpoint(u, v):
    return (u + v) / 2


def euclidean_midpoint(u, v):
    return torch.arctan2(torch.tensor(2), (1 / torch.tan(u) + 1 / torch.tan(v)))


def midpoint(u, v, manifold):
    if torch.isclose(u, v):
        return u
    elif manifold.type == "H":
        return hyperbolic_midpoint(u, v)
    elif manifold.type == "S":
        return spherical_midpoint(u, v)
    elif manifold.type == "E":
        return euclidean_midpoint(u, v)
    else:
        raise ValueError(f"No midpoint formula for manifold type '{manifold.type}'")
    