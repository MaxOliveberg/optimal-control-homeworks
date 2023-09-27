import dataclasses
from typing import Callable

import numpy as np


def psi(s):
    return 3 * (s ** 3) - 14 * (s ** 2) + 20 * s - 8


@dataclasses.dataclass
class Circle:
    point: np.ndarray
    radius: float


@dataclasses.dataclass
class Config:
    k: float = 1.0
    d: float = 1.0
    epsilon: float = 1.0
    T: float = 4
    dt: float = 0.1
    basis_N: int = 1
    distance_scaling: float = 1.0
    x_l: np.ndarray = np.array([2, 2])
    x_f: np.ndarray = np.array([1, 2])
    x_e: np.ndarray = np.array([5, 0])
    psi: Callable[[float], float] = psi
    circles: list[Circle] = None

    @property
    def x0(self) -> np.ndarray:
        ret = np.zeros(4)
        ret[:2] = self.x_f
        ret[2:] = self.x_l
        return ret

    @property
    def N(self) -> int:
        return int(self.T / self.dt)

    def g(self, beta: np.ndarray):
        beta_norm = np.sqrt(beta.dot(beta))
        if beta_norm <= self.d:
            return self.k * beta
        elif self.d < beta_norm <= self.d + self.epsilon:
            return self.psi(beta_norm) * (beta / beta_norm)
        else:
            return np.zeros(2)

    def f(self, x: np.ndarray, u: np.ndarray):
        """
        We use the convention x = [x_f, x_l] is a 4-vector, u = [u_x, u_y] is a 2-vector.
        """
        ret = np.zeros(4)
        ret[:2] = self.g(x[2:] - x[:2])
        ret[2:] = u
        return ret

    def f_0(self, x: np.ndarray, u: np.ndarray):
        return u.dot(u)

    def phi(self, x: np.ndarray):
        diff = x[:2] - self.x_e
        return diff.dot(diff)
