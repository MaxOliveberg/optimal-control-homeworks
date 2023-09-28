import dataclasses
from typing import Callable
import matplotlib.pyplot as plt
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


def split_result(x: np.ndarray, config: Config):
    ret_f = []
    ret_l = []
    for k in range(config.N):
        ret_f.append(x[4 * k:4 * k + 2])
        ret_l.append(x[4 * k + 2: 4 * (k + 1)])
    return np.array(ret_f).transpose(), np.array(ret_l).transpose()


def plot_results(results, config):
    follower, leader = split_result(results.x, config)
    plt.plot(follower[0, :], follower[1, :], label="Follower Trajectory")
    plt.plot(leader[0, :], leader[1, :], label="Leader Trajectory")
    plt.scatter([config.x_f[0]], [config.x_f[1]], label="Follower initial state")
    plt.scatter([config.x_l[0]], [config.x_l[1]], label="Leader initial state")
    plt.scatter([config.x_e[0]], [config.x_e[1]], label="Exit")

    if config.circles is not None:
        for c in config.circles:
            thetas = np.arange(0, 2 * np.pi, 0.1)
            x_s = c.radius * np.cos(thetas) + c.point[0]
            y_s = c.radius * np.sin(thetas) + c.point[1]
            plt.plot(x_s, y_s, linestyle="--", color="black")

    plt.legend()


def plot_signal(u, config):
    times = np.arange(1, config.N + 1) * config.dt
    u = u.reshape(-1, 2).transpose()
    plt.plot(times, u[0, :], label="u_x")
    plt.plot(times, u[1, :], label="u_y")
    plt.legend()
    plt.xlabel("Time")
