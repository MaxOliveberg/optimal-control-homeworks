import dataclasses

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

norm = 1  # 2 * np.pi / 360


@dataclasses.dataclass
class OptimalControlSettings:
    T: float = 0.1
    alpha: float = 0.5
    r: float = 1.0
    q: float = 1.0
    N: int = 5
    C: np.ndarray = np.array([1, 0])
    z_0: np.ndarray = np.array([0.5 * norm, 1 * norm])
    noise_std: float = 0.00
    sig_max: int = 1
    limit_sig: bool = False

    @property
    def Phi(self):
        return np.array([[1, self.T], [self.alpha * self.T, 1]])

    @property
    def Gamma(self):
        return np.array([(self.T ** 2) / 2, self.T])


def construct_H(Phi, Gamma, N, C, q, r):
    H = np.zeros((3 * N + 2, 3 * N + 2))
    H[:(2 * N + 2), :(2 * N + 2)] = q * np.identity(2 * N + 2)
    for k in range(N):  # Probably exists a better way of doing this
        cool_matrix = np.diag(C)
        H[2 * k:2 * k + 2, 2 * k:2 * k + 2, ] = q * cool_matrix
    H[(2 * N + 2):, (2 * N + 2):] = r * np.identity(N)
    print(H)
    return H


def construct_A(Phi, Gamma, N, C):
    """
    We have dynamics x_1 = Ax_0 + Bu_0
    The constraint is basically an eigenvalue equation for some A
    Ax = x -> (I -A)x = 0

    We also have some initial values z_0 which we handle by setting
    Ax[0:2] = z_0 -> A[0:2, 0:2] = I
    """
    A_0 = np.zeros(((2 * N + 2, 3 * N + 2)))
    A_0[0:2 * N + 2, :2 * N + 2] = np.identity(2 * N + 2)  # identity
    A_1 = np.zeros((2 * N + 2, 3 * N + 2))  # dynamics
    for k in range(N):  # Cheeky loop
        A_1[2 * k + 2:2 * k + 4, 2 * k:2 * k + 2] = Phi
        A_1[2 * k + 2:2 * k + 4, k + (2 * N + 2)] = Gamma
    return A_0 - A_1, A_1


def construct_A_limit(Phi, Gamma, N, C):
    """
    We have dynamics x_1 = Ax_0 + Bu_0
    The constraint is basically an eigenvalue equation for some A
    Ax = x -> (I -A)x = 0

    We also have some initial values z_0 which we handle by setting
    Ax[0:2] = z_0 -> A[0:2, 0:2] = I
    """
    A_0 = np.identity(3 * N + 2)  # identity
    A_1 = np.zeros((3 * N + 2, 3 * N + 2))  # dynamics
    for k in range(N):  # Cheeky loop
        A_1[2 * k + 2:2 * k + 4, 2 * k:2 * k + 2] = Phi
        A_1[2 * k + 2:2 * k + 4, k + (2 * N + 2)] = Gamma

    return A_0 - A_1, A_1


def cost_function(x, H):
    return (1 / 2) * (x.dot(H.dot(x)))


def get_optimal(settings: OptimalControlSettings = OptimalControlSettings()):
    # Construct H-matrix
    H = construct_H(settings.Phi, settings.Gamma, settings.N, settings.C, settings.q, settings.r)

    # Dynamics
    if settings.limit_sig:
        A, _ = construct_A_limit(settings.Phi, settings.Gamma, settings.N, settings.C)
        b = np.zeros(3 * settings.N + 2)
        b[0:2] = settings.z_0
        b_lower = b.copy()
        b_lower[2 * settings.N + 2:] = -settings.sig_max
        b_upper = b.copy()
        b_upper[2 * settings.N + 2:] = settings.sig_max
        constr = scipy.optimize.LinearConstraint(A, lb=b_lower, ub=b_upper)
    else:
        A, A_1 = construct_A(settings.Phi, settings.Gamma, settings.N, settings.C)
        b = np.zeros(2 * settings.N + 2)
        b[0:2] = settings.z_0
        constr = scipy.optimize.LinearConstraint(A, lb=b, ub=b)

    to_optim = lambda x: cost_function(x, H)  # Function to be optimised
    results = scipy.optimize.minimize(to_optim, x0=np.ones(3 * settings.N + 2), constraints=[constr])
    return results.x


def step(state, signal, settings: OptimalControlSettings, noise=True):
    # Takes one timestep
    noise = np.zeros(2) if noise is False else np.array([0, np.random.normal(0, settings.noise_std)])
    return settings.Phi.dot(state) + settings.Gamma * signal + noise


def sim(settings: OptimalControlSettings = OptimalControlSettings(), steps=100):
    # Controls the pendulum using MPC.
    state = settings.z_0
    settings = settings
    angles = [settings.z_0[0]]
    vels = [settings.z_0[1]]
    signals = []
    for k in range(steps):
        settings.z_0 = state
        x = get_optimal(settings)
        signal = x[2 * settings.N + 2]
        state = step(state, signal, settings)
        signals.append(signal)
        angles.append(state[0])
        vels.append(state[1])
    plot_sim(angles, vels, signals, settings)


def plot_sim(angles, vels, signals, settings):
    # Quick plot for realised paths.
    angles = np.array(angles)
    vel = np.array(vels)
    fig, axs = plt.subplots(3)
    axs[0].plot(angles, label="Angle")
    axs[0].legend()
    axs[1].plot(vel, label="Angular Velocity")
    axs[1].legend()
    signals = np.array(signals)
    axs[2].plot(signals, label="Signal")
    axs[2].legend()
    axs[2].set_xlabel("Time [s]")
    plt.suptitle(f"q: {settings.q}, r: {settings.r}, N: {settings.N}")
    plt.show()


def plot_x(x, N):
    # Quick plot for predictions.
    angles = np.array([x[2 * k] for k in range(N)])
    vel = np.array([x[1 + 2 * k] for k in range(N)])
    fig, axs = plt.subplots(3)
    axs[0].plot(angles)
    axs[1].plot(vel)
    signals = x[2 * N + 2:]
    axs[2].plot(signals)
    plt.show()
