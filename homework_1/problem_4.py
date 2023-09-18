import numpy as np

from homework_1.problem_4_helpers import get_optimal, plot_x, OptimalControlSettings, sim

if __name__ == "__main__":
    settings = OptimalControlSettings()
    settings.limit_sig = True
    settings.N = 25
    x = get_optimal(settings)
    sim(settings,steps=100)
