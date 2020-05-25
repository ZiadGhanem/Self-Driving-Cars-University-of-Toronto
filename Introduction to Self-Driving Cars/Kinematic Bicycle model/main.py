import numpy as np
import matplotlib.pyplot as plt

class Bicycle():
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

        self.L = 2
        self.lr = 1.2
        self.w_max = 1.22

        self.sample_time = 0.01

    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

    def step(self, v, w):
        self.xc += (v * np.cos(self.theta + self.beta)) * self.sample_time
        self.yc += (v * np.sin(self.theta + self.beta)) * self.sample_time
        self.theta += (v * np.cos(self.beta) * np.tan(self.delta) / self.L) * self.sample_time
        self.delta += max(-self.w_max, min(self.w_max, w)) * self.sample_time
        self.beta = np.arctan(self.lr * np.tan(self.delta)/ self.L)


def circular_path(path_radius, time_end):
    v = (2 * np.pi * path_radius) / time_end
    w = 0
    model = Bicycle()
    model.delta = np.arctan(model.L / path_radius)

    t_data = np.arange(0, time_end, model.sample_time)
    x_data = np.zeros_like(t_data)
    y_data = np.zeros_like(t_data)

    for i in range(t_data.shape[0]):
        x_data[i] = model.xc
        y_data[i] = model.yc
        model.step(v, w)

    plt.plot(x_data, y_data)
    plt.show()

def infinity_path(path_radius, time_end):
    model = Bicycle()

    t_data = np.arange(0, time_end, model.sample_time)
    x_data = np.zeros_like(t_data)
    y_data = np.zeros_like(t_data)

    n = t_data.shape[0]
    v = 2 * 2 * np.pi * path_radius / time_end
    max_delta = 0.993 * np.arctan(model.L / path_radius)

    for i in range(n):
        x_data[i] = model.xc
        y_data[i] = model.yc

        if (i > 5*n/8 or i < n/8) and model.delta < max_delta:
            w = model.w_max
        elif 5*n/8 > i > n/8 and model.delta > - max_delta:
            w = -model.w_max
        else:
            w = 0

        model.step(v, w)

    plt.plot(x_data, y_data)
    plt.show()


def main():
    infinity_path(8, 30)

if __name__ == "__main__":
    main()
