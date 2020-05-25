import numpy as np
import matplotlib.pyplot as plt

class Vehicle():
    def __init__(self):
        # Throttle to engine torque
        self.a_0 = 400
        self.a_1 = 0.1
        self.a_2 = -0.0002

        # Gear ratio, effective radius, mass + inertia
        self.GR = 0.35
        self.r_e = 0.3
        self.J_e = 10
        self.m = 2000
        self.g = 9.81

        # Aerodynamic and friction coefficients
        self.c_a = 1.36
        self.c_r1 = 0.01

        # Tire force
        self.c = 10000
        self.F_max = 10000

        # State variables
        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0

        self.sample_time = 0.01

    def reset(self):
        # reset state variables
        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0

    def step(self, throttle, alpha):

        self.x += self.v * self.sample_time
        self.v += self.a * self.sample_time
        self.w_e += self.w_e_dot * self.sample_time

        T_e = throttle * (self.a_0 + self.a_1 * self.w_e + self.a_2 * self.w_e ** 2)

        F_aero = self.c_a * self.v ** 2
        R_x = self.c_r1 * self.v
        F_g = self.m * self.g * np.sin(alpha)
        F_load = F_aero + R_x + F_g

        omega_w = self.GR * self.w_e
        s = (omega_w * self.r_e - self.v) / self.v
        F_x = self.c * s if abs(s) < 1 else self.F_max
        self.a = (F_x - F_load) / self.m
        self.w_e_dot = (T_e - self.GR * self.r_e * F_load) / self.J_e


def main():
    time_end = 20

    model = Vehicle()

    t_data = np.arange(0, time_end, model.sample_time)
    x_data = np.zeros_like(t_data)
    v_data = np.zeros_like(t_data)
    throttle_data = np.zeros_like(t_data)
    alpha_data = np.zeros_like(t_data)

    n = t_data.shape[0]
    throttle_data[: n // 4] = [0.2 + 0.3 * i / (n // 4) for i in range(n // 4)]
    throttle_data[n // 4:3 * n // 4] = 0.5
    throttle_data[3 * n // 4:] = [0.5 - 0.5 * i / (n // 4) for i in range(n // 4)]

    for i in range(n):

        x_data[i] = model.x
        v_data[i] = model.v
        if model.x < 60:
            alpha_data[i] = np.arctan(3 / 60)
        elif model.x < 150:
            alpha_data[i] = np.arctan(9 / 90)

        model.step(throttle_data[i], alpha_data[i])

    plt.plot(t_data, x_data)
    plt.show()



if __name__ == "__main__":
    main()