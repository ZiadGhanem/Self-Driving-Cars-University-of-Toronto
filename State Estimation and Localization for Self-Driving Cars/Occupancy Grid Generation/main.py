import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from IPython.display import HTML

def inverse_scanner(num_rows, num_columns, X, meas_phi, meas_r, rmax, alpha, beta):
    m = np.zeros((num_rows, num_columns))
    x = X[0]
    y = X[1]
    theta = X[2]
    for i in range(num_rows):
        for j in range(num_columns):
            # Find range and bearing
            r = np.sqrt((i - x)**2 + (j - y)**2)
            phi = (np.arctan2(j - y, i - x) - theta + math.pi) % (2 * math.pi) - math.pi

            # Find the range measurement
            k = np.argmin(np.abs(np.subtract(phi, meas_phi)))

            # If the range is greater than the maximum sensor range
            # or behind our range measurement
            # or outside the field of view of sensor then no information is available
            if (r > min(rmax, meas_r[k] + alpha / 2.0)) or (abs(phi - meas_phi[k]) > beta / 2.0):
                m[i, j] = 0.5

            # If the range measurement lied within this cell it is likely to be an object
            elif (meas_r[k] < rmax) and (abs(r - meas_r[k]) < alpha / 2.0):
                m[i, j] = 0.7

            # If the cell is in front of range measurement it is likely to be empty
            elif r < meas_r[k]:
                m[i, j] = 0.3

    return m

def get_ranges(true_map, X, meas_phi, rmax):
    (num_rows, num_columns) = true_map.shape
    meas_r = rmax * np.ones_like(meas_phi)
    x = X[0]
    y = X[1]
    theta = X[2]
    # Iterate over measurement bearing
    for i in range(len(meas_phi)):
        # Iterate over each unit step
        for r in range(1, rmax + 1):
            # Get coordinates of cell
            xi = int(round(x + r * math.cos(theta + meas_phi[i])))
            yi = int(round(y + r * math.sin(theta + meas_phi[i])))

            # If outside map set measurement there and stop going further
            if(xi <= 0 or xi >= num_rows-1 or yi <= 0 or yi >= num_columns-1):
                meas_r[i] = r
                break

            # If in the map, but hitting an obstacle, set the measurement range and stop ray tracing
            elif true_map[xi, yi] == 1:
                meas_r[i] = r
                break

    return meas_r



def main():
    # Simulation time initialization
    T_MAX = 150
    time_steps = np.arange(T_MAX)

    # Robot states
    x_0 = [30, 30, 0]
    X = np.zeros((3, len(time_steps)))
    X[:, 0] = x_0

    # Robot motion sequence
    u = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
    u_i = 1

    # Robot sensor rotation command
    w = np.multiply(0.3, np.ones(len(time_steps)))

    # True map
    num_rows = 50
    num_columns = 60
    true_map = np.zeros((num_rows, num_columns))
    true_map[0:10, 0:10] = 1
    true_map[30:35, 40:45] = 1
    true_map[3:6, 40:60] = 1
    true_map[20:30, 25:29] = 1
    true_map[40:50, 5:25] = 1

    # Initialize the belief map
    m = np.multiply(0.5, np.ones((num_rows, num_columns)))

    # Initialize the log odds ratio
    L0 = np.log(np.divide(num_rows, np.subtract(1, num_rows)))
    L = L0

    # parameters of sensor model
    meas_phi = np.arange(-0.4, 0.4, 0.05)
    rmax = 30
    alpha = 1
    beta = 0.05

    meas_rs = []
    meas_r = get_ranges(true_map, X[:, 0], meas_phi, rmax)
    meas_rs.append(meas_r)
    invmods = []
    invmod = inverse_scanner(num_rows, num_columns, X[:, 0], meas_phi, meas_r, rmax, alpha, beta)
    invmods.append(invmod)
    ms = []
    ms.append(m)

    # Main simulation loop.
    for t in range(1, len(time_steps)):
        # Perform robot motion.
        move = np.add(X[0:2, t - 1], u[:, u_i])
        # If we hit the map boundaries, or a collision would occur, remain still.
        if (move[0] >= num_rows - 1) or (move[1] >= num_columns - 1) or (move[0] <= 0) or (move[1] <= 0) \
                or true_map[int(round(move[0])), int(round(move[1]))] == 1:
            X[:, t] = X[:, t - 1]
            u_i = (u_i + 1) % 4
        else:
            X[0:2, t] = move
        X[2, t] = (X[2, t - 1] + w[t]) % (2 * math.pi)

        # Gather the measurement range data, which we will convert to occupancy probabilities
        # using our inverse measurement model.
        meas_r = get_ranges(true_map, X[:, t], meas_phi, rmax)
        meas_rs.append(meas_r)

        # Given our range measurements and our robot location, apply our inverse scanner model
        # to get our measure probabilities of occupancy.
        invmod = inverse_scanner(num_rows, num_columns, X[:, t], meas_phi, meas_r, rmax, alpha, beta)
        invmods.append(invmod)

        # Calculate and update the log odds of our occupancy grid, given our measured
        # occupancy probabilities from the inverse model.
        L = np.log(np.divide(invmod, np.subtract(1, invmod))) + L - L0

        # Calculate a grid of probabilities from the log odds.
        p = np.exp(L)
        m = p / (1 + p)
        ms.append(m)


if __name__=="__main__":
    main()