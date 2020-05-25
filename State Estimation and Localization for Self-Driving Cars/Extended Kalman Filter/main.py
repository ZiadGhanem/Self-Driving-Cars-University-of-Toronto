import pickle
import numpy as np
import matplotlib.pyplot as plt


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def measurement_update(d, l_k, range_measure, bearing_measure, p_predict, x_predict, R_covar):
    # Covert theta to [-pi, pi]
    x_predict[2] = wrap_to_pi(x_predict[2])

    # y_estimate[k] = h[k](x_predict[k], 0)
    d_x = l_k[0] - x_predict[0] - d * np.cos(x_predict[2])
    d_y = l_k[1] - x_predict[1] - d * np.sin(x_predict[2])
    r_estimate = np.sqrt(np.square(d_x) + np.square(d_y))
    phi_estimate = np.arctan2(d_y, d_x) - x_predict[2]
    y_estimate = np.vstack([r_estimate, wrap_to_pi(phi_estimate)])

    # Create y_measure[k]
    y_measure = np.vstack([range_measure, wrap_to_pi(bearing_measure)])

    # H[k] = dh[k]/dx[k] @x_predict, 0
    H_k = np.zeros((2, 3))
    H_k[0, 0] = -d_x / r_estimate
    H_k[0, 1] = -d_y / r_estimate
    H_k[0, 2] = d * (d_x * np.sin(x_predict[2]) - d_y * np.cos(x_predict[2])) / r_estimate
    H_k[1, 0] = d_y / r_estimate ** 2
    H_k[1, 1] = -d_x / r_estimate ** 2
    H_k[1, 2] = -1 - d * (d_y * np.sin(x_predict[2]) + d_x * np.cos(x_predict[2])) / r_estimate ** 2

    # M[k] = dh[k]/dv[k] @x_predict, 0
    M_k = np.identity(2)

    # K[k] = P_predict[k].H[k](T).(H[k].P_predict[k].H[k](T) + M[k].R[k].M[k](T))^-1
    K_k = p_predict.dot(H_k.T).dot(np.linalg.inv(H_k.dot(p_predict).dot(H_k.T) + M_k.dot(R_covar).dot(M_k.T)))

    # X_estimate[k] = X_predict[k] + K[k].(y_measure[k] - y_estimate[k])
    x_estimate = x_predict + K_k.dot(y_measure - y_estimate)

    # P_estimate[k] = (1 - K[k].H[k]).P[k]
    p_estimate = (np.identity(3) - K_k.dot(H_k)).dot(p_predict)

    return x_estimate, p_estimate



def main():
    DATA_FILE = "data/data.pickle"

    with open(DATA_FILE, 'rb') as file:
        data = pickle.load(file)

    # Timestamps [s]
    t = data['t']
    # X, Y positions of landmarks
    l = data['l']
    # Distance between vehicle center and LIDAR
    d = data['d']

    #############################INPUT DATA#############################
    # Translational velocity input [m/s]
    velocity = data['v']
    # Rotational velocity input [rad/s]
    rot_velocity = data['om']
    # Velocity variance
    velocity_var = 0.01
    # Rotational velocity variance
    rot_velocity_var = 0.01
    # Input noise covariance
    Q_covar = np.diag([velocity_var, rot_velocity_var])
    ####################################################################

    ##########################MEASUREMENT DATA##########################
    # Range measurements [m]
    range_measure = data['r']
    # Bearing to each landmark center [rad]
    bearing_measure = data['b']
    # Range measurements variance
    range_var = 0.1
    # Bearing measurement variance
    bearing_var = 10
    # Measurement data covariance
    R_covar = np.diag([range_var, bearing_var])
    #####################################################################

    #############################MODEL DATA#############################
    # Initial X position [m]
    x_init = data['x_init']
    # Initial Y position [m]
    y_init = data['y_init']
    # Initial theta position [m]
    theta_init = data['th_init']
    # Estimated States x, y, theta
    X_estimate = np.zeros([len(velocity), 3])
    X_estimate[0] = np.array([x_init, y_init, theta_init])
    # State covariance matrix
    P_estimate = np.zeros([len(velocity), 3, 3])
    P_estimate[0] = np.diag([1, 1, 0.1])
    ####################################################################

    # Set initial values
    p_estimate = P_estimate[0]
    x_estimate = X_estimate[0, :].reshape(3, 1)

    for k in range(1, len(t)):

        # Get time step
        delta_t = t[k] - t[k - 1]

        # Convert theta to [-pi, pi]
        theta = wrap_to_pi(x_estimate[2])

        # Create F matrix
        F = np.array([[np.cos(theta), 0],
                     [np.sin(theta), 0],
                     [0, 1]], dtype='float')

        # x_predict = f[k-1](x_estimate[k-1], u[k-1], 0)
        x_predict = x_estimate + F.dot(np.array([[velocity[k-1]], [rot_velocity[k-1]]])).dot(delta_t)
        # Convert theta to [-pi, pi]
        x_predict[2] = wrap_to_pi(x_predict[2])

        # F[k-1] = df[k-1]/dx[k-1] @x_estimate[k-1], u[k-1], 0
        F_jacob = np.array([[1, 0, -np.sin(theta)*delta_t*velocity[k-1]],
                            [0, 1, np.cos(theta)*delta_t*velocity[k-1]],
                            [0, 0, 1]], dtype='float')

        # L[k-1] = df[k-1]/dw[k-1] @x_estimate[k-1], u[k-1], 0
        L_jacob = np.array([[np.cos(theta)*delta_t, 0],
                        [np.sin(theta)*delta_t, 0],
                        [0, 1]], dtype='float')

        # p_predict[k] = F[k-1].p_estimate[k-1].F[k-1](T) + L[k-1].Q[k-1].L[k-1](T)
        p_predict = F_jacob.dot(p_estimate).dot(F_jacob.T) + L_jacob.dot(Q_covar).dot(L_jacob.T)

        # Loop over available measurements
        for i in range(len(range_measure[k])):
            x_estimate, p_estimate = measurement_update(d, l[i], range_measure[k, i], bearing_measure[k, i], p_predict, x_predict, R_covar)
            p_predict = p_estimate
            x_predict = x_estimate

        X_estimate[k] = x_estimate.reshape(1, 3)
        P_estimate[k] = p_estimate

    e_fig = plt.figure()
    ax = e_fig.add_subplot(111)
    ax.plot(X_estimate[:, 0], X_estimate[:, 1])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Estimated trajectory')
    plt.show()

    e_fig = plt.figure()
    ax = e_fig.add_subplot(111)
    ax.plot(t[:], X_estimate[:, 2])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('theta [rad]')
    ax.set_title('Estimated trajectory')
    plt.show()

if __name__=="__main__":
    main()