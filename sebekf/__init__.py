import numpy as np

class EKF(object):
    def __init__(self, num_states, num_sensors, x_0, P_0, Q, R, F, B, H):
        self.num_states = num_states
        self.num_sensors = num_sensors

        self.x = x_0    # state estimate
        self.P = P_0    # error covariance matrix
        self.Q = Q      # Process noise covariance matrix. dimension == num_states
        self.R = R      # Measurement noise covariance matrix. dimension == num_sensors

        self.F = F      # State transition model
        self.B = B      # Control signal transition model TODO not implemented
        self.H = H      # State measurement model

    def step(self, z):
        z = np.array([[z[0]],[z[1]],[z[2]]])
        ### predict ###
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        ## update ###
        K = np.dot(self.P, self.H.T).dot(np.linalg.inv(np.dot(self.H, self.P).dot(self.H.T) + self.R))
        self.x += np.dot(K, z - np.dot(self.H, self.x))
        self.P = (np.eye(self.num_states) - np.dot(K, self.H)).dot(self.P)

        return self.x
