import numpy as np

class EKF(object):
    def __init__(self, num_states, num_sensors, x_0, P_0, Q, R, F, B, H):
        self.num_states = num_states
        self.num_sensors = num_sensors

        self.x = x_0        # state estimate
        self.P_prior = None # error covariance matrix before data is sampled
        self.P_post = P_0   # error covariance matrix after data is sampled
        self.Q = Q          # Process noise covariance matrix. dimension == num_states
        self.R = R          # Measurement noise covariance matrix. dimension == num_sensors

        self.F = F          # State transition model
        self.B = B          # Control signal transition model TODO not implemented
        self.H = H          # State measurement model

    def step(self, z):
        '''
        z is current observation
        '''

        ### predict ###
        self.x = self.F * self.x
        self.P_prior = self.F * self.P_post * self.F.T + self.Q
        ## update ###
        K = (self.P_prior * self.H.T) * np.linalg.inv(self.H * self.P_prior * self.H.T + self.R)
        self.x = self.x + K * (np.array(z) - (self.H * self.x).T).T
        self.P_post = (np.eye(self.num_states) - K * self.H) * self.P_prior

        return self.x
