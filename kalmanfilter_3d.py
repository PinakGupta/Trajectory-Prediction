import numpy as np

class KalmanFilter3D:
    def __init__(self, x=0, y=0, z=0):
        # Initial state [x, y, z, dx, dy, dz]
        self.state = np.array([[x], [y], [z], [0], [0], [0]], dtype=float)

        # State transition matrix (6x6 for 3D position and velocity)
        self.A = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 1, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 1],  # z = z + dz
            [0, 0, 0, 1, 0, 0],  # dx = dx
            [0, 0, 0, 0, 1, 0],  # dy = dy
            [0, 0, 0, 0, 0, 1]   # dz = dz
        ], dtype=float)

        # Measurement matrix (3x6 - we observe x, y, z)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=float)

        # Process noise covariance (6x6)
        self.Q = np.eye(6) * 0.01

        # Measurement noise covariance (3x3)
        self.R = np.eye(3) * 5

        # Estimate error covariance (6x6)
        self.P = np.eye(6)

    def predict(self):
        """Predict the next state"""
        self.state = np.dot(self.A, self.state)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q
        return int(self.state[0][0]), int(self.state[1][0]), int(self.state[2][0])

    def correct(self, x, y, z):
        """Update the state with measurement"""
        Z = np.array([[x], [y], [z]])
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state += np.dot(K, (Z - np.dot(self.H, self.state)))
        self.P = np.dot((np.eye(6) - np.dot(K, self.H)), self.P)

    def get_velocity(self):
        """Get current velocity estimates"""
        return self.state[3][0], self.state[4][0], self.state[5][0]
