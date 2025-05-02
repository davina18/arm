import numpy as np
from forward_kinematics import compute_forward_kinematics
from jacobian import compute_jacobian

class RobotModel:
    def __init__(self):
        self.q = np.zeros((4, 1))
        self.ee_position = np.zeros((3, 1))
        self.jacobian = np.zeros((3, 4))

    def update(self, q):
        self.q = q
        self.ee_position = compute_forward_kinematics(q.flatten()).reshape((3, 1))
        self.jacobian = compute_jacobian(q.flatten())

    def get_ee_position(self):
        return self.ee_position

    def get_jacobian(self):
        return self.jacobian

    def get_joint_position(self, joint_idx):
        return np.array([[self.q[joint_idx, 0]]])
