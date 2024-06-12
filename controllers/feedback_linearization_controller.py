import numpy as np
from models.manipulator_model import ManiuplatorModel
from controllers.controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        self.Kd = np.array([[4,0], [0,4]])
        self.Kp = np.array([[2,0], [0,2]])

        q1, q2, q1_dot, q2_dot = x

        q = np.array([q1, q2])
        q_dot = np.array([q1_dot, q2_dot])

        v = q_r_ddot + self.Kd @ (q_r_dot - q_dot) + self.Kp @ (q_r - q)
        # v = q_r_ddot

        M = self.model.M(x)
        C = self.model.C(x)

        # tau = np.dot(M,v) + np.dot(C,x[2:])
        tau = M @ v + C @ q_r_dot

        return tau
