import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.Kp = kp
        self.Kd = kd

        A = np.array([[0,1,0],[0,0,1],[0,0,0]])
        B = np.array([[0],[self.b],[0]])
        L = np.array([[3*p],[3*p**2],[p**3]])
        W = np.array([[1,0,0]])
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.eso.set_B(np.array([[0], [b], [0]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        q, q_dot = x
        self.u_ = 0
        self.eso.update(q, self.u_)
        q_hat, q_dot_hat, f = self.eso.get_state()
        v = self.Kp * (q_d - q) + self.Kd * (q_d_dot - q_dot_hat) + q_d_ddot
        u = (v - f) / self.b
        self.u_ = u

        return u
