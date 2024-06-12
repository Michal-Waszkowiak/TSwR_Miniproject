import numpy as np

from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        p0 = p[0]
        p1 = p[1]
        self.L = np.array([[3*p0, 0],
                           [0, 3*p1],
                           [3*p0**2, 0],
                           [0, 3*p1**2],
                           [p0**3, 0],
                           [0, p1**3]])
        W = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        A = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        B = np.zeros((6,2))
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot], axis=0)
        M = self.model.M(x)
        C = self.model.C(x)
        M_ = np.linalg.inv(M)

        A = np.zeros((6,6))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 4:6] = np.eye(2)
        A[2:4, 2:4] = -M_ @ C

        B = np.zeros((6,2))
        B[2:4, :] = M_

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1,q2])
        q_dot = np.array([q1_dot,q2_dot])
        M = self.model.M(x)
        C = self.model.C(x)
        z_hat = self.eso.get_state()
        x_hat = z_hat[:2]
        x_hat_dot = z_hat[2:4]
        f = z_hat[4:]

        v = q_d_ddot + self.Kp @ (q_d-q) + self.Kd @ (q_d_dot - x_hat_dot)
        u = M @ (v - f) + C @ x_hat_dot
        self.update_params(x_hat,x_hat_dot)
        self.eso.update(q.reshape(len(q),1), u.reshape(len(u),1))

        return u
