import numpy as np
from controllers.controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3

        self.model_1 = ManiuplatorModel(Tp)
        self.model_1.m3 = 0.1
        self.model_1.r3 = 0.05

        self.model_2 = ManiuplatorModel(Tp)
        self.model_2.m3 = 0.01
        self.model_2.r3 = 0.01

        self.model_3 = ManiuplatorModel(Tp)
        self.model_3.m3 = 1.0
        self.model_3.r3 = 0.3

        self.models = [self.model_1, self.model_2, self.model_3]
        self.i = 0

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        errors = []
        for model in self.models:
            M = model.M(x)
            C = model.C(x)

            q = x[:2]
            q_dot = x[2:]

            error = np.linalg.norm(M @ q[:, np.newaxis] + C @ q_dot[:, np.newaxis])

            errors.append(error)

        self.i = np.argmin(errors)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        self.Kd = np.array([[5, 0], [0, 5]])
        self.Kp = np.array([[3, 0], [0, 3]])
        q = x[:2]
        q_dot = x[2:]

        v = q_r_ddot + self.Kd @ (q_r_dot - q_dot) + self.Kp @ (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
