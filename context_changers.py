import random

import numpy as np


class ContextChanger:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def reset_env(self, env):
        raise NotImplementedError

    def change_env(self, env):
        raise NotImplementedError


class NullContextChanger(ContextChanger):

    def reset(self):
        pass

    def reset_env(self, env):
        pass

    def change_env(self, env):
        pass


class ReacherHardContextChanger(ContextChanger):
    def __init__(self, multi_reset=True):
        self.multi_reset = multi_reset
        self.is_reset = False

        super().__init__()

    def reset(self):
        if not self.multi_reset and self.is_reset:
            return

        self.c1_pos = [random.random() * 0.6 - 0.3, random.random() * 0.6 - 0.3, 2.5e-6]
        self.c2_pos = [random.random() * 0.6 - 0.3, random.random() * 0.6 - 0.3, 2.5e-6]
        self.c3_pos = [random.random() * 0.6 - 0.3, random.random() * 0.6 - 0.3, 2.5e-6]
        self.c4_pos = [random.random() * 0.6 - 0.3, random.random() * 0.6 - 0.3, 2.5e-6]
        self.c5_pos = [random.random() * 0.6 - 0.3, random.random() * 0.6 - 0.3, 2.5e-6]

        self.c1_visible = round(random.random())
        self.c2_visible = round(random.random())
        self.c3_visible = round(random.random())
        self.c4_visible = round(random.random())
        self.c5_visible = round(random.random())

        self.c1_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, self.c1_visible]
        self.c2_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, self.c2_visible]
        self.c3_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, self.c3_visible]
        self.c4_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, self.c4_visible]
        self.c5_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, self.c5_visible]

        self.is_reset = True

    def change_env(self, env):
        env.physics.named.model.mat_texid['grid'] = -1
        env.physics.named.model.geom_rgba['ground'] = [1., 1., 1., 1]

        env.physics.named.data.geom_xpos['c1'] = self.c1_pos
        env.physics.named.data.geom_xpos['c2'] = self.c2_pos
        env.physics.named.data.geom_xpos['c3'] = self.c3_pos
        env.physics.named.data.geom_xpos['c4'] = self.c4_pos
        env.physics.named.data.geom_xpos['c5'] = self.c5_pos

        env.physics.named.model.geom_rgba['c1'] = self.c1_color
        env.physics.named.model.geom_rgba['c2'] = self.c2_color
        env.physics.named.model.geom_rgba['c3'] = self.c3_color
        env.physics.named.model.geom_rgba['c4'] = self.c4_color
        env.physics.named.model.geom_rgba['c5'] = self.c5_color

    def reset_env(self, env):
        env.physics.named.model.mat_texid['grid'] = 1
        env.physics.named.model.geom_rgba['ground'] = [0.5, 0.5, 0.5, 1]

        env.physics.named.data.geom_xpos['c1'] = [0, 0, 2.5e-6]
        env.physics.named.data.geom_xpos['c2'] = [0, 0, 2.5e-6]
        env.physics.named.data.geom_xpos['c3'] = [0, 0, 2.5e-6]
        env.physics.named.data.geom_xpos['c4'] = [0, 0, 2.5e-6]
        env.physics.named.data.geom_xpos['c5'] = [0, 0, 2.5e-6]

        env.physics.named.model.geom_rgba['c1'] = [0, 0, 0, 0]
        env.physics.named.model.geom_rgba['c2'] = [0, 0, 0, 0]
        env.physics.named.model.geom_rgba['c3'] = [0, 0, 0, 0]
        env.physics.named.model.geom_rgba['c4'] = [0, 0, 0, 0]
        env.physics.named.model.geom_rgba['c5'] = [0, 0, 0, 0]


class ReacherHardWCContextChanger(ContextChanger):
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def change_env(self, env):
        env.physics.named.model.mat_texid['grid'] = -1
        env.physics.named.model.geom_rgba['ground'] = [1., 1., 1., 1]


    def reset_env(self, env):
        env.physics.named.model.mat_texid['grid'] = 1
        env.physics.named.model.geom_rgba['ground'] = [0.5, 0.5, 0.5, 1]


class WalkerRunContextChanger(ContextChanger):

    def reset(self):
        # self.ground_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.floor = np.random.uniform([0., 0., 0., 1.], [0.5, 0.5, 0.5, 1])

    def change_env(self, env):
        pass
        # env.physics.named.model.mat_texid['grid'] = -1
        # env.physics.named.model.geom_rgba['floor'] = self.floor

    def reset_env(self, env):
        pass
        # env.physics.named.model.mat_texid['grid'] = 1
        # env.physics.named.model.geom_rgba['floor'] = [0.5, 0.5, 0.5, 1]
