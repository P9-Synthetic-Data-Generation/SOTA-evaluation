from keras import Optimizer
from keras import backend

import tensorflow as tf


def clip_norm(g, c, n):
    if c > 0:
        g = backend.switch(n >= c, g * c / n, g)
    return g


class NoisyAdam(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, noise=0.0, **kwargs):
        super(NoisyAdam, self).__init__(**kwargs)
        self.iterations = backend.Variable(0, name="iterations")
        self.lr = backend.variable(lr, name="lr")
        self.beta_1 = backend.variable(beta_1, name="beta_1")
        self.beta_2 = backend.variable(beta_2, name="beta_2")
        self.epsilon = epsilon
        self.decay = backend.variable(decay, name="decay")
        self.initial_decay = decay
        self.noise = noise

    def get_gradients(self, loss, params):
        grads = backend.gradients(loss, params)
        if hasattr(self, "clipnorm") and self.clipnorm > 0:
            norm = backend.sqrt(sum([backend.sum(backend.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, "clipvalue") and self.clipvalue > 0:
            grads = [backend.clip(g, -self.clipvalue, self.clipvalue) for g in grads]

        if self.noise > 0:
            grads = [(g + backend.random_normal(g.shape, mean=0, stddev=(self.noise * self.clipnorm))) for g in grads]
        return grads

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [backend.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= 1.0 / (1.0 + self.decay * self.iterations)

        t = self.iterations + 1
        lr_t = lr * (backend.sqrt(1.0 - backend.pow(self.beta_2, t)) / (1.0 - backend.pow(self.beta_1, t)))

        shapes = [backend.get_variable_shape(p) for p in params]
        ms = [backend.zeros(shape) for shape in shapes]
        vs = [backend.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1.0 - self.beta_2) * backend.square(g)
            p_t = p - lr_t * m_t / (backend.sqrt(v_t) + self.epsilon)

            self.updates.append(backend.update(m, m_t))
            self.updates.append(backend.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(backend.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            "lr": float(backend.get_value(self.lr)),
            "beta_1": float(backend.get_value(self.beta_1)),
            "beta_2": float(backend.get_value(self.beta_2)),
            "decay": float(backend.get_value(self.decay)),
            "epsilon": self.epsilon,
        }
        base_config = super(NoisyAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
