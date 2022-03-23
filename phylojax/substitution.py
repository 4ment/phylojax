from abc import ABC, abstractmethod

import jax.numpy as np
import jax.numpy.linalg as LA


class SubstitutionModel(ABC):
    def __init__(self, frequencies, rates=None):
        self.frequencies = frequencies
        self.rates = rates

    @abstractmethod
    def p_t(self, branch_lengths):
        pass

    @abstractmethod
    def q(self):
        pass

    @staticmethod
    def norm(Q, frequencies):
        return -np.sum(np.diagonal(Q, axis1=-2, axis2=-1) * frequencies, -1)


class JC69(SubstitutionModel):
    def __init__(self):
        super().__init__(np.array([0.25] * 4))

    def p_t(self, d):
        # d = np.expand_dims(d, axis=-1)
        a = 0.25 + 3 / 4 * np.exp(-4 / 3 * d)
        b = 0.25 - 0.25 * np.exp(-4 / 3 * d)
        return np.concatenate(
            [a, b, b, b, b, a, b, b, b, b, a, b, b, b, b, a], axis=-1
        ).reshape(d.shape[:-1] + (4, 4))

    def q(self):
        return np.array(
            [
                [-1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                [1.0 / 3, -1.0, 1.0 / 3, 1.0 / 3],
                [1.0 / 3, 1.0 / 3, -1.0, 1.0 / 3],
                [1.0 / 3, 1.0 / 3, 1.0 / 3, -1.0],
            ]
        )


def diag_last_dim(x):
    return np.expand_dims(x, -1) * np.eye(x.shape[-1])


class SymmetricSubstitutionModel(SubstitutionModel, ABC):
    def __init__(self, frequencies):
        super().__init__(frequencies)

    def p_t(self, branch_lengths):
        Q = self.q()
        Q /= np.expand_dims(
            np.expand_dims(SubstitutionModel.norm(Q, self.frequencies), -1), -1
        )
        sqrt_pi = diag_last_dim(np.sqrt(self.frequencies))
        sqrt_pi_inv = diag_last_dim(1.0 / np.sqrt(self.frequencies))
        S = sqrt_pi @ Q @ sqrt_pi_inv
        e, v = self.eigen(S)
        offset = branch_lengths.ndim - e.ndim
        return (
            np.expand_dims(sqrt_pi_inv @ v, -3)
            @ diag_last_dim(
                np.exp(
                    e.reshape(e.shape[:-1] + (1,) * offset + e.shape[-1:])
                    * branch_lengths
                )
            )
            @ np.expand_dims(LA.inv(v) @ sqrt_pi, -3)
        )

    def eigen(self, Q):
        return LA.eigh(Q)


class GTR(SymmetricSubstitutionModel):
    def __init__(self, rates, frequencies):
        super().__init__(frequencies)
        self.rates = rates

    def q(self):
        rates = np.expand_dims(self.rates, -2)
        pi = np.expand_dims(self.frequencies, -2)
        return np.concatenate(
            (
                -(
                    rates[..., 0] * pi[..., 1]
                    + rates[..., 1] * pi[..., 2]
                    + rates[..., 2] * pi[..., 3]
                ),
                rates[..., 0] * pi[..., 1],
                rates[..., 1] * pi[..., 2],
                rates[..., 2] * pi[..., 3],
                rates[..., 0] * pi[..., 0],
                -(
                    rates[..., 0] * pi[..., 0]
                    + rates[..., 3] * pi[..., 2]
                    + rates[..., 4] * pi[..., 3]
                ),
                rates[..., 3] * pi[..., 2],
                rates[..., 4] * pi[..., 3],
                rates[..., 1] * pi[..., 0],
                rates[..., 3] * pi[..., 1],
                -(
                    rates[..., 1] * pi[..., 0]
                    + rates[..., 3] * pi[..., 1]
                    + rates[..., 5] * pi[..., 3]
                ),
                rates[..., 5] * pi[..., 3],
                rates[..., 2] * pi[..., 0],
                rates[..., 4] * pi[..., 1],
                rates[..., 5] * pi[..., 2],
                -(
                    rates[..., 2] * pi[..., 0]
                    + rates[..., 4] * pi[..., 1]
                    + rates[..., 5] * pi[..., 2]
                ),
            ),
            -1,
        ).reshape(self.rates.shape[:-1] + (4, 4))


class HKY(SymmetricSubstitutionModel):
    def __init__(self, kappa, frequencies):
        super().__init__(frequencies)
        self.kappa = kappa

    def q(self):
        pi = self.frequencies
        return np.hstack(
            (
                -(pi[..., 1] + self.kappa * pi[..., 2] + pi[..., 3]),
                pi[..., 1],
                self.kappa * pi[..., 2],
                pi[..., 3],
                pi[..., 0],
                -(pi[..., 0] + pi[..., 2] + self.kappa * pi[..., 3]),
                pi[..., 2],
                self.kappa * pi[..., 3],
                self.kappa * pi[..., 0],
                pi[..., 1],
                -(self.kappa * pi[..., 0] + pi[..., 1] + pi[..., 3]),
                pi[..., 3],
                pi[..., 0],
                self.kappa * pi[..., 1],
                pi[..., 2],
                -(pi[..., 0] + self.kappa * pi[..., 1] + pi[..., 2]),
            )
        ).reshape(self.kappa.shape[:-1] + (4, 4))
