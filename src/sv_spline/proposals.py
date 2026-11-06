from abc import ABC, abstractmethod
from .spline import K_decomp
import numpy as np

class Proposal(ABC):
    def __init__(self, keys):
        self.keys = keys

    @abstractmethod
    def propose(self, params, rng):
        pass

    def log_q_ratio(self, current, candidate):
        return 0

    def adapt(self, params, accepted):
        """
        Adapte le pas de la proposition en fonction de l'acceptation ou du rejet de la proposition.
        """
        pass

class RandomWalk(Proposal):
    def __init__(self, keys, scale):
        super().__init__(keys)
        self.scale = scale

    def propose(self, params, rng):
        candidate = params.copy()
        for key in self.keys:
            candidate[key] = params[key] + rng.normal(0, self.scale[key])
        return candidate

class GProposal(Proposal):
    def __init__(self, K, kernel_width, sigma_e):
        super().__init__(keys=['g'])
        self.eigenvalues, self.eigenvectors = K_decomp(K)
        self.kernel_width = kernel_width
        self.sigma_e = sigma_e

    def propose(self, params, rng):
        g = params['g']
        n = len(g)
        coeff = rng.normal(0, 1, size=n)
        candidate = params.copy()
        eps = np.zeros(n)
        for i in range(n):
            if self.eigenvalues[i] <= 1e-8:
                eps += self.eigenvectors[:,i]*coeff[i]*self.kernel_width
            else:
                eps += self.eigenvectors[:,i]*coeff[i]*self.sigma_e/np.sqrt(self.eigenvalues[i])
        candidate['g'] = g + eps
        return candidate

class Composite(Proposal):
    def __init__(self, proposals):
        keys = []
        for proposal in proposals:
            keys.extend(proposal.keys)
        super().__init__(keys)
        self.proposals = proposals

    def propose(self, params, rng):
        candidate = params.copy()
        for proposal in self.proposals:
            candidate = proposal.propose(candidate, rng)
        return candidate

    def log_q_ratio(self, current, candidate):
        log_q = 0
        for proposal in self.proposals:
            log_q += proposal.log_q_ratio(current, candidate)
        return log_q

    def adapt(self, params, rng):
        for proposal in self.proposals:
            proposal.adapt(params, rng)