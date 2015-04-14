
class MarkovChainNode:
    """
    Distribution for a set of Markov chains, the variables in each taking on 
    discrete states defined by a Dirichlet distribution. 
    Variables are:
        z: Markov chain (states x times x copies)
        zz: two-slice marginals for transitions (states x states x 
            times - 1, copies)
        logZ: log partition function for posterior (copies)

    Note that this is not a normal node, since it does not parameterize the 
        full joint distribution over z. Rather, it is only a container for 
        the expected values of the marginals z and zz, and the log 
        partition function, logZ. 
    """
    def __init__(self, z_prior, zz_prior, logZ_prior, name='markov'):
        if len(z_prior.shape) < 2:
            raise ValueError(
                'State variable must have at least two dimensions')
        if len(zz_prior.shape) < 3:
            raise ValueError(
                'Transition marginal must have at least three dimensions')

        self.M = z_prior.shape[0]
        self.T = z_prior.shape[1]
        if len(z_prior.shape) > 2:
            self.K = z_prior.shape[2:]
        else:
            self.K = (1,)

        if zz_prior.shape[0:3] != (self.M, self.M, self.T - 1):
            raise ValueError(
                'Transition marginal has shape inconsistent with prior')
        if logZ_prior.shape != self.K:
            raise ValueError(
                'logZ has shape inconsistent with prior')

        self.name = name
        self.z = z_prior
        self.zz = zz_prior
        self.logZ = logZ_prior

    def update(self):
        raise NotImplementedError('Instances should define this method for each model')

