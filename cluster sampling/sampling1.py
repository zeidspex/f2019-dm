#%%
import math
import numpy as np
import scipy.special as sks


def csprob(ps, n, i=1, ks=np.array([]).astype('int')):
    """
    :param ps: distribution of classes in the cluster
    :param n: sample size
    :param i: iteration (used by recursion)
    :param ks: array of combinations (used by recursion)
    :return: probability that the majority class in the sample is the same as in the cluster

    Sample input:
        ps = [0.35, 0.15, 0.15, 0.15, 0.15, 0.05, 0, 0, 0, 0]; n = 5
    """
    assert (math.isclose(sum(ps), 1) and len(ps) == 10 and n >= 1)

    if i == 1:
        rng = range(int(np.ceil(n / 10)), n + 1)
    elif i < 10:
        rng = range(0, min(ks[0], n - sum(ks[0:i])) + 1)
    else:
        rng = range((n - sum(ks[0:9])), min(ks[0], n - sum(ks[0:9])) + 1)

    if i < 10:
        return np.sum([csprob(ps, n, i + 1, np.concatenate((ks, np.array([k])))) for k in rng])
    else:
        return np.sum([
            (lambda ks:
                np.prod(
                    [
                        sks.comb(n - np.sum(ks[0:i]), ks[i]) * (ps[i] ** ks[i])
                        for i in range(10)
                    ]
                ) * (1 / np.sum(ks == ks[0]))
            )(np.concatenate((ks, np.array([k]))))
            for k in rng
        ])
