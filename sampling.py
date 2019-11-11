#%%
import math
import numpy as np
import scipy.special as sks
import scipy.stats as st


def csprob(std, n, i=0, ps=None, ks=np.array([]).astype('int')):
    """
    :param std: standard deviation of distribution of classes in the cluster
    :param n: sample size
    :param ps: cluster distribution (used by recursion)
    :param i: iteration (used by recursion)
    :param ks: array of combinations (used by recursion)
    :return: probability that the majority class in the sample is the same as in the cluster
    """
    if not ps:
        ps = list(st.norm.pdf(range(-5, 5), 0, std))
        ps[5] += 1 - sum(ps)
        ps.sort(reverse=True)

    assert (math.isclose(sum(ps), 1) and len(ps) == 10 and n >= 1)

    if i == 0:
        rng = range(int(np.ceil(n / 10)), n + 1)
    elif i < 9:
        rng = range(0, min(ks[0], n - sum(ks)) + 1)
    else:
        rng = range((n - sum(ks)), min(ks[0], n - sum(ks)) + 1)

    if i < 10:
        return np.sum([csprob(std, n, i + 1, ps, np.concatenate((ks, np.array([k])))) for k in rng])
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
