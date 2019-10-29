#%%
import math
import numpy as np
import scipy.special as sks

# Number of samples selected from the cluster
n = 3

# Array for temporary storing coefficients
ks = np.array([0] * 10)

# Variable for storing result
prob = 0

# Distribution of classes in the cluster, or probabilities of selecting each class
ps = [0.35, 0.15, 0.15, 0.15, 0.15, 0.05, 0, 0, 0, 0]

# Check that probabilities add up to 1
assert math.isclose(sum(ps), 1)

for k1 in range(int(np.ceil(n / 10)), n + 1):
    ks[0] = k1
    for k2 in range(0, min(k1, n - sum(ks[0:1])) + 1):
        ks[1] = k2
        for k3 in range(0, min(k1, n - sum(ks[0:2])) + 1):
            ks[2] = k3
            for k4 in range(0, min(k1, n - sum(ks[0:3])) + 1):
                ks[3] = k4
                for k5 in range(0, min(k1, n - sum(ks[0:4])) + 1):
                    ks[4] = k5
                    for k6 in range(0, min(k1, n - sum(ks[0:5])) + 1):
                        ks[5] = k6
                        for k7 in range(0, min(k1, n - sum(ks[0:6])) + 1):
                            ks[6] = k7
                            for k8 in range(0, min(k1, n - sum(ks[0:7])) + 1):
                                ks[7] = k8
                                for k9 in range(0, min(k1, n - sum(ks[0:8])) + 1):
                                    ks[8] = k9
                                    for k10 in range((n - sum(ks[0:9])), min(k1, n - sum(ks[0:9])) + 1):
                                        ks[9] = k10

                                        # Verify that the total number of items matches n
                                        # Also verify that the first class is majority class
                                        assert sum(ks) == n
                                        assert all([ks[0] >= k for k in ks[1:]])

                                        prob_d = [
                                            sks.comb(n - np.sum(ks[0:i]), ks[i]) * (ps[i] ** ks[i])
                                            for i in range(10)
                                        ]
                                        prob += np.prod(prob_d) * (1 / sum(ks == ks[0]))


print('Probability that cluster is labeled correctly: %.3f' % prob)
