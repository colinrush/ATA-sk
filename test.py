import numpy as np
from SKutils import calculate_sk, replace_values

NANTS    = 20
NFREQS   = 192
NPOLS    = 2
NSAMPLES = 8192


for i in range(10):
    data = np.random.normal(scale=8, size=(NANTS, NFREQS, NPOLS, NSAMPLES))

    sk_array = calculate_sk(data)
    print("done SK")

    #replace_values(data, sk_array, 0.8, 1.2, "zero")

    print("done")
