import numpy as np
import time
import timeit

def branch_states(states, num_states):
    (k, l) = states.shape
    ns = np.empty((k, num_states, l+1), dtype=int)
    ns[:, :, :l] = states.reshape(k, 1, l)
    p = np.arange(num_states, dtype=int)
    ns[:, :, l] = p.reshape(1, 1, num_states)
    return ns.reshape(k * num_states, l+1)


if __name__ == '__main__':

    nstates = 10000
    lstate = 256
    num_states = 256
    
    st = np.ones((nstates, lstate), dtype=int)

    loop = 10
    tt = timeit.timeit("branch_states(st, num_states)", globals=globals(), number=loop)
    print(tt / loop)
