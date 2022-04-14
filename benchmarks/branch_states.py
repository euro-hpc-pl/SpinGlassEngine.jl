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

    nstates = 1000
    lstate = 128
    num_states = 256
    
    st = np.ones((nstates, lstate), dtype=int)

    tic = time.time()
    branch_states(st, num_states)
    toc = time.time()
    print(toc - tic)

    loop = 100
    tt = timeit.timeit("branch_states(st, num_states)", globals=globals(), number=loop)
    print(tt / loop)
