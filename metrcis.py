import numpy as np

def ACC(R):
    n_experieneces=R.shape[0]
    ACC=np.sum(R,axis=1)[-1]/n_experieneces
    return ACC

def BWT(R):
    n_experieneces=R.shape[0]
    BWT=0
    for i in range(n_experieneces):
        BWT+=R[n_experieneces-1][i]-R[i][i]
    BWT/=n_experieneces-1
    return BWT