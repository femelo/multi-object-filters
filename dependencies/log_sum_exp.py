import numpy as np

def log_sum_exp(w, axis=0):
    ''' Performs log-sum-exp trick to avoid numerical underflow
        input:  w weight vector assumed already log transformed
        output: log(sum(exp(w))) 
    '''
    if np.prod(w.shape) == 0:
        return np.array(w.shape)

    max_val = np.max(w, axis=axis)
    return np.log(np.sum(np.exp(w - max_val))) + max_val
    
