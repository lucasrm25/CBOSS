''' Author: Lucas Rath
'''

import numpy as np

def argminN(y:np.ndarray, N:int=1):
	''' Returns the indices of the sorted top N minimum values '''
	assert np.ndim(y)==1
	if N <= 0:
		return np.array([])
	elif N >= len(y):  # assert N <= len(y)
		return np.argsort(y)
	else:
		idx_min = np.argpartition(y, N)[:N]
		return idx_min[np.argsort(y[idx_min])]

def argmaxN(y:np.ndarray, N:int=1):
    return argminN(y=-y, N=N)

def minN(y:np.ndarray, N:int=1):
    return y[argminN(y=y,N=N)]

def maxN(y:np.ndarray, N:int=1):
    return y[argmaxN(y=y,N=N)]

def arg_new_unique(X_new:np.ndarray, X:np.ndarray):
	''' Returns the indices of a subdataset of X_new that are not included in X
	'''
	assert X_new.shape[1] == X_new.shape[1], 'Wrong dimensions!'
	
	# remove duplicated dataset
	X_new_unique, idx_new_unique = np.unique(X_new.astype("<U22"), return_index=True, axis=0)

	# select elements of X_new that are not in X
	a1, a2 = X_new[idx_new_unique].astype("<U22"), X.astype("<U22")
	# a1, a2 = X_new_unique, X
	a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
	a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
	idx_unique  = ~ np.in1d(a1_rows, a2_rows, assume_unique=True)

	idx = idx_new_unique[idx_unique]
	return idx

