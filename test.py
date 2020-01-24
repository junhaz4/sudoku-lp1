import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as scs # sparse matrix construction 
import scipy.linalg as scl # linear algebra algorithms
import scipy.optimize as sco # for minimization use
import matplotlib.pylab as plt # for visualization

import time
import sys
from multiprocessing import Pool
from sudoku_solver import solver

# We test the following algoritm on small data set.

if __name__ == "__main__":
	if len(sys.argv) < 2:
		data = pd.read_csv("./small2.csv") 
	else:
		data = pd.read_csv(sys.argv[1])

	corr_cnt = 0
	start = time.time()

	random_seed = 42
	sample_max  = 1000

	np.random.seed(random_seed)

	if len(data) > sample_max:
	    samples = np.random.choice(len(data), sample_max)
	else:
	    samples = np.arange(len(data))

	pool = Pool() 

	quizzes = data["quizzes"][samples]
	solutions = data["solutions"][samples]

	res = pool.map(solver, quizzes)

	corr_cnt = np.sum(1*(res == solutions)) 
	end = time.time()

	# report:
	print("Aver Time: {t:6.2f} secs. Success rate: {corr} / {all} ".format(t=(end-start)/(len(samples)), corr=corr_cnt, all=len(samples)) )

