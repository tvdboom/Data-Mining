import numpy as np
from unique import unique #we load unique in from documentation because argument axis didn't work
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

#make matrix of 1 and 0s
def make_data_set(n, nbit, randomseed = 2):
	"""
	n - number of bitstrings
	nbit - number of bits per bit string
	"""
	np.random.seed(randomseed)
	data = np.random.randint(2,size=(nbit, n))
	return data
	
#count nbit of distinct elements
def get_true_count(data):
	count = unique(data, axis = 1).shape[1]
	return count

#count nbit zeros from left to right
def check_zero(x):
	x = x[::-1]
	n = len(x)
	if sum(x) == 0: #if the list contains only zeros return length list
		return n
	else:
		p = 0
		while x[p] != 1:
			p += 1
		return p
		
#check first one bit
def check_first_one_bit(x):
	n = len(x)
	if sum(x) == 0: #if the list contains only zeros return length list
		return n
	else:
		p = 0
		while x[p] != 1:
			p += 1
		return p
		
#count position first zero
def check_first_zero(x):
	n = len(x)
	if sum(x) == n: #if the list contains only zeros return length list
		return n
	else:
		p = 0
		while x[p] != 0:
			p += 1
		return p

# convert binary list to integer
def bin2int(x):
	p = 0
	x = x[::-1]
	for i in range(len(x)):
		p += 2**i * x[i]
	return int(p)

#algorithm 4.4.2 from the book
def adaptivesampling_counting(data, nbit_bucket):
	"""
	data - list of binary numbers
	nbit_bucket - number of bits assigned to a bucket
	"""
	ts = time.time() # starting time of algorithm
	nbit = len(data[:,0])
	n = len(data[0,:])
	n_buckets = 2**nbit_bucket
	buckets = np.zeros(n_buckets) #fill buckets with zeros
	for i in range(n):
		bucket_index = bin2int(data[0:nbit_bucket,i])
		n_zero = check_zero(data[nbit_bucket:,i])
		if n_zero > buckets[bucket_index]:
			buckets[bucket_index] = n_zero
	N_groups = int(n_buckets/(3 * np.log10(nbit)/np.log10(2)))
	if N_groups == 0:
		N_groups = 1
		print 'Warning: no median or mean is used in adaptive sampling.'	
	estimate = np.median(np.array([2**np.mean(np.array_split(buckets, N_groups)[i]) for i in range(N_groups)]))
	runtime = time.time() - ts # runtime
	return estimate, runtime

#Probabilistic counting
def prob_counting(data, nbit_bucket, magic_number = 0.7735162909):
	ts = time.time() # starting time of algorithm
	nbit = len(data[:,0])
	n = len(data[0,:])
	n_buckets = 2**nbit_bucket
	buckets = np.zeros(n_buckets) #fill buckets with zeros
	P = np.zeros((n_buckets, nbit)) # create a binary number P for all buckets
	for i in range(n):
		bucket_index = bin2int(data[0:nbit_bucket,i])
		n_zero = check_first_one_bit(data[nbit_bucket:,i])
		P[bucket_index, n_zero] = 1
	R = np.mean([check_first_zero(P[i,:]) for i in range(n_buckets)])
	estimate = 2**(R) * n_buckets * magic_number
	runtime = time.time() - ts # runtime
	return estimate, runtime

#LogLog counting
def loglog_counting(data, nbit_bucket, magic_number =  0.7735162909):
	ts = time.time() # starting time of algorithm
	nbit = len(data[:,0])
	n = len(data[0,:])
	n_buckets = 2**nbit_bucket
	buckets = np.zeros(n_buckets) #fill buckets with zeros
	for i in range(n):
		bucket_index = bin2int(data[0:nbit_bucket,i])
		n_zero = check_zero(data[nbit_bucket:,i])
		if n_zero > buckets[bucket_index]:
			buckets[bucket_index] = n_zero
	R = np.mean(buckets)
	estimate = 2**(R) * n_buckets * magic_number
	runtime = time.time() - ts # runtime
	return estimate, runtime
	
def get_RAE(estimate, true_count):
	RAE = np.abs(true_count - estimate)/true_count
	return RAE
	
def plot_RAE(relerrors, nbins = 25):
	
	# plot 1 - all in 1 plot
	colors = ['r','g','b']
	labels = ['Adaptive sampling', 'Probabilistic', 'LogLog']
	bins = np.linspace(0,np.max(relerrors), nbins)
	for i in range(3): plt.hist(relerrors[:,i], bins, color = colors[i], label = labels[i], alpha = 0.6)
	plt.legend(loc=2)
	plt.xlabel('RAE', size = 15)
	plt.ylabel('Frequency', size = 15)
	plt.savefig('RAE1.pdf')
	plt.savefig('RAE1.png')
	plt.show()
	
	# plot 2 - all three in different plots
	colors = ['r','g','b']
	labels = ['Adaptive sampling', 'Probabilistic', 'LogLog']
	plt.figure(figsize=(18,5))
	gs = gridspec.GridSpec(1, 3)
	for i in range(3):
		ax = plt.subplot(gs[0:1, i-1:i])
		if i == 1: ax.set_ylabel('Frequency', size = 15)
		ax.hist(relerrors[:,i], color = colors[i], alpha = 0.6)
		ax.set_title(labels[i], size = 15)
		ax.set_xlabel('RAE', size = 15)
	plt.savefig('RAE2.pdf')
	plt.savefig('RAE2.png')
	plt.show()
	
def main(n, nbit, nbit_bucket, n_experiments, plot = True):
	"""
	n - number of bitstrings
	nbit - number of bits per bit string
	nbit_bucket - number of bits assigned to a bucket
	n_experiments - amount of experiments
	"""
	print '-----------------------------------'
	print 'AidM Assignment 2'
	print '-----------------------------------'
	print 'Running',n_experiments,' experiments ( n = ',n,', nbit = ',nbit,', nbit per buckets = ',nbit_bucket,')'
	print ''
	randomseeds = np.arange(n_experiments) # create randomseeds for all experiments
	relerrors = np.zeros((n_experiments,3)) # allocate memory for the relative errors
	std = np.zeros((n_experiments,3)) # allocate memory for the std's
	estimates = np.zeros((n_experiments, 3)) # allocate memory for the estimates
	runtimes = np.zeros((n_experiments, 3)) # allocate memory for the runtimes
	for m in range(n_experiments):
		print m+1
		data = make_data_set(n, nbit, randomseeds[m]) # create random data set
		true_count = get_true_count(data) # evaluate the true count
		estimates[m,0], runtimes[m,0] = adaptivesampling_counting(data, nbit_bucket) # save estimates
		estimates[m,1], runtimes[m,1] = prob_counting(data, nbit_bucket)
		estimates[m,2], runtimes[m,2] = loglog_counting(data, nbit_bucket)
		for r in range(3): relerrors[m,r] = get_RAE(estimates[m,r], true_count) # get RAE corresponding to all estimates
	print ''
	print 'Average errors of',n_experiments,'trials: ' # print results in terminal
	print ''
	print 'Adaptive sampling: RAE = ',np.mean(relerrors[:,0]),', average runtime: ',np.mean(runtimes[:,0])
	print 'Probabilistic counting: RAE = ',np.mean(relerrors[:,1]),'( Theoretical: 0.78/sqrt(2**nbit): ',0.78/np.sqrt(2**nbit_bucket),')',', average runtime: ',np.mean(runtimes[:,1])
	print 'Loglog counting: RAE = ',np.mean(relerrors[:,2]),'( Theoretical: 1.30/sqrt(2**nbit): ',1.30/np.sqrt(2**nbit_bucket),')',', average runtime: ',np.mean(runtimes[:,2])
	print ''
	if plot is True: plot_RAE(relerrors) # plot the results

# run all experiments
main(n=100000, nbit=32, nbit_bucket=10, n_experiments=50)
