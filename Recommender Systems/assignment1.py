import numpy as np
import scipy.sparse
import statsmodels.api as sm
import time

def column(array, i):
	return [row[i] for row in array]

def import_data():
	#inlezen data en variabeles toekennen
	movies = np.genfromtxt('movies.dat', dtype=None, comments='#', delimiter='::', usecols = (0,1))
	ratings = np.genfromtxt('ratings.dat', dtype=None, comments='#', delimiter='::', usecols = (0,1,2,3))

	movie_id = column(movies, 0)
	movie_title = column(movies,1)

	data = [[] for i in range(3)]

	data[0] = column(ratings, 0)	#user_id
	data[1] = column(ratings, 1)	#movie_id
	data[2] = column(ratings, 2)	#rating

	#save data
	np.save('data.npy', data)

def main():
	data = np.load('data.npy').T
	
	MAE_avg = np.zeros(5)
	MAE_useravg = np.zeros(5)
	MAE_movieavg = np.zeros(5)
	MAE_linreg = np.zeros(5)
	MAE_incrementalSVD = np.zeros(5)
	
	RMSE_avg = np.zeros(5)
	RMSE_useravg = np.zeros(5)
	RMSE_movieavg = np.zeros(5)
	RMSE_linreg = np.zeros(5)
	RMSE_incrementalSVD = np.zeros(5)

	# shuffle data set randomly
	np.random.seed(1)
	np.random.shuffle(data)
	data = data.T
	
	for i in range(5):

		splitted_data = np.array_split(data, 5, axis = 1)
		test_set = splitted_data.pop(i)
		training_set = np.hstack(splitted_data)
		
		# get utility matrices
		utilitymatrix_training = init_utilitymatrix(training_set)
		utilitymatrix_test = init_utilitymatrix(test_set)
		
		# get models and MAE
		startTime = time.time()
		MAE_avg[i], RMSE_avg[i], prediction_matrix_avg_model = avg_model(utilitymatrix_training, utilitymatrix_test)
		print time.time() - startTime
		startTime = time.time()
		MAE_useravg[i], RMSE_useravg[i], predictionmatrix_useravg = useravg_model(utilitymatrix_training, utilitymatrix_test, prediction_matrix_avg_model)
		print time.time() - startTime
		startTime = time.time()
		MAE_movieavg[i], RMSE_movieavg[i], predictionmatrix_movieavg = movieavg_model(utilitymatrix_training, utilitymatrix_test, prediction_matrix_avg_model)
		print time.time() - startTime
		startTime = time.time()
		MAE_linreg[i], RMSE_linreg[i] = linreg_model(utilitymatrix_training, utilitymatrix_test, predictionmatrix_useravg, predictionmatrix_movieavg)
		print time.time() - startTime
		startTime = time.time()
		MAE_incrementalSVD[i], RMSE_incrementalSVD[i] = model_incrementalSVD(utilitymatrix_training, utilitymatrix_test)
		print time.time() - startTime
		startTime = time.time()
		
	print ''
	print 'Final results: '
	print ''
	print 'MAE global average: ',np.mean(MAE_avg)
	print 'MAE user average: ',np.mean(MAE_useravg)
	print 'MAE movie average: ',np.mean(MAE_movieavg)
	print 'MAE linear regression: ',np.mean(MAE_linreg)
	print 'MAE incremental SVD: ',np.mean(MAE_incrementalSVD)
	print ''
	print 'RMSE global average: ',np.mean(RMSE_avg)
	print 'RMSE user average: ',np.mean(RMSE_useravg)
	print 'RMSE movie average: ',np.mean(RMSE_movieavg)
	print 'RMSE linear regression: ',np.mean(RMSE_linreg)
	print 'RMSE incremental SVD: ',np.mean(RMSE_incrementalSVD)
	
	# save everyhting to a text file
	header = '# 1 MAE global, user, movie, linreg, incremental SVD 2 RMSE global, user, movie, linref, incremental SVD'
	data = np.array([[np.mean(MAE_avg), np.mean(MAE_useravg), np.mean(MAE_movieavg), np.mean(MAE_linreg), np.mean(MAE_incrementalSVD)],
	[np.mean(RMSE_avg), np.mean(RMSE_useravg), np.mean(RMSE_movieavg), np.mean(RMSE_linreg), np.mean(RMSE_incrementalSVD)]])
	np.savetxt('results', data, header = header)
			
def init_utilitymatrix(data):
	
	utilitymatrix = np.zeros((6040,3952)) # max movie_id and user_id -1
	utilitymatrix[:] = np.NAN
	
	for j in range(len(data[0,:])):
		utilitymatrix[data[0,j]-1,data[1,j]-1] = data[2,j]
	
	return utilitymatrix
	
def avg_model(utilitymatrix_training, utilitymatrix_test):
	
	predictionmatrix = np.zeros((6040,3952))
	predictionmatrix[:] = np.nanmean(utilitymatrix_training)
	
	MAE = get_MAE(predictionmatrix, utilitymatrix_test)
	RMSE = get_RMSE(predictionmatrix, utilitymatrix_test)
	
	return MAE, RMSE, predictionmatrix
	
def useravg_model(utilitymatrix_training, utilitymatrix_test, prediction_matrix_avg_model):
	
	predictionmatrix = np.copy(prediction_matrix_avg_model)

	useravg = np.nanmean(utilitymatrix_training, axis = 1)
	for j in range(6040):
		if not np.isnan(useravg[j]):
			predictionmatrix[j,:] = useravg[j]

	MAE = get_MAE(predictionmatrix, utilitymatrix_test)
	RMSE = get_RMSE(predictionmatrix, utilitymatrix_test)
	
	return MAE, RMSE, predictionmatrix
	
def movieavg_model(utilitymatrix_training, utilitymatrix_test, prediction_matrix_avg_model):
	
	predictionmatrix = np.copy(prediction_matrix_avg_model)
	
	movieavg = np.nanmean(utilitymatrix_training, axis = 0)
	for j in range(3952):
		if not np.isnan(movieavg[j]):
			predictionmatrix[:,j] = movieavg[j]
	
	MAE = get_MAE(predictionmatrix, utilitymatrix_test)
	RMSE = get_RMSE(predictionmatrix, utilitymatrix_test)
	
	return MAE, RMSE, predictionmatrix
	
def linreg_model(utilitymatrix_training, utilitymatrix_test, predictionmatrix_useravg, predictionmatrix_movieavg, summary = True):
	
	# ravel arrays
	x1 = predictionmatrix_useravg.ravel()
	x2 = predictionmatrix_movieavg.ravel()
	y = np.array(utilitymatrix_training).ravel()
	X_int = np.array([x1,x2]).T
	X_int = sm.add_constant(X_int)
	
	# remove NaN entries
	x1 = x1[np.logical_not(np.isnan(y))]
	x2 = x2[np.logical_not(np.isnan(y))]
	y = y[np.logical_not(np.isnan(y))]
	
	# create right shape for sm
	X = np.array([x1,x2]).T
	X = sm.add_constant(X)
	
	# evaluate model
	model = sm.OLS(y, X).fit() # sm.OLS(output, input)
	if summary: print model.summary()
	predictions = model.predict(X_int)
	predictionmatrix = predictions.reshape((6040,3952))
	
	# evaluate statistics
	MAE = get_MAE(predictionmatrix, utilitymatrix_test)
	RMSE = get_RMSE(predictionmatrix, utilitymatrix_test)
	
	return MAE, RMSE

def get_MAE(predictionmatrix, utilitymatrix_test):
	
	diff = predictionmatrix - utilitymatrix_test # take difference
	diff = diff[np.logical_not(np.isnan(diff))] # remove nans
	MAE = 1/float(diff.size) * np.sum(np.absolute(diff))
	
	return MAE

def get_RMSE(predictionmatrix, utilitymatrix_test):
	
	diff = predictionmatrix - utilitymatrix_test # take difference
	diff = diff[np.logical_not(np.isnan(diff))] # remove nans
	RMSE = np.sqrt( 1/float(diff.size) * np.sum(diff**2) )
	
	return RMSE
	
def model_incrementalSVD(utilitymatrix_training, utilitymatrix_test):
	
	# initalize some important parameters from MyMedialite website
	num_iterations = 75
	num_factors = 10
	regularization = 0.05
	learn_rate = 0.005
	
	# number of user and movie IDs
	num_user_IDs = 6040
	num_movie_IDs = 3952
	
	# initalize U and M matrices with random numbers between 1 and 5
	np.random.seed(2)
	U = np.random.normal(loc=0.0, scale=0.1, size=(num_user_IDs, num_factors))
	M = np.random.normal(loc=0.0, scale=0.1, size=(num_factors, num_movie_IDs))
	
	print ''
	print 'Running incremental SVD algorithm: '
	print ''
	for i in range(num_iterations):
		
		startTime = time.time()
		print 'Iteration (',i+1,'/',num_iterations,')'
		
		for n in range(num_user_IDs):
			for m in range(num_movie_IDs):
			
				rating = utilitymatrix_training[n,m]
				if not np.isnan(rating):
					
					# caculate the error
					error = rating - np.dot(U[n,:], M[:,m])
					
					# update the values in U and M
					U[n,:] = U[n,:] + learn_rate * (2. * error * M[:,m] - regularization * U[n,:])
					M[:,m] = M[:,m] + learn_rate * (2. * error * U[n,:] - regularization * M[:,m])
		
		print ((time.time() - startTime) )* 75
					

	# get prediction matrix		
	predictionmatrix = np.dot(U, M)
	
	#set all values in U and M between 1 and 5
	predictionmatrix[predictionmatrix < 1] = 1.
	predictionmatrix[predictionmatrix > 5] = 5.
	
	
	# calculate MAE and RMSE
	MAE = get_MAE(predictionmatrix, utilitymatrix_test)
	RMSE = get_RMSE(predictionmatrix, utilitymatrix_test)

	return MAE, RMSE
		
#import_data()
main()
