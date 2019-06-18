import numpy as np
import tensorflow as tf
DTYPE=tf.float32
from kernels import *

def KL(q_mu, q_Delta, type_var):

	############################################
	#### Kullback-Liebler divergence term ######
	############################################

	KL_term = tf.constant(0.0, dtype=DTYPE)
	KL_term += - 2.0 * tf.reduce_sum(tf.log(tf.diag_part(q_Delta)))
	
	if type_var=='full':

		KL_term += tf.trace(tf.matmul(q_Delta,q_Delta,transpose_b=True))
	
	else:

		KL_term += tf.trace(tf.square(q_Delta))
    
	KL_term += tf.matmul(q_mu,q_mu ,transpose_a=True) - tf.cast(tf.shape(q_mu)[0],DTYPE )

	return 0.5 * KL_term

def bernoulli(x, p):

	#################################
	### bernoulli log-likelihood ####
	#################################

	return tf.log(tf.where(tf.equal(x, 1), p, 1-p))

def re_error(X, Y, full_cov, kernel_type, dim_input, position_interaction, type_var, num_data):

	########################################
	#### model fit part of the ELBO ########
	########################################
	Kmm = kernel(X1 = X, X2 = X, kernel_type = kernel_type, dim_input = dim_input,
		position_interaction = position_interaction, full_cov = True)
	Kmm = condition(Kmm)
	L  = tf.cholesky(Kmm)
	with tf.variable_scope('model',reuse=True):

			q_sqrt_real = tf.get_variable('q_sqrt_real',dtype=DTYPE)
			q_mu = tf.get_variable('q_mu',dtype=DTYPE)

	if type_var=='full':

		q_sqrt = tf.matrix_band_part(q_sqrt_real,-1,0) 	
	else:
		q_sqrt = tf.square(q_sqrt_real)
		
	f_mean = tf.matmul(L,q_mu)
	f_var_sqrt = tf.matmul(L,q_sqrt)
	f_var = tf.diag_part(tf.square(f_var_sqrt))
	f_var = tf.reshape(f_var,[num_data,1])
	
	if full_cov:

		sample_y = f_mean + tf.matmul(tf.cholesky(condition(f_var)),tf.random_normal(shape=(tf.shape(f_mean)[0],1),dtype=tf.float32))
	
	else:

		sample_y = f_mean + tf.multiply(tf.sqrt(f_var),tf.random_normal(shape=(tf.shape(f_mean)[0],1),dtype=tf.float32))

	
	return tf.reduce_sum(bernoulli(Y,tf.sigmoid(sample_y)))


def build_likelihood(X, Y, kernel_type, dim_input, position_interaction, type_var, num_data):

	###########################################################################################################
	##### helper function to combine model fit and KL divergence terms to arrive at final form of the ELBO ####
	###########################################################################################################

	bound = re_error(X, Y, kernel_type, dim_input, position_interaction, type_var, num_data)
	with tf.variable_scope('model',reuse=True):

		q_sqrt_real = tf.get_variable('q_sqrt_real',dtype=DTYPE)
		q_mu = tf.get_variable('q_mu',dtype=DTYPE)

	if type_var=='full':

		q_sqrt = tf.matrix_band_part(q_sqrt_real,-1,0) 	
	else:
		q_sqrt = tf.square(q_sqrt_real)
		
	bound-= KL(q_mu, q_sqrt, type_var)

	return -bound
