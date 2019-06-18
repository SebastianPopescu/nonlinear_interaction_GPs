import numpy as np
import tensorflow as tf
from kernels import *
DTYPE=tf.float32


def conditional(Xnew, X, kernel_type, dim_input, position_interaction , type_var, white=True, full_cov=False):
	
	###########################################################
	### helper function to implement posterior GP equations ###
	###########################################################

	#### partially inspired fron ononymous function from GPflow #####

	num_data = tf.shape(X)[0]  # M
	Kmm = kernel(X,X,True,kernel_type, dim_input, position_interaction)
	Kmm = condition(Kmm)
	Kmn = kernel(X, Xnew,True,kernel_type, dim_input, position_interaction)
	Knn = kernel(Xnew,Xnew,full_cov,kernel_type, dim_input, position_interaction)
	with tf.variable_scope('model',reuse=True):

			q_sqrt_real = tf.get_variable('q_sqrt_real',dtype=DTYPE)
			q_mu = tf.get_variable('q_mu',dtype=DTYPE)

	if type_var=='full':

		q_sqrt = tf.matrix_band_part(q_sqrt_real,-1,0) 	
	else:
		q_sqrt = tf.square(q_sqrt_real)
			   
	return base_conditional(Kmn, Kmm, Knn, q_mu, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

def base_conditional(Kmn, Kmm, Knn, f, full_cov, q_sqrt=None, white=False):
    
	###########################################################
	### helper function to implement posterior GP equations ###
	###########################################################

	#### partially inspired fron ononymous function from GPflow #####

	Lm = tf.cholesky(Kmm)

	A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)
	if full_cov:

		fvar = Knn - tf.matmul(A, A, transpose_a=True)

	else:
		fvar = Knn - tf.transpose(tf.reduce_sum(tf.square(A), 0,keep_dims=True))

	if not white:
		A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

	fmean = tf.matmul(A, f, transpose_a=True)
	if full_cov:

		LTA= tf.matmul(tf.transpose(q_sqrt),A)
		fvar = fvar + tf.matmul(LTA,LTA,transpose_a=True)
	else:
		LTA= tf.matmul(tf.transpose(q_sqrt),A)
		fvar = fvar + tf.transpose(tf.reduce_sum(tf.square(LTA),0,keep_dims=True))

	return fmean, fvar


def conditional_interaction(Xnew, X, type_var, kernel_type, plot_effect, position_interaction, white=True, full_cov=False):

	num_data = tf.shape(X)[0]  # M
	Kmm = kernel_interaction(X1 = X, X2 = X, full_cov = True, kernel_type = kernel_type,
		plot_effect = plot_effect, position_interaction = position_interaction)
	Kmm = condition(Kmm)
	Kmn = kernel_interaction(X1 = X, X2 = Xnew, full_cov = True, kernel_type = kernel_type,
		plot_effect = plot_effect, position_interaction = position_interaction)
	Knn = kernel_interaction(X1 = Xnew, X2 = Xnew, full_cov = full_cov, kernel_type = kernel_type,
		plot_effect = plot_effect, position_interaction = position_interaction)
	
	with tf.variable_scope('model',reuse=True):

		q_sqrt_real = tf.get_variable('q_sqrt_real', dtype=DTYPE)
		q_mu = tf.get_variable('q_mu', dtype=DTYPE)

	if type_var=='full':

		q_sqrt = tf.matrix_band_part(q_sqrt_real,-1,0) 	
	
	else:
	
		q_sqrt = tf.square(q_sqrt_real)
		   
	return base_conditional(Kmn, Kmm, Knn, q_mu, full_cov=full_cov, q_sqrt=q_sqrt, white=white)


def conditional_additive(Xnew, X, dim_input, type_var, white=True, full_cov=False):
	
	num_data = tf.shape(X)[0]  # M
	Kmm = kernel_additive(X1 = X, X2 = X, full_cov = True, dim_input = dim_input)
	Kmm = condition(Kmm)
	Kmn = kernel_additive(X1 = X, X2 = Xnew, full_cov = True, dim_input = dim_input)
	Knn = kernel_additive(X1 = Xnew, X2 = Xnew, full_cov = full_cov, dim_input = dim_input)
	
	with tf.variable_scope('model',reuse=True):

			q_sqrt_real = tf.get_variable('q_sqrt_real',dtype=DTYPE)
			q_mu = tf.get_variable('q_mu',dtype=DTYPE)

	if type_var=='full':

		q_sqrt = tf.matrix_band_part(q_sqrt_real,-1,0) 	
	
	else:
	
		q_sqrt = tf.square(q_sqrt_real)
		   
	return base_conditional(Kmn, Kmm, Knn, q_mu, full_cov=full_cov, q_sqrt=q_sqrt, white=white)
