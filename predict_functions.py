import numpy as np
import tensorflow as tf
from conditional_GP import *
DTYPE=tf.float32


def build_predict(X_new, X_train, kernel_type, dim_input, position_interaction, type_var):

	###################################################################
	#### get overall predictive mean and variance at testing time #####
	###################################################################

	f_mean,f_var = conditional(Xnew = X_new, X = X_train, kernel_type = kernel_type, dim_input = dim_input,
		position_interaction = position_interaction, type_var = type_var, full_cov=False, white=True)

	return tf.sigmoid(f_mean), f_var


def build_predict_interaction(X_new, X_train, type_var, kernel_type, plot_effect, position_interaction):

	############################################################################
	#### get interaction term predictive mean and variance at testing time #####
	############################################################################

	f_mean, f_var = conditional_interaction(Xnew = X_new, X = X_train, type_var = type_var,
		kernel_type = kernel_type, plot_effect = plot_effect, position_interaction = position_interaction, full_cov=False, white=True)

	return tf.sigmoid(f_mean), f_var


def build_predict_additive(X_new, X_train, type_var , dim_input):

	############################################################################
	###### get main effects predictive mean and variance at testing time #######
	############################################################################

	f_mean, f_var = conditional_additive(Xnew = X_new, X = X_train, dim_input = dim_input,
		type_var = type_var, full_cov=False, white=True)

	return tf.sigmoid(f_mean),f_var