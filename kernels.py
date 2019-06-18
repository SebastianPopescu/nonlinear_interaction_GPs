import numpy as np
import tensorflow as tf

def condition(X):

	######################################################################
	#### helper function to ensure cholesy decomposition is sucessful ####
	######################################################################

	return X + tf.eye(tf.shape(X)[0]) * 1e-3


def RBF(X1,X2,log_lengthscales,log_variance_kernel):

	##############################################################################
	##### Squared Exponential kernel with Automatic Relevance Determination ######
	##############################################################################

	X1 = X1 / tf.exp(log_lengthscales)
	X2 = X2 / tf.exp(log_lengthscales)
	X1s = tf.reduce_sum(tf.square(X1),1)
	X2s = tf.reduce_sum(tf.square(X2),1)       

	cov_matrix = tf.exp(log_variance_kernel) * tf.exp(-(-2.0 * tf.matmul(X1,tf.transpose(X2)) + tf.reshape(X1s,(-1,1)) + tf.reshape(X2s,(1,-1)))/2)      

	return cov_matrix

def kernel(X1, X2, full_cov, kernel_type, dim_input, position_interaction):

	############################################################
	#### helper function to construct the additive kernels #####
	############################################################

	with tf.variable_scope('model', reuse=True):

		if kernel_type == 'interaction':

			##############################################
			######## Three-way Interaction term ##########
			##############################################

			log_lengthscale = tf.get_variable('log_lengthscales')
			log_variance_kernel = tf.get_variable('log_variance_kernel')
			
			if full_cov:

				output_kernel = RBF(X1,X2,log_lengthscale,log_variance_kernel)
			
			else:
			
				output_kernel = RBF_Kdiag(X1,log_variance_kernel)

			###############################################
			####### Two-way Interaction terms #############
			###############################################

			##### reminder -- there are going to be three two-way interaction terms #####

			interaction_positions=[[0,1],[1,2],[0,2]]

			for _ in range(3):

				log_variance_kernel = tf.get_variable('log_variance_kernel_two_way_'+str(_))
				log_lengthscale = tf.get_variable('log_lengthscales_two_way_'+str(_))
				
				input_x1 = tf.gather(X1,axis=-1,indices=interaction_positions[_])
				input_x2 = tf.gather(X2,axis=-1,indices=interaction_positions[_])
				
				if full_cov:

					output_kernel += RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			
				else:
					output_kernel += RBF_Kdiag(input_x1,log_variance_kernel)	

			###############################################
			######## Additive part ########################
			###############################################

			for _ in range(dim_input):

				log_variance_kernel = tf.get_variable('log_variance_kernel_'+str(_))
				log_lengthscale = tf.get_variable('log_lengthscales_'+str(_))
				
				input_x1 = tf.slice(X1,[0,_],[-1,1])
				input_x2 = tf.slice(X2,[0,_],[-1,1])
				
				if full_cov:

					output_kernel += RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			
				else:
					output_kernel += RBF_Kdiag(input_x1,log_variance_kernel)	

		elif kernel_type == 'mixed-additive-interaction':

			#########################################################
			########## Two-way Interaction term #####################
			#########################################################

			log_variance_kernel = tf.get_variable('log_variance_kernel')
			log_lengthscale = tf.get_variable('log_lengthscales')
			
			input_x1 = tf.gather(X1,axis=-1,indices=position_interaction)
			input_x2 = tf.gather(X2,axis=-1,indices=position_interaction)

			if full_cov:

				output_kernel = RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			
			else:
				output_kernel = RBF_Kdiag(input_x1,log_variance_kernel)	

			######################################
			######## Additive part ###############
			######################################

			for _ in range(dim_input):

				log_variance_kernel = tf.get_variable('log_variance_kernel_'+str(_))
				log_lengthscale = tf.get_variable('log_lengthscales_'+str(_))

				input_x1 = tf.gather(X1,axis=-1,indices=[ _ ])
				input_x2 = tf.gather(X2,axis=-1,indices=[ _ ])

				if full_cov:

					output_kernel += RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			
				else:
					output_kernel += RBF_Kdiag(input_x1,log_variance_kernel)	


		else:
			#############################
			##### Additive kernels ######
			#############################

			log_variance_kernel = tf.get_variable('log_variance_kernel_0')
			log_lengthscale = tf.get_variable('log_lengthscales_0')
			
			input_x1 = tf.gather(X1,axis=-1,indices=[0])
			input_x2 = tf.gather(X2,axis=-1,indices=[0])
			
			if full_cov:

				output_kernel = RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			
			else:
				output_kernel = RBF_Kdiag(input_x1,log_variance_kernel)	

			for _ in range(1, dim_input):

				log_variance_kernel = tf.get_variable('log_variance_kernel_'+str(_),)
				log_lengthscale = tf.get_variable('log_lengthscales_'+str(_),)
				
				input_x1 = tf.gather(X1,axis=-1,indices=[ _ ])
				input_x2 = tf.gather(X2,axis=-1,indices=[ _ ])
				
				if full_cov:

					output_kernel += RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			
				else:
					output_kernel += RBF_Kdiag(input_x1,log_variance_kernel)	


	return output_kernel

def RBF_Kdiag(X,variance_kernel):
	
	### computes the diagonal covariance matrix of Kff
	
	return tf.ones((tf.shape(X)[0],1),dtype=tf.float32) * tf.exp(variance_kernel)	

def kernel_interaction(X1, X2, full_cov, kernel_type, plot_effect, position_interaction):

	##############################################################################
	##### supports both "three-way interactions " and "two-way interactions" #####
	##############################################################################

	with tf.variable_scope('model', reuse=True):

		if kernel_type=='interaction':

			if plot_effect == 'three-way':

				###############################################
				######## Three-way interaction term  ##########
				###############################################

				log_lengthscale = tf.get_variable('log_lengthscales')
				log_variance_kernel = tf.get_variable('log_variance_kernel')
				
				if full_cov:

					output_kernel = RBF(X1,X2,log_lengthscale,log_variance_kernel)
				
				else:
				
					output_kernel = RBF_Kdiag(X1,log_variance_kernel)
			
			elif plot_effect == 'two-way':

				##################################################
				#### Summation of two-way interaction terms ######
				##################################################
				interaction_positions=[[0,1],[1,2],[0,2]]

				log_variance_kernel = tf.get_variable('log_variance_kernel_two_way_'+str(0))
				log_lengthscale = tf.get_variable('log_lengthscales_two_way_'+str(0))
				
				input_x1 = tf.gather(X1,axis=-1,indices=interaction_positions[0])
				input_x2 = tf.gather(X2,axis=-1,indices=interaction_positions[0])
				
				if full_cov:

					output_kernel = RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			
				else:
					output_kernel = RBF_Kdiag(input_x1,log_variance_kernel)	

				for _ in range(1,3):

					log_variance_kernel = tf.get_variable('log_variance_kernel_two_way_'+str(_))
					log_lengthscale = tf.get_variable('log_lengthscales_two_way_'+str(_))
					
					input_x1 = tf.gather(X1,axis=-1,indices=interaction_positions[_])
					input_x2 = tf.gather(X2,axis=-1,indices=interaction_positions[_])

					if full_cov:

						output_kernel += RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			

					else:

						output_kernel += RBF_Kdiag(input_x1,log_variance_kernel)	


		elif kernel_type=='mixed-additive-interaction':

			#########################################################
			########## Two-way interaction term #####################
			#########################################################

			log_variance_kernel = tf.get_variable('log_variance_kernel')
			log_lengthscale = tf.get_variable('log_lengthscales')
			
			input_x1 = tf.gather(X1,axis=-1,indices=position_interaction)
			input_x2 = tf.gather(X2,axis=-1,indices=position_interaction)

			if full_cov:

				output_kernel = RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			
	
			else:
	
				output_kernel = RBF_Kdiag(input_x1,log_variance_kernel)	

	return output_kernel


def kernel_additive(X1, X2, full_cov, dim_input):

	##################################################
	###### we get the additive univariate kernels ####
	##################################################

	with tf.variable_scope('model',reuse=True):

		log_variance_kernel = tf.get_variable('log_variance_kernel_0')
		log_lengthscale = tf.get_variable('log_lengthscales_0')
		
		input_x1 = tf.gather(X1,axis=-1,indices=[0])
		input_x2 = tf.gather(X2,axis=-1,indices=[0])
		
		if full_cov:

			output_kernel = RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			

		else:

			output_kernel = RBF_Kdiag(input_x1,log_variance_kernel)	

		for _ in range(1,dim_input):

			log_variance_kernel = tf.get_variable('log_variance_kernel_'+str(_))
			log_lengthscale = tf.get_variable('log_lengthscales_'+str(_))
			
			input_x1 = tf.gather(X1,axis=-1,indices=[ _ ])
			input_x2 = tf.gather(X2,axis=-1,indices=[ _ ])
			
			if full_cov:

				output_kernel += RBF(input_x1,input_x2,log_lengthscale,log_variance_kernel)			
			
			else:
			
				output_kernel += RBF_Kdiag(input_x1,log_variance_kernel)	


	return output_kernel
