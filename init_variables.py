import numpy as np
import tensorflow as tf
DTYPE=tf.float32


def init_variables(num_data, type_var, kernel_type, dim_input ):	


	with tf.variable_scope('model'):

		q_mu = tf.get_variable(initializer = tf.zeros_initializer(),shape=(num_data,1),
			dtype=DTYPE,name='q_mu')				

		if type_var=='full':
			
			identity_matrix = tf.eye(num_data,dtype=DTYPE)
			q_sqrt_real = tf.get_variable(initializer = identity_matrix,
				dtype=DTYPE, name='q_sqrt_real')
							
		else:

			identity_matrix = tf.eye(num_data,dtype=DTYPE)
			q_sqrt_real = tf.get_variable(initializer = identity_matrix ,
				dtype=DTYPE, name='q_sqrt_real')	

		if kernel_type == 'interaction':

			##############################################
			######### Three-way Interaction term #########
			##############################################

			log_variance_kernel = tf.get_variable(initializer = tf.constant(0.301),dtype=tf.float32,
				name='log_variance_kernel')
		
			log_lengthscales = tf.get_variable(initializer = tf.constant([0.301 for _ in range(dim_input)]),
				dtype=tf.float32, name='log_lengthscales')
			
			
			##############################################
			########### Two-way Interaction terms ########
			##############################################

			####################################################################
			#### Reminder there are going to be 3 two-way interaction terms ####
			####################################################################

			for _ in range(3):

				log_variance_kernel = tf.get_variable(initializer = tf.constant(0.301),dtype=tf.float32,
					name='log_variance_kernel_two_way_'+str(_))

				log_lengthscales = tf.get_variable(initializer = tf.constant([0.301 for _ in range(2)]),
					dtype=tf.float32,name='log_lengthscales_two_way_'+str(_))


			##############################################
			######### Additive terms #####################
			##############################################

			for _ in range(dim_input):
	
				log_variance_kernel = tf.get_variable(initializer = tf.constant(0.301,dtype=DTYPE),dtype=DTYPE,
					name='log_variance_kernel_'+str(_))
				
				log_lengthscales = tf.get_variable(initializer = tf.constant([0.301],dtype=DTYPE),
					dtype=DTYPE,name='log_lengthscales_'+str(_))
			
		elif kernel_type == 'mixed-additive-interaction':
			
			############################################
			######### Two-way Interaction term #########
			############################################

			log_variance_kernel = tf.get_variable(initializer = tf.constant(0.301),dtype=tf.float32,
				name='log_variance_kernel')
			
			log_lengthscales = tf.get_variable(initializer = tf.constant([0.301 for _ in range(2)]),
				dtype=tf.float32,name='log_lengthscales')
			

			#####################################
			######### Additive terms ############
			#####################################

			for _ in range(dim_input):
	
				log_variance_kernel = tf.get_variable(initializer = tf.constant(0.301,dtype=DTYPE),dtype=DTYPE,
					name='log_variance_kernel_'+str(_))
				
				log_lengthscales = tf.get_variable(initializer = tf.constant([0.301],dtype=DTYPE),
					dtype=DTYPE,name='log_lengthscales_'+str(_))
		

		elif kernel_type == 'additive':
			
			#####################################
			######### Additive terms ############
			#####################################

			for _ in range(dim_input):
	
				log_variance_kernel = tf.get_variable(initializer = tf.constant(0.301,dtype=DTYPE),dtype=DTYPE,
					name='log_variance_kernel_'+str(_))
				
				log_lengthscales = tf.get_variable(initializer = tf.constant([0.301],dtype=DTYPE),
					dtype=DTYPE,name='log_lengthscales_'+str(_))

		else:

			pass
