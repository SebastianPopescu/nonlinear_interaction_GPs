# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from collections import defaultdict
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import time
import argparse
import os
import math
from sklearn.cluster import  KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
DTYPE=tf.float32
import pandas as pd
import seaborn as sns
sns.set_style('white')
from init_variables import *
from losses import *
from kernels import *
from conditional_GP import *
from predict_functions import *

class VariationalGaussianProcessClassifier(object):

	def __init__(self, num_data,
		dim_input, dim_output,
		num_iterations, type_var,
		full_cov,
		kernel_type, position_interaction,
		names, model, anchor):

		self.model = model
		self.anchor = anchor			
		self.kernel_type = kernel_type
		self.position_interaction = position_interaction
		self.full_cov = full_cov
		self.num_data = num_data
		self.type_var = type_var
		self.dim_input = dim_input
		self.dim_output = dim_output
		self.num_iterations = num_iterations
		self.names = names

		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess= tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))		
		self.X = tf.placeholder(tf.float32,shape=(None,dim_input),name='Input_ph_train')
		self.X_test = tf.placeholder(tf.float32,shape=(None,dim_input),name='Input_ph_test')
		self.Y = tf.placeholder(tf.float32,shape=(None,dim_output),name='Output_ph_train')
		self.Y_test = tf.placeholder(tf.float32,shape=(None,dim_output),name='Output_ph_test')


	def session_TF(self, X_training, Y_training, X_testing, Y_testing):

		##############################################################
		#### main function that executes the code ####################
		##############################################################

		############################
		#### get model fit cost ####
		############################

		re_cost = re_error(self.X, self.Y, self.full_cov, self.kernel_type,
			self.dim_input, self.position_interaction, self.type_var, self.num_data)
		
		with tf.variable_scope('model',reuse=True):

				q_sqrt_real = tf.get_variable('q_sqrt_real',dtype=DTYPE)
				q_mu = tf.get_variable('q_mu',dtype=DTYPE)

		if self.type_var=='full':

			q_sqrt = tf.matrix_band_part(q_sqrt_real,-1,0) 	
		else:
			q_sqrt = tf.square(q_sqrt_real)

		#####################
		#### get KL cost ####
		#####################

		kl_cost = KL(q_mu, q_sqrt, self.type_var)

		########################
		#### get final cost ####
		########################

		cost = - re_cost + kl_cost
		opt = tf.train.AdamOptimizer(1e-4)
		train_op = opt.minimize(cost)

		##########################################################
		#### get predictive mean and variance at testing time ####	
		##########################################################

		#############################
		#### overall predictions ####
		#############################

		predictions_mean, predictions_var = build_predict(self.X_test, self.X,
			self.kernel_type, self.dim_input, self.position_interaction, self.type_var)
		
		if self.kernel_type=='interaction':

			predictions_mean_three_way_interaction, predictions_var_three_way_interaction = build_predict_interaction(self.X_test,
				self.X, self.type_var, self.kernel_type, 'three-way', self.position_interaction)

			predictions_mean_two_way_interaction, predictions_var_two_way_interaction = build_predict_interaction(self.X_test,
				self.X, self.type_var, self.kernel_type, 'two-way', self.position_interaction)

			predictions_mean_additive, predictions_var_additive = build_predict_additive(self.X_test, self.X, self.type_var,
				self.dim_input)

		elif self.kernel_type=='mixed-additive-interaction':

			predictions_mean_two_way_interaction, predictions_var_two_way_interaction = build_predict_interaction(self.X_test,self.X, 
				self.type_var, self.kernel_type, 'two-way', self.position_interaction)

			predictions_mean_additive, predictions_var_additive = build_predict_additive(self.X_test,self.X, self.type_var,
				self.dim_input)

		test_log_likelihood = tf.reduce_sum(bernoulli(self.Y_test, predictions_mean))
	
		tf.summary.scalar(tensor = tf.squeeze(kl_cost),name = 'kl_cost')
		tf.summary.scalar(tensor = tf.squeeze(re_cost),name = 're_cost')
		tf.summary.scalar(tensor = tf.squeeze(test_log_likelihood),name = 'test_ll')
		merged = tf.summary.merge_all()

		if self.kernel_type=='mixed-additive-interaction':

			train_writer = tf.summary.FileWriter('./tensorboard_'+str(self.kernel_type)+'_'+str(self.model)+'/'+str(self.names[self.position_interaction[0]])+'_'+str(self.names[self.position_interaction[1]]),self.sess.graph)

		else:

			train_writer = tf.summary.FileWriter('./tensorboard_'+str(self.kernel_type)+'_'+str(self.model),self.sess.graph)

		self.sess.run(tf.global_variables_initializer())

		for i in range(self.num_iterations):

			_,cost_np,re_cost_np, kl_cost_np, summary  = self.sess.run([train_op,cost,re_cost,kl_cost,merged],
				feed_dict={self.X:X_training, self.Y:Y_training,
				self.X_test:X_testing, self.Y_test:Y_testing})
			train_writer.add_summary(summary,i)
			
			if i % 1000 == 0  and i != 0:

				preds_now = self.sess.run(predictions_mean,feed_dict={self.X_test:X_testing, self.X:X_training})
				predictii_bi = binarize(preds_now,0.5)
				print('***** validation accuracy is '+str(accuracy_score(Y_testing,predictii_bi))+' ******')

			print('at iteration '+str(i) + ' we have nll : '+str(cost_np) + 're cost :'+str(re_cost_np)+' kl cost :'+str(kl_cost_np))
			
		preds_now, vars_now = self.sess.run([predictions_mean, predictions_var], feed_dict={self.X_test:X_testing, self.X:X_training})
		predictii_bi = binarize(preds_now,0.5)
		print('***** final accuracy is '+str(accuracy_score(Y_testing,predictii_bi))+' ******')

		x_hippo = np.linspace(3000,10000,500)
		x_ptau = np.linspace(10.0,100.0,500)
		x_ab_pet = np.linspace(0.8,2.5,500)
		x_ab_csf = np.linspace(70,300,500)
		x_brain_pad = np.linspace(-20.0,20.0,500)

		if self.model=='model1':

			x_dim1 = np.linspace(-20.0,20.0,500)
			x_dim2 = np.linspace(70,300,500)
			x_dim3 = np.linspace(3000,10000,500)

		elif self.model=='model2':

			x_dim1 = np.linspace(-20.0,20.0,500)
			x_dim2 = np.linspace(0.8,2.5,500)
			x_dim3 = np.linspace(3000,10000,500)

		elif self.model=='model3':

			x_dim3 = np.linspace(3000,10000,500)
			x_dim2 = np.linspace(10.0,100.0,500)
			x_dim1 = np.linspace(0.8,2.5,500)


		grid1 = np.meshgrid(x_dim3,x_dim2)
		dim3,dim2 = grid1
		dim3 = dim3.reshape((-1,1))
		dim2 = dim2.reshape((-1,1))

		grid2 = np.meshgrid(x_dim3,x_dim1)
		plm,dim1 = grid2
		dim1 = dim1.reshape((-1,1))

		low_anchor_value = np.percentile(X_training_original[:,self.anchor],25)
		medium_anchor_value = np.percentile(X_training_original[:,self.anchor],50)
		high_anchor_value = np.percentile(X_training_original[:,self.anchor],75)

		################################################################
		### create synthetic testing input for low value for anchor ####
		################################################################

		scalar  = np.ones((500*500,1)) * low_anchor_value
		
		if self.anchor == 0:

			low_anchor_input  = np.concatenate((scalar,dim2,dim3),axis=1)

		elif self.anchor == 1:

			low_anchor_input  = np.concatenate((dim1,scalar,dim3),axis=1)
		
		elif self.anchor == 2:

			low_anchor_input  = np.concatenate((dim1,dim2,scalar),axis=1)

		##################################################################
		### create synthetic testing input for medium value for anchor ###
		##################################################################

		scalar  = np.ones((500*500,1)) * medium_anchor_value
		
		if self.anchor==0:

			medium_anchor_input  = np.concatenate((scalar,dim2,dim3),axis=1)

		elif self.anchor==1:

			medium_anchor_input  = np.concatenate((dim1,scalar,dim3),axis=1)
		
		elif self.anchor==2:

			medium_anchor_input  = np.concatenate((dim1,dim2,scalar),axis=1)

		##################################################################
		### create synthetic testing input for high value for anchor #####
		##################################################################
		
		scalar  = np.ones((500*500,1)) * high_anchor_value
		
		if self.anchor==0:

			high_anchor_input  = np.concatenate((scalar,dim2,dim3),axis=1)

		elif self.anchor==1:

			high_anchor_input  = np.concatenate((dim1,scalar,dim3),axis=1)
		
		elif self.anchor==2:

			high_anchor_input  = np.concatenate((dim1,dim2,scalar),axis=1)

		#### Stratified on Anchor ####
		#################################

		lista_anchor_input = [low_anchor_input,medium_anchor_input,high_anchor_input]
		current_anchor = ['Low','Medium','High']
		if self.anchor==0:
			remainder_values = [dim2,dim3] 
			remainder_positions = [1,2]

		elif self.anchor==1:
			remainder_values = [dim1,dim3]
			remainder_positions = [0,2]

		elif self.anchor==2:
			remainder_values = [dim1,dim2]
			remainder_positions = [0,1]

		if self.kernel_type=='interaction':

			########################################
			##### predictive mean dictionaries #####
			########################################

			dict_mean_pred = defaultdict()
			for iterator in range(4):
				dict_mean_pred[iterator] = defaultdict()

			############################################
			##### predictive variance dictionaries #####
			############################################

			dict_var_pred = defaultdict()
			for iterator in range(4):
				dict_var_pred[iterator] = defaultdict()
			
			for anchor_input,anchor_type in zip(lista_anchor_input,current_anchor):
				
				anchor_input_transformed  = scaler.transform(anchor_input)

				predictions_mean_np, predictions_mean_three_way_interaction_np, predictions_mean_two_way_interaction_np, predictions_mean_additive_np, predictions_var_np, predictions_var_three_way_interaction_np, predictions_var_two_way_interaction_np, predictions_var_additive_np = self.sess.run([predictions_mean,predictions_mean_three_way_interaction,
					predictions_mean_two_way_interaction, predictions_mean_additive, predictions_var, predictions_var_three_way_interaction,
					predictions_var_two_way_interaction, predictions_var_additive], feed_dict={ self.X_test:anchor_input_transformed,
					self.X:X_training })
				
				predictions_mean_np = np.abs(1.0 - predictions_mean_np)
				predictions_mean_three_way_interaction_np = np.abs(1.0 - predictions_mean_three_way_interaction_np) - 0.5
				predictions_mean_two_way_interaction_np = np.abs(1.0 -predictions_mean_two_way_interaction_np) - 0.5
				predictions_mean_additive_np = np.abs(1.0 - predictions_mean_additive_np) -0.5

				dict_mean_pred[2][anchor_type] = predictions_mean_three_way_interaction_np
				dict_mean_pred[1][anchor_type] = predictions_mean_two_way_interaction_np
				dict_mean_pred[0][anchor_type] = predictions_mean_additive_np
				dict_mean_pred[3][anchor_type] = predictions_mean_np

				dict_var_pred[2][anchor_type] = predictions_var_three_way_interaction_np
				dict_var_pred[1][anchor_type] = predictions_var_two_way_interaction_np
				dict_var_pred[0][anchor_type] = predictions_var_additive_np
				dict_var_pred[3][anchor_type] = predictions_var_np

			culoare = 'black'
			marker_size = 240
			
			fig, ax = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True, figsize=(90,60))
			
			########################################
			####### Predictive Mean Plots ##########
			########################################

			for iterator in range(4):
				for iterator_anchor in range(3):
		                                                                     
					im = ax[iterator,iterator_anchor].contourf(remainder_values[0].reshape((500,500)), remainder_values[1].reshape((500,500)),
						dict_mean_pred[iterator][current_anchor[iterator_anchor]].reshape((500,500)),
						levels = 20,
						#levels=(0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0),
						#cmap='coolwarm', vmin=0.0, vmax=1.0)
						cmap='coolwarm')
					#cb = plt.colorbar()

					cb = fig.colorbar(im,ax=ax[iterator,iterator_anchor])
					cb.ax.tick_params(labelsize=50)
					cb

					ax[iterator,iterator_anchor].autoscale(False)
					m = ['o', 'x']
					control_pmci = 0
					control_smci = 0
					names = ['pMCI','sMCI']
					for i in range(X_training.shape[0]):

						if current_anchor[iterator_anchor]=='Low':
							if X_training_original[i,self.anchor] < low_anchor_value:        
								if control_smci==0 and int(Y_training[i,0])==1:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_smci+=1
								elif control_pmci==0 and int(Y_training[i,0])==0:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_pmci+=1            
								else:

									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,s=marker_size)                                                            
							else:
								pass

						elif current_anchor[iterator_anchor]=='Medium':

							if X_training_original[i,self.anchor] > low_anchor_value and X_training_original[i,self.anchor] < high_anchor_value:
								if control_smci==0 and int(Y_training[i,0])==1:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_smci+=1
								elif control_pmci==0 and int(Y_training[i,0])==0:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_pmci+=1            
								else:

									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,s=marker_size)                                                            
							else:
								pass

						elif current_anchor[iterator_anchor]=='High':
							if X_training_original[i,self.anchor] > high_anchor_value:        
								if control_smci==0 and int(Y_training[i,0])==1:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_smci+=1
								elif control_pmci==0 and int(Y_training[i,0])==0:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_pmci+=1            
								else:

									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,s=marker_size)                                                            
							else:
								pass
				
					ax[iterator,iterator_anchor].legend(numpoints=1, fontsize=40)
					ax[iterator,iterator_anchor].set_xlabel(self.names[remainder_positions[0]], fontsize=60)
					ax[iterator,iterator_anchor].set_ylabel(self.names[remainder_positions[1]], fontsize=60)
					ax[0,iterator_anchor].set_title(str(self.kernel_type)+' '+str(current_anchor[iterator_anchor])+' '+str(self.names[self.anchor]), fontsize=80)
					ax[iterator,iterator_anchor].tick_params(axis='both', which='major', labelsize=50)
					ax[iterator,iterator_anchor].tick_params(axis='both', which='minor', labelsize=50)
			
			fig.tight_layout()
			plt.savefig('./predictive_mean_'+str(self.model)+'_'+str(self.kernel_type)+'_'+str(self.names[self.anchor])+'.jpeg')   
			plt.close()

		elif self.kernel_type=='mixed-additive-interaction':
			
			########################################			
			##### predictive mean dictionaries #####
			########################################

			dict_mean_pred = defaultdict()
			
			############################################
			##### predictive variance dictionaries #####
			############################################

			dict_var_pred = defaultdict()			

			for iterator in range(3):
			
				dict_mean_pred[iterator] = defaultdict()
				dict_var_pred[iterator] = defaultdict()
			
			for anchor_input,anchor_type in zip(lista_anchor_input,current_anchor):
				
				anchor_input_transformed  = scaler.transform(anchor_input)

				predictions_mean_np, predictions_mean_two_way_interaction_np, predictions_mean_additive_np, predictions_var_np, predictions_var_two_way_interaction_np, predictions_var_additive_np = self.sess.run([predictions_mean,
					predictions_mean_two_way_interaction, predictions_mean_additive, predictions_var, 
					predictions_var_two_way_interaction, predictions_var_additive], feed_dict={self.X_test:anchor_input_transformed,
					self.X:X_training})
				
				predictions_mean_np = np.abs(1.0 - predictions_mean_np)
				predictions_mean_two_way_interaction_np = np.abs(1.0 -predictions_mean_two_way_interaction_np)
				predictions_mean_additive_np = np.abs(1.0 - predictions_mean_additive_np)
				
				dict_mean_pred[1][anchor_type] = predictions_mean_two_way_interaction_np
				dict_mean_pred[0][anchor_type] = predictions_mean_additive_np
				dict_mean_pred[2][anchor_type] = predictions_mean_np

				dict_var_pred[1][anchor_type] = predictions_var_two_way_interaction_np
				dict_var_pred[0][anchor_type] = predictions_var_additive_np
				dict_var_pred[2][anchor_type] = predictions_var_np

			culoare = 'black'
			marker_size = 240
			
			fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(90,60))
			
			########################################
			####### Predictive Mean Plots ##########
			########################################

			for iterator in range(3):
				for iterator_anchor in range(3):
		                                                                     
					im = ax[iterator,iterator_anchor].contourf(remainder_values[0].reshape((500,500)), remainder_values[1].reshape((500,500)),
						dict_mean_pred[iterator][current_anchor[iterator_anchor]].reshape((500,500)),
						levels = 20,
						#levels=(0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0),
						#cmap='coolwarm', vmin=0.0, vmax=1.0)
						cmap='coolwarm')
					#cb = plt.colorbar()

					cb = fig.colorbar(im,ax=ax[iterator,iterator_anchor])
					cb.ax.tick_params(labelsize=50)
					cb

					ax[iterator,iterator_anchor].autoscale(False)
					m = ['o', 'x']
					control_pmci = 0
					control_smci = 0
					names = ['pMCI','sMCI']
					for i in range(X_training.shape[0]):

						if current_anchor[iterator_anchor]=='Low':
							if X_training_original[i,self.anchor] < low_anchor_value:        
								if control_smci==0 and int(Y_training[i,0])==1:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_smci+=1
								elif control_pmci==0 and int(Y_training[i,0])==0:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_pmci+=1            
								else:

									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,s=marker_size)                                                            
							else:
								pass

						elif current_anchor[iterator_anchor]=='Medium':

							if X_training_original[i,self.anchor] > low_anchor_value and X_training_original[i,self.anchor] < high_anchor_value:
								if control_smci==0 and int(Y_training[i,0])==1:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_smci+=1
								elif control_pmci==0 and int(Y_training[i,0])==0:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_pmci+=1            
								else:

									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,s=marker_size)                                                            
							else:
								pass

						elif current_anchor[iterator_anchor]=='High':
							if X_training_original[i,self.anchor] > high_anchor_value:        
								if control_smci==0 and int(Y_training[i,0])==1:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_smci+=1
								elif control_pmci==0 and int(Y_training[i,0])==0:
									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
									control_pmci+=1            
								else:

									ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
										marker=m[int(Y_training[i,0])],color=culoare,s=marker_size)                                                            
							else:
								pass
				
					ax[iterator,iterator_anchor].legend(numpoints=1, fontsize=40)
					ax[iterator,iterator_anchor].set_xlabel(self.names[remainder_positions[0]], fontsize=60)
					ax[iterator,iterator_anchor].set_ylabel(self.names[remainder_positions[1]], fontsize=60)
					ax[0,iterator_anchor].set_title(str(self.kernel_type)+' '+str(current_anchor[iterator_anchor])+' '+str(self.names[self.anchor]), fontsize=80)
					ax[iterator,iterator_anchor].tick_params(axis='both', which='major', labelsize=50)
					ax[iterator,iterator_anchor].tick_params(axis='both', which='minor', labelsize=50)
			
			fig.tight_layout()
			plt.savefig('./predictive_mean_'+str(self.model)+'_'+str(self.kernel_type)+'_'+str(self.names[self.anchor])+'.jpeg')   
			plt.close()

		elif self.kernel_type=='additive':

			########################################
			##### predictive mean dictionaries #####
			########################################

			dict_mean_pred_additive = defaultdict()

			############################################
			##### predictive variance dictionaries #####
			############################################

			dict_var_pred_additive = defaultdict()
		
			for anchor_input,anchor_type in zip(lista_anchor_input,current_anchor):
				
				anchor_input_transformed  = scaler.transform(anchor_input)

				predictions_mean_np, predictions_var_np = self.sess.run([predictions_mean, predictions_var],
					feed_dict={self.X_test:anchor_input_transformed,
					self.X:X_training})
				
				predictions_mean_np = np.abs(1.0 - predictions_mean_np)
	
				dict_mean_pred[anchor_type] = predictions_mean_np

				dict_var_pred[anchor_type] = predictions_var_np

			culoare = 'black'
			marker_size = 240
			
			fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(90,60))
			
			########################################
			####### Predictive Mean Plots ##########
			########################################

			iterator = 0
			for iterator_anchor in range(3):
	                                                                     
				im = ax[iterator,iterator_anchor].contourf(remainder_values[0].reshape((500,500)), remainder_values[1].reshape((500,500)),
					dict_mean_pred[current_anchor[iterator_anchor]].reshape((500,500)),
					levels = 20,
					#levels=(0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0),
					#cmap='coolwarm', vmin=0.0, vmax=1.0)
					cmap='coolwarm')
				#cb = plt.colorbar()

				cb = fig.colorbar(im,ax=ax[iterator,iterator_anchor])
				cb.ax.tick_params(labelsize=50)
				cb

				ax[iterator,iterator_anchor].autoscale(False)
				m = ['o', 'x']
				control_pmci = 0
				control_smci = 0
				names = ['pMCI','sMCI']
				for i in range(X_training.shape[0]):

					if current_anchor[iterator_anchor]=='Low':
						if X_training_original[i,self.anchor] < low_anchor_value:        
							if control_smci==0 and int(Y_training[i,0])==1:
								ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
									marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
								control_smci+=1
							elif control_pmci==0 and int(Y_training[i,0])==0:
								ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
									marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
								control_pmci+=1            
							else:

								ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
									marker=m[int(Y_training[i,0])],color=culoare,s=marker_size)                                                            
						else:
							pass

					elif current_anchor[iterator_anchor]=='Medium':

						if X_training_original[i,self.anchor] > low_anchor_value and X_training_original[i,self.anchor] < high_anchor_value:
							if control_smci==0 and int(Y_training[i,0])==1:
								ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
									marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
								control_smci+=1
							elif control_pmci==0 and int(Y_training[i,0])==0:
								ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
									marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
								control_pmci+=1            
							else:

								ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
									marker=m[int(Y_training[i,0])],color=culoare,s=marker_size)                                                            
						else:
							pass

					elif current_anchor[iterator_anchor]=='High':
						if X_training_original[i,self.anchor] > high_anchor_value:        
							if control_smci==0 and int(Y_training[i,0])==1:
								ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
									marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
								control_smci+=1
							elif control_pmci==0 and int(Y_training[i,0])==0:
								ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
									marker=m[int(Y_training[i,0])],color=culoare,label=names[int(Y_training[i,0])],s=marker_size)
								control_pmci+=1            
							else:

								ax[iterator,iterator_anchor].scatter(X_training_original[i,remainder_positions[0]], X_training_original[i,remainder_positions[1]], 
									marker=m[int(Y_training[i,0])],color=culoare,s=marker_size)                                                            
						else:
							pass
			
				ax[iterator,iterator_anchor].legend(numpoints=1, fontsize=40)
				ax[iterator,iterator_anchor].set_xlabel(self.names[remainder_positions[0]], fontsize=60)
				ax[iterator,iterator_anchor].set_ylabel(self.names[remainder_positions[1]], fontsize=60)
				ax[0,iterator_anchor].set_title(str(self.kernel_type)+' '+str(current_anchor[iterator_anchor])+' '+str(self.names[self.anchor]), fontsize=80)
				ax[iterator,iterator_anchor].tick_params(axis='both', which='major', labelsize=50)
				ax[iterator,iterator_anchor].tick_params(axis='both', which='minor', labelsize=50)
			
			fig.tight_layout()
			plt.savefig('./predictive_mean_'+str(self.model)+'_'+str(self.kernel_type)+'_'+str(self.names[self.anchor])+'.jpeg')   
			plt.close()


if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--kernel_type',type=str,default='interaction',help='can be additive;interaction;mixed-additive-interaction')
	parser.add_argument('--position_interaction',type=int,nargs='+')
	parser.add_argument('--model',type=str,default='model1',help='can be model1;model2;model3')
	parser.add_argument('--anchor',type=int,default=0)
	parser.add_argument('--num_iterations',type=int,default=5001)
	args = parser.parse_args()

	num_cv = 5
	data = np.genfromtxt('./csv_classification_overall.csv',delimiter=',',skip_header=1)
	### delete the row number ###
	#############################
	data = np.delete(data,[0],axis=1)
	print(data[:,1])
	indices_smci = np.where(data[:,1]==1)[0]
	indices_pmci = np.where(data[:,1]==0)[0]
	print(indices_smci)

	data_smci = data[indices_smci,]
	print('shape of sMCI data')
	print(data_smci.shape)
	
	data_pmci = data[indices_pmci,]
	print('shape of pMCI data')
	print(data_pmci.shape)
	
	### subset the sMCI ###########
	np.random.seed(1)
	np.random.shuffle(indices_smci)
	print(indices_smci)

	np.random.shuffle(data_smci)
	data_smci = data_smci[:data_pmci.shape[0],:]
	print(data_smci)
	data = np.concatenate((data_pmci,data_smci),axis=0)	
	print('shape of data after merging -- it should be class balanced')
	print(data.shape)

	kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)
	control=0
	for i_train_kkt,i_test_kkt in kf.split(data,data[:,1]):
		if control==num_cv:
			i_train,i_test = i_train_kkt,i_test_kkt
			control+=1
		else:
			control+=1

	data_training = data[i_train,]
	data_testing = data[i_test,]
	Y_training = data_training[:,1]
	Y_training = Y_training.reshape((-1,1))

	Y_testing = data_testing[:,1]
	Y_testing = Y_testing.reshape((-1,1))
	## "ADNI_ID","conversion.m36","cor.brain_PAD","amyloid","age","sex","APOE4.bl","PTAU.bl","TAU.bl","ABETA.bl","Hippocampus.bl"
	##     0            1                2            3       4     5       6          7         8       9            10
	if args.model=='model1':

		X_training = np.delete(data_training,[0,1,3,4,5,6,7,8],axis=1)
		X_testing = np.delete(data_testing,[0,1,3,4,5,6,7,8],axis=1)
		names = ['Brain-PAD','AB-CSF','Hippocampus']

	elif args.model=='model2':

		X_training = np.delete(data_training,[0,1,4,5,6,7,8,9],axis=1)
		X_testing = np.delete(data_testing,[0,1,4,5,6,7,8,9],axis=1)
		names = ['Brain-PAD','AB-PET','Hippocampus']

	elif args.model=='model3':

		X_training = np.delete(data_training,[0,1,2,4,5,6,8,9],axis=1)
		X_testing = np.delete(data_testing,[0,1,2,4,5,6,8,9],axis=1)
		names = ['AB-PET','P-TAU','Hippocampus']


	scaler=StandardScaler().fit(X_training)
	X_training_original  =X_training
	X_training = scaler.transform(X_training)
	X_testing = scaler.transform(X_testing)

	X_training = X_training.astype(np.float64)
	X_testing = X_testing.astype(np.float64)
	X_training_original = X_training_original.astype(np.float64)
	Y_training = Y_training.astype(np.float64)
	Y_testing = Y_testing.astype(np.float64)

	svgpc = VariationalGaussianProcessClassifier(num_data=X_training.shape[0],dim_input=X_training.shape[1],dim_output=1,
		num_iterations=args.num_iterations,type_var="full",
		full_cov=False,
		position_interaction=[int(args.position_interaction[0]),int(args.position_interaction[1])],
		kernel_type=args.kernel_type,
		names=names,model=args.model,
		anchor=args.anchor)

	##### plot effect -- only to be used with kernel_type='interaction'
	##### plot effect -- can be 'two-way' or 'three-way'
	init_variables(num_data = X_training.shape[0], type_var='full', kernel_type=args.kernel_type, dim_input=X_training.shape[1])	
	svgpc.session_TF(X_training,Y_training,X_testing,Y_testing)
	




