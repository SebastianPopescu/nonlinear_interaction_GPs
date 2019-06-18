# -*- coding: utf-8 -*-
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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import os
DTYPE=tf.float32
from init_variables import *
from losses import *
from kernels import *
from conditional_GP import *
from predict_functions import *


class VariationalGaussianProcessClassifier(object):

	#### implementation inspired by "Scalable Variational Gaussian Process Classification by Hensman et al.,2015" ####################
	#### this code extends the above paper to the non-sparse setting, thereby using all the training points and no inducing points ###

	def __init__(self, num_data, dim_input, dim_output,
		num_iterations, type_var, full_cov,
		kernel_type, position_interaction,
		num_smci_sample, num_cv,
		model,
		run,
		names):

		self.run = run ### integer which defines the bootstrapped dataset and the CV fold ####
		self.model = model ###### can be either model1;model2 or model3
		self.num_smci_sample = num_smci_sample ### the number of the bootstraped dataset
		self.num_cv = num_cv  ###### number of the cross-validation set used for the current bootstrapped dataset #####
		self.kernel_type = kernel_type ##### could be 'interaction' -- where we apply all the possible bivariate pairwise interaction kernel and on top a three-way interaction kernel
		##### 'mixed-additive-interaction' -- where we add on top of the additive kernels an interaction kernel specificed by self.position_interaction
		##### 'additive' -- where we use just additive kernels without any interaction between features ###
		self.position_interaction = position_interaction ### if self.kernel_type='mixed-additive-interaction' then this encodes the bivariate interaction to use -- list of the form [position_biomarker1, position_biomarker2]
		self.full_cov = full_cov ### specifies the type of 
		self.num_data = num_data ### number of training points
		self.type_var = type_var ####  can be "full" -- full parametrization of the variational variance ; if "diagonal" -- uses just a diagonal parametrization
		self.dim_input = dim_input #### number of dimensions of input data
		self.dim_output = dim_output #### number of dimensions of output data -- in our case this is always 1
		self.num_iterations = num_iterations #### number of training iterations
		self.names = names ##### list of names of biomarkers used in current model

		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess= tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))				
		self.X = tf.placeholder(tf.float32,shape=(None,dim_input),name='Input_ph_train')
		self.X_test = tf.placeholder(tf.float32,shape=(None,dim_input),name='Input_ph_test')
		self.Y = tf.placeholder(tf.float32,shape=(None,dim_output),name='Output_ph_train')
		self.Y_test = tf.placeholder(tf.float32,shape=(None,dim_output),name='Output_ph_test')


	def session_TF(X_training, Y_training, X_testing, Y_testing,
		indices_testing):

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

		predictions_mean, predictions_var = build_predict(self.X_test, self.X,
			self.kernel_type, self.dim_input, self.position_interaction, self.type_var)
		
		#### get log-likelihood at testing time
		test_log_likelihood = tf.reduce_sum(bernoulli(self.Y_test,predictions_mean))
	
		tf.summary.scalar(tensor = tf.squeeze(kl_cost), name = 'kl_cost')
		tf.summary.scalar(tensor = tf.squeeze(re_cost), name = 're_cost')
		tf.summary.scalar(tensor = tf.squeeze(test_log_likelihood), name = 'test_ll')
		merged = tf.summary.merge_all()

		##### save the training process for later use with tensorboard 
		if self.kernel_type=='mixed-additive-interaction':

			train_writer = tf.summary.FileWriter('./tensorboard_'+str(self.num_iterations)+'_'+str(self.model)+'/run_'+str(self.run)+'/'+str(self.kernel_type)+'/'+str(names[self.position_interaction[0]])+'_'+str(names[self.position_interaction[1]]),self.sess.graph)

		else:

			train_writer = tf.summary.FileWriter('./tensorboard_'+str(self.num_iterations)+'_'+str(self.model)+'/run_'+str(self.run)+'/'+str(self.kernel_type),self.sess.graph)

		#### initialize tf variables #### 
		self.sess.run(tf.global_variables_initializer())

		#### training process ####
		for i in range(self.num_iterations):

			_, cost_np, re_cost_np, kl_cost_np, summary  = self.sess.run([train_op,cost,re_cost,kl_cost,merged],
				feed_dict={self.X:X_training,self.Y:Y_training,
				self.X_test:X_testing,self.Y_test:Y_testing})
			train_writer.add_summary(summary,i)
			
			if i % 1000 ==0  and i !=0:

				preds_now = self.sess.run(predictions_mean,feed_dict={self.X_test:X_testing,self.X:X_training})
				predictii_bi = binarize(preds_now,0.5)
				print('***** validation accuracy is '+str(accuracy_score(Y_validation,predictii_bi))+' ******')

			print('at iteration '+str(i) + ' we have nll : '+str(cost_np) + 're cost :'+str(re_cost_np)+' kl cost :'+str(kl_cost_np))

		#########################################
		#### get predictions at testing time ####
		#########################################

		preds_now, vars_now = self.sess.run([predictions_mean,predictions_var], feed_dict={self.X_test:X_testing, self.X:X_training})
		predictii_bi = binarize(preds_now,0.5)
		print('***** final accuracy is '+str(accuracy_score(Y_testing,predictii_bi))+' ******')
		Y1=[]
		Y2=[]
		num_yes=0
		num_no=0

		for i in range(Y_testing.shape[0]):
			if Y_testing[i]==1:
				Y1.append('yes')
				num_yes+=1
			else:
				Y1.append('no')
				num_no+=1
			if predictii_bi[i]==1:
				Y2.append('yes')
			else:
				Y2.append('no')
		print(confusion_matrix(Y1,Y2,labels=["yes","no"]))

		###################################
		### save the predictions ##########
		###################################

		text_to_write = ''
		for i in range(Y_testing.shape[0]):

			text_to_write+=str(indices_testing[i,0])+','+str(Y_testing[i,0])+','+str(predictii_bi[i,0])+','+str(preds_now[i,0])+','+str(vars_now[i,0])+'\n'

		if self.kernel_type=='mixed-additive-interaction':

			with open('./results_'+str(self.num_iterations)+'/'+str(self.model)+'_'+str(self.kernel_type)+'/num_sample_'+str(self.num_smci_sample)+'/cv_'+str(self.num_cv)+'/'+str(names[self.position_interaction[0]])+'_'+str(names[self.position_interaction[1]])+'/results.txt','w') as f:
				f.write(text_to_write)

		else:

			with open('./results_'+str(self.num_iterations)+'/'+str(self.model)+'_'+str(self.kernel_type)+'/num_sample_'+str(self.num_smci_sample)+'/cv_'+str(self.num_cv)+'/results.txt','w') as f:
				f.write(text_to_write)

		### get the test log likelihood ###
		###################################

		test_log_likelihood_now = self.sess.run(test_log_likelihood,feed_dict={self.X_test:X_testing,self.X:X_training,self.Y_test:Y_testing})
		text_to_write = str(test_log_likelihood_now)

		if self.kernel_type=='mixed-additive-interaction':

			with open('./results_'+str(self.num_iterations)+'/'+str(self.model)+'_'+str(self.kernel_type)+'/num_sample_'+str(self.num_smci_sample)+'/cv_'+str(self.num_cv)+'/'+str(names[self.position_interaction[0]])+'_'+str(names[self.position_interaction[1]])+'/test_log_likelihood.txt','w') as f:
				f.write(text_to_write)

		else:

			with open('./results_'+str(self.num_iterations)+'/'+str(self.model)+'_'+str(self.kernel_type)+'/num_sample_'+str(self.num_smci_sample)+'/cv_'+str(self.num_cv)+'/test_log_likelihood.txt','w') as f:
				f.write(text_to_write)
	

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--kernel_type',type=str,default='additive',help='can be additive;interaction;mixed-additive-interaction')
	parser.add_argument('--num_iterations',type=int,default=5000,help='the number of iterations')
	parser.add_argument('--position_interaction',type=list,default=None,help='list with the position of the bivariate interaction, only to be used when kernel_type=mixed-additive-interaction')
	parser.add_argument('--num_sample',type=int,default=0,help='defines which bootstrapped dataset and which fold of the CV framework to use for current model, it goes from 0 to 999 for 100 bootstrapped datasets with a 10-fold CV framework')
	parser.add_argument('--type_features',type=str,default='model1', help='should be either model1; model2; model3; the configuration of thse models are specificed in the paper')
	args = parser.parse_args()

	#######################################################
	#### create the folders where to store the results ####
	#######################################################

	cmd='mkdir -p ./results_'+str(args.num_iterations)
	os.system(cmd)

	num_smci_sample = args.num_sample // 10
	num_cv = args.num_sample % 10

	cmd='mkdir -p ./results_'+str(args.num_iterations)+'/'+str(args.type_features)+'_'+str(args.kernel_type)
	os.system(cmd)
	cmd='mkdir -p ./results_'+str(args.num_iterations)+'/'+str(args.type_features)+'_'+str(args.kernel_type)+'/num_sample_'+str(num_smci_sample)
	os.system(cmd)
	cmd='mkdir -p ./results_'+str(args.num_iterations)+'/'+str(args.type_features)+'_'+str(args.kernel_type)+'/num_sample_'+str(num_smci_sample)+'/cv_'+str(num_cv)
	os.system(cmd)

	##########################
	##### load the data ######
	##########################

	data = np.genfromtxt('./csv_classification_overall.csv',delimiter=',',skip_header=1)

	### delete the row number ###
	#############################
	data = np.delete(data,[0],axis=1)
	print(data[:,1])
	### get position of stable MCI subjects ###
	indices_smci = np.where(data[:,1]==1)[0]
	### get position of progressive MCI subjects ###
	indices_pmci = np.where(data[:,1]==0)[0]

	### dataset of just sMCI subjects ###
	data_smci = data[indices_smci,]
	
	### dataset of just pMCI subjects ###
	data_pmci = data[indices_pmci,]
	
	### subsample the sMCI subjects ###
	### to arrive at class balance between sMCI and pMCI ####
	np.random.seed(num_smci_sample)
	np.random.shuffle(indices_smci)

	np.random.shuffle(data_smci)
	data_smci = data_smci[:data_pmci.shape[0],:]
	data = np.concatenate((data_pmci,data_smci),axis=0)	

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

	## names of columns in original csv file ##### 
	## "ADNI_ID","conversion.m36","cor.brain_PAD","amyloid","age","sex","APOE4.bl","PTAU.bl","TAU.bl","ABETA.bl","Hippocampus.bl"
	##     0            1                2            3       4     5       6          7         8       9            10
	if args.type_features=='model1':

		#### biomarker configuration corresponding to Model 1 as described in the paper ####

		X_training = np.delete(data_training,[0,1,3,4,5,6,7,8],axis=1)
		X_testing = np.delete(data_testing,[0,1,3,4,5,6,7,8],axis=1)
		names = ['Brain-PAD','AB-CSF','Hippocampus']

	elif args.type_features=='model2':

		#### biomarker configuration corresponding to Model 2 as described in the paper ####

		X_training = np.delete(data_training,[0,1,4,5,6,7,8,9],axis=1)
		X_testing = np.delete(data_testing,[0,1,4,5,6,7,8,9],axis=1)
		names = ['Brain-PAD','AB-PET','Hippocampus']

	elif args.type_features=='model3':

		#### biomarker configuration corresponding to Model 3 as described in the paper ####

		X_training = np.delete(data_training,[0,1,2,4,5,6,8,9],axis=1)
		X_testing = np.delete(data_testing,[0,1,2,4,5,6,8,9],axis=1)
		names = ['AB-PET','P-TAU','Hippocampus']

	if args.kernel_type=='mixed-additive-interaction':

		cmd='mkdir -p ./results_'+str(args.num_iterations)+'/'+str(args.type_features)+'_'+str(args.kernel_type)+'/num_sample_'+str(num_smci_sample)+'/cv_'+str(num_cv)+'/'+str(names[int(args.position_interaction[0])])+'_'+str(names[int(args.position_interaction[1])])
		os.system(cmd)

	scaler=StandardScaler().fit(X_training)
	X_training_original  =X_training
	X_training = scaler.transform(X_training)
	X_testing = scaler.transform(X_testing)

	svgpc = VariationalGaussianProcessClassifier(num_data=X_training.shape[0], dim_input=X_training.shape[1], dim_output=1,
		num_iterations=args.num_iterations, type_var="full",
		full_cov=False, kernel_type=args.kernel_type,
		position_interaction=[int(args.position_interaction[0]),int(args.position_interaction[1])],
		num_smci_sample = num_smci_sample, num_cv = num_cv,
		model= args.type_features,
		run=args.num_sample, names=names)

	init_variables(num_data = X_training.shape[0], type_var='full', kernel_type=args.kernel_type, dim_input=X_training.shape[1])
	svgpc.session_TF(X_training, Y_training, X_testing, Y_testing)
	





	













