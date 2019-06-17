import numpy as np
from collections import defaultdict
import argparse
from  sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

if __name__=='__main__':


	#############################################################################################################
	#### creates a csv file with summary statistics for each of the bootstrapped datasets for a given model #####
	#############################################################################################################

	parser = argparse.ArgumentParser()
	parser.add_argument('--model',type=str, help ='can be either model1; model2; model3')
	parser.add_argument('--kernel_type',type=str,help='can be additive; interaction; mixed-additive-interaction')
	parser.add_argument('--num_smci_sample',type=int,default=500,help='number of bootstrapped datasets * number of folds used for Cross-Validation')
	parser.add_argument('--total_cv',type=int,default=10, help = 'number of folds used for Cross-Validation')
	parser.add_argument('--num_input',type=int,default=3, help = 'number of dimensions of input data; in paper everything is size 3')
	parser.add_argument('--position_interaction',type=list, help = 'a list that contains the positions of the two biomarkers that we want to model an interaction for')
	parser.add_argument('--num_iterations',type=int,default=5001)
	args = parser.parse_args()

	if args.model=='model1':

		names = ['Brain-PAD','AB-CSF','Hippocampus']

	elif args.model=='model2':

		names = ['Brain-PAD','AB-PET','Hippocampus']

	elif args.model=='model3':

		names = ['AB-PET','P-TAU','Hippocampus']


	accuracy = []
	sensitivity = []
	specificity = []
	auc_values = []
	list_overall_test_log_likelihoods = []

	### iterate over each num_smci_sample ###
	for i in range(args.num_smci_sample):
		print(i)
		### iterate over each cross validation run ###
		true_observations = []
		predictions = []
		prob_pred = []
		overall_test_log_Likelihood = 0.0
		
		if args.kernel_type=='mixed-additive-interaction':

			data = np.genfromtxt('./results_'+str(args.num_iterations)+'/'+str(args.model)+'_'+str(args.kernel_type)+'/num_sample_'+str(i)+'/cv_'+str(0)+'/'+str(names[int(args.position_interaction[0])])+
			'_'+str(names[int(args.position_interaction[1])])+'/results.txt',delimiter=',')

		else:

			data = np.genfromtxt('./results_'+str(args.num_iterations)+'/'+str(args.model)+'_'+str(args.kernel_type)+'/num_sample_'+str(i)+'/cv_'+str(0)+'/results.txt',delimiter=',')
		
		for j in range(1,args.total_cv):

			### get the predictions results from this instance ###
			######################################################	
			
			if args.kernel_type=='mixed-additive-interaction':

				data_temp = np.genfromtxt('./results_'+str(args.num_iterations)+'/'+str(args.model)+'_'+str(args.kernel_type)+'/num_sample_'+str(i)+'/cv_'+str(j)+'/'+str(names[int(args.position_interaction[0])])+
				'_'+str(names[int(args.position_interaction[1])])+'/results.txt',delimiter=',')

			else:

				data_temp = np.genfromtxt('./results_'+str(args.num_iterations)+'/'+str(args.model)+'_'+str(args.kernel_type)+'/num_sample_'+str(i)+'/cv_'+str(j)+'/results.txt',delimiter=',')

			if j != 0:

				data = np.concatenate((data,data_temp),axis=0)
				
			for _ in range(data_temp.shape[0]):
				true_observations.append(int(data_temp[_,1]))
				predictions.append(int(data_temp[_,2]))
				prob_pred.append(data_temp[_,3])

			### get the test log-likelihood from this instance ###
			######################################################

		
			if args.kernel_type=='mixed-additive-interaction':

				with open('./results_'+str(args.num_iterations)+'/'+str(args.model)+'_'+str(args.kernel_type)+'/num_sample_'+str(i)+'/cv_'+str(j)+'/'+str(names[int(args.position_interaction[0])])+
				'_'+str(names[int(args.position_interaction[1])])+'/test_log_likelihood.txt') as f:

					lines = f.readlines()
					line = lines[0]
					print(line)
					overall_test_log_Likelihood += float(line)


			else:
				
				with open('./results_'+str(args.num_iterations)+'/'+str(args.model)+'_'+str(args.kernel_type)+'/num_sample_'+str(i)+'/cv_'+str(j)+'/test_log_likelihood.txt') as f:

					lines = f.readlines()
					line = lines[0]
					print(line)
					overall_test_log_Likelihood += float(line)

		### calculate the accuracy statistics on this num_smci_sample ###
		#################################################################
		auc_values.append(roc_auc_score(y_true = true_observations, y_score = prob_pred))
		sensitivity.append(recall_score(y_true = true_observations, y_pred = predictions, pos_label=0))
		tn, fp, fn, tp = confusion_matrix(true_observations, predictions).ravel()
		specificity.append(tp/(tp+fn))
		accuracy.append(accuracy_score(true_observations,predictions))

		if args.kernel_type=='mixed-additive-interaction':

			np.savetxt('./predictions_'+str(args.model)+'_'+str(args.kernel_type)+'_'+str(names[int(args.position_interaction[0])])+
				'_'+str(names[int(args.position_interaction[1])])+'_num_smci_sample_'+str(i)+'.csv',data,delimiter=',',fmt='%1.3f')			

		else:

			np.savetxt('./predictions_'+str(args.model)+'_'+str(args.kernel_type)+'_num_smci_sample_'+str(i)+'.csv',data,delimiter=',',fmt='%1.3f')


		list_overall_test_log_likelihoods.append(overall_test_log_Likelihood) 

	### dump the accuracy statistics per num_smci_sample in an overall file ###
	text_de_printat ='Accuracy,Sensitivity,Specificity,AUC,Log-likelihood\n'
	for _ in range(args.num_smci_sample):

		text_de_printat+=str(accuracy[_])+','+str(sensitivity[_])+','+str(specificity[_])+','+str(auc_values[_])+','+str(list_overall_test_log_likelihoods[_])+'\n'

	if args.kernel_type=='mixed-additive-interaction':

		with open('./accuracy_statistics_'+str(args.model)+'_'+str(args.kernel_type)+'_'+str(names[int(args.position_interaction[0])])+
				'_'+str(names[int(args.position_interaction[1])])+'.csv','w') as f:

			f.write(text_de_printat)

	else:

		with open('./accuracy_statistics_'+str(args.model)+'_'+str(args.kernel_type)+'.csv','w') as f:

			f.write(text_de_printat)

	print('overall results')
	print(np.mean(accuracy))
	print(np.mean(sensitivity))
	print(np.mean(specificity))
	print(np.mean(auc_values))
	print(np.mean(list_overall_test_log_likelihoods))


