# nonlinear_interaction_GPs
Accompanying code for "Non-linear interactions between biomarkers in conversion from Mild Cognitive Impairment to Alzheimer's Disease"

Non-linear interaction plots 

in command line:
bash send_decision_boundaries.sh

this will create folders with .jpeg images similar to the ones in the paper.

For additive models, a single row is created with the overall decision boundary.
For two-way interaction models, 2 rows are created with the relative contribution of the additive part in the first row, respectively the relative contribution of the two-way interaction on the second row
For three-way interaction models, 3 rows are created with the relative contribution of the additive part in the first row, the second row encodes the summed releative contributions of all the possible two-way pairwise interactions, respectively the third row encodes the three-way interaction.

Running bootstrapped models to detect interactions

in command line:
 bash send_GP.sh $number_of_bootstraps
 
 we recommend using $number_of_bootstraps=5 for faster results. in paper this is set to 100.
 
 after this command is finished in command line run:
 bash send_gather_results.sh
 
 this latter command will create several accuracy_$model_$type_interaction.csv files which fold basic summaries of the bootstrapped runs of the respective model.
 
 
