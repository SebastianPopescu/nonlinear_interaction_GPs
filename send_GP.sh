#!/bin/sh

##########################
######## Model 1 #########
##########################

###################################
####### Additive model ############
###################################

mkdir -p results
for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='additive' --type_features='model1' --num_iterations=5001 --position_interaction=00
done

###################################
####### Two-way interactions ######
###################################

for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='mixed-additive-interaction' --type_features='model1' --num_iterations=5001 --position_interaction=01
done


for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='mixed-additive-interaction' --type_features='model1' --num_iterations=5001 --position_interaction=02

for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='mixed-additive-interaction' --type_features='model1' --num_iterations=5001 --position_interaction=12
done

######################################
####### Three-way interaction ########
######################################

for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='interaction' --type_features='model1' --num_iterations=5001 --position_interaction=00
done






##########################
######## Model 2 #########
##########################

###################################
####### Additive model ############
###################################


for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='additive' --type_features='model2' --num_iterations=5001 --position_interaction=00
done

###################################
####### Two-way interactions ######
###################################

for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='mixed-additive-interaction' --type_features='model2' --num_iterations=5001 --position_interaction=01
done


for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='mixed-additive-interaction' --type_features='model2' --num_iterations=5001 --position_interaction=02

for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='mixed-additive-interaction' --type_features='model2' --num_iterations=5001 --position_interaction=12
done

######################################
####### Three-way interaction ########
######################################

for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='interaction' --type_features='model2' --num_iterations=5001 --position_interaction=00
done




##########################
######## Model 3 #########
##########################

###################################
####### Additive model ############
###################################


for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='additive' --type_features='model3' --num_iterations=5001 --position_interaction=00
done

###################################
####### Two-way interactions ######
###################################

for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='mixed-additive-interaction' --type_features='model3' --num_iterations=5001 --position_interaction=01
done


for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='mixed-additive-interaction' --type_features='model3' --num_iterations=5001 --position_interaction=02

for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='mixed-additive-interaction' --type_features='model3' --num_iterations=5001 --position_interaction=12
done

######################################
####### Three-way interaction ########
######################################

for((i=1;i<=$1;i++))
do
        echo $i
        python VGPC.py --num_sample=$i --kernel_type='interaction' --type_features='model3' --num_iterations=5001 --position_interaction=00
done