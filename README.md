# rotation_NNN

A simple repository to test cifar10 keras model on rotation dataset

## Usage:
Add the train and test sets then 

 ./docker-build.sh
 
 ./docker-run.sh
 
 python cifar10.py
 
 This will generate the model on ./model/cifar10
 
 Then you can predict the results of the test with
 
 python predict.py
 
 this will generate the test.preds.csv
