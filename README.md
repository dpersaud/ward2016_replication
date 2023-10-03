# ward2016_replication

Replication Steps

1. Relavent files were downloaded from the original paper's supplementary information
   Including:
	- Magpie/  			(the original software package)
	- datasets/			(folder containing the required datasets)
		bandgap.data: ICSD entries in OQMD use to build the heirarchical model according the original manuscript
		oqmd.data: OQMD v1.0 used to train the heirarchical model to make predictions
	- hierarchical-bandgap-model.in (text interface input file to create propsed bandgap model)

2. Modified hierarchical-bandgap-model.in to export base models and vary the random state
   Note: random state needed to be changed on line 19 and 25
	- made 10 different models with this input file for random seed sensitivity test
		--> run this using ./makeModels.sh
	- exports dataset template to ensure the same features get created for any other data used with these models

3. Created train-model.in to import a base model and and retrain with the perscribed training set (OQMD v1)
	- ran this script 10 times for each of the models with the varrying random seeds for the sensitivity test
		--> run this using ./trainModels.sh
