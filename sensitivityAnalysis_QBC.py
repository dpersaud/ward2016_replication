#%%
# IMPORT DEPENDENCIES--------------------------------------------------------------------------------------------------
'''
Investigating the sensitivity of the original models to pseudo-random seeds, with the theory being that the test set 
is outside the distribution of the training set and that is the reason the predictions are so sensitive to the seed.

author: Daniel Persaud
date:   2023.11.15
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost  as xgb

from sklearn.model_selection import cross_val_score

print ('imported dependencies!')

#%%
# IMPORT DATA----------------------------------------------------------------------------------------------------------

# import icsd data
dfIcsd = pd.read_csv('datasets/bandgap_features.csv', header=0)
# drop entries with missing data
dfIcsd.dropna(inplace=True)

# split the icsd data into features and target
dfIcsd_x = dfIcsd.drop(['Class'], axis=1)
srIcsd_y = dfIcsd['Class']

# import featurized training data
dfTrain = pd.read_csv('datasets/oqmd_all_features.csv', header=0)
# drop entries with missing data
dfTrain.dropna(inplace=True)

# split the training data into features and target
dfTrain_x = dfTrain.drop(['Class'], axis=1)
srTrain_y = dfTrain['Class']

# import featurized test data
dfTest = pd.read_csv('datasets/test_set_features.csv', header=0)

# split the test data into features and target
dfTest_x = dfTest.drop(['Class'], axis=1)
srTest_y = dfTest['Class']

#%%
# 10-FOLD RANDOM CV RANDOM FOREST REGRESSOR WITH DEFAULT PARAMETERS----------------------------------------------------

# create a pipeline for the random forest regressor
rfReg = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=0, n_jobs=-1))

# perfrom 10-fold random CV on the training data
rfReg_10cv = cross_val_score(rfReg, dfIcsd_x, srIcsd_y, cv=10, scoring='neg_root_mean_squared_error')

# print the mean and standard deviation of the 10-fold random CV
print('Random Forest Regressor with default parameters')
print('Mean RMSE: ', (rfReg_10cv.mean()*-1))
print('Standard Deviation of RMSE: ', rfReg_10cv.std())

#%%
# 10-FOLD RANDOM CV XGBOOST REGRESSOR WITH DEFAULT PARAMETERS----------------------------------------------------------

# create a pipeline for the xgboost regressor
xgbReg = make_pipeline(StandardScaler(), xgb.XGBRegressor(random_state=0, n_jobs=-1))

# perfrom 10-fold random CV on the training data
xgbReg_10cv = cross_val_score(xgbReg, dfIcsd_x, srIcsd_y, cv=10, scoring='neg_root_mean_squared_error')

# print the mean and standard deviation of the 10-fold random CV
print('XGBoost Regressor with default parameters')
print('Mean RMSE: ', (xgbReg_10cv.mean()*-1))
print('Standard Deviation of RMSE: ', xgbReg_10cv.std())

#%%
# PULL THE SAME STATISTICS FROM THE THE ORIGINAL MODEL-----------------------------------------------------------------

# pull the validation RMSE from the model with random seed 0
with open('./models/baseModels/logs/hierarchical-bandgap-model-seed0.out', 'r') as f:
    # read the file
    lstLines_temp = f.readlines()
    # get the line with the results
    strRMSE_temp = lstLines_temp[43]
    # split the line by spaces
    lstRMSE_temp = strRMSE_temp.split(' ')
    fltRMSE = float(lstRMSE_temp[1])

# print the mean RMSE across 10-fold random CV for the original model
print('Original Model')
print('Mean RMSE: ', fltRMSE)


'''
The lines below pull the validation RMSE from 10-fold random CV for each seed from the log files of the original model
and calculate the mean and standard deviation of the mean RMSE and std of the RMSE across all 10 random seeds.
'''
# # initialize a dictionary to store the results
# dicOriginalValidation = {}

# # for each file in ./models/baseModels/logs/
# for strFile_temp in os.listdir('./models/baseModels/logs/'):
#     # get the seed from the file name (for all files that start with hierarchical-bandgap-model-seed)
#     if strFile_temp.startswith('hierarchical-bandgap-model-seed'):
#         # get the number between 'seed' and '.out'
#         intSeed = int(strFile_temp[31:-4])
#         # in the file, pull the results from line 44
#         with open('./models/baseModels/logs/' + strFile_temp, 'r') as f:
#             # read the file
#             lstLines_temp = f.readlines()
#             # get the line with the results
#             strRMSE_temp = lstLines_temp[43]
#             # split the line by spaces
#             lstRMSE_temp = strRMSE_temp.split(' ')
#             fltRMSE_temp = float(lstRMSE_temp[1])
#             # add the results to the dictionary
#             dicOriginalValidation[intSeed] = fltRMSE_temp

# # print the mean and standard deviation of the 10-fold random CV
# print('Original Model')
# print('Mean RMSE: ', np.mean(list(dicOriginalValidation.values())))
# print('Standard Deviation of RMSE: ', np.std(list(dicOriginalValidation.values())))

#%%
# GENERATE A LIST OF PSEUDO-RANDOM SEEDS-------------------------------------------------------------------------------

# set the random seed generator seed so that the results are reproducible
np.random.seed(0)
# randomly generate 10 seeds
lstSeeds = np.random.randint(0, 1000, 10)

#%%
# PSUEDO-RANDOM SEED SENSITIVITY ANALYSIS ON RANDOM FOREST REGRESSOR---------------------------------------------------

# # for each seed, initialize a random forest regressor, train the model and make predictions on the test set
# lstRfRegPredictions = []
# for intSeed_temp in lstSeeds:
#     # initialize the random forest regressor
#     rfReg_temp = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=intSeed_temp, n_jobs=-1))
#     # fit the model to the training data
#     rfReg_temp.fit(dfTrain_x, srTrain_y)
#     # make predictions on the test set
#     lstRfRegPredictions.append(rfReg_temp.predict(dfTest_x))

# # initialize a dataframe to store the predictions
# dfTest_rfRegPredictions = pd.DataFrame(lstRfRegPredictions).T
# # rename the columns to be the seed used
# dfTest_rfRegPredictions.columns = lstSeeds
# # add the composition to the dataframe
# dfTest_rfRegPredictions['Composition'] = ['ScHg4Cl7','V2Hg3Cl7','Mn6CCl8','Hf4S11Cl2','VCu5Cl9']
# # add the target to the dataframe
# dfTest_rfRegPredictions['Class'] = dfTest['Class']
# # add the mean of the predictions to the dataframe
# dfTest_rfRegPredictions['Mean'] = dfTest_rfRegPredictions.iloc[:, :10].mean(axis=1)
# # add the standard deviation of the predictions to the dataframe
# dfTest_rfRegPredictions['Std'] = dfTest_rfRegPredictions.iloc[:, :10].std(axis=1)
# # add the min of the predictions to the dataframe
# dfTest_rfRegPredictions['Min'] = dfTest_rfRegPredictions.iloc[:, :10].min(axis=1)
# # add the max of the predictions to the dataframe
# dfTest_rfRegPredictions['Max'] = dfTest_rfRegPredictions.iloc[:, :10].max(axis=1)
# # add the difference between the max and min of the predictions to the dataframe
# dfTest_rfRegPredictions['Max-Min'] = dfTest_rfRegPredictions['Max'] - dfTest_rfRegPredictions['Min']
# # save the predictions to a csv file
# dfTest_rfRegPredictions.to_csv('predictions/sensitivityAnalysis/testData_predictions_rfReg.csv', index=False)

'''
This takes a long time to run so I just pull the data in from the csv file - if you want to run it, create the 
features using the make-features.in script and uncomment the code
'''
dfTest_rfRegPredictions = pd.read_csv('predictions/sensitivityAnalysis/testData_predictions_rfReg.csv', header=0)

#%%
# PSUEDO-RANDOM SEED SENSITIVITY ANALYSIS ON XGBOOST REGRESSOR----------------------------------------------------------

# # for each seed, initialize a xgboost regressor, train the model and make predictions on the test set
# lstXgbRegPredictions = []
# for intSeed_temp in lstSeeds:
#     # initialize the xgboost regressor
#     xgbReg_temp = make_pipeline(StandardScaler(), xgb.XGBRegressor(random_state=intSeed_temp, n_jobs=-1))
#     # fit the model to the training data
#     xgbReg_temp.fit(dfTrain_x, srTrain_y)
#     # make predictions on the test set
#     lstXgbRegPredictions.append(xgbReg_temp.predict(dfTest_x))

# # initialize a dataframe to store the predictions
# dfTest_xgbRegPredictions = pd.DataFrame(lstXgbRegPredictions).T
# # rename the columns to be the seed used
# dfTest_xgbRegPredictions.columns = lstSeeds
# # add the composition to the dataframe
# dfTest_xgbRegPredictions['Composition'] = ['ScHg4Cl7','V2Hg3Cl7','Mn6CCl8','Hf4S11Cl2','VCu5Cl9']
# # add the target to the dataframe
# dfTest_xgbRegPredictions['Class'] = dfTest['Class']
# # add the mean of the predictions to the dataframe
# dfTest_xgbRegPredictions['Mean'] = dfTest_xgbRegPredictions.iloc[:, :10].mean(axis=1)
# # add the standard deviation of the predictions to the dataframe
# dfTest_xgbRegPredictions['Std'] = dfTest_xgbRegPredictions.iloc[:, :10].std(axis=1)
# # add the min of the predictions to the dataframe
# dfTest_xgbRegPredictions['Min'] = dfTest_xgbRegPredictions.iloc[:, :10].min(axis=1)
# # add the max of the predictions to the dataframe
# dfTest_xgbRegPredictions['Max'] = dfTest_xgbRegPredictions.iloc[:, :10].max(axis=1)
# # add the difference between the max and min of the predictions to the dataframe
# dfTest_xgbRegPredictions['Max-Min'] = dfTest_xgbRegPredictions['Max'] - dfTest_xgbRegPredictions['Min']
# # save the predictions to a csv file
# dfTest_xgbRegPredictions.to_csv('predictions/sensitivityAnalysis/testData_predictions_xgbReg.csv', index=False)

'''
This takes a long time to run so I just pull the data in from the csv file - if you want to run it, create the 
features using the make-features.in script and uncomment the code
'''
dfTest_xgbRegPredictions = pd.read_csv('predictions/sensitivityAnalysis/testData_predictions_xgbReg.csv', header=0)


#%%
# PLOT PREDICTIONS-----------------------------------------------------------------------------------------------------

# take the predictions for Composition = ScHg4Cl7 and convert them to a list
lstScHg4Cl7_rfRegPredictions = dfTest_rfRegPredictions[:1].values.tolist()[0][:10]
lstScHg4Cl7_xgbRegPredictions = dfTest_xgbRegPredictions[:1].values.tolist()[0][:10]
# take the predictions for Composition = V2Hg3Cl7 and convert them to a list
lstV2Hg3Cl7_rfRegPredictions = dfTest_rfRegPredictions[1:2].values.tolist()[0][:10]
lstV2Hg3Cl7_xgbRegPredictions = dfTest_xgbRegPredictions[1:2].values.tolist()[0][:10]
# take the predictions for Composition = Mn6CCl8 and convert them to a list
lstMn6CCl8_rfRegPredictions = dfTest_rfRegPredictions[2:3].values.tolist()[0][:10]
lstMn6CCl8_xgbRegPredictions = dfTest_xgbRegPredictions[2:3].values.tolist()[0][:10]
# take the predictions for Composition = Hf4S11Cl2 and convert them to a list
lstHf4S11Cl2_rfRegPredictions = dfTest_rfRegPredictions[3:4].values.tolist()[0][:10]
lstHf4S11Cl2_xgbRegPredictions = dfTest_xgbRegPredictions[3:4].values.tolist()[0][:10]
# take the predictions for Composition = VCu5Cl9 and convert them to a list
lstVCu5Cl9_rfRegPredictions = dfTest_rfRegPredictions[4:5].values.tolist()[0][:10]
lstVCu5Cl9_xgbRegPredictions = dfTest_xgbRegPredictions[4:5].values.tolist()[0][:10]

# put the predictions into a list for the violin plot
lstViolinData_rfReg = [lstScHg4Cl7_rfRegPredictions,
                       lstV2Hg3Cl7_rfRegPredictions,
                       lstMn6CCl8_rfRegPredictions,
                       lstHf4S11Cl2_rfRegPredictions,
                       lstVCu5Cl9_rfRegPredictions]
lstViolinData_xgbReg = [lstScHg4Cl7_xgbRegPredictions,
                        lstV2Hg3Cl7_xgbRegPredictions,
                        lstMn6CCl8_xgbRegPredictions,
                        lstHf4S11Cl2_xgbRegPredictions,
                        lstVCu5Cl9_xgbRegPredictions]

# initialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor='w', edgecolor='k', dpi = 800, constrained_layout=True)

# plot the orgPredictions with red x's
ax.scatter(np.arange(1, 6, 1), dfTest['Class'], marker='x', color='tab:red', label ='original predictions')

# plot the rfReg predictions
pltVP_rf = ax.violinplot(lstViolinData_rfReg, showextrema=True, showmedians=True)
# add violinplot labels
pltVP_rf['bodies'][0].set_label('RF random seed sensitivity')
# adjust transparency of the violins and the median lines
for body in pltVP_rf['bodies']:
    body.set_alpha(0.2)
    body.set_edgecolor('black')
    body.set_facecolor('tab:purple')
    body.set_edgecolor('black')

# Make all the violin statistics marks red:
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = pltVP_rf[partname]
    vp.set_edgecolor('tab:purple')
    vp.set_linewidth(2)

pltVP_rf['cbars'].set_alpha(0.5)
pltVP_rf['cmaxes'].set_alpha(0.5)
pltVP_rf['cmins'].set_alpha(0.5)
pltVP_rf['cmedians'].set_alpha(0.5)

# plot the xgbReg predictions
pltVP_xgb = ax.violinplot(lstViolinData_xgbReg, showextrema=True, showmedians=True)
# add violinplot labels
pltVP_xgb['bodies'][0].set_label('XGB random seed sensitivity')
# adjust transparency of the violins and the median lines
for body in pltVP_xgb['bodies']:
    body.set_alpha(0.2)
    body.set_edgecolor('black')
    body.set_facecolor('tab:cyan')
    body.set_edgecolor('black')

# Make all the violin statistics marks blue:
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = pltVP_xgb[partname]
    vp.set_edgecolor('tab:cyan')
    vp.set_linewidth(3)

pltVP_xgb['cbars'].set_alpha(0.5)
pltVP_xgb['cmaxes'].set_alpha(0.5)
pltVP_xgb['cmins'].set_alpha(0.5)
pltVP_xgb['cmedians'].set_alpha(0.5)

# set the xticks
ax.set_xticks(np.arange(1, 6, 1))
# make a list of the compositions in the test dataset with the subscripts formatted for LaTeX
lstCompositionsFormated = ['$\mathdefault{ScHg_4Cl_7}$',
                            '$\mathdefault{V_2Hg_3Cl_7}$',
                            '$\mathdefault{Mn_6C Cl_8}$',
                            '$\mathdefault{Hf_4S_{11}Cl_2}$',
                            '$\mathdefault{VCu_5Cl_9}$']
# set the xticklabels
ax.set_xticklabels(lstCompositionsFormated, fontdict={'fontsize': 12, 'fontweight': 'medium'})
# set the x-axis label
ax.set_xlabel('Composition', fontdict={'fontsize': 14, 'fontweight': 'medium'})

# set the y-axis label
ax.set_ylabel('Bandgap (eV)', fontdict={'fontsize': 14, 'fontweight': 'medium'})
# set the y-axis limits from 0 to 2 in increments of 0.5
ax.set_yticks(np.arange(0, 2.5, 0.5))

# set the title
ax.set_title('Test Set Predictions - Modern Models', fontdict={'fontsize': 20, 'fontweight': 'medium'})
# reorder the legend entries
handles, labels = ax.get_legend_handles_labels()
handles = [handles[1], handles[2], handles[0]]
labels = [labels[1], labels[2], labels[0]]
# add the legend
ax.legend(handles, labels, )

# save the figure
plt.savefig('plots/sensitivityAnalysis/testData_modernModelPredictions.png', dpi=800, bbox_inches='tight')
# save the figure as a .tiff    
plt.savefig('plots/sensitivityAnalysis/testData_modernModelPredictions.tif', dpi=600, bbox_inches='tight')


# %%
