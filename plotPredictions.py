#%%
# IMPORT DEPENDENCIES--------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


print('numpy version: ', np.__version__)
print('pandas version: ', pd.__version__)
print('matplotlib version: ', matplotlib.__version__)

#%%
# LOAD DATA------------------------------------------------------------------------------------------------------------

# make a list of all random seeds used in the model building
lstRandomSeeds = [0, 1, 15, 42, 58, 66, 73, 152, 982, 8653]

# make a dataframe for the 'test' dataset - table 2 from the original paper
dfTest = pd.DataFrame({'strComposition' : ['ScHg4Cl7',
                                           'V2Hg3Cl7',
                                           'Mn6CCl8',
                                           'Hf4S11Cl2',
                                           'VCu5Cl9'],
                       'orgPredictions' : [1.26,
                                           1.16,
                                           1.28,
                                           1.11,
                                           1.19]})

# make a list of the compositions in the test dataset with the subscripts formatted for LaTeX
lstCompositionsFormated = ['$\mathdefault{ScHg_4Cl_7}$',
                            '$\mathdefault{V_2Hg_3Cl_7}$',
                            '$\mathdefault{Mn_6C Cl_8}$',
                            '$\mathdefault{Hf_4S_{11}Cl_2}$',
                            '$\mathdefault{VCu_5Cl_9}$']

# for each random seed, load the predictions for the test dataset and add them to the dfTest dataframe as a new column
for i in lstRandomSeeds:
    dfTemp = pd.read_csv('predictions/testData_predictions_seed' + str(i) + '.prop', sep=',', engine='python')
    # rename the column 'Entry' to 'strComposition'
    dfTemp.rename(columns={'Entry' : 'strComposition'}, inplace=True)
    # add the bandgap_predicted of dfTemp to dfTest based on the composition
    dfTest = dfTest.merge(dfTemp[['strComposition', 'bandgap_predicted']], on='strComposition', how='left')
    # rename the column 'bandgap_predicted' to 'repPredictions'
    dfTest.rename(columns={'bandgap_predicted' : 'repPredictions_seed' + str(i)}, inplace=True)

# find the mean of the repPredictions for each composition
dfTest['repPredictions_mean'] = dfTest.iloc[:, 2:].mean(axis=1)
# find the min of the repPredictions for each composition
dfTest['repPredictions_min'] = dfTest.iloc[:, 2:].min(axis=1)
# find the max of the repPredictions for each composition
dfTest['repPredictions_max'] = dfTest.iloc[:, 2:].max(axis=1)

# find the difference between the mean of the repPredictions and the orgPredictions (absolute value)
dfTest['repPredictions_mean_diff'] = abs(dfTest['repPredictions_mean'] - dfTest['orgPredictions'])


# export the dfTest dataframe to a csv file
dfTest.to_csv('predictions/testData_predictions.csv', index=False)

#%%
# PLOT PREDICTIONS-----------------------------------------------------------------------------------------------------

lstScHg4Cl7_repPredictions = dfTest[:1].values.tolist()[0][2:]
lstV2Hg3Cl7_repPredictions = dfTest[1:2].values.tolist()[0][2:]
lstMn6CCl8_repPredictions = dfTest[2:3].values.tolist()[0][2:]
lstHf4S11Cl2_repPredictions = dfTest[3:4].values.tolist()[0][2:]
lstVCu5Cl9_repPredictions = dfTest[4:5].values.tolist()[0][2:]

lstViolinData = [lstScHg4Cl7_repPredictions,
                 lstV2Hg3Cl7_repPredictions,
                 lstMn6CCl8_repPredictions,
                 lstHf4S11Cl2_repPredictions,
                 lstVCu5Cl9_repPredictions]

# initialize a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor='w', edgecolor='k', dpi = 300, tight_layout=True)

# plot the repPredictions
pltVP = ax.violinplot(lstViolinData, showextrema=True, showmedians=True,)
# add violin plot labels
pltVP['bodies'][0].set_label('random seed sensitivity')
# adjust transparency of the violins and the median lines
for body in pltVP['bodies']:
    body.set_alpha(0.2)
    body.set_edgecolor('black')
pltVP['cbars'].set_alpha(0.5)
pltVP['cmaxes'].set_alpha(0.5)
pltVP['cmins'].set_alpha(0.5)
pltVP['cmedians'].set_alpha(0.5)
# plot the orgPredictions with red x's
ax.scatter(np.arange(1, 6, 1), dfTest['orgPredictions'], marker='x', color='tab:red', label ='original predictions')
# plot the points for repPredictions_seed1 with green o's
ax.scatter([1,2,3,4,5], dfTest['repPredictions_seed0'], s = 20, marker="^", color='tab:green', label ='replicated predictions')

# set the xticks
ax.set_xticks(np.arange(1, 6, 1))
# set the xticklabels
ax.set_xticklabels(lstCompositionsFormated, fontdict={'fontsize': 12, 'fontweight': 'medium'})
# set the x-axis label
ax.set_xlabel('Composition', fontdict={'fontsize': 14, 'fontweight': 'medium'})

# set the y-axis label
ax.set_ylabel('Bandgap (eV)', fontdict={'fontsize': 14, 'fontweight': 'medium'})

# set the title
ax.set_title('Predictions for the test dataset', fontdict={'fontsize': 20, 'fontweight': 'medium'})

# reorder the legend entries
handles, labels = ax.get_legend_handles_labels()
handles = [handles[1], handles[2], handles[0]]
labels = [labels[1], labels[2], labels[0]]
# add the legend
ax.legend(handles, labels, )

# save the figure
fig.savefig('predictions/testData_predictions.png', dpi=300, bbox_inches='tight')
# %%
