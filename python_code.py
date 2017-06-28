#Importing the required libraries

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
import pickle

#Change the directory
os.chdir('I:\\Data science\\Data science competition\\SMS digital\\New_file\\POC1')

#===================================================
#TRAIN MEASUREMENT DATASET transposing
#===================================================
#Reading the training data
measurement000 = pd.read_csv('MEASUREMENT0.csv', sep = ',')
measurement001 = pd.read_csv('MEASUREMENT1.csv', sep = ',')
measurement000.head()
measurement001.head()

#Combining the two dataframes
measurement = pd.concat([measurement000 , measurement001])
measurement.describe()

#Dropping of the columns that are not needed
measurement.drop(['MeasurementID', 'SensorID', 'SensorStatus'], inplace=True, axis = 1)

#Transposing the Key and Value pair 
#Creating empty train data frame
key = measurement['Key'].unique()
column_names = ['SlabID', 'MeasurementSequence'] + key.tolist()
transposed_measure_train = pd.DataFrame(columns = column_names)

z=0
for slab in measurement['SlabID'].unique():
  print(slab)
  trim_dataset = measurement[measurement['SlabID']==slab]
  trim_dataset = trim_dataset.reset_index()
  mat_row = len(trim_dataset['MeasurementSequence'].unique())
  j = 0
  k = 173
  for i in range(mat_row):
    a = trim_dataset.loc[j:k, ['Value']].values.tolist()
    a = [item for sub_item in a for item in sub_item]
    Value = [slab] + [i+1] + a
    transposed_measure_train.loc[z] = Value
    j = k+1
    k = k+174
    z = z+1
    
#===================================================
#TEST MEASUREMENT DATASET transposing
#===================================================
#Reading the Test measurement data
measurement = pd.read_csv('Test_MEASUREMENT0.csv', sep=',')
measurement.head() 

#Dropping of the columns that are not needed
measurement.drop(['MeasurementID', 'SensorID', 'SensorStatus'], inplace=True, axis = 1)

#Transposing the Key and Value pair 
#Creating a empty test data frame
key = measurement['Key'].unique()
column_names = ['SlabID', 'MeasurementSequence'] + key.tolist()
transposed_measure_test = pd.DataFrame(columns = column_names)

z=0
for slab in measurement['SlabID'].unique():
  print(slab)
  trim_dataset = measurement[measurement['SlabID']==slab]
  trim_dataset = trim_dataset.reset_index()
  mat_row = len(trim_dataset['MeasurementSequence'].unique())
  j = 0
  k = 173
  for i in range(mat_row):
    a = trim_dataset.loc[j:k, ['Value']].values.tolist()
    a = [item for sub_item in a for item in sub_item]
    Value = [slab] + [i+1] + a
    transposed_measure_test.loc[z] = Value
    j = k+1
    k = k+174
    z = z+1

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
######################################################################
#Section 2 :
#In this part we are going to be merging all the train and test supporting tables 
#together to form the final train and test dataframes
######################################################################
#===================================================
#Train data merging
#===================================================
#Reading the train data required for merging

measurement_t = transposed_measure_train.copy()
defect = pd.read_csv('DEFECT.csv', sep=';')
heat = pd.read_csv('HEAT.csv', sep=';')
slab = pd.read_csv('SLAB.csv', sep=';')

#Merging the transposed train measurement with defect
measurement_t["DefectID"] = np.nan
measurement_t["HeatID"] = np.nan
measurement_t["DefectType"] = np.nan
measurement_t["DefectSlabSide"] = np.nan
measurement_t["DefectDetected"] = np.nan
measurement_t["DefectStart_X"] = np.nan
measurement_t["DefectEnd_X"] = np.nan
measurement_t["DefectStart_Y"] = np.nan
measurement_t["DefectEnd_Y"] = np.nan
measurement_t["target"] = 0

for rows in range(len(defect)):
    slab_id = defect.SlabID[rows]
    print (slab_id)
    temp_df = measurement_t[measurement_t['SlabID']==slab_id]
    for i in temp_df.index :
        measurement_t.loc[i,["DefectID"]] = defect.loc[rows,].DefectID
        measurement_t.loc[i,["HeatID"]] = defect.loc[rows,].HeatID
        measurement_t.loc[i,["DefectType"]] = defect.loc[rows,].DefectType
        measurement_t.loc[i,["DefectSlabSide"]] = defect.loc[rows,].DefectSlabSide
        measurement_t.loc[i,["DefectDetected"]] = defect.loc[rows,].DefectDetected
        measurement_t.loc[i,["DefectStart_X"]] = defect.loc[rows,].DefectStart_X
        measurement_t.loc[i,["DefectEnd_X"]] = defect.loc[rows,].DefectEnd_X
        measurement_t.loc[i,["DefectStart_Y"]] = defect.loc[rows,].DefectStart_Y
        measurement_t.loc[i,["DefectEnd_Y"]] = defect.loc[rows,].DefectEnd_Y
        if (measurement_t.loc[i,].ActCastLength >= defect.loc[rows,].DefectStart_Y) & (measurement_t.loc[i,].ActCastLength <= defect.loc[rows,].DefectEnd_Y):
            measurement_t.loc[i,["target"]] = 1  
        
measurement_t_defect = measurement_t.copy()

#Changing the colname
colnames = measurement_t_defect.columns.values.tolist()
colnames[180] = 'PCA_DefectDetected'
measurement_t_defect.columns = colnames

#Merging the train slab with the measurement_t_defect
measurement_t_defect_slab = pd.merge(measurement_t_defect, slab,
                                     how= "left",
                                     on = "SlabID")

#Removing the extra StrandID, HeatID and Changing the column names 
measurement_t_defect_slab.drop(['StrandID_y','HeatID_x'], axis = 1, inplace = True)
colnames = measurement_t_defect_slab.columns.values.tolist()
colnames[45]= 'StrandID'
colnames[185] = 'HeatID'
measurement_t_defect_slab.columns = colnames

#Merging the measurement_t_defect_slab to heat
measurement_t_defect_slab_heat = pd.merge(measurement_t_defect_slab, heat,
                                          how = 'left', 
                                          on = 'HeatID')
#Change the target column type to category
measurement_t_defect_slab_heat["target"] = measurement_t_defect_slab_heat["target"].astype('category')

#===================================================
#Test merging
#===================================================
heat = pd.read_csv('TEST_HEAT.csv', sep=';')
slab = pd.read_csv('TEST_SLAB.csv', sep=';')

#Merging the transposed test measurement with the test slab
transposed_measure_test_slab = pd.merge(transposed_measure_test, slab,
                                        how = 'left',
                                        on = 'SlabID')

#Removing the extra StrandID and changing the colname 
transposed_measure_test_slab.drop('StrandID_y', axis = 1, inplace= True)
colnames = transposed_measure_test_slab.columns.values.tolist()
colnames[45] = 'StrandID'
transposed_measure_test_slab.columns = colnames

#Merging the measurement_t_defect_slab to heat
transposed_measure_test_slab_heat = pd.merge(transposed_measure_test_slab, heat,
                                             how = 'left',
                                             on = 'HeatID')


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#===================================================
#Section3:
#Now that the train and the test dataframes are ready we can proceed 
#to building the model with a little bit of prepocessing
#===================================================
train = measurement_t_defect_slab_heat.copy()
test = transposed_measure_test_slab_heat.copy()

train['DefectType'].fillna(value = 'not_sure',inplace = True)
train['DefectSlabSide'].fillna(value = 'not_sure', inplace = True)
train['PCA_DefectDetected'].fillna(value = False, inplace = True)

#Dropping the columns that are constant
train.drop(['ActMouldThickn', 'StatusCastingActv', 'StatusEmergencyStopActv',
            'PowderType', 'StatusTailoutActv', 'StatusTundishChange',
            'StatusTundishOpen', 'Thickness', 'StartTimestamp',
            'EndTimestamp', 'DefectID'], axis = 1, inplace = True)

#Dropping the columns to predict the defect
#The below columns are not included for each predictions. 
#They are dropped while building the model

drop_defect_predict = ['SlabID', 'DefectStart_X', 'DefectStart_Y', 'DefectEnd_X', 'DefectEnd_Y', 'DefectType',
                        'DefectSlabSide', 'PCA_DefectDetected', 'HasDefect', 'CreationDate']
drop_defect_side_predict = ['SlabID', 'DefectStart_X', 'DefectStart_Y', 'DefectEnd_X', 'DefectEnd_Y', 'DefectType',
                             'PCA_DefectDetected', 'HasDefect', 'CreationDate', 'target']
drop_X_position_predict = ['SlabID', 'DefectEnd_X', 'DefectStart_Y', 'DefectEnd_Y', 'DefectType', 
                            'DefectSlabSide', 'PCA_DefectDetected', 'HasDefect', 'CreationDate', 'target']

#Training the model with only those in test slabs 
test_slabs = test['SlabID'].unique()
train = train[train['SlabID'].isin(test_slabs)]

#======================================================================= 
#===================TO PREDICT IF DEFECT PRESENT OR NOT=================
#=======================================================================
#Random Forest
x_defect_pred = train.columns[~train.columns.isin(drop_defect_predict + ['target'])]
y_defect_pred = train['target']

clf_defect_pred = RandomForestClassifier(random_state = 100)
clf_defect_pred.fit(train[x_defect_pred], y_defect_pred)
test['predicted_defect'] = clf_defect_pred.predict(test[x_defect_pred])

#======================================================================= 
#======================TO PREDICT THE DEFECT SLAB SIDE==================
#======================================================================= 
#on prod test
train['DefectSlabSide'] = train['DefectSlabSide'].astype('category')
train_defect_side = train[train['DefectSlabSide']!='not_sure']

x_slab_side = train_defect_side.columns[~train_defect_side.columns.isin(drop_defect_side_predict + ['DefectSlabSide'])]
y_slab_side = train_defect_side['DefectSlabSide']

clf_defect_side = RandomForestClassifier(random_state = 100)
clf_defect_side.fit(train_defect_side[x_slab_side], y_slab_side)

test['predicted_defect_side'] = clf_defect_side.predict(test[x_slab_side])

#=======================================================================  
#=================TO PREDICT THE X POSITION OF THE DEFECT===============
#=======================================================================
#on prod dataset
train_X_position = train[train['DefectStart_X'].notnull()]

x_defect_x_pos = train_X_position.columns[~train_X_position.columns.isin(drop_X_position_predict + ['DefectStart_X'])]
y_defect_x_pos = train_X_position['DefectStart_X']

clf_defect_x_pos = RandomForestRegressor(random_state = 100)
clf_defect_x_pos.fit(train_X_position[x_defect_x_pos], y_defect_x_pos)

test['predicted_defect_X'] = clf_defect_x_pos.predict(test[x_defect_x_pos])

#=======================================================================
#Forming the final predicted defect 
defect_slab = []
defect_slab_side = []
defect_start_Y = []
defect_End_Y = []
defect_start_X = []
defect_End_X = []
HeatID = []

for slab in test['SlabID'].unique():
    print (slab)
    test_slab = test[test['SlabID'] == slab]
    test_slab = test_slab.reset_index()
    active_defect = 0
    X_position = 0
    
    for i in range(len(test_slab)):
        if ((test_slab.predicted_defect[i] == 1 ) & (active_defect == 0)):
            defect_slab.append(slab)
            defect_slab_side.append(test.predicted_defect_side[i])
            defect_start_Y.append(test_slab.ActCastLength[i])
            defect_start_X.append(test_slab.predicted_defect_X[i])
            active_defect = 1
            HeatID.append(test_slab.HeatID[i])
            
        elif ((test_slab.predicted_defect[i]== 0 ) & (active_defect == 1)):
            defect_End_Y.append(test_slab.ActCastLength[i-1])
            defect_End_X.append(test_slab.predicted_defect_X[i-1])
            active_defect = 0
            
        elif ((test_slab.predicted_defect[i]==0) & (active_defect == 1) & (i==len(test_slab)-1)):
            defect_End_Y.append(test_slab.ActCastLength[i])
            defect_End_X.append(test_slab.predicted_defect_X[i])
            active_defect = 0
        elif ((test_slab.predicted_defect[i]==1) & (active_defect == 1) & (i==len(test_slab)-1)):
            defect_End_Y.append(test_slab.ActCastLength[i])
            defect_End_X.append(test_slab.predicted_defect_X[i])
            active_defect = 0
        else:
            next
  
X_defect = np.mean(np.array([defect_start_X, defect_End_X]), axis = 0).tolist()
         
final_sub_df = {'DefectID' : list(range(1,len(defect_slab)+1)),
              'SlabID' : defect_slab,
              'HeatID' : HeatID,
              'DefectType' : ['LFC']*len(defect_slab),
              'DefectSlabSide' : defect_slab_side, 
              'DefectDetected' : ['TRUE']*len(defect_slab),
              'DefectStart_X' : X_defect,
              'DefectStart_Y' : defect_start_Y,
              'DefectEnd_X' : X_defect,
              'DefectEnd_Y' : defect_End_Y}

Defect_slabs = pd.DataFrame(final_sub_df, columns=['DefectID','SlabID', 'HeatID','DefectType', 
                                               'DefectSlabSide', 'DefectDetected','DefectStart_X', 
                                               'DefectStart_Y', 'DefectEnd_X','DefectEnd_Y'])
    
    
