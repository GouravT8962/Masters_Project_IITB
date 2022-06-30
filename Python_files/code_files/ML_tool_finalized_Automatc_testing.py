import time
first_time = time.time()

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
# from sklearn.model_selection import GridSearchCV#,StratifiedKFold
from sklearn.svm import SVR
# from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# import pickle
import os

# Input Directory/Folder Name here
directory = 'Finalized_Model_Automatic_Testing'           # Enter he name of directory to be created
try: 
    os.mkdir(directory) 
except: 
    print("Directory with name " + directory + " already exists")         

#Reading CSV (Comma Separted Value) File
CSV_filename = 'combined_dataset_5_6_7_8'               # Write file name of .csv file without extension
CSV_filename1 = CSV_filename + '.csv'
CSV_filename2 = directory + '/' + CSV_filename + '_processed.csv'
CSV_filename3 = directory + '/' + CSV_filename + '_scaled.csv'

train_df = pd.read_csv(CSV_filename1,header = 0);p=0;           # Run this file if started from scratch
# train_df = pd.read_csv(CSV_filename2,header = 0);p=1;         # Run this file if only unwanted columns are removed
# train_df = pd.read_csv(CSV_filename3,header = 0);p=2;         # Run this file if all unwanted columns are removed and data scaling is done

input_cols_values = ['L_Mn_drive','L_Mn_tail1','L_Mp_stg1','mul_cap','nf_ibias',	'nf_Mn_drive',	'nf_Mn_tail1',	'nf_Mn_tail2',	'nf_Mp_stg2']
output_cols_values = ['CMRR_DC', 'Noise_10M', 'Noise_1M', 'Noise_DC', 'Phase Margin', 'Phase Margin Frequency', '3dB Bandwidth', 'UGB', 'DC_Power', 'DC_Gain', 'Slew Rate Value', 'PSRR_DC']

input_cols = []
output_cols = []

for col_value in input_cols_values:
    if col_value in train_df.columns:
        input_cols.append(col_value)

for col_value in output_cols_values:
    if col_value in train_df.columns:
        output_cols.append(col_value)
        
total_cols = input_cols + output_cols
        
#Removing the columns that are unnecessary from dataframe
label = ['Point','Corner','Region_Drive_1','Region_Drive_2','Region_Mn_tail1','Region_Mn_tail2','Region_Mp_stg1_1','Region_Mp_stg1_2','Region_Mp_stg2','Region_Mp_Rz','Region_Mn2','Region_Mp3','Region_Mp4','Region_Mn_ibias','OPAMP_RAK:OPAMP_DC:1','nf_Mp_stg1','nf_Rz','nf_Mp3','nf_Mp4']
for label_value in label:
    if label_value in train_df.columns:
        train_df = train_df.drop(labels=label_value, axis=1)
        
if 'Pass/Fail' in train_df.columns:
    pass_fail_values = train_df.loc[:,'Pass/Fail']

if p!=1 and p!=2:
    print("Data Processing started")
    for i in range(0,pass_fail_values.shape[0]):
        if (pass_fail_values[i] == 'fail') or (pass_fail_values[i] == 'near'):
            #print(i)
            train_df = train_df.drop(labels = i, axis=0)
    
    train_df = train_df.drop(labels='Pass/Fail', axis=1)
    print("Data Processing done")
    processed = input("Do you want to create a .csv file for saving processed data?, Y/N : ")
    if processed == "Y":
        train_df[total_cols].to_csv(CSV_filename2)
else: 
    print("Data Processing already done")
    
for col_value in train_df.columns:
    train_df[col_value] = train_df[col_value].astype(float)

#processing data points which are either very small or very large
print("Data Scaling started")
if p!=2:
    for col in output_cols:
          if (col == 'Phase Margin Frequency'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]/1e7
          elif (col == 'Slew Rate Value'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]/1e7
          elif (col == 'UGB'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]/1e7
          elif (col == '3dB Bandwidth'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]/1e6
          elif (col == 'DC_Power'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]*1e4
          elif (col == 'Noise_DC'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]*1e5
          elif (col == 'Noise_10M'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]*1e9
          elif (col =='Noise_1M'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]*1e9
          print(col)
   
    for col in input_cols:
          if (col == 'L_Mn_drive'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]*1e7
          elif (col == 'L_Mn_tail1'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]*1e7
          elif (col == 'L_Mp_stg1'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]*1e7
          elif (col == 'nf_Mn_tail2'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]*0.1
          elif (col =='nf_Mp_stg2'):
                for i in train_df.index:
                  train_df[col][i] = train_df[col][i]*0.1
          print(col)
          
    scaled = input("Do you want to create a .csv file for saving scaled data?, Y/N : ")
    if scaled == "Y":
        train_df[total_cols].to_csv(CSV_filename3)

mul_PMF = 1e7
mul_SR = 1e7
mul_UGB = 1e7
mul_3db = 1e6
mul_power = 1e-4
mul_gain = 1
mul_PM = 1
mul_NDC = 1e-5
mul_N1M = 1e-9
mul_N10M = 1e-9

mul_Ldrive = 1e-7
mul_Ltail1 = 1e-7
mul_Lstg1 = 1e-7
mul_mulcap = 1
mul_nf_ibias = 1
mul_nf_Mn_drive = 1
mul_nf_Mn_tail1 = 1
mul_nf_Mn_tail2 = 10
mul_nf_Mp_stg2 = 10

print("Data Scaling done")

# summary of dataset
print(train_df[total_cols]) 
print(train_df[total_cols].describe())

# creating correlation matrix for all inouts and outputs
corrMatrix = train_df.corr(method='spearman')
fig,ax = plt.subplots(figsize=(10,8))
sns.set(font_scale=1)
sns.heatmap(abs(corrMatrix),annot=True,cmap="YlGnBu")
plt.show()

train_x = pd.DataFrame((train_df[input_cols]))
train_y = pd.DataFrame((train_df[output_cols]))

outp = output_cols

errors_lists = []
for dv in output_cols:
  x_train, x_test, y_train, y_test = train_test_split(train_x, train_y[dv],test_size = 0.25)

  if dv in ['Phase Margin Frequency', 'Slew Rate Value', 'DC_Power', 'UGB', 'Noise_1M', 'Noise_10M','Phase Margin', '3dB Bandwidth']:
      svr = SVR(kernel = "rbf", C=50, epsilon=0.01, gamma=0.01)
  elif dv in ['Noise_DC', 'DC_Gain']:
      svr = SVR(kernel = "rbf", C=50, epsilon=0.01, gamma=0.001)
  else:
      svr = SVR()
  svr.fit(x_train,y_train)
  y_pred = svr.predict(x_test)
  errors = mean_squared_error(y_test, y_pred, squared=False) #gives RMSE vales
  # pd.DataFrame([y_test,y_pred],columns = ['Actual Value of ' + dv, 'Predicted Value of ' + dv]).to_csv(directory + '/' + dv + '.csv')
  errors_lists.append(errors)
  print(dv)
  
muls_op_list = [mul_N10M, mul_N1M, mul_NDC, mul_PM, mul_PMF, mul_3db, mul_UGB, mul_power, mul_gain, mul_SR]
error_std = [errors_lists[0]/train_y['Noise_10M'].describe().loc['std'],
              errors_lists[1]/train_y['Noise_1M'].describe().loc['std'],
              errors_lists[2]/train_y['Noise_DC'].describe().loc['std'],
              errors_lists[3]/train_y['Phase Margin'].describe().loc['std'],
              errors_lists[4]/train_y['Phase Margin Frequency'].describe().loc['std'],
              errors_lists[5]/train_y['3dB Bandwidth'].describe().loc['std'],
              errors_lists[6]/train_y['UGB'].describe().loc['std'],
              errors_lists[7]/train_y['DC_Power'].describe().loc['std'],
              errors_lists[8]/train_y['DC_Gain'].describe().loc['std'],
              errors_lists[9]/train_y['Slew Rate Value'].describe().loc['std']]

df = pd.DataFrame(list(zip(output_cols, np.array(errors_lists)*np.array(muls_op_list), error_std)),
                columns =['Target Variable', 'RMSE', 'RMSE wrt. STDEV'])
print(df,"\n")

errors_lists = []
for dv in input_cols:
  x_train, x_test, y_train, y_test = train_test_split(train_y, train_x[dv])

  if dv in ['nf_Mn_tail1', 'nf_Mn_tail2', 'L_Mp_stg1', 'nf_Mp_stg2', 'L_Mn_drive', 'nf_Mn_drive', 'mul_cap']:
      svr = SVR(kernel = "rbf", C=50, epsilon=0.01, gamma=0.01)
  elif dv in ['nf_ibias', 'L_Mn_tail1']:
      svr = SVR(kernel = "rbf", C=50, epsilon=0.01, gamma=0.1)
  else:
      svr = SVR()
  svr.fit(x_train,y_train)
  y_pred = svr.predict(x_test)
  errors = mean_squared_error(y_test, y_pred, squared=False) #gives RMSE vales
  # pd.DataFrame([y_test,y_pred],columns = ['Actual Value of ' + dv, 'Predicted Value of ' + dv]).to_csv(directory + '/' + dv + '.csv')
  errors_lists.append(errors)
  print(dv)
  
muls_ip_list = [mul_Ldrive, mul_Ltail1, mul_Lstg1, mul_mulcap , mul_nf_ibias, mul_nf_Mn_drive, mul_nf_Mn_tail1, mul_nf_Mn_tail2, mul_nf_Mp_stg2]
error_std = [errors_lists[0]/train_x['L_Mn_drive'].describe().loc['std'],
              errors_lists[1]/train_x['L_Mn_tail1'].describe().loc['std'],
              errors_lists[2]/train_x['L_Mp_stg1'].describe().loc['std'],
              errors_lists[3]/train_x['mul_cap'].describe().loc['std'],
              errors_lists[4]/train_x['nf_ibias'].describe().loc['std'],
              errors_lists[5]/train_x['nf_Mn_drive'].describe().loc['std'],
              errors_lists[6]/train_x['nf_Mn_tail1'].describe().loc['std'],
              errors_lists[7]/train_x['nf_Mn_tail2'].describe().loc['std'],
              errors_lists[8]/train_x['nf_Mp_stg2'].describe().loc['std']]

df = pd.DataFrame(list(zip(input_cols, np.array(errors_lists)*np.array(muls_ip_list), error_std)),
                columns =['Target Variable', 'RMSE', 'RMSE wrt. STDEV'])
print(df)

final_time = time.time()

print("Total Time required to run the code : ", final_time - first_time)