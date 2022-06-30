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
# import os
#import cv2
#from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV #,StratifiedKFold
# from sklearn.metrics import classification_report,roc_auc_score
#from sklearn import svm
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
# from sklearn import metrics 
from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import VarianceThreshold

#Reading CSV (Comma Separted Value) File
train_df = pd.read_csv('combined_dataset_5_6_7_8_scaled.csv',header = 0);p=2;
# train_df = pd.read_csv('combined_dataset_5_6_7_8_processed.csv',header = 0);p=1;
# train_df = pd.read_csv('combined_dataset_5_6_7_8.csv',header = 0);p=0;

#display(train_df)

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
        train_df.to_csv('models_tt2/combined_dataset_5_6_7_8_processed.csv')
else: 
    print("Data Processing already done")
    
for col_value in train_df.columns:
    train_df[col_value] = train_df[col_value].astype(float)

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
    # train_df.to_csv('combined_dataset_5_6_7_8_scaled.csv')
    scaled = input("Do you want to create a .csv file for saving scaled data?, Y/N : ")
    if scaled == "Y":
        train_df.to_csv('models_tt2/combined_dataset_5_6_7_8_scaled.csv')

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
# print(train_df) 
# print(train_df.describe())

# creating correlation matrix for all inouts and outputs
# corrMatrix = train_df.corr(method='spearman')
# fig,ax = plt.subplots(figsize=(10,8))
# sns.set(font_scale=1)
# sns.heatmap(abs(corrMatrix),annot=True,cmap="YlGnBu")
# plt.show()

train_x = pd.DataFrame((train_df[input_cols]))
train_y = pd.DataFrame((train_df[output_cols]))

outp = output_cols

# errors_lists = []
# for dv in output_cols:
#   x_train, x_test, y_train, y_test = train_test_split(train_x, train_y[dv])

#   svr = SVR()
#   #svr1 = SVR(kernel = "linear", C=10, epsilon=0.1, degree=3, gamma=2)
#   svr.fit(x_train,y_train)
#   y_pred = svr.predict(x_test)
#   errors = mean_squared_error(y_test, y_pred, squared=False)
#   #print("Errors for :: "+dv+" :: "+str(errors))
#   errors_lists.append(errors)
#   print(dv)
  
# muls_op_list = [mul_N10M, mul_N1M, mul_NDC, mul_PM, mul_PMF, mul_3db, mul_UGB, mul_power, mul_gain, mul_SR]
# error_std = [errors_lists[0]/train_y['Noise_10M'].describe().loc['std'],
#              errors_lists[1]/train_y['Noise_1M'].describe().loc['std'],
#              errors_lists[2]/train_y['Noise_DC'].describe().loc['std'],
#              errors_lists[3]/train_y['Phase Margin'].describe().loc['std'],
#              errors_lists[4]/train_y['Phase Margin Frequency'].describe().loc['std'],
#              errors_lists[5]/train_y['3dB Bandwidth'].describe().loc['std'],
#              errors_lists[6]/train_y['UGB'].describe().loc['std'],
#              errors_lists[7]/train_y['DC_Power'].describe().loc['std'],
#              errors_lists[8]/train_y['DC_Gain'].describe().loc['std'],
#              errors_lists[9]/train_y['Slew Rate Value'].describe().loc['std']]

# df = pd.DataFrame(list(zip(output_cols, np.array(errors_lists)*np.array(muls_op_list), error_std)),
#                columns =['Target Variable', 'RMSE', 'RMSE wrt. STDEV'])
# print(df,"\n")

# errors_lists = []
# for dv in input_cols:
#   x_train, x_test, y_train, y_test = train_test_split(train_y, train_x[dv])

#   svr = SVR()
#   #svr1 = SVR(kernel = "linear", C=10, epsilon=0.1, degree=3, gamma=2)
#   svr.fit(x_train,y_train)
#   y_pred = svr.predict(x_test)
#   errors = mean_squared_error(y_test, y_pred, squared=False)
#   #print("Errors for :: "+dv+" :: "+str(errors))
#   errors_lists.append(errors)
#   print(dv)
  
# muls_ip_list = [mul_Ldrive, mul_Ltail1, mul_Lstg1, mul_mulcap , mul_nf_ibias, mul_nf_Mn_drive, mul_nf_Mn_tail1, mul_nf_Mn_tail2, mul_nf_Mp_stg2]
# error_std = [errors_lists[0]/train_x['L_Mn_drive'].describe().loc['std'],
#              errors_lists[1]/train_x['L_Mn_tail1'].describe().loc['std'],
#              errors_lists[2]/train_x['L_Mp_stg1'].describe().loc['std'],
#              errors_lists[3]/train_x['mul_cap'].describe().loc['std'],
#              errors_lists[4]/train_x['nf_ibias'].describe().loc['std'],
#              errors_lists[5]/train_x['nf_Mn_drive'].describe().loc['std'],
#              errors_lists[6]/train_x['nf_Mn_tail1'].describe().loc['std'],
#              errors_lists[7]/train_x['nf_Mn_tail2'].describe().loc['std'],
#              errors_lists[8]/train_x['nf_Mp_stg2'].describe().loc['std']]

# df = pd.DataFrame(list(zip(input_cols, np.array(errors_lists)*np.array(muls_ip_list), error_std)),
#                columns =['Target Variable', 'RMSE', 'RMSE wrt. STDEV'])
# print(df)

for dv in ['DC_Gain', 'Noise_DC', 'Noise_10M']:
# ['Slew Rate Value', 'DC_Power', 'UGB', 'Phase Margin', 'Phase Margin Frequency', '3dB Bandwidth', 'Noise_1M']:  # output variables hypertuning
  x_train, x_test, y_train, y_test = train_test_split(train_x, train_y[dv], test_size = 0.25)
  param_grid = [{'C': [1, 10, 50], 'gamma': [0.1,0.01,0.001],'kernel': ['rbf'], 'epsilon' : [0.01]}] #,
                # {'C': [0.1, 1, 10, 50], 'kernel': ['linear']}]
  
  grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3)
  grid.fit(x_train, y_train)
  print(dv)
  print(grid.best_params_)  # print best parameter after tuning
  print(grid.best_estimator_)  # print how our model looks after hyper-parameter tuning
  #grid_predictions = grid.predict(x_test)
  print(grid.score(x_test, y_test))# print classification report
  filename = 'Finalized_Model_Automatic_Testing/' + str(dv) + '.txt'
  f = open(filename,'w')
  f.write(str(grid.best_params_))
  f.close()

  
for dv in ['L_Mn_drive', 'L_Mp_stg1', 'nf_ibias']:
# ['nf_Mp_stg2', 'nf_Mn_tail1', 'nf_Mn_tail2', 'nf_Mn_drive', 'L_Mn_tail1', 'mul_cap']: # input variables hypertuning
# ['nf_Mp_stg2', 'nf_Mn_tail1', 'mul_cap', 'nf_ibias', 'L_Mn_tail1', 'L_Mn_drive']:
  x_train, x_test, y_train, y_test = train_test_split(train_y, train_x[dv])
  param_grid = [{'C': [1, 10, 50], 'gamma': [0.1,0.01,0.001],'kernel': ['rbf'], 'epsilon' : [0.01]}]#,
                # {'C': [0.1, 1, 10, 50], 'kernel': ['linear']}]
  
  grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3)
  grid.fit(x_train, y_train)
  print(dv)
  print(grid.best_params_)  # print best parameter after tuning
  print(grid.best_estimator_)  # print how our model looks after hyper-parameter tuning
  #grid_predictions = grid.predict(x_test)
  print(grid.score(x_test, y_test))# print classification report
  filename = 'Finalized_Model_Automatic_Testing/' + str(dv) + '.txt'
  f = open(filename,'w')
  f.write(str(grid.best_params_))
  f.close()
  
"""
#creating classifiers for different Outputs
classifiers = int(input("Do you want to load classifiers or create new ones (for load, type 0 and for create, type 1) : "))
# classifiers = 1
pre_time = time.time()

if classifiers == 1:
    
    train_x_tr, train_x_tst, train_y_tr, train_y_tst = train_test_split(train_x, train_y, test_size=0.3, random_state=42)
    (pd.DataFrame(train_x_tr)).to_csv("models_tt2/train_x_tr.csv")
    (pd.DataFrame(train_x_tst)).to_csv("models_tt2/train_x_tst.csv")
    (pd.DataFrame(train_y_tr)).to_csv("models_tt2/train_y_tr.csv")
    (pd.DataFrame(train_y_tst)).to_csv("models_tt2/train_y_tst.csv")
    
    print("creating classifiers")
    
    # Output Classifiers creation
    svr1 = make_pipeline(SVR(kernel = "linear"))
    svr1.fit(train_x_tr[input_cols], train_y_tr['DC_Gain'])
    pickle.dump(svr1, open("models_tt2/DC_Gain.sav", 'wb'))
    print('classifier created for DC_Gain\t',time.time()-pre_time)
    
    svr2 = make_pipeline(SVR(kernel = "linear"))
    svr2.fit(train_x_tr[input_cols], train_y_tr['Slew Rate Value'])
    pickle.dump(svr2, open("models_tt2/Slew Rate Value.sav", 'wb'))
    print('classifier created for Slew Rate Values\t',time.time()-pre_time)
    
    svr3 = make_pipeline(SVR(kernel = "linear"))
    svr3.fit(train_x_tr[input_cols], train_y_tr['UGB'])
    pickle.dump(svr3, open("models_tt2/UGB.sav", 'wb'))
    print('classifier created for UGB\t',time.time()-pre_time)
    
    svr4 = make_pipeline(SVR(kernel = "linear"))
    svr4.fit(train_x_tr[input_cols], train_y_tr['Phase Margin'])
    pickle.dump(svr4, open("models_tt2/Phase Margin.sav", 'wb'))
    print('classifier created for Phase Margin\t',time.time()-pre_time)
    
    svr5 = make_pipeline(SVR(kernel = "linear"))
    svr5.fit(train_x_tr[input_cols], train_y_tr['DC_Power'])
    pickle.dump(svr5, open("models_tt2/DC_Power.sav", 'wb'))
    print('classifier created for DC_Power\t',time.time()-pre_time)
    
    svr6 = make_pipeline(SVR(kernel = "linear"))
    svr6.fit(train_x_tr[input_cols], train_y_tr['3dB Bandwidth'])
    pickle.dump(svr6, open("models_tt2/3dB Bandwidth.sav", 'wb'))
    print('classifier created for 3dB Bandwidth\t',time.time()-pre_time)
    
    svr7 = make_pipeline(SVR(kernel = "linear"))
    svr7.fit(train_x_t[input_cols], train_y_tr['Noise_DC'])
    pickle.dump(svr7, open("models_tt2/Noise_DC.sav", 'wb'))
    print('classifier created for Noise_DC\t',time.time()-pre_time)
    
    svr8 = make_pipeline(SVR(kernel = "linear"))
    svr8.fit(train_x_tr[input_cols], train_y_tr['Noise_1M'])
    pickle.dump(svr8, open("models_tt2/Noise_1M.sav", 'wb'))
    print('classifier created for Noise_1M\t',time.time()-pre_time)
    
    svr9 = make_pipeline(SVR(kernel = "linear"))
    svr9.fit(train_x_tr[input_cols], train_y_tr['Noise_10M'])
    pickle.dump(svr9, open("models_tt2/Noise_10M.sav", 'wb'))
    print('classifier created for Noise_10M\t',time.time()-pre_time)
    
    # Input Classifiers creation
    svr11 = make_pipeline(SVR(kernel = "linear"))
    svr11.fit(train_y_tr[output_cols], train_x_tr['nf_Mn_drive'])
    pickle.dump(svr11, open("models_tt2/nf_Mn_drive.sav", 'wb'))
    print('classifier created for nf_Mn_drive\t',time.time()-pre_time)
    
    svr12 = make_pipeline(SVR(kernel = "linear"))
    svr12.fit(train_y_tr[output_cols], train_x_tr['nf_Mn_tail1'])
    pickle.dump(svr12, open("models_tt2/nf_Mn_tail1.sav", 'wb'))
    print('classifier created for nf_Mn_tail1\t',time.time()-pre_time)
    
    svr13 = make_pipeline(SVR(kernel = "linear"))
    svr13.fit(train_y_tr[output_cols], train_x_tr['nf_Mn_tail2'])
    pickle.dump(svr13, open("models_tt2/nf_Mn_tail2.sav", 'wb'))
    print('classifier created for nf_Mn_tail2\t',time.time()-pre_time)
    
    svr14 = make_pipeline(SVR(kernel = "linear"))
    svr14.fit(train_y_tr[output_cols], train_x_tr['nf_Mp_stg2'])
    pickle.dump(svr14, open("models_tt2/nf_Mp_stg2.sav", 'wb'))
    print('classifier created for nf_Mp_stg2\t',time.time()-pre_time)
    
    svr15 = make_pipeline(SVR(kernel = "linear"))
    svr15.fit(train_y_tr[output_cols], train_x_tr['nf_ibias'])
    pickle.dump(svr15, open("models_tt2/nf_ibias.sav", 'wb'))
    print('classifier created for nf_ibias\t',time.time()-pre_time)
    
    svr16 = make_pipeline(SVR(kernel = "linear"))
    svr16.fit(train_y_tr[output_cols], train_x_tr['L_Mn_drive'])
    pickle.dump(svr16, open("models_tt2/L_Mn_drive.sav", 'wb'))
    print('classifier created for L_Mn_drive\t',time.time()-pre_time)
    
    svr17 = make_pipeline(SVR(kernel = "linear"))
    svr17.fit(train_y_tr[output_cols], train_x_tr['L_Mn_tail1'])
    pickle.dump(svr17, open("models_tt2/L_Mn_tail1.sav", 'wb'))
    print('classifier created for L_Mn_tail1\t',time.time()-pre_time)
    
    svr18 = make_pipeline(SVR(kernel = "linear"))
    svr18.fit(train_y_tr[output_cols], train_x_tr['L_Mp_stg1'])
    pickle.dump(svr18, open("models_tt2/L_Mp_stg1.sav", 'wb'))
    print('classifier created for L_Mp_stg1\t',time.time()-pre_time)
    
    svr19 = make_pipeline(SVR(kernel = "linear"))
    svr19.fit(train_y_tr[output_cols], train_x_tr['mul_cap'])
    pickle.dump(svr19, open("models_tt2/mul_cap.sav", 'wb'))
    print('classifier created for mul_cap\t',time.time()-pre_time)
    
    post_time = time.time()
    
elif (classifiers==0):
    # input to output classifiers
    train_x_tr = pd.read_csv('models_tt2/train_x_tr.csv',header = 0)
    train_x_tst = pd.read_csv('models_tt2/train_x_tst.csv',header = 0)
    train_y_tr = pd.read_csv('models_tt2/train_y_tr.csv',header = 0)
    train_y_tst = pd.read_csv('models_tt2/train_y_tst.csv',header = 0)
    train_keys = train_x_tr['mul_cap'].keys()
    test_keys = train_x_tst['mul_cap'].keys()
    
    svr1 = pickle.load(open("models_tt2/DC_Gain.sav", 'rb'))
    svr2 = pickle.load(open("models_tt2/Slew Rate Value.sav", 'rb'))
    svr3 = pickle.load(open("models_tt2/UGB.sav", 'rb'))
    svr4 = pickle.load(open("models_tt2/Phase Margin.sav", 'rb'))
    svr5 = pickle.load(open("models_tt2/DC_Power.sav", 'rb'))
    svr6 = pickle.load(open("models_tt2/3dB Bandwidth.sav", 'rb'))
    svr7 = pickle.load(open("models_tt2/Noise_DC.sav", 'rb'))
    svr8 = pickle.load(open("models_tt2/Noise_1M.sav", 'rb'))
    svr9 = pickle.load(open("models_tt2/Noise_10M.sav", 'rb'))
    
    
    # output to input classifiers
    svr11 = pickle.load(open("models_tt2/nf_Mn_drive.sav", 'rb'))
    svr12 = pickle.load(open("models_tt2/nf_Mn_tail1.sav", 'rb'))
    svr13 = pickle.load(open("models_tt2/nf_Mn_tail2.sav", 'rb'))
    svr14 = pickle.load(open("models_tt2/nf_Mp_stg2.sav", 'rb'))
    svr15 = pickle.load(open("models_tt2/nf_ibias.sav", 'rb'))
    svr16 = pickle.load(open("models_tt2/L_Mn_drive.sav", 'rb'))
    svr17 = pickle.load(open("models_tt2/L_Mn_tail1.sav", 'rb'))
    svr18 = pickle.load(open("models_tt2/L_Mp_stg1.sav", 'rb'))
    svr19 = pickle.load(open("models_tt2/mul_cap.sav", 'rb'))
    
    post_time = time.time()

if classifiers == 1:
    print("Classfiers Created, time required : ", post_time - pre_time)
else:
    print("Classfiers Loaded, time required : ", post_time - pre_time)
    
"""
  
# Input values of specification to be taken from user

# required_DC_Gain = 45                         # in dB
# required_Slew_Rate = 200                      # in V/usec
# required_UGB = 200                            # in MHz
# required_Phase_Margin = 65                    # in degees
# required_DC_Power = 2                         # in mW
# required_3dB_Bandwidth = 0.5                  # in MHz
# required_Noise_DC = 30                        # in uV/sqrt(Hz)
# required_Noise_1M = 10                        # in nV/sqrt(Hz)
# required_Noise_10M = 7                        # in nV/sqrt(Hz)
# required_PMFreq = 0.9875 * required_UGB       # in MHz

# test_out = [[required_Noise_10M /1e9 /mul_N10M, 
#             required_Noise_1M /1e9 /mul_N1M, 
#             required_Noise_DC /1e6 /mul_NDC, 
#             required_Phase_Margin /mul_PM, 
#             required_PMFreq * 1e6 / mul_UGB, 
#             required_3dB_Bandwidth *1e6 /mul_3db, 
#             required_UGB *1e6 /mul_UGB, 
#             required_DC_Power /1e3 /mul_power, 
#             required_DC_Gain /mul_gain, 
#             required_Slew_Rate *1e6 /mul_SR]]

# # test_out = [[48.93,15.9,16.5,62.01]]

# nf_Mn_drive_predict = round(float(svr11.predict(test_out)))
# nf_Mn_tail1_predict = round(float(svr12.predict(test_out)))
# nf_Mn_tail2_predict = round(float(svr13.predict(test_out)))
# nf_Mp_stg2_predict = round(float(svr14.predict(test_out)))
# nf_ibias_predict = round(float(svr15.predict(test_out)))
# L_Mn_drive_predict = round(float(svr16.predict(test_out)))
# L_Mn_tail1_predict = round(float(svr17.predict(test_out)))
# L_Mp_stg1_predict = round(float(svr18.predict(test_out)))
# mul_cap_predict = round(float(svr19.predict(test_out)))

# predicted_output = [[L_Mn_drive_predict, L_Mn_tail1_predict, L_Mp_stg1_predict, mul_cap_predict, nf_ibias_predict, nf_Mn_drive_predict, nf_Mn_tail1_predict, nf_Mn_tail2_predict, nf_Mp_stg2_predict]]
# predicted_output_scaled = [[L_Mn_drive_predict *mul_Ldrive, 
#                             L_Mn_tail1_predict *mul_Ltail1, 
#                             L_Mp_stg1_predict *mul_Lstg1, 
#                             mul_cap_predict *mul_mulcap, 
#                             nf_ibias_predict *mul_nf_ibias, 
#                             nf_Mn_drive_predict *mul_nf_Mn_drive, 
#                             nf_Mn_tail1_predict *mul_nf_Mn_tail1, 
#                             nf_Mn_tail2_predict *mul_nf_Mn_tail2, 
#                             nf_Mp_stg2_predict *mul_nf_Mp_stg2]]

# # print(predicted_output_scaled)

# predicted_DC_Gain = svr1.predict(predicted_output)
# predicted_SR = svr2.predict(predicted_output)
# predicted_UGB = svr3.predict(predicted_output)
# predicted_PM = svr4.predict(predicted_output)
# predicted_DC_power = svr5.predict(predicted_output)
# predicted_3dB_BW = svr6.predict(predicted_output)
# predicted_NoiseDC = svr7.predict(predicted_output)
# predicted_Noise1M = svr8.predict(predicted_output)
# predicted_Noise10M = svr9.predict(predicted_output)

# errors = [[abs(predicted_Noise10M - test_out[0][0]) * mul_N10M,
#            abs(predicted_Noise1M - test_out[0][1]) * mul_N1M,
#            abs(predicted_NoiseDC - test_out[0][2]) * mul_NDC,
#            abs(predicted_PM - test_out[0][3]) * mul_PM,
#            abs(predicted_3dB_BW - test_out[0][5]) * mul_3db,
#            abs(predicted_UGB - test_out[0][6]) * mul_UGB,
#            abs(predicted_UGB*0.9875 - test_out[0][4]) * mul_UGB,
#            abs(predicted_DC_power - test_out[0][7]) * mul_power,
#            abs(predicted_DC_Gain - test_out[0][8]) * mul_gain,
#            abs(predicted_SR - test_out[0][9]) * mul_SR]]


# output_df = pd.DataFrame(np.array(predicted_output_scaled).reshape(1,9),columns= input_cols)
# print(output_df.head())

# specifications_errors_df = pd.DataFrame(np.array(errors).reshape(1,10),columns= output_cols)
# print(specifications_errors_df.head())

# final_time = time.time()

# print("Total Time required to run the code : ", final_time - first_time)
# print("Error in DC_Gain : \t",abs(predicted_DC_Gain - test_out[0][8]) * mul_gain)
# print("Error in Slew_Rate : \t",abs(predicted_SR - test_out[0][9]) * mul_SR)
# print("Error in UGB : \t",abs(predicted_UGB - test_out[0][6]) * mul_UGB)
# print("Error in Phase Margin : \t",abs(predicted_PM - test_out[0][3]) * mul_PM)
# print("Error in DC_Power : \t",abs(predicted_DC_power - test_out[0][7]) * mul_power)
# print("Error in 3dB Bandwidth : \t",abs(predicted_3dB_BW - test_out[0][5]) * mul_3db)
# print("Error in Noise_DC : \t",abs(predicted_NoiseDC - test_out[0][2]) * mul_NDC)
# print("Error in Noise_1M : \t",abs(predicted_Noise1M - test_out[0][1]) * mul_N1M)
# print("Error in Noise_10M : \t",abs(predicted_Noise10M - test_out[0][0]) * mul_N10M)


#user inputs of specification

# nf_ibias_values = np.arange(min(train_x['nf_ibias']),max(train_x['nf_ibias'])+1,1,dtype=np.uint16)
# nf_Mn_drive_values = np.arange(min(train_x['nf_Mn_drive']),max(train_x['nf_Mn_drive'])+1,10,dtype=np.uint16)
# nf_Mn_tail1_values = np.arange(min(train_x['nf_Mn_tail1']),max(train_x['nf_Mn_tail1'])+1,10,dtype=np.uint16)
# nf_Mn_tail2_values = np.arange(min(train_x['nf_Mn_tail2']),max(train_x['nf_Mn_tail2'])+1,25,dtype=np.uint16)
# nf_Mp_stg2_values = np.arange(min(train_x['nf_Mp_stg2']),max(train_x['nf_Mp_stg2'])+1,25,dtype=np.uint16)

# total_len = len(nf_ibias_values) * len(nf_Mn_drive_values) * len(nf_Mn_tail1_values) * len(nf_Mn_tail2_values) * len(nf_Mp_stg2_values)

# test_data = np.zeros((total_len, train_x.shape[1]))

# i=0
# for a in nf_ibias_values:
#   for b in nf_Mn_drive_values:
#     for c in nf_Mn_tail1_values:
#       for d in nf_Mn_tail2_values:
#         for e in nf_Mp_stg2_values:
#           test_data[i] = [a,b,c,d,e]
#           i=i+1

# test_data.astype(np.uint16)
# test_data = [[2,13,22,115,216]]
# Gain_predict = svr1.predict(test_data)
# print(Gain_predict)

# test_data_output = np.zeros((test_data.shape[0],4))
# test_data_output[:,0] = svr1.predict(test_data) #predictions_DC_Gain
# print("Predictions DC_Gain")
# test_data_output[:,1] = svr2.predict(test_data) #predictions_SR
# print("Predictions SR")
# test_data_output[:,2] = svr3.predict(test_data) #predictions_UGB
# print("Predictions UGB")
# test_data_output[:,3] = svr4.predict(test_data) #predictions_PM
# print("Predictions PM")

# print("\nDC_Gain : \tmin = ",min(test_data_output[:,0]),"\tmax = ",max(test_data_output[:,0]))
# print("SR : \tmin = ",min(test_data_output[:,1])*10,"\tmax = ",max(test_data_output[:,1])*10)
# print("UGB : \tmin = ",min(test_data_output[:,2])*10,"\tmax = ",max(test_data_output[:,2])*10)
# print("PM : \tmin = ",min(test_data_output[:,3]),"\tmax = ",max(test_data_output[:,3]))


# test_data_inputs_outputs_updated = np.zeros((total_len,test_data.shape[1]+test_data_output.shape[1]))

# for i in range(0, total_len):
#     test_data_inputs_outputs_updated[i][0:5] = test_data[i]
#     test_data_inputs_outputs_updated[i][5] = test_data_output[i][0]
#     test_data_inputs_outputs_updated[i][6] = test_data_output[i][1]*1e7
#     test_data_inputs_outputs_updated[i][7] = test_data_output[i][2]*1e7
#     test_data_inputs_outputs_updated[i][8] = test_data_output[i][3]
    
# lookup_table = pd.DataFrame(test_data_inputs_outputs_updated)
# lookup_table.to_csv('LookUp_Table1.csv')

# final_time = time.time()
# print("\n Total Time Required for training : ",final_time - first_time)