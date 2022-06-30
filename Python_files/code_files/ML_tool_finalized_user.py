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
#from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
import pickle
import os

# Input Directory/Folder Name here
directory = 'Finalized_Model_User'           # Enter he name of directory to be created
try: 
    os.mkdir(directory) 
except: 
    print("Directory with name " + directory + " already exists")         

#Reading CSV (Comma Separted Value) File
CSV_filename = 'combined_dataset_5_6_7_8'               # Write file name of .csv file without extension
CSV_filename1 = CSV_filename + '.csv'
CSV_filename2 = directory + '/' + CSV_filename + '_processed.csv'
CSV_filename3 = directory + '/' + CSV_filename + '_scaled.csv'

# train_df = pd.read_csv(CSV_filename1,header = 0);p=0;           # Run this file if started from scratch
# train_df = pd.read_csv(CSV_filename2,header = 0);p=1;         # Run this file if only unwanted columns are removed
train_df = pd.read_csv(CSV_filename3,header = 0);p=2;         # Run this file if all unwanted columns are removed and data scaling is done

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
mul_Lstg2 = 1e-7
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

#creating classifiers for different Outputs
models = int(input("Do you want to load models or create new ones (for load, type 0 and for create, type 1) : "))
pre_time = time.time()

if models == 1:
    print("Creating Models")
    
    # Output Regression Model creation
    svr1 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.001))
    svr1.fit(train_x, train_y['DC_Gain'])
    pickle.dump(svr1, open(directory + "/DC_Gain.sav", 'wb'))
    print('classifier created for DC_Gain\t',time.time()-pre_time)

    svr2 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr2.fit(train_x, train_y['Slew Rate Value'])
    pickle.dump(svr2, open(directory + "/Slew Rate Value.sav", 'wb'))
    print('classifier created for Slew Rate Values\t',time.time()-pre_time)

    svr3 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr3.fit(train_x, train_y['UGB'])
    pickle.dump(svr3, open(directory + "/UGB.sav", 'wb'))
    print('classifier created for UGB\t',time.time()-pre_time)

    svr4 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr4.fit(train_x, train_y['Phase Margin'])
    pickle.dump(svr4, open(directory + "/Phase Margin.sav", 'wb'))
    print('classifier created for Phase Margin\t',time.time()-pre_time)

    svr5 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr5.fit(train_x, train_y['DC_Power'])
    pickle.dump(svr5, open(directory + "/DC_Power.sav", 'wb'))
    print('classifier created for DC_Power\t',time.time()-pre_time)

    svr6 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr6.fit(train_x, train_y['3dB Bandwidth'])
    pickle.dump(svr6, open(directory + "/3dB Bandwidth.sav", 'wb'))
    print('classifier created for 3dB Bandwidth\t',time.time()-pre_time)

    svr7 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.001))
    svr7.fit(train_x, train_y['Noise_DC'])
    pickle.dump(svr7, open(directory + "/Noise_DC.sav", 'wb'))
    print('classifier created for Noise_DC\t',time.time()-pre_time)

    svr8 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr8.fit(train_x, train_y['Noise_1M'])
    pickle.dump(svr8, open(directory + "/Noise_1M.sav", 'wb'))
    print('classifier created for Noise_1M\t',time.time()-pre_time)

    svr9 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr9.fit(train_x, train_y['Noise_10M'])
    pickle.dump(svr9, open(directory + "/Noise_10M.sav", 'wb'))
    print('classifier created for Noise_10M\t',time.time()-pre_time)
    
    # Input Regression Model creation
    svr11 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr11.fit(train_y[outp], train_x['nf_Mn_drive'])
    pickle.dump(svr11, open(directory + "/nf_Mn_drive.sav", 'wb'))
    print('classifier created for nf_Mn_drive\t',time.time()-pre_time)
    
    svr12 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr12.fit(train_y[outp], train_x['nf_Mn_tail1'])
    pickle.dump(svr12, open(directory + "/nf_Mn_tail1.sav", 'wb'))
    print('classifier created for nf_Mn_tail1\t',time.time()-pre_time)
    
    svr13 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr13.fit(train_y[outp], train_x['nf_Mn_tail2'])
    pickle.dump(svr13, open(directory + "/nf_Mn_tail2.sav", 'wb'))
    print('classifier created for nf_Mn_tail2\t',time.time()-pre_time)
    
    svr14 = make_pipeline(SVR(kernel = "rbf", C=50, epsilon=0.01, gamma=0.01))
    svr14.fit(train_y[outp], train_x['nf_Mp_stg2'])
    pickle.dump(svr14, open(directory + "/nf_Mp_stg2.sav", 'wb'))
    print('classifier created for nf_Mp_stg2\t',time.time()-pre_time)
    
    svr15 = make_pipeline(SVR(kernel = "rbf", C=50, epsilon=0.01, gamma=0.1))
    svr15.fit(train_y[outp], train_x['nf_ibias'])
    pickle.dump(svr15, open(directory + "/nf_ibias.sav", 'wb'))
    print('classifier created for nf_ibias\t',time.time()-pre_time)
    
    svr16 = make_pipeline(SVR(kernel = "rbf", C=50, epsilon=0.01, gamma=0.01))
    svr16.fit(train_y[outp], train_x['L_Mn_drive'])
    pickle.dump(svr16, open(directory + "/L_Mn_drive.sav", 'wb'))
    print('classifier created for L_Mn_drive\t',time.time()-pre_time)
    
    svr17 = make_pipeline(SVR(kernel = "rbf", C=50, epsilon=0.01, gamma=0.1))
    svr17.fit(train_y[outp], train_x['L_Mn_tail1'])
    pickle.dump(svr17, open(directory + "/L_Mn_tail1.sav", 'wb'))
    print('classifier created for L_Mn_tail1\t',time.time()-pre_time)
    
    svr18 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr18.fit(train_y[outp], train_x['L_Mp_stg1'])
    pickle.dump(svr18, open(directory + "/L_Mp_stg1.sav", 'wb'))
    print('classifier created for L_Mp_stg1\t',time.time()-pre_time)
    
    svr19 = make_pipeline(SVR(kernel="rbf", C=50, epsilon=0.01, gamma=0.01))
    svr19.fit(train_y[outp], train_x['mul_cap'])
    pickle.dump(svr19, open(directory + "/mul_cap.sav", 'wb'))
    print('classifier created for mul_cap\t',time.time()-pre_time)
    
    post_time = time.time()
    
elif (models==0):
    # input to output Regression Model
    svr1 = pickle.load(open(directory + "/DC_Gain.sav", 'rb'))
    svr2 = pickle.load(open(directory + "/Slew Rate Value.sav", 'rb'))
    svr3 = pickle.load(open(directory + "/UGB.sav", 'rb'))
    svr4 = pickle.load(open(directory + "/Phase Margin.sav", 'rb'))
    svr5 = pickle.load(open(directory + "/DC_Power.sav", 'rb'))
    svr6 = pickle.load(open(directory + "/3dB Bandwidth.sav", 'rb'))
    svr7 = pickle.load(open(directory + "/Noise_DC.sav", 'rb'))
    svr8 = pickle.load(open(directory + "/Noise_1M.sav", 'rb'))
    svr9 = pickle.load(open(directory + "/Noise_10M.sav", 'rb'))
    
    # output to input classifiers
    svr11 = pickle.load(open(directory + "/nf_Mn_drive.sav", 'rb'))
    svr12 = pickle.load(open(directory + "/nf_Mn_tail1.sav", 'rb'))
    svr13 = pickle.load(open(directory + "/nf_Mn_tail2.sav", 'rb'))
    svr14 = pickle.load(open(directory + "/nf_Mp_stg2.sav", 'rb'))
    svr15 = pickle.load(open(directory + "/nf_ibias.sav", 'rb'))
    svr16 = pickle.load(open(directory + "/L_Mn_drive.sav", 'rb'))
    svr17 = pickle.load(open(directory + "/L_Mn_tail1.sav", 'rb'))
    svr18 = pickle.load(open(directory + "/L_Mp_stg1.sav", 'rb'))
    svr19 = pickle.load(open(directory + "/mul_cap.sav", 'rb'))
    
    post_time = time.time()

if models == 1:
    print("Classfiers Created, time required : ", post_time - pre_time)
else:
    print("Classfiers Loaded, time required : ", post_time - pre_time)
    
# Input values of specification to be taken from user
required_DC_Gain = 50                        # in dB
required_UGB = 500                            # in MHz
required_Phase_Margin = 65                   # in degees
required_DC_Power = 1.5                         # in mW
required_Noise_1M = 10                        # in nV/sqrt(Hz)
required_Slew_Rate = 400 #round(train_df['Slew Rate Value'].describe().loc['mean']*mul_SR/train_df['UGB'].describe().loc['mean']/mul_UGB * required_UGB, 4)          # in V/us
required_Noise_DC = round(train_df['Noise_DC'].describe().loc['mean']*mul_NDC/train_df['Noise_1M'].describe().loc['mean']/mul_N1M/1000 * required_Noise_1M, 4)                       # in uV/sqrt(Hz)
required_Noise_10M = round(train_df['Noise_10M'].describe().loc['mean']*mul_N10M/train_df['Noise_1M'].describe().loc['mean']/mul_N1M * required_Noise_1M, 4)                        # in nV/sqrt(Hz)
required_3dB_Bandwidth = round(train_df['3dB Bandwidth'].describe().loc['mean']*mul_3db/train_df['UGB'].describe().loc['mean']/mul_UGB * required_UGB, 4)                  # in MHz
required_PMFreq = round(train_df['Phase Margin Frequency'].describe().loc['mean']*mul_PMF/train_df['UGB'].describe().loc['mean']/mul_UGB * required_UGB, 4)       # in MHz

test_out = [[required_Noise_10M /1e9 /mul_N10M, 
            required_Noise_1M /1e9 /mul_N1M, 
            required_Noise_DC /1e6 /mul_NDC, 
            required_Phase_Margin /mul_PM, 
            required_PMFreq * 1e6 / mul_UGB, 
            required_3dB_Bandwidth *1e6 /mul_3db, 
            required_UGB *1e6 /mul_UGB, 
            required_DC_Power /1e3 /mul_power, 
            required_DC_Gain /mul_gain, 
            required_Slew_Rate *1e6 /mul_SR]]

test_out_df = pd.DataFrame(np.array(test_out).reshape(1,10),columns = output_cols)

nf_Mn_drive_predict = round(float(svr11.predict(test_out)))
nf_Mn_tail1_predict = round(float(svr12.predict(test_out)))
nf_Mn_tail2_predict = round(float(svr13.predict(test_out)),1)
nf_Mp_stg2_predict  = round(float(svr14.predict(test_out)),1)
nf_ibias_predict    = round(float(svr15.predict(test_out)))
L_Mn_drive_predict  = round(float(svr16.predict(test_out)),2)
L_Mn_tail1_predict  = round(float(svr17.predict(test_out)),2)
L_Mp_stg1_predict   = round(float(svr18.predict(test_out)),2)
mul_cap_predict     = round(float(svr19.predict(test_out)))

# nf_Mn_drive_predict = (float(svr11.predict(test_out)))
# nf_Mn_tail1_predict = (float(svr12.predict(test_out)))
# nf_Mn_tail2_predict = (float(svr13.predict(test_out)))
# nf_Mp_stg2_predict  = (float(svr14.predict(test_out)))
# nf_ibias_predict    = (float(svr15.predict(test_out)))
# L_Mn_drive_predict  = (float(svr16.predict(test_out)))
# L_Mn_tail1_predict  = (float(svr17.predict(test_out)))
# L_Mp_stg1_predict   = (float(svr18.predict(test_out)))
# mul_cap_predict     = (float(svr19.predict(test_out)))

predicted_output = [[L_Mn_drive_predict, L_Mn_tail1_predict, L_Mp_stg1_predict, mul_cap_predict, nf_ibias_predict, nf_Mn_drive_predict, nf_Mn_tail1_predict, nf_Mn_tail2_predict, nf_Mp_stg2_predict]]
predicted_output_scaled = [[L_Mn_drive_predict *mul_Ldrive, 
                            L_Mn_tail1_predict *mul_Ltail1, 
                            L_Mp_stg1_predict *mul_Lstg2, 
                            mul_cap_predict *mul_mulcap, 
                            nf_ibias_predict *mul_nf_ibias, 
                            nf_Mn_drive_predict *mul_nf_Mn_drive, 
                            nf_Mn_tail1_predict *mul_nf_Mn_tail1, 
                            nf_Mn_tail2_predict *mul_nf_Mn_tail2, 
                            nf_Mp_stg2_predict *mul_nf_Mp_stg2]]

# print(predicted_output_scaled)

predicted_DC_Gain = svr1.predict(predicted_output)
predicted_SR = svr2.predict(predicted_output)
predicted_UGB = svr3.predict(predicted_output)
predicted_PM = svr4.predict(predicted_output)
predicted_DC_power = svr5.predict(predicted_output)
predicted_3dB_BW = svr6.predict(predicted_output)
predicted_NoiseDC = svr7.predict(predicted_output)
predicted_Noise1M = svr8.predict(predicted_output)
predicted_Noise10M = svr9.predict(predicted_output)

errors = [[abs(predicted_Noise10M - test_out[0][0]) * mul_N10M,
           abs(predicted_Noise1M - test_out[0][1]) * mul_N1M,
           abs(predicted_NoiseDC - test_out[0][2]) * mul_NDC,
           abs(predicted_PM - test_out[0][3]) * mul_PM,
           abs(predicted_UGB*0.9916 - test_out[0][4]) * mul_UGB,
           abs(predicted_3dB_BW - test_out[0][5]) * mul_3db,
           abs(predicted_UGB - test_out[0][6]) * mul_UGB,
           abs(predicted_DC_power - test_out[0][7]) * mul_power,
           abs(predicted_DC_Gain - test_out[0][8]) * mul_gain,
           abs(predicted_SR - test_out[0][9]) * mul_SR]]

predicted_values = [[(predicted_Noise10M) * mul_N10M,
                     (predicted_Noise1M) * mul_N1M,
                     (predicted_NoiseDC) * mul_NDC,
                     (predicted_PM) * mul_PM,
                     (predicted_UGB*0.9916) * mul_UGB,
                     (predicted_3dB_BW) * mul_3db,
                     (predicted_UGB) * mul_UGB,
                     (predicted_DC_power) * mul_power,
                     (predicted_DC_Gain) * mul_gain,
                     (predicted_SR) * mul_SR]]


output_df = pd.DataFrame(np.array(predicted_output_scaled).reshape(1,9),columns= input_cols)
print("\noutput_df\n", output_df.head())

specifications_errors_df = pd.DataFrame(np.array(errors).reshape(1,10),columns= output_cols)
print("\nspecifications_errors_df\n", specifications_errors_df.head())

predicted_specifications = pd.DataFrame(np.array(predicted_values).reshape(1,10),columns= output_cols)
print("\npredicted_specifications\n", predicted_specifications.head())

final_time = time.time()

print("Total Time required to run the code : ", round(final_time - first_time,3))