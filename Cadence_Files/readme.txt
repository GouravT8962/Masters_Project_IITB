******************************************************************************************
This is the readme file for Cadence Files.
******************************************************************************************


******************************************************************************************
Note: The below steps are for aiCAS Lab server only, to work with VLSI lab servers, 
some changes in paths have to made.
******************************************************************************************


******************************************************************************************
Steps to get the Cadence files:
1. Extract the "OPAMP_RAK_Gourav.zip" file in aiCAS Lab server in 
   your working directory.
2. When extracted, Folder name "OPAMP_RAK" appears.
3. Open Cadence using the commands given by aiCAS Lab admins.
4. Import this folder as seperate library in working directory using 
   "Add Library" method in "Library Manager" of Cadence Virtuoso Software.
5. After it gets established as a normal library, we can work 
   with umc65 technology library which is already attached with it.
6. Click on Library "OPAMP_RAK", then open cell "OPAMP_1_ENDED", then 
   click on "adexl_1" to open an ADEXL window.
7. After ADEXL window opens, click on "Global Variables" on the left side 
   of window in "Data View".
8. This is where the values of varibles used in this thesis is to be defined.

The variables that need to edited in this script are:
a) nf_ibias_values - for value given by ML tool for "nf_ibias"
b) nf_Mn_drive_values - for value given by ML tool for "nf_Mn_drive"
c) nf_Mn_tail1_values - for value given by ML tool for "nf_Mn_tail1"
d) nf_Mn_tail2_values - for value given by ML tool for "nf_Mn_tail2"
e) nf_Mp_stg2_values - for value given by ML tool for "nf_Mp_stg2"
f) L_Mn_drive - for value given by ML tool for "L_Mn_drive "
g) L_Mn_tail1 - for value given by ML tool for "L_Mn_tail1"
h) L_Mp_stg1 - for value given by ML tool for "L_Mp_stg1"
i) mul_cap - for value given by ML tool for "mul_cap"
******************************************************************************************


******************************************************************************************
After this part of Cadence Virtuoso is set up, we can move to python UI 
window for generating circuit sizes from ML tool.
For Python UI window, see readme file in "Python_files/code_files" directory.
******************************************************************************************