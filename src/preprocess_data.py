import pandas as pd
import numpy as np

df = pd.DataFrame(columns=['HADMID', 'Arterial_BP_mean', 'Arterial_SBP', 'Arterial_DBP', 'HR',
                           'NI_BP_mean', 'NI_SBP', 'NI_DBP', 'Resp_rate', 'SpO2'])

a_bp_mean = pd.read_csv('data/MIMIC_preprocessed/Arterial_blood_pressure_mean.csv')
#a_bp_mean.dropna(axis=0, inplace=True)
a_dbp = pd.read_csv('data/MIMIC_preprocessed/Arterial_diastolic_blood_pressure.csv')
#a_dbp.dropna(axis=0, inplace=True)
a_sbp = pd.read_csv('data/MIMIC_preprocessed/Arterial_systolic_blood_pressure.csv')
#a_sbp.dropna(axis=0, inplace=True)
hr = pd.read_csv('data/MIMIC_preprocessed/Heart_rate.csv')
#hr.dropna(axis=0, inplace=True)
ni_bp_mean = pd.read_csv('data/MIMIC_preprocessed/Mean_noninvasive_blood_pressure.csv')
#ni_bp_mean.dropna(axis=0, inplace=True)
ni_sbp = pd.read_csv('data/MIMIC_preprocessed/Noninvasive_systolic_blood_pressure.csv')
#ni_sbp.dropna(axis=0, inplace=True)
ni_dbp = pd.read_csv('data/MIMIC_preprocessed/Noninvasive_diastolic_blood_pressure.csv')
#ni_dbp.dropna(axis=0, inplace=True)
resp_rate = pd.read_csv('data/MIMIC_preprocessed/Respiratory_rate.csv')
#resp_rate.dropna(axis=0, inplace=True)
SpO2 = pd.read_csv('data/MIMIC_preprocessed/Peripheral_capillary_oxygen_saturation.csv')
#SpO2.dropna(axis=0, inplace=True)

print(a_bp_mean.shape, a_dbp.shape, a_sbp.shape, hr.shape, ni_bp_mean.shape, ni_sbp.shape, ni_dbp.shape, resp_rate.shape, SpO2.shape)