import pandas as pd
import numpy as np

df = pd.DataFrame(
    columns=[
        "HADM_ID",
        "Arterial_BP_mean",
        "Arterial_SBP",
        "Arterial_DBP",
        "HR",
        "NI_BP_mean",
        "NI_SBP",
        "NI_DBP",
        "Resp_rate",
        "SpO2",
    ]
)

a_bp_mean = pd.read_csv("data/mimic-iii_preprocessed/arterial_blood_pressure_mean.csv")
a_dbp = pd.read_csv("data/mimic-iii_preprocessed/arterial_diastolic_blood_pressure.csv")
a_sbp = pd.read_csv("data/mimic-iii_preprocessed/arterial_systolic_blood_pressure.csv")
hr = pd.read_csv("data/mimic-iii_preprocessed/heart_rate.csv")
ni_bp_mean = pd.read_csv("data/mimic-iii_preprocessed/mean_noninvasive_blood_pressure.csv")
ni_sbp = pd.read_csv("data/mimic-iii_preprocessed/noninvasive_systolic_blood_pressure.csv")
ni_dbp = pd.read_csv("data/mimic-iii_preprocessed/noninvasive_diastolic_blood_pressure.csv")
resp_rate = pd.read_csv("data/mimic-iii_preprocessed/respiratory_rate.csv")
SpO2 = pd.read_csv("data/mimic-iii_preprocessed/peripheral_capillary_oxygen_saturation.csv")

print(
    a_bp_mean.shape,
    a_dbp.shape,
    a_sbp.shape,
    hr.shape,
    ni_bp_mean.shape,
    ni_sbp.shape,
    ni_dbp.shape,
    resp_rate.shape,
    SpO2.shape,
)
