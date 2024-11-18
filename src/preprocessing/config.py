import os

CHUNK_SIZE = 1_000_000
NUM_THREADS = os.cpu_count() * 4

VITALS_CSV_PATH = "data/mimic-iii/CHARTEVENTS.csv"
PATIENTS_CSV_PATH = "data/mimic-iii/DIAGNOSES_ICD.csv"
PATIENTS_PREPROCESSED_CSV_PATH = "data/mimic-iii_preprocessed/heart_failure_patients.csv"

PICKLE_INPUT_DIRS = [
    "data/mimic-iii_preprocessed/vitals_5_measurements_hf",
    "data/mimic-iii_preprocessed/vitals_5_measurements_not_hf",
]

OUTPUT_DIRS = [
    "data/mimic-iii_preprocessed",
    "data/mimic-iii_preprocessed/vitals",
    "data/mimic-iii_preprocessed/vitals_5_measurements_hf",
    "data/mimic-iii_preprocessed/vitals_5_measurements_not_hf",
]
OUTPUT_CSV_PATH = "data/mimic-iii_preprocessed/heart_failure_patients.csv"


ITEM_ID_DICT = {
    220052: "arterial_blood_pressure_mean",
    220050: "arterial_systolic_blood_pressure",
    220051: "arterial_diastolic_blood_pressure",
    220045: "heart_rate",
    220210: "respiratory_rate",
    220277: "peripheral_capillary_oxygen_saturation",
    220181: "mean_noninvasive_blood_pressure",
    220179: "noninvasive_systolic_blood_pressure",
    220180: "noninvasive_diastolic_blood_pressure",
}

TARGET_ICD9_CODES = [
    "40201",
    "40211",
    "40291",
    "40401",
    "40403",
    "40411",
    "40413",
    "40491",
    "40493",
    "4280",
    "42820",
    "42821",
    "42822",
    "42823",
    "42830",
    "42831",
    "42832",
    "42833",
    "42840",
    "42841",
    "42842",
    "42843",
    "4289",
]

VITALS_FILE_PATHS = {
    "a_bp_mean": "data/mimic-iii_preprocessed/vitals/arterial_blood_pressure_mean.csv",
    "a_dbp": "data/mimic-iii_preprocessed/vitals/arterial_diastolic_blood_pressure.csv",
    "a_sbp": "data/mimic-iii_preprocessed/vitals/arterial_systolic_blood_pressure.csv",
    "hr": "data/mimic-iii_preprocessed/vitals/heart_rate.csv",
    "ni_bp_mean": "data/mimic-iii_preprocessed/vitals/mean_noninvasive_blood_pressure.csv",
    "ni_sbp": "data/mimic-iii_preprocessed/vitals/noninvasive_systolic_blood_pressure.csv",
    "ni_dbp": "data/mimic-iii_preprocessed/vitals/noninvasive_diastolic_blood_pressure.csv",
    "resp_rate": "data/mimic-iii_preprocessed/vitals/respiratory_rate.csv",
    "SpO2": "data/mimic-iii_preprocessed/vitals/peripheral_capillary_oxygen_saturation.csv",
}
