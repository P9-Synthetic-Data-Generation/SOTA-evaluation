import pandas as pd
import os

file_paths = {
    "a_bp_mean": "data/mimic-iii_preprocessed/hf_vitals/arterial_blood_pressure_mean.csv",
    "a_dbp": "data/mimic-iii_preprocessed/hf_vitals/arterial_diastolic_blood_pressure.csv",
    "a_sbp": "data/mimic-iii_preprocessed/hf_vitals/arterial_systolic_blood_pressure.csv",
    "hr": "data/mimic-iii_preprocessed/hf_vitals/heart_rate.csv",
    "ni_bp_mean": "data/mimic-iii_preprocessed/hf_vitals/mean_noninvasive_blood_pressure.csv",
    "ni_sbp": "data/mimic-iii_preprocessed/hf_vitals/noninvasive_systolic_blood_pressure.csv",
    "ni_dbp": "data/mimic-iii_preprocessed/hf_vitals/noninvasive_diastolic_blood_pressure.csv",
    "resp_rate": "data/mimic-iii_preprocessed/hf_vitals/respiratory_rate.csv",
    "SpO2": "data/mimic-iii_preprocessed/hf_vitals/peripheral_capillary_oxygen_saturation.csv",
}

filtered_dfs = {}

for name, path in file_paths.items():
    df = pd.read_csv(path)

    hadm_counts = df["HADM_ID"].value_counts()

    valid_hadm_ids = hadm_counts[hadm_counts >= 5].index

    filtered_df = df[df["HADM_ID"].isin(valid_hadm_ids)]

    sorted_df = filtered_df.sort_values(by=["HADM_ID", "CHARTTIME"])

    trimmed_df = sorted_df.groupby("HADM_ID").head(5)

    filtered_dfs[name] = trimmed_df

    print("here")

output_dir = "data/mimic-iii_preprocessed/hf_vitals_5_measurements"

os.makedirs(output_dir, exist_ok=True)

for name, df in filtered_dfs.items():
    output_file = os.path.join(output_dir, f"{name}_5_measurements.csv")

    df.to_csv(output_file, index=False)

    print(f"Saved {name} to {output_file}")
