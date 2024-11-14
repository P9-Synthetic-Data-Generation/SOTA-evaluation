import pandas as pd
import os

file_paths = {
    "a_bp_mean": "data/mimic-iii_preprocessed/vitals_5_measurements/a_bp_mean_5_measurements.csv",
    "a_dbp": "data/mimic-iii_preprocessed/vitals_5_measurements/a_dbp_5_measurements.csv",
    "a_sbp": "data/mimic-iii_preprocessed/vitals_5_measurements/a_sbp_5_measurements.csv",
    "hr": "data/mimic-iii_preprocessed/vitals_5_measurements/hr_5_measurements.csv",
    "ni_bp_mean": "data/mimic-iii_preprocessed/vitals_5_measurements/ni_bp_mean_5_measurements.csv",
    "ni_sbp": "data/mimic-iii_preprocessed/vitals_5_measurements/ni_sbp_5_measurements.csv",
    "ni_dbp": "data/mimic-iii_preprocessed/vitals_5_measurements/ni_dbp_5_measurements.csv",
    "resp_rate": "data/mimic-iii_preprocessed/vitals_5_measurements/resp_rate_5_measurements.csv",
    "SpO2": "data/mimic-iii_preprocessed/vitals_5_measurements/SpO2_5_measurements.csv",
}


def find_common_hadm_ids():
    common_hadm_ids = None

    for name, path in file_paths.items():
        df = pd.read_csv(path)

        hadm_ids = set(df["HADM_ID"].unique())

        if common_hadm_ids is None:
            common_hadm_ids = hadm_ids
        else:
            common_hadm_ids &= hadm_ids

    print(f"Number of HADM_IDs found in all files: {len(common_hadm_ids)}")
    return common_hadm_ids


def split_data(common_hadm_ids):
    os.makedirs(os.path.join("data", "mimic-iii_preprocessed", "vitals_5_measurements_hf"), exist_ok=True)
    os.makedirs(os.path.join("data", "mimic-iii_preprocessed", "vitals_5_measurements_not_hf"), exist_ok=True)

    hf_admissions = set(
        pd.read_csv(os.path.join("data", "mimic-iii_preprocessed", "heart_failure_patients.csv"))["HADM_ID"]
    )

    for name, path in file_paths.items():
        hf_csv_path = os.path.join("data", "mimic-iii_preprocessed", "vitals_5_measurements_hf", f"{name}.csv")
        not_hf_csv_path = os.path.join("data", "mimic-iii_preprocessed", "vitals_5_measurements_not_hf", f"{name}.csv")

        df_vitals = pd.read_csv(path)
        unique_hadm_ids = df_vitals["HADM_ID"].unique()

        for hadm_id in unique_hadm_ids:
            if hadm_id in common_hadm_ids:
                filtered_df = df_vitals[df_vitals["HADM_ID"] == hadm_id]
                if hadm_id in hf_admissions:
                    hf_write_header = not os.path.exists(hf_csv_path)
                    filtered_df.to_csv(
                        hf_csv_path, mode="a", index=False, header=hf_write_header, columns=["HADM_ID", "VALUE"]
                    )
                else:
                    not_hf_write_header = not os.path.exists(not_hf_csv_path)
                    filtered_df.to_csv(
                        not_hf_csv_path, mode="a", index=False, header=not_hf_write_header, columns=["HADM_ID", "VALUE"]
                    )

        print(f"{name} finished")


if __name__ == "__main__":
    split_data(find_common_hadm_ids())
