import pandas as pd

input_csv_path = "data/mimic-iii/DIAGNOSES_ICD.csv"
output_csv_path = "data/mimic-iii_preprocessed/heart_failure_patients.csv"

target_icd9_codes = [
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

filtered_rows = []

chunk_size = 1_000_000
for chunk in pd.read_csv(
    input_csv_path, chunksize=chunk_size, usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"], dtype={"ICD9_CODE": str}
):
    filtered_chunk = chunk[chunk["ICD9_CODE"].isin(target_icd9_codes)]
    filtered_rows.append(filtered_chunk)

filtered_df = pd.concat(filtered_rows, ignore_index=True)

filtered_df.drop_duplicates(subset=["SUBJECT_ID", "ICD9_CODE"], inplace=True)

unique_subject_count = filtered_df["SUBJECT_ID"].nunique()
print(f"Number of unique SUBJECT_IDs: {unique_subject_count}")

filtered_df.sort_values(by="SUBJECT_ID", inplace=True)

filtered_df.to_csv(output_csv_path, index=False)

print(f"Filtered and sorted diagnoses saved to {output_csv_path}")
