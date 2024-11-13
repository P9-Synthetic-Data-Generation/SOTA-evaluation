import pandas as pd

# Define the file paths
input_csv_path = 'data/MIMIC/DIAGNOSES_ICD.csv'
output_csv_path = 'data/MIMIC_preprocessed/filtered_diagnoses.csv'

# List of target ICD-9 codes
target_icd9_codes = [
    '40201', '40211', '40291', '40401', '40403', '40411', '40413', 
    '40491', '40493', '428', '2811', '42820', '42821', '42822', '42823', 
    '42830', '42831', '42832', '42833', '42840', '42841', '42842', '42843', 
    '4289'
]

# Initialize an empty list to hold filtered rows
filtered_rows = []

# Read the CSV file in chunks for memory efficiency
chunk_size = 1_000_000
for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size, dtype={'ICD9_CODE': str}):
    # Filter rows with the target ICD-9 codes
    filtered_chunk = chunk[chunk['ICD9_CODE'].isin(target_icd9_codes)]
    # Append the filtered rows to the list
    filtered_rows.append(filtered_chunk)

# Concatenate all filtered chunks into a single DataFrame
filtered_df = pd.concat(filtered_rows, ignore_index=True)

# Drop duplicate rows based on SUBJECT_ID and ICD9_CODE
filtered_df.drop_duplicates(subset=['SUBJECT_ID', 'ICD9_CODE'], inplace=True)

# Sort the DataFrame by SUBJECT_ID
filtered_df.sort_values(by='SUBJECT_ID', inplace=True)

# Save the sorted and filtered DataFrame to a new CSV file
filtered_df.to_csv(output_csv_path, index=False)

print(f"Filtered and sorted diagnoses saved to {output_csv_path}")
