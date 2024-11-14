import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

csv_file_path = "data/mimic-iii/CHARTEVENTS.csv"
output_dir = "data/mimic-iii_preprocessed"
chunk_size = 1_000_000
num_threads = 16

os.makedirs(output_dir, exist_ok=True)

item_dict = {
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


def process_chunk(chunk, item_id):
    filtered_chunk = chunk[chunk["ITEMID"] == item_id]
    filtered_csv_path = os.path.join(output_dir, f"{item_dict[item_id]}.csv")
    write_header = not os.path.exists(filtered_csv_path)
    filtered_chunk.to_csv(filtered_csv_path, mode="a", index=False, header=write_header)


def process_chunk_for_all_item_ids(chunk):
    for item_id in item_dict.keys():
        process_chunk(chunk, item_id)


with ThreadPoolExecutor(max_workers=num_threads) as executor:
    chunk_iter = pd.read_csv(
        csv_file_path,
        chunksize=chunk_size,
        usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "VALUE"],
    )

    futures = [executor.submit(process_chunk_for_all_item_ids, chunk) for chunk in chunk_iter]

    for future in as_completed(futures):
        future.result()

print("Filtering and saving completed.")
