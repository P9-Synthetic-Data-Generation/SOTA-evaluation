import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

csv_file_path = 'data/MIMIC/CHARTEVENTS.csv'
output_dir = 'data/MIMIC_preprocessed'
chunk_size = 1_000_000
num_threads = 12

os.makedirs(output_dir, exist_ok=True)

item_ids = [220052, 220050, 220051, 220045, 220210, 220277, 220181, 220179, 220180]
item_dict = {
    220052: 'Arterial_blood_pressure_mean',
    220050: 'Arterial_systolic_blood_pressure',
    220051: 'Arterial_diastolic_blood_pressure',
    220045: 'Heart_rate',
    220210: 'Respiratory_rate',
    220277: 'Peripheral_capillary_oxygen_saturation',
    220181: 'Mean_noninvasive_blood_pressure',
    220179: 'Noninvasive_systolic_blood_pressure',
    220180: 'Noninvasive_diastolic_blood_pressure'
}

def process_chunk(chunk, item_id):
    filtered_chunk = chunk[chunk['ITEMID'] == item_id]
    filtered_csv_path = os.path.join(output_dir, f'{item_dict[item_id]}.csv')
    write_header = not os.path.exists(filtered_csv_path)
    filtered_chunk.to_csv(filtered_csv_path, mode='a', index=False, header=write_header)

def process_chunk_for_all_item_ids(chunk):
    for item_id in item_ids:
        process_chunk(chunk, item_id)

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    chunk_iter = pd.read_csv(csv_file_path, chunksize=chunk_size)
    
    futures = [executor.submit(process_chunk_for_all_item_ids, chunk) for chunk in chunk_iter]

    for future in as_completed(futures):
        future.result()

print("Filtering and saving completed.")