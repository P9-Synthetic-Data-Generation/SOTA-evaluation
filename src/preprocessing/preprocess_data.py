import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from config import *


def make_data_dirs(output_dirs: list[str]):
    """
    Creates directories for storing processed data if they do not already exist.

    Args:
        output_dirs (list[str]): List of directory paths to be created.
    """
    for dir in output_dirs:
        os.makedirs(dir, exist_ok=True)

    print("Data directories created")


def filter_hf_data(
    input_csv_path: str, num_threads: int, chunk_size: int, item_id_dict: dict[int, str], output_dir: str
):
    """
    Filters heart failure-related data by ITEMID and writes the results to separate CSV files.

    Args:
        input_csv_path (str): Path to the input CSV file containing raw data.
        num_threads (int): Number of threads for concurrent processing.
        chunk_size (int): Number of rows to read per chunk.
        item_id_dict (dict[int, str]): Dictionary mapping ITEMIDs to their corresponding output file names.
        output_dir (str): Directory where the filtered data files will be saved.
    """
    print(f"Filtering heart failure data from {input_csv_path}...")

    def _process_chunk(chunk):
        for item_id in item_id_dict.keys():
            filtered_chunk = chunk[chunk["ITEMID"] == item_id]
            filtered_csv_path = os.path.join(output_dir, f"{item_id_dict[item_id]}.csv")
            write_header = not os.path.exists(filtered_csv_path)
            filtered_chunk.to_csv(filtered_csv_path, mode="a", index=False, header=write_header)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        chunk_iter = pd.read_csv(
            input_csv_path,
            chunksize=chunk_size,
            usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "VALUE", "CHARTTIME"],
        )

        futures = [executor.submit(_process_chunk, chunk) for chunk in chunk_iter]

        for future in as_completed(futures):
            future.result()

    print(f"Finished filtering heart failure data to {output_dir}")


def filter_hf_patients(input_csv_path: str, output_csv_path: str, target_icd9_codes: list[str]):
    """
    Filters patient data based on heart failure ICD-9 codes and writes the results to a CSV file.

    Args:
        input_csv_path (str): Path to the input CSV file containing patient data.
        output_csv_path (str): Path to the output CSV file for filtered patient data.
        target_icd9_codes (list[str]): List of target ICD-9 codes to filter by.
    """
    filtered_rows = []

    data = pd.read_csv(
        input_csv_path,
        usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
        dtype={"ICD9_CODE": str},
    )
    filtered_data = data[data["ICD9_CODE"].isin(target_icd9_codes)]
    filtered_rows.append(filtered_data["HADM_ID"])

    filtered_df = pd.concat(filtered_rows, ignore_index=True)
    filtered_df.drop_duplicates(inplace=True)
    filtered_df.sort_values(inplace=True)
    filtered_df.to_csv(output_csv_path, index=False)

    print(f"Finished filtering patients data to {output_csv_path}")


def filter_5_measurements(file_paths: str, input_csv_path: str, output_dirs: str):
    """
    Filters heart failure data to keep only the first five measurements for HADM_IDs that are common across all vitals.

    Args:
        file_paths (dict[str, str]): Dictionary of file paths for input data, keyed by dataset name.
        input_csv_path (str): Path to the preprocessed patients CSV file.
        output_dirs (list[str]): List of output directories for saving filtered data.
    """
    print("Filtering heart failure data to keep only five first rows with common HADM_ID...")
    filtered_dfs = {}

    for name, path in file_paths.items():
        df = pd.read_csv(path)

        hadm_counts = df["HADM_ID"].value_counts()
        valid_hadm_ids = hadm_counts[hadm_counts >= 5].index
        filtered_df = df[df["HADM_ID"].isin(valid_hadm_ids)]
        sorted_df = filtered_df.sort_values(by=["HADM_ID", "CHARTTIME"])
        trimmed_df = sorted_df.groupby("HADM_ID").head(5)
        filtered_dfs[name] = trimmed_df

    def _find_common_hadm_ids(filtered_dfs: dict[str, pd.DataFrame]) -> list[str]:
        """
        Identifies HADM_IDs common to all vitals.

        Args:
            filtered_dfs (dict[str, pd.DataFrame]): Dictionary of filtered dataframes, keyed by dataset name.

        Returns:
            set[str]: Set of common HADM_IDs found across all vitals.
        """
        common_hadm_ids = None

        for name in filtered_dfs.keys():
            df = filtered_dfs[name]

            hadm_ids = set(df["HADM_ID"].unique())

            if common_hadm_ids is None:
                common_hadm_ids = hadm_ids
            else:
                common_hadm_ids &= hadm_ids

        print(f"Number of HADM_IDs found in all files: {len(common_hadm_ids)}")
        return common_hadm_ids

    hf_admissions = set(pd.read_csv(os.path.join(input_csv_path))["HADM_ID"])

    common_hadm_ids = _find_common_hadm_ids(filtered_dfs)

    for name in filtered_dfs.keys():
        hf_csv_path = os.path.join(output_dirs[0], f"{name}.csv")
        not_hf_csv_path = os.path.join(output_dirs[1], f"{name}.csv")

        df_vitals = filtered_dfs[name]
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


def combine_5_measurements(input_dirs: list[str]):
    """
    Combines multiple 5-measurement CSV files from different directories into a single DataFrame,
    and saves the combined data along with labels into pickle files.

    This function processes CSV files from each directory in `input_dirs`, where each directory contains
    data about vital measurements. It extracts specific columns ('HADM_ID' and 'VALUE') from each file,
    and appends them into a combined DataFrame. Depending on the directory name, it assigns binary labels
    (1 for "vitals_5_measurements_hf" and 0 for "vitals_5_measurements_not_hf") to the rows.
    The combined data and labels are then saved in pickle format.

    Args:
        input_dirs (list[str]): A list of directory paths containing the 5-measurement CSV files to process.
            Each directory should contain CSV files with 'HADM_ID' and 'VALUE' columns.

    Saves:
        - A pickle file containing the combined data (features).
        - A pickle file containing the associated labels (binary: 1 for heart failure, 0 for non-heart failure).
    """
    full_df = pd.DataFrame()
    labels = []

    for dir in input_dirs:
        combined_df = pd.DataFrame()
        for filename in os.listdir(dir):
            filepath = os.path.join(dir, filename)
            df = pd.read_csv(filepath)

            if "HADM_ID" not in combined_df.columns:
                combined_df["HADM_ID"] = df["HADM_ID"]

            combined_df[filename.split(".")[0]] = df["VALUE"]

        if os.path.split(dir)[1] == "vitals_5_measurements_hf":
            labels += [1 for _ in range(0, len(combined_df), 5)]
        elif os.path.split(dir)[1] == "vitals_5_measurements_not_hf":
            labels += [0 for _ in range(0, len(combined_df), 5)]

        full_df = pd.concat([full_df, combined_df], ignore_index=True)

    data = full_df.to_numpy()
    labels = np.array(labels)

    os.makedirs(os.path.join("data", "mimic-iii_preprocessed", "pickle_data"), exist_ok=True)

    print(
        f"Data combined and saved in pickle format. Shape of data: {data.shape}. True labels: {sum(labels)}/{len(labels)}."
    )

    with open(os.path.join(os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "data.pkl")), "wb") as f:
        pickle.dump(data, f)

    with open(os.path.join(os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "labels.pkl")), "wb") as f:
        pickle.dump(labels, f)


if __name__ == "__main__":
    # make_data_dirs(output_dirs=OUTPUT_DIRS)
    # filter_hf_data(
    #     input_csv_path=VITALS_CSV_PATH,
    #     num_threads=NUM_THREADS,
    #     chunk_size=CHUNK_SIZE,
    #     item_id_dict=ITEM_ID_DICT,
    #     output_dir=OUTPUT_DIRS[1],
    # )
    # filter_hf_patients(
    #     input_csv_path=PATIENTS_CSV_PATH, output_csv_path=OUTPUT_CSV_PATH, target_icd9_codes=TARGET_ICD9_CODES
    # )
    # filter_5_measurements(
    #     file_paths=VITALS_FILE_PATHS, input_csv_path=PATIENTS_PREPROCESSED_CSV_PATH, output_dirs=OUTPUT_DIRS[3:5]
    # )
    combine_5_measurements(input_dirs=PICKLE_INPUT_DIRS)
