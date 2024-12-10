import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from config import *
from sklearn.model_selection import train_test_split


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
    input_csv_path: str,
    num_threads: int,
    chunk_size: int,
    item_id_dict: dict[int, str],
    output_dir: str,
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


def clean_data(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Cleans the data by:
    - Ensuring the 'VALUE' column is numeric.
    - Removing NaN values and values outside the range of -500 to 500.

    Args:
        df (pd.DataFrame): Dataframe containing the heart failure data.

    Returns:
        pd.DataFrame: Cleaned dataframe with NaN and out-of-threshold values removed.
    """
    len_before = df["VALUE"].shape[0]

    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df[~df["VALUE"].apply(lambda x: pd.isna(x) or x <= 0 or x > 500)]

    len_after = df["VALUE"].shape[0]

    print(f"Cleaning data from {name}. {len_before - len_after} Rows removed.")

    return df


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

        df = clean_data(df, name)

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
                        hf_csv_path,
                        mode="a",
                        index=False,
                        header=hf_write_header,
                        columns=["HADM_ID", "VALUE"],
                    )
                else:
                    not_hf_write_header = not os.path.exists(not_hf_csv_path)
                    filtered_df.to_csv(
                        not_hf_csv_path,
                        mode="a",
                        index=False,
                        header=not_hf_write_header,
                        columns=["HADM_ID", "VALUE"],
                    )

        print(f"{name} finished")


def combine_5_measurements(input_dirs: list[str], train_test_split_ratio: float):
    """
    Processes multiple directories containing 5-measurement CSV files, combines the data into a 3D array,
    assigns binary labels based on directory names, and saves the resulting data and labels as pickle files.

    The function creates a 3D numpy array with shape (x, y, 5), where:
        - x: number of chunks (groups of 5 measurements),
        - y: number of files (CSV files processed),
        - 5: measurements in each chunk.

    Labels are binary:
        - 1 for directories named "vitals_5_measurements_hf" (indicating heart failure),
        - 0 for "vitals_5_measurements_not_hf" (indicating non-heart failure).

    Args:
        input_dirs (list[str]): List of directory paths containing the 5-measurement CSV files to process.

    Output:
        - Saves a pickle file `data.pkl` containing the 3D numpy array of measurements.
        - Saves a pickle file `labels.pkl` containing the binary labels corresponding to the directories.

    Directory structure of the saved pickle files:
        data/mimic-iii_preprocessed/pickle_data/
            - data.pkl
            - labels.pkl
    """
    data_list = []
    labels = []

    for dir in input_dirs:
        dir_data = []
        for filename in os.listdir(dir):
            filepath = os.path.join(dir, filename)
            df = pd.read_csv(filepath)

            if "VALUE" not in df.columns:
                raise ValueError(f"'VALUE' column not found in file: {filepath}")

            if "HADM_ID" in df.columns:
                df = df.drop(columns=["HADM_ID"])

            chunks = [
                df["VALUE"].iloc[i : i + 5].to_list()
                for i in range(0, len(df["VALUE"]), 5)
                if len(df["VALUE"].iloc[i : i + 5]) == 5
            ]
            dir_data.append(chunks)

        dir_data = np.array(dir_data).transpose(1, 0, 2)

        reshaped_dir_data = dir_data.reshape(dir_data.shape[0], -1)

        data_list.append(reshaped_dir_data)

        if os.path.basename(dir) == "vitals_5_measurements_hf":
            labels += [1] * dir_data.shape[0]
        elif os.path.basename(dir) == "vitals_5_measurements_not_hf":
            labels += [0] * dir_data.shape[0]

    combined_data = np.concatenate(data_list, axis=0)

    save_dir = os.path.join("data", "mimic-iii_preprocessed", "pickle_data")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "original_data.pkl"), "wb") as f:
        pickle.dump(combined_data, f)

    train_data, test_data, train_labels, test_labels = train_test_split(
        combined_data, np.array(labels), train_size=train_test_split_ratio
    )

    with open(os.path.join(save_dir, "train_data.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(save_dir, "train_labels.pkl"), "wb") as f:
        pickle.dump(train_labels, f)

    with open(os.path.join(save_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_data, f)
    with open(os.path.join(save_dir, "test_labels.pkl"), "wb") as f:
        pickle.dump(test_labels, f)

    print(f"Data and labels saved in '{save_dir}'")


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
    #     file_paths=VITALS_FILE_PATHS, input_csv_path=PATIENTS_PREPROCESSED_CSV_PATH, output_dirs=OUTPUT_DIRS[2:5]
    # )
    combine_5_measurements(input_dirs=PICKLE_INPUT_DIRS, train_test_split_ratio=0.9)
