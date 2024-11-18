import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from config import *


def make_data_dirs(output_dirs: list[str]):
    """
    Make the different directories to store the processed data in.

    Args:
        output_dirs (list[str]): List of the paths to the directories.
    """
    for dir in output_dirs:
        os.makedirs(dir, exist_ok=True)

    print("Data directories created")


def filter_hf_data(input_csv_path: str, num_threads: int, chunk_size: int, item_id_dict: dict[str], output_dir: str):
    """_summary_

    Args:
        input_csv_path (str): _description_
        num_threads (int): _description_
        chunk_size (int): _description_
        item_id_dict (dict[str]): _description_
        output_dir (str): _description_
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


def filter_hf_patients(input_csv_path: str, output_csv_path: str, target_icd9_codes: dict[str]):
    """_summary_

    Args:
        input_csv_path (str): _description_
        output_csv_path (str): _description_
        target_icd9_codes (dict[str]): _description_
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
    """_summary_

    Args:
        file_paths (str): _description_
        input_csv_path (str): _description_
        output_dirs (str): _description_
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

    def _find_common_hadm_ids(filtered_dfs: dict[pd.DataFrame]) -> list[str]:
        """_summary_

        Args:
            filtered_dfs (dict[pd.DataFrame]): _description_

        Returns:
            list[str]: _description_
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


if __name__ == "__main__":
    make_data_dirs(output_dirs=OUTPUT_DIRS)
    filter_hf_data(
        input_csv_path=VITALS_CSV_PATH,
        num_threads=NUM_THREADS,
        chunk_size=CHUNK_SIZE,
        item_id_dict=ITEM_ID_DICT,
        output_dir=OUTPUT_DIRS[1],
    )
    filter_hf_patients(
        input_csv_path=PATIENTS_CSV_PATH, output_csv_path=OUTPUT_CSV_PATH, target_icd9_codes=TARGET_ICD9_CODES
    )
    filter_5_measurements(
        file_paths=VITALS_FILE_PATHS, input_csv_path=PATIENTS_PREPROCESSED_CSV_PATH, output_dirs=OUTPUT_DIRS[3:5]
    )
