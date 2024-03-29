import pandas as pd
import os
import numpy as np
import subprocess
import time


def split_table_in_batches(table, number_of_batches=10):
    """
    Split table into batches of size batch_size
    :param table: pandas dataframe
    :param number_of_batches: number of batches to split the table into
    """

    # Fill second column missing values with index
    try:
        table[1] = table[1].fillna(table.index.to_series()).astype(int).astype(str)
    except Exception as e:
        print("The table is already filled with index")
    # Split table into batches according to number_of_batches
    split_table = np.array_split(table, number_of_batches)
    return split_table


def prepare_batches(input_batch_list, dataset_name="DDS"):
    """
    Take a list of batches and save them as .smi files in the fpadmet_results folder
    :param input_batch_list: list containing batches of pandas dataframes
    :param dataset_name: name of the dataset
    """
    smi_files_list = []
    for index, object in enumerate(input_batch_list):
        batch_number = str(index).zfill(3)  # pad with zeros
        smi_batch_path = f"{FPADMET_RESULTS}/{dataset_name}/{batch_number}.smi"
        for current_parameter in range(1, 59):
            string_parameter = str(current_parameter)
            fp_batch_path = f"{FPADMET_RESULTS}/{dataset_name}/{batch_number}_{string_parameter}_fps.txt"
            object.to_csv(fp_batch_path, sep="\t", header=None, index=False)
        object.to_csv(smi_batch_path, sep="\t", header=None, index=False)
        smi_files_list.append(smi_batch_path)
    return smi_files_list


def run_fp_admet(smi_files):
    """
    Run fp-admet on the list of smi files
    :param smi_files: list of smi files
    """
    os.chdir("{HOME}/fpadmet")
    # base_path = smi_files[0].split(".")[0]
    for current_file in smi_files:
        compound_command, commands_count = "", 1
        current_batch_file = current_file.split(".")[0]
        for current_parameter in range(1, 59):
            prepared_command = f"bash run_fpadmet.sh -f {current_file} -p {current_parameter} -a -o {current_batch_file} & "
            predicted_file = f"{current_batch_file}_{current_parameter}_predicted.txt"
            if os.path.exists(predicted_file):
                print(f"File {predicted_file} already exists")
                continue
            compound_command += prepared_command
            commands_count += 1
            if commands_count == 5:
                result = subprocess.run(compound_command, shell=True)
                time.sleep(100)
                compound_command, commands_count = "", 1
                # Check the result
                if result.returncode == 0:
                    print("Bash command executed successfully")
                else:
                    print(f"Bash command failed with return code {result.returncode}")
    os.chdir("..")


HOME = "/home/ec2-user/np-clinical-trials"
FPADMET_RESULTS = f"{HOME}/data/output/fpadmet_results"

DDS_50_smi = pd.read_csv(f"{HOME}/data/DDS-50.smi", sep="\t", header=None)
COCONUT_smi = pd.read_csv(f"{HOME}/data/COCONUT_DB.smi", sep=" ", header=None)

DDS_split_tables = split_table_in_batches(DDS_50_smi, number_of_batches=30)
DDS_split_files = prepare_batches(DDS_split_tables, dataset_name="DDS")

run_fp_admet(DDS_split_files)

COCONUT_split_tables = split_table_in_batches(COCONUT_smi, number_of_batches=30)
COCONUT_split_files = prepare_batches(COCONUT_split_tables, dataset_name="COCONUT")

run_fp_admet(COCONUT_split_files)
