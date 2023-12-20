import pandas as pd
import os
import numpy as np
import subprocess
import time


def split_table_in_batches(table, added_name="", number_of_batches=10):
    """Split table into batches of size batch_size"""

    # Fill second column missing values with index
    table[1] = table[1].fillna(table.index.to_series()).astype(int).astype(str)
    # Split table into batches according to number_of_batches
    split_table = np.array_split(table, number_of_batches)
    # return [table[i : i + batch_size] for i in range(0, len(table), batch_size)]
    return split_table


def prepare_batches(input_batch_list, dataset_name="DDS"):
    """
    Take a list of batches and save them as .smi files in the fpadmet_results folder
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


def run_fp_admet(smi_files, smi_tables):
    os.chdir("fpadmet")
    base_path = smi_files[0].split(".")[0]
    for current_file, current_table in zip(smi_files, smi_tables):
        compound_command, commands_count = "", 1
        for current_parameter in range(1, 59):
            prepared_command = f"bash runadmet_customized.sh -f {current_file} -p {current_parameter} -a -o {base_path} & "
            predicted_file = f"{base_path}_{current_parameter}_predicted.txt"
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
    print(os.getcwd())


FPADMET_RESULTS = "/home/ec2-user/np-clinical-trials/fpadmet_results"

DDS_50_smi = pd.read_csv("support/DDS-50.smi", sep="\t", header=None)
COCONUT_smi = pd.read_csv("support/COCONUT_DB.smi", sep="\t", header=None)

DDS_split_tables = split_table_in_batches(DDS_50_smi, number_of_batches=30)
DDS_split_files = prepare_batches(DDS_split_tables, dataset_name="DDS")

run_fp_admet(DDS_split_files, DDS_split_tables)

COCONUT_split_tables = split_table_in_batches(COCONUT_smi, number_of_batches=30)
COCONUT_split_files = prepare_batches(COCONUT_split_tables)

run_fp_admet(COCONUT_split_files, COCONUT_split_tables, dataset_name="COCONUT")
