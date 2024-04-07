"""Manually process all old bpod nwb files

Each time we decide to populate new analysis to the old bpod sessions, we should 
1. "Reproducible Run" this script manually (by enabling `python -u process_nwbs_bpod_manual_batch.py "$@"` in the run script)
2. Create a data asset from the results folder
3. Attache the data asset to "Collect and Upload" capsule and run it

See https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline?tab=readme-ov-file#notes-on-manually-re-process-all-nwbs-and-overwrite-s3-database-and-thus-the-streamlit-app

IMPORTANT NOTES: 
If you want to debug in VS Code rather than "Reproducible Run", you may run into a problem that nwb files cannot be loaded.
If this happens, try copying the nwb files to the /scratch folder.
See my frustrating experience here: https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-basic/issues/28#issuecomment-2041350746

"""

import glob
import os
import logging
import shutil
from pathlib import Path
import json
import sys

import multiprocessing as mp
import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s [%(filename)s:%(funcName)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(f"/root/capsule/results/capsule.log")
                              ])

from process_nwbs import process_one_nwb

logger = logging.getLogger(__name__)

def _reformat_string(s):
    if s.count('_') < 2:  # No suffix
        return s
    subject_date, time_part = s.rsplit('_', 1)
    
    if len(time_part) < 5:  # Old suffix (0, 1, 2, ...)
        return s
    
    time_part = time_part.zfill(6)
    formatted_time = f'{time_part[:2]}-{time_part[2:4]}-{time_part[4:6]}'
    return f'{subject_date}_{formatted_time}'


def get_nwb_to_process(nwb_folder, nwb_processed_folder, if_exclude_exist=True, if_exclude_error=True):
    # The simplest solution: find nwb files that have not been processed
    nwb = [f_name.split('/')[-1].split('.')[0] for f_name in glob.glob(f'{nwb_folder}/*.nwb')]
    logger.info(f'Total {len(nwb)} nwbs in {nwb_folder}')
    logger.info(f'   examples: {nwb[:3]}')

    if if_exclude_exist:
        nwb_processed = [f_name.split('/')[-1].split('.')[0] for f_name in glob.glob(f'{nwb_processed_folder}/*')]
        # Per this issue https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-basic/issues/1, 
        # I have to revert the processed file name back to json name, e.g.: 'xxxxx_2023-11-08_92908' to 'xxxxx_2023-11-08_09-29-08'
        nwb_processed = [_reformat_string(s) for s in nwb_processed]
        logger.info(f'{len(nwb_processed)} processed nwbs already in {nwb_processed_folder}')
        logger.info(f'   examples: {nwb_processed[:3]}')
    else:
        nwb_processed = []

    f_error = f'{nwb_processed_folder}/error_files.json'
    if if_exclude_error and Path(f_error).exists():
        nwb_errors = [f_name.split('/')[-1].split('.')[0] for f_name in json.load(open(f'{nwb_processed_folder}/error_files.json'))]
    else:
        nwb_errors = []
    logger.info(f'previous {len(nwb_errors)} errors to skip')
        
    nwb_to_process = [f'{nwb_folder}/{f}.nwb' for f in list(set(nwb) - set(nwb_processed) - set(nwb_errors))]
        
    return nwb_to_process
    

if __name__ == '__main__':
    
    data_folder = '/root/capsule/data/foraging_nwb_bpod'
    result_folder = '/root/capsule/results'

    # Make sure `foraging_nwb_bpod` is attached
    # nwb_file_names = glob.glob(f'{data_folder}/**/*.nwb', recursive=True)
    nwb_file_names = get_nwb_to_process(
        nwb_folder=data_folder,
        nwb_processed_folder='/root/capsule/data/foraging_nwb_bpod_processed',
    )
    
    # DEBUG OVERRIDE
    # nwb_file_names = [f'{data_folder}/447921_2019-09-11_13-56-46.nwb']  # Test bpod session
    
    logger.info(f'{len(nwb_file_names)} nwb files to process.')

    # Note that in Code Ocean, mp.cpu_count() is not necessarily the number of cores available in this session.
    # For example, even in the environment setting with 4 cores, mp.cpu_count() returns 16.
    # To really speed up when manually re-do all sessions, make sure to set core = 16 or more in the environment setting.``
    try:
        n_cpus = int(sys.argv[1])  # Input from pipeline
    except:
        n_cpus = mp.cpu_count()
    
    if n_cpus > 1:
        logger.info(f'Starting multiprocessing with {n_cpus} cores...')
        with mp.Pool(processes=n_cpus) as pool:
            jobs = [pool.apply_async(process_one_nwb, args=(nwb_file_name, result_folder)) for nwb_file_name in nwb_file_names]
            for job in tqdm.tqdm(jobs):
                job.get()
    else:
        logger.info('Starting single processing...')
        for nwb_file_name in tqdm.tqdm(nwb_file_names):
            process_one_nwb(nwb_file_name, result_folder)