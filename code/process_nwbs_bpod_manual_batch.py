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

if __name__ == '__main__':
    
    data_folder = '/root/capsule/data/foraging_nwb_bpod'
    result_folder = '/root/capsule/results'

    # Make sure `foraging_nwb_bpod` is attached
    nwb_file_names = glob.glob(f'{data_folder}/**/*.nwb', recursive=True)
    
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