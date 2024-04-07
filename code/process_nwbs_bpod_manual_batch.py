"""Manually process all old bpod nwb files

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
    scratch_folder = '/scratch/foraging_nwb_bpod'
    result_folder = '/root/capsule/results'

    # Somehow for converted old bpod nwbs, I have to copy them to scratch...
    # See https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-basic/issues/28#issuecomment-2041309175
    if not os.path.exists(scratch_folder):
        logger.info('Copying files to scratch...')
        shutil.copytree(data_folder, scratch_folder)

    # By default, process all nwb files under /data/foraging_nwb_bonsai folder
    # In the CO pipeline, upstream capsule will assign jobs by putting nwb files to this folder
    nwb_file_names = glob.glob(f'{scratch_folder}/**/*.nwb', recursive=True)
    
    # DEBUG OVERRIDE
    # nwb_file_names = ['/root/capsule/data/foraging_nwb_bpod/452272_2019-11-08_18-29-44.nwb']  # Test bpod session
    
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