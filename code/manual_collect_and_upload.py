"""Manual collect and upload
During development, it is faster to do parallel computing in this capsule locally
and then collect and upload the results to the cloud. 

Usage:
1. Run process_nwbs.py to generate the nwb files to \results
2. Run this script to register a manual data asset and manually call the `collect_and_upload` capsule
"""

import os
import time
from aind_codeocean_api.codeocean import CodeOceanClient

COLLECT_AND_UPLOAD_CAPSULE_ID = '3b851d69-5e4f-4718-b0e5-005ca531aaeb'

co_client = CodeOceanClient(domain=os.getenv('API_KEY'),
                            token=os.getenv('API_SECRET'))
        

def manual_collect_and_upload():
    # Get date and time
    date_time = time.strftime("%Y%m%d-%H%M%S")
    
    # ---- Manually register data asset before this !!! ----
    result_asset_id = co_client.get_data_asset(data_asset_id)(
        # computation_id=pipeline_job_id, 
        asset_name=f'foraging_behavior_bonsai_pipeline_results_manual_{date_time}',
        mount='foraging_behavior_bonsai_pipeline_results',
        tags=['foraging', 'behavior', 'bonsai', 'hanhou', 'manual_output']
        ).json()['id']
    
    # ---- Run foraging_behavior_bonsai_pipeline_collect_and_upload_results ----
    upload_capsule_id = co_client.run_capsule(
        capsule_id=COLLECT_AND_UPLOAD_CAPSULE_ID,
        data_assets=[dict(id=result_asset_id,
                          mount='foraging_behavior_bonsai_pipeline_results')]).json()['id']
    
    if_completed = False          
    while not if_completed:
        status = co_client.get_computation(computation_id=upload_capsule_id).json()
        print(status)
        if_completed = status['state'] == 'completed' and status['end_status'] == 'succeeded'
        time.sleep(5)        
    
    print('Upload Done!')
    

if __name__ == '__main__':
    manual_collect_and_upload()