#%%
import os, subprocess

def setup_aws_cli():
    aws_dir = os.path.expanduser("~/.aws")

    ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
    SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')

    # ensure .aws directory exists
    if not os.path.exists(aws_dir):
        os.makedirs(aws_dir)

    # write credentials file
    with open(os.path.join(aws_dir, 'credentials'), 'w') as f:
        f.write('[default]\n')
        f.write(f'aws_access_key_id = {ACCESS_KEY}\n')
        f.write(f'aws_secret_access_key = {SECRET_KEY}\n')
        # if needed
        # f.write('aws_session_token = YOUR_SESSION_TOKEN\n')

    # write config file
    with open(os.path.join(aws_dir, 'config'), 'w') as f:
        f.write('[default]\n')
        f.write(f'region = {DEFAULT_REGION}\n')
        f.write('output = json\n')
        
        
def aws_s3_sync(source_directory, destination_bucket):
    try:
        subprocess.check_output(['aws', 's3', 'sync', source_directory, destination_bucket])
        print(f'Successfully synced {source_directory} to {destination_bucket}')
    except subprocess.CalledProcessError as error:
        print(f'Failed to sync {source_directory} to {destination_bucket}. The error message is {error.output}')


def upload_result_to_s3(local_folder, bucket_name):
    setup_aws_cli()
    aws_s3_sync(local_folder, bucket_name)  