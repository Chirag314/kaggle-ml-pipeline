import os
import json

def submit_to_kaggle(submission_file, competition, message="Auto submission"):
    os.system(f'kaggle competitions submit -c {competition} -f {submission_file} -m "{message}"')
