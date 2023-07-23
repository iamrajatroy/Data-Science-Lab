import requests
from tqdm import tqdm

n = 20
print(f'Running API {n} times.')

for _ in tqdm(range(n)):
    url = "http://localhost:5005/train"

    payload={
    'experiment_name': 'diabetes',
    'label': 'Outcome'}
    files=[
    ('train_file',('diabetes.csv',open('/Users/rajatroy/Desktop/MyWorkspace/data/diabetes.csv','rb'),'text/csv'))
    ]
    headers = {}
    requests.request("POST", url, headers=headers, data=payload, files=files, timeout=1200)

