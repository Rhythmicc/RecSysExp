
## DownloadData

```python
import os, subprocess

if not os.path.exists('./data/'):
    subprocess.call('mkdir ./data/', shell=True)

def download_data(url):
    print('Down loading data from ' + url)
    status = subprocess.call('cd ./data/ && curl -O ' + url, shell=True)
    print('Sucess!' if status == 0 else 'Error.')

if not os.path.exists('/data/ml-100k.zip'):
    download_data('https://rec-exp2.s3.didiyunapi.com/ml-100k.zip')
    status = subprocess.call('cd ./data/ && unzip ml-100k.zip', shell=True)
```
