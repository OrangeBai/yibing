import zipfile
import pandas as pd
import requests
import os
from itertools import product 
FRY9C_DOWNLOAD_PREFIX = "http://www.ffiec.gov/npw/FinancialReport/ReturnBHCFZipFiles/?zipfilename="
FRY9C_DOWNLOAD_PATH = '/mnt/g/Yibing/Data/lcr/FFIEC/fry9c_zip'
FRY9C_PATH = '/mnt/g/Yibing/Data/lcr/FFIEC/fry9c'
CALL_REPORT_DOWNLOAD_PATH = '/mnt/g/Yibing/Data/lcr/FFIEC/call_report_zip'
CALL_REPORT_PATH = '/mnt/g/Yibing/Data/lcr/FFIEC/call_report'
FRY9C_QUARTER_MAP = {
    1: '0331',
    2: '0630',
    3: '0930',
    4: '1231'
}
HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
}



def _download_fry_9c(year, quarter):
    quarter_date = FRY9C_QUARTER_MAP[quarter]
    file_name = f'BHCF{year}{quarter_date}.ZIP'
    response = requests.get(FRY9C_DOWNLOAD_PREFIX + file_name, stream=True, headers=HEADER)  # Stream the response
    os.makedirs(FRY9C_DOWNLOAD_PATH, exist_ok=True)
    with open(f"{FRY9C_DOWNLOAD_PATH}/{file_name}", "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
            file.write(chunk)

def _unzip_files(source, destination):
    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(destination)
    return

def unzip_fry9c():
    # for file in
        pass


if __name__ == "__main__":
    # for args in product(range(2011, 2023), range(1, 5)):
    #     _download_fry_9c(*args)
    pass
