import pandas as pd

DICT_PATH = f"/mnt/g/Yibing/Data/lcr/MDRM/MDRM_CSV.csv"


def filter_dict(prefix=None, code=None):
    data = pd.read_csv(DICT_PATH)
    if prefix is not None:
        data = data[data.Mnemonic == prefix.upper()]
    if code is not None:
        data = data[data["Item Code"] == code]
    return data
