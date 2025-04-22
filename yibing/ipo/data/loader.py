import os
import pandas as pd
import numpy as np
from yibing.ipo.data.metadata import (
    JQ_INFO,
    DS_INFO,
    SUMMARY_TABLE_COLMNS_MAP,
    GEM_OFFLINE_RAW_DATA_COLUMMNS_MAP,
    MEAN_TABEL_RENAME,
    SUMMARY2_TABLE,
    SUMMARY_TABLE_COLUMNS_MAP_REFRESH,
    SUBSCRIPTION_RATIO_COLUMNS,
    UMDERWRITER_COLUMNS,
)
import logging
from tqdm import tqdm
from yibing.utils.file_system import get_lib

logger = logging.getLogger(__name__)

QUOTE_PATH = "/mnt/f/Yibing/Data/_02IPO/quote"
QUOTE_NEW_PATH = "/mnt/f/Yibing/Data/_02IPO/quote_new"
IPO_DATA = "/mnt/f/Yibing/Data/_02IPO/summary/ChiNext.xlsx"
SUMMARY_PATH = ""

PROJECT = "IPO"
COLUMNS = ["TARGET", "PRICE", "AMOUNT", "INV_NAME", "INV_TYPE"]
DS_INDEX = {
    "公司名称": "DISPLAY_NAME",
    "上市时间": "START_DATE",
    "承销商": "IB",
    "发行股数（万股）": "SHARES",
    "发行价格（元）": "IPO_PRICE",
}


def _read_file(file_path):
    ticker = ".".join(file_path.split("/")[-1].split(".")[:-1])
    data = pd.read_excel(file_path, index_col=0).reset_index(drop=True)
    data.columns = COLUMNS
    data["TICKER"] = ticker
    data = data[data.TARGET.notna()]
    data = data[~data.TARGET.str.contains("数据来源：Wind")]
    return data


def load_all_data(refresh=False):
    """
    Load all data as a single dataframe
    """
    ml = get_lib(PROJECT, "data")
    if refresh or not ml.has_symbol("data/quote"):
        files = os.listdir(QUOTE_PATH)
        datas = [_read_file(os.path.join(QUOTE_PATH, f)) for f in files]
        return pd.concat(datas)
    else:
        return ml.read("data/quote").data


def load_all_data_new(refresh=False) -> pd.DataFrame:
    ml = get_lib(PROJECT, "data")
    if refresh or not ml.has_symbol("data/quote_new"):
        files = [
            os.path.join(dirpath, f)
            for dirpath, dirnames, file_names in os.walk(QUOTE_NEW_PATH)
            for f in file_names
        ]
        datas = [_read_file(f) for f in tqdm(files)]
        datas = pd.concat(datas).set_index("TICKER")
        ml.write("data/quote_new", datas)
        return datas
    else:
        data = ml.read("data/quote_new").data
        data = data.rename(columns={"TICKER": "SecCode"})  # type: ignore
        return data.set_index("SecCode")  # type: ignore


def read_metadata():
    metadata = pd.concat(
        [pd.Series(v).rename(k).rename(index=DS_INDEX) for k, v in DS_INFO.items()],
        axis=1,
    ).T
    metadata = metadata.astype(
        {"DISPLAY_NAME": str, "SHARES": np.float64, "IPO_PRICE": np.float64}
    )
    metadata["START_DATE"] = pd.to_datetime(metadata.START_DATE)
    return metadata


def refresh_data():
    lib = get_lib(PROJECT, "data")

    data = load_all_data()
    lib.write("data/quote", data)

    metadata = read_metadata()
    lib.write("data/metadata", metadata)


def read_summary():
    IPO_DATA = "/mnt/f/Yibing/Data/_02IPO/summary/ChiNext.xlsx"
    summary = pd.read_excel(IPO_DATA).rename(columns=SUMMARY_TABLE_COLMNS_MAP)
    summary = summary.dropna(thresh=1)
    summary = summary[summary.SecCode.str.contains(".SZ")]
    return summary.set_index("SecCode")


def read_offline_table():
    data = pd.read_excel(
        "/mnt/f/Yibing/Data/_02IPO/summary/GEMOflRawData.xlsx", engine="openpyxl"
    ).rename(columns=GEM_OFFLINE_RAW_DATA_COLUMMNS_MAP)
    data = data[data.SecCode.notna()]
    data = data[~data.SecCode.str.contains("Wind")]
    data = data.set_index("SecCode")
    return data


def read_mean_table():
    data = pd.read_excel(
        "/mnt/f/Yibing/Data/_02IPO/summary/GEMOflAvg.xlsx", engine="openpyxl"
    ).rename(columns=MEAN_TABEL_RENAME)
    data = data[data.SecCode.notna()]
    data = data[~data.SecCode.str.contains("Wind")]
    data = data.set_index("SecCode")
    return data


def read_summary2():
    data = pd.read_excel(
        "/mnt/f/Yibing/Data/_02IPO/summary/GEMIPOInd.xlsx", engine="openpyxl"
    ).rename(columns=SUMMARY2_TABLE)
    data = data[data.SecCode.notna()]
    data = data[~data.SecCode.str.contains("Wind")]
    data = data.set_index("SecCode")
    return data


def read_summary_refresh():
    summary = pd.read_excel("/mnt/f/Yibing/Data/_02IPO/summary/ChiNext_rr.xlsx").rename(
        columns=SUMMARY_TABLE_COLUMNS_MAP_REFRESH
    )
    summary = summary.dropna(thresh=1)
    summary = summary[summary.SecCode.str.contains(".SZ")]
    return summary.set_index("SecCode")


def read_sub_ratio():
    data = pd.read_excel("/mnt/f/Yibing/Data/_02IPO/summary/SubRatio.xlsx").rename(
        columns=SUBSCRIPTION_RATIO_COLUMNS
    )
    data = data.dropna(thresh=1)
    data = data[data.SecCode.str.contains(".SZ")]
    return data.set_index("SecCode")


def read_uw():
    data = pd.read_excel(
        "/mnt/f/Yibing/Data/_02IPO/summary/GEMIssueMethodAndUnderWritter.xlsx",
        engine="openpyxl",
    ).rename(columns=UMDERWRITER_COLUMNS)
    data = data.dropna(thresh=1)
    data = data[data.SecCode.str.contains(".SZ")]
    data = data.set_index("SecCode")
    return data


# if __name__ == "__main__":
#     refresh_data()
