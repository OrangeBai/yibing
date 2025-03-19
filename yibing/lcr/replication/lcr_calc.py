import pandas as pd
from dataclasses import dataclass
from typing import Any, Union, Set, List
import ray
from lcr.replication.constants import *

@dataclass
class Node():
    param: Union[Set[str], List[str], str]

    def __call__(self, df) -> pd.Series:
        if type(self.param) == str:
            return df[self.param]
        elif type(self.param) == set:
            return df[self.param].sum(axis=1)
        elif type(self.param) == list:
            return df[self.param].max(1)
        else:
            raise ValueError

@dataclass
class NodeWrapper():
    df: pd.DataFrame
    def __call__(self, params):
        if type(params) == str:
            return self.df[params]
        elif type(params) == set:
            return self.df[params].sum(axis=1)
        elif type(params) == list:
            return self.df[params].max(1)
        else:
            raise ValueError

    def sum(self, l): 
        return sum([self.__call__(p) for p in l])
        

def compute_inflow(df):
    nd = NodeWrapper(df)
    loan_1y = nd(LOANS_AND_LEASES_1Y) + nd(REPRICING_DATA_3M) + nd(REPRICING_DATA_1Y)
    deri_inflow = nd.sum([
        DERI_INT_RATE_FAIR_VALUE_POS_HOLD_FOR_TRADING,
        DERI_FOR_EXCH_FAIR_VALUE_POS_HOLD_FOR_TRADING,
        DERI_EQU_DERI_FAIR_VALUE_POS_HOLD_FOR_TRADING,
        DERI_COM_OTHE_FAIR_VALUE_POS_HOLD_FOR_TRADING,

        DERI_INT_RATE_FAIR_VALUE_POS_HOLD_NOT_TRADING,
        DERI_FOR_EXCH_FAIR_VALUE_POS_HOLD_NOT_TRADING,
        DERI_EQU_DERI_FAIR_VALUE_POS_HOLD_NOT_TRADING,
        DERI_COM_OTHE_FAIR_VALUE_POS_HOLD_NOT_TRADING
    ])
    return loan_1y + deri_inflow

def compute_outflow(df):
    nd = NodeWrapper(df)
    uninsured_ratio = nd(UNINSURED_DEPOSIT) / nd(DEPOSIT)
    retail_deposit =  nd(TRANSACTION_IND_COR)  * S1_RETAIL + nd(TIME_DEPOSIT_LESS_250K_LESS_3M) * S1_1M_Q + nd(SAVING_DEPOSIT) * S1_RETAIL

    stable_retail_deposit = 0.05 * (1 - uninsured_ratio) * retail_deposit
    less_stable_retail_deposit = 0.10 * uninsured_ratio * retail_deposit

    operational_deposit = nd(TRANSACTION_IND_COR) * (1 - S1_RETAIL) + nd(TRANSACTION_DOM_GOV) + nd(TRANSACTION_SUB_GOV) + nd(TRANSACTION_DOM_BNK) + nd(TRANSACTION_FOR_BNK)  + nd(TRANSACTION_FOR_GOV) + nd(DEPOSIT_FOREIGN_OFC) * S1_1M_Y
    
    stable_operational_deposit = operational_deposit * 0.05 * (1 - uninsured_ratio)
    less_stable_operational_deposit = operational_deposit * 0.25 * uninsured_ratio



    
# class InflowCalculator(LCRCalculator):
#     def __init__(self, df):
#         self.df = df

#     def __call__(self):
#         load_1y = Node(LOANS_AND_LEASES_1Y)(self.df) + Node(REPRICING_DATA_3M)(self.df)
# def compute_graph(df):
import os

cur_dir = '/mnt/g/Yibing/Data/lcr/FFIEC/FFIEC CDR Call Bulk All Schedules 09302024'
files = os.listdir(cur_dir)
all_dfs = [
    pd.read_csv(os.path.join(cur_dir, file), sep='\t', low_memory=False, converters={'IDRSSD': str}, skiprows=lambda x: x in [1]).set_index('IDRSSD') for file in files if '09302024' in file and "Schedule" in file
]
df = pd.concat(all_dfs, axis=1, join='inner')
# loan_1y = extract.remote(LOANS_AND_LEASES_1Y, df)
# ray.get(loan_1y)
