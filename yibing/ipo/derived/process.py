import pandas as pd
from yibing.ipo.data.metadata import INV_RENAME, PROTECTED

def normalized_bid_std(g, column='IPOPrice', low_bound=0.05, high_bound=0.95):
    # g = g.drop_duplicates('INV_NAME')
    low = g.PRICE.quantile(low_bound)
    high = g.PRICE.quantile(high_bound)
    g['ratio'] = g['PRICE'].div(g[column])
    g = g[g.PRICE.gt(low) & g.PRICE.lt(high)]
    return g.ratio.std()


def merge_data(left, right, columns):
    cur_data = left.copy()
    cur_data = cur_data.merge(right[columns], on='SecCode')
    return cur_data

def replace_inv_types(data, protected=False):
    for k, v in INV_RENAME.items():
        data = data.replace(k, v)
    if not protected:
        return data
    for k in PROTECTED:
        data = data.replace(k, 'protected')
    return data

def build_inv_type_ratio(data, summary, column='IPOPrice'):
    # Average bid benchmarked on IPO Price for each type of investors
    cur_data = merge_data(data, summary, [column, 'IPOListDt'])
    cur_data['RATIO'] = cur_data["PRICE"].div(cur_data[column])
    cur_data = replace_inv_types(cur_data)

    cur_data = cur_data.reset_index().drop_duplicates(['INV_NAME', 'SecCode'])
    inv_type = cur_data.groupby(["INV_TYPE", "SecCode", 'IPOListDt']).apply(lambda x: x['RATIO'].mean()).rename('Mean').reset_index()
    return inv_type

def build_4num_ratio(mean_data, summary, column='IPOPrice'):
    # Ratio of 4 numbers to IPO price.
    mean_table = mean_data[['OflBidWAvgAll', 'OflBidWAvgMF_SS_PF_EA_Ins_QFII', 'OflBidMedAll', 'OflBidMedMF_SS_PF_EA_Ins_QFII']].div(summary[column], axis=0).dropna(thresh=1)
    mean_table = merge_data(mean_table, summary, ["IPOListDt"])
    return mean_table

def build_high_low_ratio(offline_data, summary, column='IPOPrice'):
    cur_data = merge_data(offline_data[['OflBidLowLmtYuan', 'OflBidUpLmtYuan']], summary, ['IPOListDt', column])
    cur_data['LowRatio'] = cur_data['OflBidLowLmtYuan'].div(cur_data[column])
    cur_data['HighRatio'] = cur_data['OflBidUpLmtYuan'].div(cur_data[column])
    return cur_data


def statistics(ts):
    return {
        'mean': ts.mean(),
        'std': ts.std(),
        'median': ts.median(),
        'high': ts.max(),
        'low': ts.min()
    }