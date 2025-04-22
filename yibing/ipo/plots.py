# %% init
from yibing.ipo.data.metadata import INV_RENAME
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from yibing.ipo.data.metadata import (
    DS_INFO,
    JQ_INFO,
    GEM_OFFLINE_RAW_DATA_COLUMMNS_MAP,
    INV_RENAME,
)
from yibing.ipo.data.loader import (
    load_all_data_new,
    read_offline_table,
    read_mean_table,
    read_summary_refresh,
    read_sub_ratio,
    read_uw 
)
from yibing.utils.file_system import get_lib
from yibing.ipo.derived.process import normalized_bid_std, merge_data, replace_inv_types, build_4num_ratio, check_valid
import pandas as pd
B1 = pd.Timestamp('2021-10-20')
B2 = pd.Timestamp('2023-10-20')
lib = get_lib("ipo", "data")
data = load_all_data_new(False)
summary = read_summary_refresh()
mean_data = read_mean_table()
offline_data = read_offline_table()
sub_ratio = read_sub_ratio()
uw = read_uw()

# %% Std plot
import seaborn as sns

cur_data = merge_data(data, summary, ['IPOPrice'])
std = cur_data.groupby('SecCode').apply(normalized_bid_std).rename('STD').to_frame()
std_ts = merge_data(std, summary, ['IPOListDt']).set_index("IPOListDt").STD.sort_index()
ax = sns.scatterplot(std_ts) # type: ignore
ax.axvline(pd.Timestamp('2021-10-20'), color='black') # type: ignore

# %% black pink
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from yibing.ipo.derived.process import build_inv_type_ratio, build_4num_ratio, build_high_low_ratio


summary['Day1ClosePrice'] = summary['IPOPrice'] * (1 + summary['Day1ChgPct'] / 100)
inv_type_ratio = build_inv_type_ratio(data, summary, "Day1ClosePrice")
min_4num_ratio = build_4num_ratio(mean_data, summary, 'Day1ClosePrice').set_index("IPOListDt").sort_index()
high_low_ratio = build_high_low_ratio(offline_data, summary, "Day1ClosePrice").set_index("IPOListDt").sort_index()
ipo_price_ratio = summary['IPOPrice'].div(summary['Day1ClosePrice']).to_frame().merge(summary["IPOListDt"], on='SecCode').set_index('IPOListDt')
fix, ax = plt.subplots(2, 1, figsize=(18, 12))
sns.scatterplot(inv_type_ratio, x='IPOListDt', y='Mean', hue='INV_TYPE', ax=ax[0])

ax[0].axvline(pd.Timestamp('2021-10-20'), color='black', linewidth=2)
ax[0].axhline(1, color='pink', linewidth=2)
ax[0].plot(min_4num_ratio['OflBidWAvgMF_SS_PF_EA_Ins_QFII'], label='Avg MF_SS_PF_EA', ls='--', color='black', linewidth=3, alpha=0.7)

ax[0].plot(ipo_price_ratio.sort_index(), label='IPOPrice', linewidth=2)
# ax[0].plot(min_4num_ratio.set_index("IPOListDt").sort_index()['OflBidWAvgAll'], label='Avg All', linewidth=2)
# ax[0].plot(high_low_ratio['HighRatio'], linestyle='-.', color='pink', linewidth=2, alpha=0.7, label='High Cutoff')
# ax[0].plot(high_low_ratio['LowRatio'], linestyle='--', color='black', label='Low Cutoff')
ax[0].set_xlim(summary.IPOListDt.min(), summary.IPOListDt.max())

ax[0].legend()
ax[0].set_title('Mean of Bid/IPO Price')

day1_ret = summary[['Day1ChgPct', 'IPOListDt']].copy()
# day1_ret['Day1ChgPct'] = np.log( 1 + day1_ret["Day1ChgPct"] / 100) 
sns.scatterplot(day1_ret, x='IPOListDt', y='Day1ChgPct', ax=ax[1])
ax[1].axvline(pd.Timestamp('2021-10-20'), color='black', linewidth=2)
ax[1].axhline(0, color='pink', linewidth=2)

ax[1].set_xlim(summary.IPOListDt.min(), summary.IPOListDt.max())
ax[1].set_title("First Day Return")

# %% Plot one IPO per month
summary['ym'] = summary.IPOListDt.dt.strftime('%Y-%m')
tickers = summary.groupby('ym').apply(lambda x: list(x.index))

for ym, tks in tickers.items():
    for tk in tks:
        if tk in data.index:
            cur_d = data.loc[tk]
            ax = cur_d.hist(bins=30, figsize=(16, 8))
            ax[0][0].get_figure().suptitle(f"{ym} | {tk}")
            break

# %% Plot Q std
def cal_q_std(g):
    return g['AMOUNT'].div(g.AMOUNT.max()).std()

ax = data.groupby(level=0).apply(cal_q_std).rename('Q_STD').to_frame().merge(summary['IPOListDt'], left_index=True, right_index=True).set_index("IPOListDt").sort_index().plot()
ax.axvline(pd.Timestamp('2021-10-20'), color='black')


# %%  Q STD per group
def cal_std_group(g):
    q_max = g.AMOUNT.max()
    return g.groupby("INV_TYPE").apply(lambda x: pd.Series({'Q_STD': (x.AMOUNT / q_max).std(), 'N_INV': len(x)}))

groupby_q = data.copy().groupby(level=0).apply(cal_std_group).reset_index(level=1)
groupby_q = merge_data(groupby_q, summary, ['IPOListDt']).pipe(replace_inv_types)
ax = sns.scatterplot(groupby_q[groupby_q.Q_STD.notna()], x="IPOListDt", y="Q_STD", hue="INV_TYPE")
ax.axvline(pd.Timestamp('2021-10-20'), color='black')
fig = ax.get_figure()
fig.suptitle('Q STD')
fig.set_size_inches(16, 8)


# %% Plot Number of Institutions
cur_data = data.copy().reset_index().drop_duplicates(['SecCode', 'INV_NAME'])
cur_data = cur_data.groupby(["SecCode", "INV_TYPE"]).size().rename('Number').reset_index(level=1)
cur_data = merge_data(cur_data, summary, "IPOListDt").pipe(replace_inv_types)
ax = sns.scatterplot(
    cur_data, x='IPOListDt', y="Number", hue="INV_TYPE"
)
fig = ax.get_figure()
fig.set_size_inches(16, 8)

# %% Statistics
def statistics(ts):
    return {
        'mean': ts.mean(),
        'std': ts.std(),
        'median': ts.median(),
        'high': ts.max(),
        'low': ts.min(),
        'N': len(ts)
    }
cur_data = merge_data(data, summary, ["IPOListDt"])
d1 = cur_data[cur_data.IPOListDt.lt(B1)]
d2 = cur_data[cur_data.IPOListDt.gt(B1) & cur_data.IPOListDt.lt(B2)]
d3 = cur_data[cur_data.IPOListDt.gt(B2)]
ss = pd.DataFrame(
    {
        'Price - 1': statistics(d1.PRICE),
        'Price - 2': statistics(d2.PRICE),
        'Price - 3': statistics(d3.PRICE),
        'Amount - 1': statistics(d1.AMOUNT),
        'Amount - 2': statistics(d2.AMOUNT),
        'Amount - 3': statistics(d3.AMOUNT),
    }
)


d1_p = d1.groupby(["SecCode", 'INV_TYPE']).apply(lambda x: pd.Series(statistics(x.drop_duplicates('INV_NAME').PRICE))).reset_index(level=1).pipe(replace_inv_types).pipe(lambda x: merge_data(x, summary, ["IPOListDt"]))

d2_p = d2.groupby(["SecCode", 'INV_TYPE']).apply(lambda x: pd.Series(statistics(x.drop_duplicates('INV_NAME').PRICE))).reset_index(level=1).pipe(replace_inv_types).pipe(lambda x: merge_data(x, summary, ["IPOListDt"]))

d1_q = d1.groupby(["SecCode", 'INV_TYPE']).apply(lambda x: pd.Series(statistics(x.drop_duplicates('INV_NAME').AMOUNT))).reset_index(level=1).pipe(replace_inv_types).pipe(lambda x: merge_data(x, summary, ["IPOListDt"]))

d2_q = d2.groupby(["SecCode", 'INV_TYPE']).apply(lambda x: pd.Series(statistics(x.drop_duplicates('INV_NAME').AMOUNT))).reset_index(level=1).pipe(replace_inv_types).pipe(lambda x: merge_data(x, summary, ["IPOListDt"]))



# %% Day1 Change Pct
from matplotlib import pyplot as plt

temp = mean_data[['OflBidWAvgMF_SS_PF_EA_Ins_QFII', 'OflBidWAvgAll', 'OflBidMedMF_SS_PF_EA_Ins_QFII', 'OflBidMedAll']].min(1).rename('Min').to_frame().merge(summary[['IPOListDt', 'IPOPrice', 'Day1ChgPct']], left_index=True, right_index=True)
temp['Day1NormedChange'] = ((temp['Day1ChgPct'] / 100 + 1) * temp['IPOPrice'] / temp['Min'] - 1 ) * 100

fig, ax = plt.subplots(1, 1, figsize=(16, 6))
sns.scatterplot(temp, x='IPOListDt', y='Day1ChgPct', ax=ax)
sns.scatterplot(temp, x='IPOListDt', y='Day1NormedChange', ax=ax, alpha=0.5, size=0.1)

# %% Over Subscription
import seaborn as sns
ax = sns.scatterplot(sub_ratio, x='IPOListDt', y='OflOverSubMultiple')
ax.axvline(B1, color='black', linewidth=2)
ax.axvline(B2, color='black', linewidth=2)
ax.set_title("Over subscription")
ax.grid()
fig = ax.get_figure()
fig.set_size_inches(16, 6)
# %% check corr-bid
cur_data = data.groupby(['SecCode', 'INV_NAME']).apply(lambda x: x.PRICE.mean() if len(x.PRICE.unique()) == 1 else np.nan)
cur_data = cur_data.rename('PRICE').reset_index()
cur_data = cur_data[cur_data.SecCode.isin(list(offline_data.index))]
cur_data = merge_data(cur_data, summary, ['IPOListDt', 'IPOPrice'])
cur_data = merge_data(cur_data, offline_data, ['OflBidLowLmtYuan', 'OflBidUpLmtYuan'])
std = cur_data.groupby('SecCode').apply(normalized_bid_std).rename('Std').to_frame()
cur_data = merge_data(cur_data, std, ['Std'])
cur_data['Ratio'] = cur_data['PRICE'].div(cur_data["IPOPrice"])
# cur_data = cur_data.set_index("SecCode")

def remove_outlier_z_score(ts, p):
    # ts = ts[ts.le(ts.quantile()) & ts.ge(ts.quantile(0.05))]
    return ts / ts.mean()

def check_corr(cur_data, gt):
    df = cur_data.pivot_table(index="SecCode", columns="INV_NAME", values='Ratio')
    cols = df.count().gt(gt).replace(False, np.nan).dropna().index
    # df = df.div(cur_data.IPOPrice.drop_duplicates(), axis=0)[cols]
    df = df[cols].apply(lambda x: (x - x.mean()).div(x.std()), axis=1)
    corr = df[cols].corr()
    corr = corr.where(np.triu(np.ones(corr.shape[0]), k=1).astype(bool)).stack().rename_axis(['inv1', 'inv2']).rename('corr')
    return corr

fig, ax = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
# cur_data = cur_data.set_index("SecCode")
corr_all = check_corr(cur_data, 100).hist(ax=ax[0][0])
corr_p1 = check_corr(cur_data[cur_data.IPOListDt < pd.Timestamp('2021-10-20')], 100).hist(ax=ax[0][1])
corr_p2 = check_corr(cur_data[(cur_data.IPOListDt < pd.Timestamp('2023-10-20')) & (cur_data.IPOListDt > pd.Timestamp('2021-10-20'))], 100).hist(ax=ax[1][0])
corr_p3 = check_corr(cur_data[cur_data.IPOListDt > pd.Timestamp('2023-10-20')], 30).hist(ax=ax[1][1])
ax[0][0].set_title('All')
ax[0][1].set_title('before  2021-10-20')
ax[1][0].set_title('between  2021-10-20 to 2023-10-20')
ax[1][1].set_title('after 2023-10-20')
fig.suptitle('Ratio')

# def check_win(g):
#     name = g.name
#     win = g.PRICE.le(offline_data.loc[name, 'OflBidUpLmtYuan']) & g.PRICE.ge(offline_data.loc[g.name, 'OflBidLowLmtYuan'])
#     g['win'] = win
#     return g

# df = cur_data.groupby('SecCode').apply(check_win).reset_index(drop=True).pivot(index='SecCode', columns='INV_NAME',values='win')
# cols = df.count().gt(100).replace(False, np.nan).dropna().index
# cur_df = df[cols].replace(True, 1).replace(False, 0)
# ax = cur_df.corr().stack().sort_values().replace(1, np.nan).dropna().hist()
# ax.set_title('Win pairs')

# %%
corr = check_corr(cur_data[(cur_data.IPOListDt < pd.Timestamp('2023-10-20')) & (cur_data.IPOListDt > pd.Timestamp('2021-10-20'))], 100)
high_corr = corr[corr > 0.90].reset_index()
high_players = (high_corr.inv1.value_counts() > 5).replace(False, np.nan).dropna().index
high_players = high_players.union((high_corr.inv2.value_counts() > 5).replace(False, np.nan).dropna().index)

def plot_players(player):
    high_corr[(high_corr.inv1 == player) | (high_corr.inv2 == player)].sort_values(by='corr')
    col_players = high_corr[(high_corr.inv1 == player) | (high_corr.inv2 == player)].sort_values(by='corr').tail(5)
    col_players = list(col_players.inv1) + list(col_players.inv2)
    cc_data = cur_data[cur_data.INV_NAME.isin(col_players)]
    cc_data = cc_data[cc_data.Ratio.notna()]
    cc_data = cc_data.sort_values('IPOListDt')

    def fill_1std(g):
        return pd.Series({
            'Upper': g.Ratio.mean() + g.Std.iloc[0],
            "lower": g.Ratio.mean() - g.Std.iloc[0],
            "IPOListDt": g.IPOListDt.iloc[0]
        })
    cc_1sigma = cc_data.groupby('SecCode').apply(fill_1std).sort_values("IPOListDt")
    fig, ax = plt.subplots(1, 1, figsize=(16, 8)) 
    
    ax = sns.scatterplot(cc_data.reset_index(), x='IPOListDt', y='Ratio', hue='INV_NAME', legend=False)
    ax.plot(cc_data["IPOListDt"], cc_data["OflBidUpLmtYuan"].div(cc_data['IPOPrice']), color='black', alpha=0.5, label='high')
    ax.plot(cc_data["IPOListDt"], cc_data["OflBidLowLmtYuan"].div(cc_data['IPOPrice']), color='black', alpha=0.5, label='low')

    ax.fill_between(cc_1sigma['IPOListDt'], cc_1sigma["lower"], cc_1sigma['Upper'], color='pink', alpha=0.5, label='1 sigma')
    # ax.plot(1 - cc_data['Std'], color='pink', alpha=0.5)
    ax.set_ylim(0.5, 2)
    ax.grid()
    ax.legend()
for i, p in enumerate(high_players):
    if i < 20:
        plot_players(p)


# %% Bids after high/low
# cur_data = data.reset_index().groupby(['SecCode', 'INV_NAME']).apply(lambda x: x if len(x.PRICE.unique()) == 1 else pd.DataFrame()).reset_index(drop=True).drop_duplicates(['SecCode', 'INV_NAME']).set_index('SecCode')
# cur_data = merge_data(cur_data, offline_data, ["IPOListDt", "OflBidUpLmtYuan", "OflBidLowLmtYuan"])
# cur_data = merge_data(cur_data, summary, ['IPOPrice'])
# lib.write('data/quote_new/remove_multiple_quote', cur_data)
ipo_seq = summary["IPOListDt"]

cur_data = lib.read('data/quote_new/remove_multiple_quote').data
cur_d2 = cur_data[cur_data.IPOListDt.gt(pd.Timestamp('2021-10-20')) & cur_data.IPOListDt.lt(pd.Timestamp('2023-10-31'))]
idx =cur_d2.INV_NAME.value_counts().gt(150).replace(False, np.nan).dropna().index
cur_d2 = cur_d2[cur_d2.INV_NAME.isin(list(idx))]

def find_next_5(ipo_seq, ticker):
    idx = ipo_seq.index.get_loc(ticker)
    if idx < len(ipo_seq) - 6:
        return list(ipo_seq.index[idx + 1: min(idx + 6, len(ipo_seq))])
    else:
        return []

all_bid_high = {}
for n, g in cur_d2.groupby('INV_NAME'):
    high = g[g.PRICE.gt(g.OflBidUpLmtYuan)]
    if high.empty:
         continue
    cur_ipo_seq = ipo_seq[ipo_seq.index.intersection(g.index.unique())]
    for _, cur_high in high.iterrows():
        cur_ratio = cur_high.PRICE / cur_high.IPOPrice
        next_5bids = find_next_5(cur_ipo_seq, cur_high.name) 
        all_bid_high[(n, cur_high.name)] = pd.Series([cur_ratio] + list(g.loc[next_5bids, 'PRICE'] / g.loc[next_5bids, 'IPOPrice']))

all_bid_low = {}
for n, g in cur_d2.groupby('INV_NAME'):
    high = g[g.PRICE.lt(g.OflBidLowLmtYuan)]
    if high.empty:
         continue
    cur_ipo_seq = ipo_seq[ipo_seq.index.intersection(g.index.unique())]
    for _, cur_high in high.iterrows():
        cur_ratio = cur_high.PRICE / cur_high.IPOPrice
        next_5bids = find_next_5(cur_ipo_seq, cur_high.name) 
        all_bid_low[(n, cur_high.name)] = pd.Series([cur_ratio] + list(g.loc[next_5bids, 'PRICE'] / g.loc[next_5bids, 'IPOPrice']))


all_bid_low = pd.DataFrame(all_bid_low)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(all_bid_low.mean(axis=1))
ax.fill_between(all_bid_low.index, all_bid_low.mean(axis=1) + 0.5 * all_bid_low.std(axis=1), all_bid_low.mean(axis=1) - 0.5 * all_bid_low.std(axis=1), alpha=0.2)
ax.grid()
ax.set_title('5 Bids after ruled out by bidding too low | benchmarked on IPO Price')

all_bid_high = pd.DataFrame(all_bid_high)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(all_bid_high.mean(axis=1))
ax.fill_between(all_bid_high.index, all_bid_high.mean(axis=1) + 0.5 * all_bid_high.std(axis=1), all_bid_high.mean(axis=1) - 0.5 * all_bid_high.std(axis=1), alpha=0.2)
ax.grid()
ax.set_title('5 Bids after ruled out by bidding too high | benchmarked on IPO Price')

# %% new players
cur_data = lib.read('data/quote_new/remove_multiple_quote').data
cur_d2 = cur_data[cur_data.IPOListDt.gt(pd.Timestamp('2021-10-20')) & cur_data.IPOListDt.lt(pd.Timestamp('2023-10-31'))]
cur_d2['Win'] = cur_d2["PRICE"].gt(cur_d2['OflBidLowLmtYuan']) & cur_d2["PRICE"].lt(cur_d2['OflBidUpLmtYuan'])
cc = cur_d2.pivot(columns='INV_NAME', values='Win')

ncc = cc.count()

players3 = ncc[ncc > 200].index
# players2 = ncc[(ncc <= 200) & (ncc > 100)].index
# players1 = ncc[(ncc <= 100) & (ncc > 50)].index
players0 = ncc[ncc <= 200].index

win_rate = pd.concat([
    cc[p].sum(1) / cc[p].count(1).replace(0, np.nan) for p in [players0, players3]
], keys=['0', '1', '2', '3'], axis=1)

sns.histplot(win_rate, kde=True)

# %%
cur_data = lib.read('data/quote_new/remove_multiple_quote').data
cur_d2 = cur_data[cur_data.IPOListDt.gt(pd.Timestamp('2021-10-20')) & cur_data.IPOListDt.lt(pd.Timestamp('2023-10-31'))]
cur_d2['Win'] = cur_d2["PRICE"].gt(cur_d2['OflBidLowLmtYuan']) & cur_d2["PRICE"].lt(cur_d2['OflBidUpLmtYuan'])
cc = cur_d2.pivot(columns='INV_NAME', values='Win')

win_rate_table = pd.concat([
    (cc.sum() / cc.count()).rename('Win Rate'), 
    cc.count().rename('Played')
    ], axis=1
).replace(0, np.nan)
ax = win_rate_table.pipe(np.log).plot.scatter(x='Win Rate', y='Played')
ax.set_title(f"Corr: {win_rate_table['Win Rate'].corr(win_rate_table['Played'])}")

# ncc = cc.count()
# players3 = ncc[ncc > 200].index
# # players2 = ncc[(ncc <= 200) & (ncc > 100)].index
# # players1 = ncc[(ncc <= 100) & (ncc > 50)].index
# players0 = ncc[ncc <= 200].index

# win_rate = pd.concat([
#     cc[p].sum(1) / cc[p].count(1).replace(0, np.nan) for p in [players0,  players3]
# ], keys=['0', '1', '2', '3'], axis=1)

# %% True values
cur_data = lib.read('data/quote_new/remove_multiple_quote').data
cur_data = cur_data[cur_data.IPOListDt.gt(B2)]
summary['Day1ClosePrice'] = summary['IPOPrice'] * (1 + summary['Day1ChgPct'] / 100)
cur_data = merge_data(cur_data, summary, ["Day1ClosePrice"])

overprice = cur_data[cur_data['IPOPrice'] > cur_data['Day1ClosePrice']]
w_op = overprice[overprice.PRICE < overprice.OflBidLowLmtYuan].groupby("INV_NAME").size()
l_op = overprice[overprice.PRICE > overprice.OflBidLowLmtYuan].groupby("INV_NAME").size()

underprice = cur_data[cur_data['IPOPrice'] < cur_data['Day1ClosePrice']]
w_up = underprice[underprice.PRICE.lt(underprice.OflBidUpLmtYuan) & (underprice.PRICE.gt(underprice.OflBidLowLmtYuan))].groupby("INV_NAME").size()
l_up = underprice[underprice.PRICE.gt(underprice.OflBidUpLmtYuan) | (underprice.PRICE.lt(underprice.OflBidLowLmtYuan))].groupby("INV_NAME").size()


total_bids = cur_data.groupby("INV_NAME").size()
win_rate = pd.concat([w_op, l_op, w_up, l_up, total_bids], keys=['W_OP', 'L_OP', 'W_UP', 'L_UP', 'total_bids'], axis=1)

win_rate['OP_win_rate'] = win_rate['W_OP'] / (win_rate['W_OP'] + win_rate['L_OP'])
win_rate['UP_win_rate'] = win_rate['W_UP'] / (win_rate['W_UP'] + win_rate["L_UP"])
win_rate['Total_win_rate'] = (win_rate['W_OP'] + win_rate['W_UP']).div(total_bids)
# win_rate['Total_win_rate'] = (win_rate['W_UP'] + win_rate['W_OP']) / 
win_rate = win_rate.reset_index().merge(data.reset_index()[['INV_NAME', 'INV_TYPE']].drop_duplicates(), left_on='INV_NAME', right_on='INV_NAME')

win_rate = win_rate.pipe(replace_inv_types)
win_rate['Q'] = win_rate.total_bids.apply(lambda x: x // 100)
# pt, ax = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
# for i, v in enumerate(INV_RENAME.values()):
#     ii = i // 3
#     jj = i % 3
#     cur_ax = ax[ii][jj]
#     sns.scatterplot(win_rate[win_rate.INV_TYPE == v], x='OP_win_rate', y='UP_win_rate', hue='INV_TYPE', ax=cur_ax)

# %% 4 num
cur_data = lib.read('data/quote_new/remove_multiple_quote').data
# cur_data = cur_data[cur_data.IPOListDt.ge(B1) & cur_data.IPOListDt.le(B2)]

cur_data = merge_data(cur_data, mean_data, ['OflBidWAvgAll', 'OflBidWAvgMF_SS_PF_EA_Ins_QFII', 'OflBidMedAll', 'OflBidMedMF_SS_PF_EA_Ins_QFII'])
# min_4num = build_4num_ratio(mean_data, summary)

def _find_between(g):
    min_4num = g[['OflBidWAvgAll', 'OflBidWAvgMF_SS_PF_EA_Ins_QFII', 'OflBidMedAll', 'OflBidMedMF_SS_PF_EA_Ins_QFII']].min(axis=1)
    protected = (g.PRICE.ge(g.IPOPrice) & g.PRICE.le(min_4num)).sum()
    num = len(g)
    return pd.Series({
        "Protected": protected,
        "Num": num,
        'Ratio': protected / num
        })

grouped = cur_data.groupby(['SecCode', 'INV_TYPE', 'IPOListDt']).apply(_find_between).reset_index().pipe(replace_inv_types)
ptc = grouped.pivot_table(index="IPOListDt", values='Protected', columns='INV_TYPE')
num = grouped.pivot_table(index="IPOListDt", values='Num', columns='INV_TYPE')
ratio = grouped.pivot_table(index="IPOListDt", values='Ratio', columns='INV_TYPE')

from matplotlib import pyplot as plt
fig, axs = plt.subplots(3, 3, figsize=(24, 12))
for i, itype in enumerate(INV_RENAME.values()):
    ii, jj = i // 3, i % 3
    g = grouped[grouped == itype]
    axs[ii][jj].plot(ptc[itype], label='protected')
    axs[ii][jj].plot(num[itype], label='total num')
    axs[ii][jj].twinx().fill_between(ratio.index, 0, ratio[itype], label='ratio', color='grey', alpha=0.6)
    axs[ii][jj].axvline(B1, color='black')
    axs[ii][jj].axvline(B2, color='black')
    axs[ii][jj].set_title(itype)
    axs[ii][jj].grid()

# %% 
cur_data = lib.read('data/quote_new/remove_multiple_quote').data
cur_data = cur_data[cur_data.IPOListDt.gt(B1) & cur_data.IPOListDt.lt(B2)]
summary['Day1ClosePrice'] = summary['IPOPrice'] * (1 + summary['Day1ChgPct'] / 100)
cur_data = merge_data(cur_data, summary, ["Day1ClosePrice"])
cur_data['Log Ret'] = (np.log(cur_data.PRICE) - np.log(cur_data.Day1ClosePrice)).abs()
cur_data["Win Rate"] = cur_data.PRICE.gt(cur_data['Day1ClosePrice'] * 0.90) & cur_data.PRICE.lt(cur_data['Day1ClosePrice'] * 1.1)

win_rate1 = cur_data.groupby("INV_NAME").apply(lambda x: pd.Series({'win rate': x['Win Rate'].sum() / len(x), 'mea': x['Log Ret'].abs().mean(), 'played': len(x)}))


# %% 
cur_data = lib.read('data/quote_new/remove_multiple_quote').data
cur_data = cur_data[cur_data.IPOListDt.gt(B1) & cur_data.IPOListDt.lt(B2)]
summary['Day1ClosePrice'] = summary['IPOPrice'] * (1 + summary['Day1ChgPct'] / 100)
cur_data = merge_data(cur_data, summary, ["Day1ClosePrice"])
cur_data['fall'] = cur_data.IPOPrice.gt(cur_data.Day1ClosePrice)
cur_data['bid_within'] = cur_data.PRICE.lt(cur_data.OflBidUpLmtYuan) & cur_data.PRICE.gt(cur_data.OflBidLowLmtYuan)
cur_data['Win'] = (cur_data.fall & cur_data.PRICE.lt(cur_data.OflBidLowLmtYuan)) | (~cur_data.fall & cur_data['bid_within'])

invs = cur_data.groupby('INV_NAME').size().gt(200).replace(False, np.nan).dropna().index
cur_data = cur_data[cur_data.INV_NAME.isin(list(invs))]

def _build(g):
    g = g.sort_values('IPOListDt').reset_index()
    g['chunk'] = g.index // 40
    return g.groupby('chunk').apply(lambda x: x.bid_within.sum() / len(x))

res = cur_data.groupby('INV_NAME').apply(_build)
res = res.rename('WR').reset_index()

axs = res.pivot(index='INV_NAME', columns='chunk', values="WR").hist()
fig = axs[0][0].get_figure()
fig.set_size_inches(12, 12)
fig.suptitle('Win Rate every 50 IPOs')
fig.tight_layout(rect=(0, 0, 1, 0.97))

# %% pairs
cur_data = lib.read('data/quote_new/remove_multiple_quote').data
cur_data = cur_data[cur_data.IPOListDt.gt(B2)]
ipo_numbers = uw[uw.IPOListDt.gt(B2)].IPOLeadUW.value_counts().sort_values()

cur_data = merge_data(cur_data, uw, ['IPOLeadUW', 'IPOIssueMethod'])
cur_data['Win'] = cur_data["PRICE"].ge(cur_data['OflBidLowLmtYuan']) & cur_data["PRICE"].le(cur_data['OflBidUpLmtYuan'])
pairs = cur_data.groupby(['INV_NAME', 'IPOLeadUW']).apply(lambda x: pd.Series({'Win Rate': x.Win.sum() / len(x), 'N_Pair': len(x)}))
pairs = pairs.reset_index().merge(ipo_numbers, left_on='IPOLeadUW', right_on='IPOLeadUW')
# invs1 = pairs[(pairs.IPOLeadUW == '华泰联合证券有限责任公司') & (pairs.N_Pair > 5) & (pairs['Win Rate'] ==1)].INV_NAME

pairs[(pairs.IPOLeadUW == '华泰联合证券有限责任公司') & (pairs.N_Pair > 3)].sort_values('Win Rate')['Win Rate'].hist()
# %%
cur_data = lib.read('data/quote_new/remove_multiple_quote').data
cur_data = cur_data[cur_data.IPOListDt.lt(B1)]
ipo_numbers = uw[uw.IPOListDt.lt(B1)].IPOLeadUW.value_counts().sort_values()
cur_data = merge_data(cur_data, uw, ['IPOLeadUW', 'IPOIssueMethod'])
cur_data['Win'] = cur_data["PRICE"].ge(cur_data['OflBidLowLmtYuan']) & cur_data["PRICE"].le(cur_data['OflBidUpLmtYuan'])
pairs = cur_data.groupby(['INV_NAME', 'IPOLeadUW']).apply(lambda x: pd.Series({'Win Rate': x.Win.sum() / len(x), 'N_Pair': len(x)}))
pairs = pairs.reset_index().merge(ipo_numbers, left_on='IPOLeadUW', right_on='IPOLeadUW')
# invs2 = pairs[(pairs.IPOLeadUW == '华泰联合证券有限责任公司') & (pairs.N_Pair > 5) & (pairs['Win Rate'] == 1)].INV_NAME


# %% 
cur_data = lib.read('data/quote_new/remove_multiple_quote').data
cur_data = cur_data[(cur_data.IPOListDt.lt(B2)) & (cur_data.IPOListDt.gt(B1))]
cur_data['Win'] = cur_data["PRICE"].ge(cur_data['OflBidLowLmtYuan']) & cur_data["PRICE"].le(cur_data['OflBidUpLmtYuan'])
win_rate = cur_data.groupby(level=0).apply(lambda x: x.Win.sum() / x.shape[0]).rename('X Sec Win Rate')
win_rate = merge_data(win_rate.to_frame(), summary, ['IPOListDt'])

n_players = cur_data.groupby(level=0).apply(lambda x: len(x.INV_NAME.unique())).rename('n_players')
n_players = merge_data(n_players.to_frame(), summary, ["IPOListDt"])


# %% Q STD Total
cur_data = lib.read('data/quote_new/remove_multiple_quote').data
std = cur_data.groupby(level=0).apply(lambda x: (x.AMOUNT / x.AMOUNT.max()).std()).rename("std").to_frame()
std = merge_data(std, summary, ["IPOListDt"])
std.set_index('IPOListDt').sort_index().plot(title='Std of Q for all bidders', grid=True)

# %%  valid rate
cur_data = merge_data(data, summary, ["IPOListDt"])
cur_data = merge_data(cur_data,  offline_data, ['OflBidLowLmtYuan', 'OflBidUpLmtYuan'])
cur_data['valid'] = cur_data['PRICE'].le(cur_data['OflBidUpLmtYuan']) & cur_data['PRICE'].ge(cur_data['OflBidLowLmtYuan'])
cur_data = cur_data[cur_data.IPOListDt.lt(B2) & cur_data.IPOListDt.gt(B1)]
cur_data.groupby(level=0).apply(lambda x: pd.Series({'valid_rate':  x["AMOUNT"].sum() / (x['valid'] * x['AMOUNT']).sum(), 'date': x.IPOListDt.iloc[0]})).set_index('date').sort_index().plot()

# %% regression
import statsmodels.api as sm

cur_data = check_valid(data, summary, offline_data)
cur_data = cur_data[cur_data.IPOListDt.ge(B1) & cur_data.IPOListDt.le(B2)]
cur_data['QE'] = cur_data.IPOListDt + pd.offsets.QuarterEnd(0)
def _build_win_rate_and_played(gg):
    gg = gg.reset_index()
    gg = gg.drop_duplicates(['SecCode', "INV_NAME"])
    return pd.Series(
        {
            "valid_rate": gg.valid.sum() / gg.shape[0],
            "played": gg.shape[0]
        }
    )
df = cur_data.groupby(['QE', 'INV_NAME']).apply(_build_win_rate_and_played)
ipo_per_q = cur_data.groupby('QE').apply(lambda x: len(x.index.unique())).rename('N_IPOs')
over_sub = cur_data.groupby('QE').apply(lambda g: g.groupby(level=0).apply(lambda x: x.AMOUNT.sum() / x[x.valid].AMOUNT.sum()).mean()).rename('over_sub')
df = df.merge(ipo_per_q, left_index=True, right_index=True).merge(over_sub, left_index=True, right_index=True)

def _build_pre_play(g):
    g = g.sort_index()
    g['pre_played'] = g.played.cumsum().shift(1).fillna(0)
    return g

df = df.groupby(level=1).apply(_build_pre_play).reset_index(level=2, drop=True)
# df = df[(df.N_IPOs > 10) & (df.played > 5)]



# x = df[['over_sub', 'pre_played']]
# x['over_sub'] = np.log(x['over_sub'])
# x['pre_played'] = np.log(x['pre_played'] + 1)
# x = sm.add_constant(x)
# y = df['valid_rate']
# model = sm.OLS(y,x)
# results = model.fit()
# results.pvalues
# %%