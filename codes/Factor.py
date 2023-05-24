import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

# 将期权wind code转换成常用格式 'future'+'C/P'+'strike'
def get_optname(s:Series) -> list:
    return [(s[i].rsplit('.')[0]).replace('-','') for i in range(len(s)) ]


# 从期权名字得到标的期货代码、行权价
def get_futname(s:Series) -> tuple:
    symbol: list = []
    strike: list = []
    for string in s:
        match = re.search(r'(.*?)(C|P)(\d+)$', string)
        if match:
            prefix, sep, suffix = match.groups()
            symbol.append(prefix)
            strike.append(float(suffix))
        else:
            symbol.append(np.nan)
            strike.append(np.nan)
    return symbol,strike

# 从期货代码中得到品种代码
def get_instrument(s:str) -> str:
    instrument: str  =  ''
    if isinstance(s,str):
        for ss in s:
            if ss.isalpha():
                instrument +=ss
        return instrument
    else:
        return np.nan

# 大于某值的为1，小于的为np.nan
def filter(value: float,threshold: float) -> float:
    if isinstance(value,float):
        if value > threshold:
            return 1
        else:
            return np.nan
    else:
        return np.nan


# 得到指定期货合约指定行权价的合成期货前置dataframe，用于计算溢价率
def CalculateRatio(future: str,strike: float,opt: DataFrame,rf: float,future_price: DataFrame,opt_price: DataFrame,opt_volume: DataFrame) -> DataFrame:
    if future in future_price.columns:
        calloption: str = future + 'C' + str(int(strike))
        putoption: str = future + 'P' + str(int(strike))
        if (calloption in opt.index) and (putoption in opt.index):
            if (calloption in opt_price.columns) and (putoption in opt_price.columns):
                future_price: Series = future_price[future]
                future_price.dropna(inplace=True)
                call_price: Series = opt_price[calloption]
                call_price.dropna(inplace=True)
                put_price:Series = opt_price[putoption]
                put_price.dropna(inplace=True)
                call_volume: Series = opt_volume[calloption]
                call_volume.dropna(inplace=True)
                put_volume: Series = opt_volume[putoption]
                put_volume.dropna(inplace=True)

                result: DataFrame = pd.concat([future_price,call_price,put_price,call_volume,put_volume],axis=1)
                column_name: list = ['Fut_price','Call_price','Put_price','Call_volume','Put_volume']
                result.columns = column_name
                result['Strike_price'] = strike
                result['contract'] = future

                delvrydate = opt.loc[calloption,'delvry_date']
                result['delvry_date'] = (delvrydate - result.index).days/365
                result['risk_free'] = rf
                result = result[result['delvry_date']>=0]
                # print(result)
                return result
            else:
                print('没有%s或%s的数据!'%(calloption,putoption))
                return pd.DataFrame()

        else:
            print('没有%s的行权日数据'%future)
            return pd.DataFrame()
    else:
        print('没有%s的价格数据!'%future)
        return pd.DataFrame()


# 计算某天某主力合约筛选后的复权溢价率
def process2(main_contract: str,date: str,future_price_df: DataFrame,combine: DataFrame,percent: float) -> float:
    try:
        future_price: float = future_price_df.loc[date,main_contract] #可能没有期货价格
        premiumrate_df: DataFrame = combine.loc[(date,main_contract)].copy()
        premiumrate_df: DataFrame = premiumrate_df[(future_price*(1-percent) < premiumrate_df['Strike_price']) & (premiumrate_df['Strike_price'] < future_price*(1+percent))]
        premiumrate_df['Volume'] = (premiumrate_df['Call_volume'] + premiumrate_df['Put_volume'])/2
        premiumrate_df['Weight'] = premiumrate_df['Volume']/premiumrate_df['Volume'].sum()
        premiumrate_df['Weight_Rate'] = premiumrate_df['Premium_rate']*premiumrate_df['Weight']
        s = pd.Series(premiumrate_df['Weight_Rate'])
        if s.isna().all():
            premium = np.nan
            return premium
        else:

            premium = s.sum()
            return premium
    except:
        premium = np.nan
        return premium

def process3(s: Series, future_price_df: DataFrame, combine: DataFrame, percent: float) -> Series:
    date: str = s.name
    # print(date)

    value: list = []
    for contract in s:
        value.append(process2(contract,date,future_price_df,combine,percent))
    return Series(value,index=s.index)

def process4(s: Series, df: DataFrame) -> Series:
    date: str = s.name
    value: list = []
    for contract in s:
        try:
            v = df.loc[date,contract]
        except:
            v = np.nan
        value.append(v)
    return Series(value,index=s.index)

s1 = time.time()

# 读取数据
opt_data: DataFrame = pd.read_csv(r'D:\Internship\KF\raw_data\opt_data.csv')
future_data: DataFrame = pd.read_csv(r'D:\Internship\KF\raw_data\daily_bar.csv')
opt = pd.read_csv(r'D:\Internship\KF\raw_data\opt.csv')



# 处理数据，方便后面使用
opt_data.rename(columns={'trade_date':'trade_day'},inplace=True)
opt_data['trade_day'] = pd.to_datetime(opt_data['trade_day'],format='%Y%m%d')
opt_data['option'] = get_optname(opt_data['wind_code'])
opt_data['fut'],opt_data['strike'] = get_futname(opt_data['option'])
opt_data['insrument'] = opt_data['fut'].apply(get_instrument)
opt_data['amount_sum'] = opt_data.groupby(['trade_day', 'insrument', 'fut'])['amount'].transform('sum')
# 得到各期货所有的行权价{''：[]，}
strike_price: dict = opt_data.groupby('fut')['strike'].apply(lambda x: sorted(list(set(x)))).to_dict()

future_data['trade_day'] = pd.to_datetime(future_data['trade_day'])
opt['delvry_date'] = pd.to_datetime(opt['delvry_date'],format='%Y%m%d')
opt.set_index('exch_code',inplace=True)

# 获得各种数据
opt_price: DataFrame = pd.pivot(opt_data,index='trade_day',columns='option',values='close')
opt_volume: DataFrame = pd.pivot(opt_data,index='trade_day',columns='option',values='volume')
future_amount: DataFrame = pd.pivot(future_data,index='trade_day',columns='contract',values='amount')
future_volume: DataFrame = pd.pivot(future_data,index='trade_day',columns='contract',values='volume')
future_price: DataFrame = pd.pivot(future_data,index='trade_day',columns='contract',values='close')
future_open: DataFrame = pd.pivot(future_data,index='trade_day',columns='contract',values='open')
future_preopen: DataFrame = pd.pivot(future_data,index='trade_day',columns='contract',values='pre_open')
future_return: DataFrame = future_open/future_preopen - 1


# 按'trade_day'和'instrument'进行分组，并找到每个分组内的amount_sum列的最大值和第二大值所在的行索引
grouped_max_idx = opt_data.groupby(['trade_day', 'instrument'])['amount_sum'].idxmax()
grouped_second_max_idx = opt_data.groupby(['trade_day', 'instrument'])['amount_sum'].nlargest(2).index.get_level_values(1)[1]

# 获取最大值和第二大值所对应的行
max_rows = opt_data.loc[grouped_max_idx]
second_max_rows = opt_data.loc[grouped_second_max_idx]

# 用期权成交额定义的“主力合约”和“次主力合约”，以及它们的期权成交额之和
MainContract = pd.pivot(max_rows,index='trade_day',columns='insrument',values='fut')
SubContract = pd.pivot(second_max_rows,index='trade_day',columns='insrument',values='fut')
MainOptAmount = pd.pivot(max_rows,index='trade_day',columns='insrument',values='amount_sum')
SubOptAmount = pd.pivot(second_max_rows,index='trade_day',columns='insrument',values='amount_sum')
# 去除贵金属
List = ['AU','AG']
MainContract = MainContract[[name for name in MainContract.columns if name not in List]]
SubContract = SubContract[[name for name in SubContract.columns if name not in List]]
MainOptAmount = MainOptAmount[[name for name in MainOptAmount.columns if name not in List]]
SubOptAmount = SubContract[[name for name in SubOptAmount.columns if name not in List]]










# 合成期货
df_list = []
for fut in strike_price.keys():
    for strike in strike_price[fut]:
        df_list.append(CalculateRatio(fut,strike,opt,0.03,future_price,opt_price,opt_volume))
#过滤空dataframe
df_list = [df for df in df_list if not df.empty]

ToatalRatio = pd.concat(df_list,axis=0)
ToatalRatio['Combine_price'] = ToatalRatio['Strike_price']*np.exp(-ToatalRatio['delvry_date']*ToatalRatio['risk_free']) + ToatalRatio['Call_price'] - ToatalRatio['Put_price']
ToatalRatio['Premium_rate'] = ToatalRatio['Combine_price']/ToatalRatio['Fut_price'] - 1  
ToatalRatio.reset_index(inplace=True)
ToatalRatio.set_index(['trade_day','contract'],inplace=True)
ToatalRatio.sort_index(inplace=True)

ToatalRatio.to_csv(r'D:\Internship\KF\data\combine.csv')

ToatalRatio = pd.read_csv(r'D:\Internship\KF\data\combine.csv')
ToatalRatio['trade_day'] = pd.to_datetime(ToatalRatio['trade_day'])
ToatalRatio.set_index(['trade_day','contract'],inplace=True)

def sharpratio(returns,rf):
    cumulative_returns = list((returns + 1).cumprod())
    num_trading_days = len(returns)
    average_annual_return = cumulative_returns[-1] ** (252 / num_trading_days) - 1
    annualized_std = returns.std() * np.sqrt(252)
    ratio = (average_annual_return-rf)/annualized_std

    return ratio

def result(FutureContract:DataFrame,FuturePrice:DataFrame,FutureAmount:DataFrame,OptAmount:DataFrame,FutureReturn:DataFrame,ToatalRatio:DataFrame,Percent:float,Threshold:float):
    # 得到各日期各合约的复权溢价率
    Ratio = FutureContract.apply(lambda x: process3(x,FuturePrice,ToatalRatio,Percent),axis=1)
    Ratio.dropna(axis=0,how='all',inplace=True)
    Ratio.dropna(axis=1,how='all',inplace=True)


    # 用剩余行列筛选
    FutureAmount = FutureContract.apply(lambda x: process4(x,FutureAmount),axis=1)
    FutureAmount:DataFrame = FutureAmount.loc[Ratio.index,Ratio.columns]

    # volume = main_contract.apply(lambda x: process4(x,future_volume),axis=1)
    # volume:DataFrame = volume.loc[factor.index,factor.columns]
    OptAmount:DataFrame = OptAmount.loc[Ratio.index,Ratio.columns]

    openreturn = FutureContract.apply(lambda x: process4(x,FutureReturn),axis=1)
    openreturn:DataFrame = openreturn.loc[Ratio.index,Ratio.columns]

    # 用期货成交量过滤
    FutureAmount = FutureAmount.applymap(lambda x: filter(x, Threshold))
    filter_factor = FutureAmount*Ratio

    #用期权成交量归一化
    # normalize_factor:  DataFrame = abs(filter_factor*volume)
    normalize_factor:  DataFrame = abs(filter_factor*OptAmount)
    normalize_factor = normalize_factor.div(normalize_factor.sum(axis=1), axis=0)

    # 乘以收益率
    ret = normalize_factor*openreturn
    ret['PNL'] = ret.sum(axis=1)

    return ret


# 两个列表作为参数
percent_list = [0.01,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.5]
# amount_list = [5e8,15e8,35e8]
# amount_list = [5e8]

sharp_Dict = {}
# 创建一个总图
fig = plt.figure()

# 定义子图的行数和列数
num_rows = 3
num_cols = 3

# 双重循环生成子图并进行绘制
for i in range(len(percent_list)):
        # 添加子图
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        ret = result(main_contract,future_price,future_amount,opt_amount,future_return,ToatalRatio,percent_list[i],5e8)
        # ret = result(main_contract,future_price,future_amount,future_volume,future_return,combine,percent_list[i],5e8)
        sharp_Dict[percent_list[i]] = sharpratio(ret['PNL'],0.03)
        # ret = pd.read_csv(f'D:\\Internship\\KF\data\\ret-{percent_list[i]}-{amount_list[j]}.csv')
        # ret.set_index('trade_day')
        # ret['PNL'].cumsum().plot(xlabel='trade_day',ylabel='PNL',ax=ax)
        # ax.set_title(f'{percent_list[i]}')

        # 绘制数据
        # ret['PNL'].cumsum().plot(xlabel='trade_day',ylabel='PNL',ax=ax)
        # text = f"{percent_list[i]} {amount_list[j]}"
        # ax.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left')
        # 设置子图标题
        # ax.set_title(f'percent={percent_list[i]} amount={amount_list[j]}')

print(sharp_Dict)
# 调整子图之间的间距
# fig.tight_layout()
# 显示图形
# plt.show()
# plt.savefig(r'D:\Internship\KF\data\PNL.png')

# print(time.time()-s1)

# 'AU' result、future price 那一天那个合约所有的合成期权的数据 

# 价内、价外因子，筛选幅度percent
# 成交额限制threshold

