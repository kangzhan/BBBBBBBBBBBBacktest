from StrategyBacktest import *


sp=StrategyBacktest()
# open =pd.read_csv('open.csv',index_col=0,parse_dates = True)
# print(open.index>='20100303')
# alpha=sp.LoadOneAlpha()
sp.start_money=1
close=sp.LoadData('close')
open=sp.LoadData('open')
vol=sp.LoadData('volume')
swing=sp.LoadData('swing')
vwap=sp.LoadData('vwap')
turn=sp.LoadData('turn')
alpha=ts_mean(1/df_replace_zero(vol),15)


# sp.ShowData()
# print(alpha)
sp.GeneratePerformance(alpha)
# print(alpha)
# print(sp.position)
# print(np.sum(sp.position,axis=1))
# print(sp.position)
print(sp.pnl/sp.start_money)
# print(sp.position)
# print(sp.position)
# print(np.sum(sp.position,axis=1))
# print(sp.pnl)

# print(sp.accumulate)
# plt.plot(sp.accumulate)

# print(sp.turnover)
# print(np.argmax(np.maximum.accumulate(sp.accumulate.values) - sp.accumulate.values))
