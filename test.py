from StrategyBacktest import *


sp=StrategyBacktest()
# open =pd.read_csv('open.csv',index_col=0,parse_dates = True)
# print(open.index>='20100303')
alpha=sp.LoadOneAlpha()
close=sp.LoadData('close')
open=sp.LoadData('open')
vol=sp.LoadData('volume')
alpha =2/df_replace_zero(vol)
# sp.ShowData()
# print(alpha)
sp.GeneratePerformance(alpha)

# print(sp.position)
# print(np.sum(sp.position,axis=1))
# print(sp.position)
# print(sp.pnl)
# print(sp.position)
# print(sp.position)
# print(np.sum(sp.position,axis=1))
# print(sp.pnl)

# print(sp.accumulate)
# plt.plot(sp.accumulate)

# print(sp.turnover)
# print(np.argmax(np.maximum.accumulate(sp.accumulate.values) - sp.accumulate.values))
