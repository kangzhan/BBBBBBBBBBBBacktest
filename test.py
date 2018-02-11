from main import *



sp=StrategyBacktest()
# open =pd.read_csv('open.csv',index_col=0,parse_dates = True)
# print(open.index>='20100303')
close=sp.LoadData('close')
open=sp.LoadData('open')
alpha=close/open


sp.GeneratePerformance(alpha)
# print(sp.position)

sp.pnl.to_csv('pnl.csv')
# print(sp.accumulate)
# plt.plot(sp.accumulate)

# print(sp.turnover)
# print(np.argmax(np.maximum.accumulate(sp.accumulate.values) - sp.accumulate.values))
