from StrategyBacktest import *


sp=FutureBacktest(freq='1H')
# buy=[0,5,1,8,3,4,5,0]
# sell=[1,4,2,5,0,2,3,0]
# close=[0,0,1,0,1,0,0,1]

bsig=1000*np.random.randn(len(sp.index))
ssig=1000*np.random.randn(len(sp.index))
csig=Cross(ts_mean(sp.open,5),sp.close)
# csig[csig<=0]=0
# csig[csig>0]=1
# print(ts_corr(sp.open,sp.close,5))
sp.GeneratePerformance(bsig,ssig,csig)

# print(sp.pnl)
# print(sp.buy)
# print(sp.sell)
# print(sp.close)
# print(sp.position)
# print(sp.cash)
# print(sp.account)
# print(sp.data)
#
# print(sp.open)
# print(sp.high)
# print(sp.low)
# print(sp.close)
# print(sp.volume)