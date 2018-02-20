from Information import *
from StockFunction import *
from FutureFunction import *
import matplotlib.pyplot as plt
import datetime



class StrategyBacktest:
    def __init__(self,commission=0.001,start_money=10000000,stock='Test',start_time='20100101',end_time = '20180210'):
        self.commission = commission
        self.stock=stock
        self.start_money = start_money
        self.column=Column[stock]
        self.start_time=start_time
        self.end_time = end_time
        self.open =pd.read_csv('Data\\'+'open.csv',index_col=0,parse_dates = True)
        self.index=self.open.index[self.open.index>=self.start_time][self.open.index<=self.end_time]
        self.open=self.open.loc[self.index][self.column]
    def SetDate(self,start_time,end_time):
        self.start_time=start_time
        self.end_time=end_time
    def SetMoney(self,start_money):
        self.start_money=start_money
    def SetCost(self,commission):
        self.commission=commission
    def LoadData(self,datatype):
        return np.array(pd.read_csv('Data\\'+datatype+'.csv',index_col=0,parse_dates = True).loc[self.index][self.column].values)

    def ShowData(self):
        print(Data_Set)

    # def daily_gain(self,alpha):
    #     self.position=alpha.div(np.sum(alpha,axis=1),axis=1)*self.start_money
    #     self.pnl = pd.DataFrame(self.position[:-1].values*self.open[1:].values/self.open[:-1].values, index=alpha.index[:-1])
    #     return self.pnl
    def Clean(self,alpha):
        # alpha[np.isnan(alpha)] = 0.000000001
        # alpha[np.isinf(alpha)] = 0.000000001
        # alpha[alpha<0]=0.000000001
        # return pd.DataFrame(alpha,index=self.index,columns=self.column)
        alpha=pd.DataFrame(alpha)
        alpha_min = np.min(alpha, axis=1)
        alpha_min[np.isnan(alpha_min)] = 1
        alpha_min[np.isinf(alpha_min)] = 1
        min_matrix = np.array(list(alpha_min) * len(alpha.columns)).reshape((len(alpha.columns), len(alpha.index))).T
        alpha[np.isnan(alpha)] = min_matrix
        alpha[np.isinf(alpha)] = min_matrix
        mmin = np.min(alpha, axis=1)
        mmax = np.max(alpha, axis=1)
        alpha[mmax != mmin] = cs_percent(alpha[mmax != mmin])
        # alpha.loc[alpha[mmax == mmin][mmin == 0].index] = 1
        alpha.iloc[alpha[(mmax-mmin)**2+mmin**2==0].index]=1
        return pd.DataFrame(alpha.values,index=self.index,columns=self.column)

    def LoadOneAlpha(self):
        return np.ones((len(self.index),len(self.column)))
    def MaxDrawdown(self,time):
        i = np.argmax(np.maximum.accumulate(self.accumulate) - self.accumulate)  # end of the period
        j = np.argmax(self.accumulate[:i])
        return (self.accumulate[i]-self.accumulate[j])/self.accumulate[i]

    def GeneratePerformance(self,alpha):

        alpha=self.Clean(alpha)
        self.position=alpha.div(np.sum(alpha,axis=1),axis=0)*self.start_money
        self.position=self.position-self.position.diff(1)
        self.position.iloc[0] = self.start_money/len(self.column)
        all_expense=self.commission*np.abs(self.position.diff(1)[1:].values)
        all_gross=self.position[:-1].values*self.open[1:].values/self.open[:-1].values
        self.expense=pd.DataFrame(np.sum(self.commission*np.abs(self.position.diff(1)[1:].values),axis=1), index=alpha.index[1:])
        self.gross = pd.DataFrame(np.sum(self.position[:-1].values*self.open[1:].values/self.open[:-1].values,axis=1)-self.start_money, index=alpha.index[1:])
        self.pnl=self.gross-self.expense
        # self.IC = pd.DataFrame(np.array(pd.DataFrame((all_gross - all_expense).T).corrwith(pd.DataFrame(alpha.T.values)).iloc[:-1]),index=alpha.index[:-1])
        self.accumulate=np.cumsum(self.pnl)
        self.maxdrawdown_value=[]
        self.maxdrawdown_percent = []
        accu=self.accumulate.values[:,0]
        for t in range(len(accu)-2):
            i = t+np.argmax(np.maximum.accumulate(accu[t:t+3]) - accu[t:t+3])  # end of the period
            j = t+np.argmax(accu[t:i+1])
            self.maxdrawdown_value.append((accu[j] - accu[i]) )
            self.maxdrawdown_percent.append((accu[j] - accu[i]) /(self.start_money))
        self.maxdrawdown_value.append(min(accu[-2] - accu[-1],0))
        self.maxdrawdown_value.append(0)
        self.maxdrawdown_percent.append(min(accu[-2] - accu[-1],0)/(self.start_money))
        self.maxdrawdown_percent.append(0)
        self.maxdrawdown_value=pd.DataFrame(self.maxdrawdown_value,index=alpha.index[1:])
        self.maxdrawdown_percent = pd.DataFrame(self.maxdrawdown_percent, index=alpha.index[1:])

        self.turnover=np.sum(np.abs(self.position.diff(1)/self.start_money),axis=1)
        self.turnover[0]=0
        years = np.arange(self.pnl.index[0].year, self.pnl.index[-1].year + 1)
        self.conclude = pd.DataFrame(0.0, index=years, columns=['retn', 'sharpe', 'maxdd','tovr'])


        for year in years:
            self.conclude.loc[year]['retn'] = self.pnl[str(year)].values.sum() / self.start_money
            self.conclude.loc[year]['sharpe'] = self.conclude.loc[year]['retn'] / (
                        (self.pnl[str(year)].values / self.start_money).std() * np.sqrt(len(self.pnl[str(year)])))
            self.conclude.loc[year]['maxdd'] =self.maxdrawdown_value[str(year)].max()/self.start_money
            self.conclude.loc[year]['tovr'] =self.turnover[str(year)].values.mean()
            # self.conclude.loc[year]['IC'] = self.IC[str(year)].values.mean()
        self.conclude.loc['All']={  'retn':self.pnl.values.sum()/self.start_money/len(years),
                                    'sharpe':(self.pnl.values.sum()/self.start_money/len(years)) / (
                                             (self.pnl.values / self.start_money).std() * np.sqrt(len(self.pnl))/np.sqrt(len(years))),
                                    'maxdd': self.maxdrawdown_value.values.max()/self.start_money,
                                    'tovr':self.turnover.mean()
                                    # 'IC':self.IC.values.mean()
                                    }


        # print(self.IC)
        self.ShowPerformance()
    def ShowPerformance(self):
        fig = plt.figure(figsize=(12,5))

        accumulate_pnl=fig.add_subplot(111)
        accumulate_pnl.plot(self.accumulate.index,self.accumulate.values/self.start_money+1,label='accumulate',color='Red')
        accumulate_pnl.set_ylabel('Account')
        mdd=accumulate_pnl.twinx()
        mdd.plot(self.maxdrawdown_percent.index, self.maxdrawdown_percent.values , label='maxdrawdown',
                 color='Blue')
        # mdd.plot(self.maxdrawdown_value.index,self.maxdrawdown_value.values/self.start_money,label='maxdrawdown',color='Blue')
        mdd.set_ylabel('MaxDrawdown')

        print(self.conclude)
        plt.show()

class FutureBacktest:
    def __init__(self,commission=0.001,start_money=100000,freq='D',stock='Test',start_time='20100101',end_time = '20180210'):
        self.commission = commission
        self.start_money=start_money
        self.freq = freq
        data = pd.read_csv('data\\IF2015.1m.csv')
        data['ttime'] = (data['date'] + ' ' + data['time']).map(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        d = data.set_index('ttime')
        del d['date']
        del d['time']
        self.data=d.resample(freq).first().dropna()
        self.cash=0
        self.open=[1,3,2,1,4,1,5,7]
        self.account=[]
        self.index=self.data.index
        self.open=np.array(d.resample(freq).first().dropna()['open'].values)
        self.high = np.array(d.resample(freq).max().dropna()['high'].values)
        self.low = np.array(d.resample(freq).min().dropna()['low'].values)
        self.close = np.array(d.resample(freq).last().dropna()['close'].values)
        self.volume = np.array(d.resample(freq).sum().dropna()['volume'].values)
    def GetPosition(self, bsig,ssig,csig):
        self.bsig=np.array(bsig)
        self.ssig=np.array(ssig)
        self.csig=np.array(csig)
        self.csig[-1]=1
        self.position=[]
        if (self.csig[0] == 0):
            self.position.append(self.bsig[0] - self.ssig[0])
        else:
            self.position.append(0)
        for i in np.arange(1,len(self.bsig)):
            if(self.csig[i]==0):
                self.position.append(self.position[i-1]+self.bsig[i]-self.ssig[i])
            else:
                self.position.append(0)
        self.position=np.array(self.position)
        return self.position
    def GeneratePerformance(self,bsig,ssig,csig):
        self.position=self.GetPosition(bsig,ssig,csig)
        self.cash=[]
        self.cash.append(self.start_money-self.position[0]*self.open[0])
        for i in np.arange(1,len(bsig)):
            self.cash.append(self.cash[i-1]-self.open[i]*(self.position[i]-self.position[i-1]))
        self.cash=np.array(self.cash)
        self.account=pd.DataFrame(self.position*self.open+self.cash,index=self.index)
        self.ShowPerformance()
        self.pnl=self.account.diff(1)
        self.pnl.iloc[0]=0
        self.pnl=pd.DataFrame(self.pnl,index=self.index)
    def ShowPerformance(self):
        fig = plt.figure(figsize=(12, 5))
        plt.plot(self.account-self.start_money)
        plt.show()

