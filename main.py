import numpy as np
import pandas as pd
from Column import Column
import matplotlib.pyplot as plt



class StrategyBacktest:
    def __init__(self,commission=0.00001,start_money=10000000,stock='Test',start_time='20100301',end_time = '20170701'):
        self.commission = commission
        self.stock=stock
        self.start_money = start_money
        self.column=Column[stock]
        self.start_time=start_time
        self.end_time = end_time
        self.open =pd.read_csv('open.csv',index_col=0,parse_dates = True)
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
        return pd.read_csv(datatype+'.csv',index_col=0,parse_dates = True).loc[self.index][self.column]

    # def daily_gain(self,alpha):
    #     self.position=alpha.div(np.sum(alpha,axis=1),axis=1)*self.start_money
    #     self.pnl = pd.DataFrame(self.position[:-1].values*self.open[1:].values/self.open[:-1].values, index=alpha.index[:-1])
    #     return self.pnl
    def Clean(self,alpha):
        return alpha
    def LoadZeroAlpha(self):
        return pd.DataFrame(0,index=self.index,columns=self.column)
    def MaxDrawdown(self,time):
        i = np.argmax(np.maximum.accumulate(self.accumulate) - self.accumulate)  # end of the period
        j = np.argmax(self.accumulate[:i])
        return (self.accumulate[i]-self.accumulate[j])/self.accumulate[i]

    def GeneratePerformance(self,alpha):
        self.position=alpha.div(np.sum(alpha,axis=1),axis=0)*self.start_money
        self.expense=pd.DataFrame(np.sum(self.commission*np.abs(self.position.diff(1)[1:].values),axis=1), index=alpha.index[1:])
        self.gross = pd.DataFrame(np.sum(self.position[:-1].values*self.open[1:].values/self.open[:-1].values,axis=1)-self.start_money, index=alpha.index[1:])
        self.pnl=self.gross-self.expense
        self.accumulate=np.cumsum(self.pnl)
        self.maxdrawdown_value=[]
        # self.maxdrawdown_percent = []
        accu=self.accumulate.values[:,0]
        for t in range(len(accu)-2):
            i = t+np.argmax(np.maximum.accumulate(accu[t:t+20]) - accu[t:t+20])  # end of the period
            j = t+np.argmax(accu[t:i+1])
            self.maxdrawdown_value.append((accu[j] - accu[i]) )
            # self.maxdrawdown_percent.append((accu[j] - accu[i]) /accu[j])
        self.maxdrawdown_value.append((accu[-2] - accu[-1]))
        self.maxdrawdown_value.append(0)
        # self.maxdrawdown_percent.append((accu[-2] - accu[-1])/accu[-2])
        # self.maxdrawdown_percent.append(0)
        self.maxdrawdown_value=pd.DataFrame(self.maxdrawdown_value,index=alpha.index[1:])
        # self.maxdrawdown_percent = pd.DataFrame(self.maxdrawdown_percent, index=alpha.index[1:])

        self.turnover=np.sum(np.abs(self.position.diff(1)/self.start_money),axis=1)
        self.turnover.iloc[0]=0
        years = np.arange(self.pnl.index[0].year, self.pnl.index[-1].year + 1)
        self.conclude = pd.DataFrame(0.0, index=years, columns=['retn', 'sharpe', 'maxdd','tovr'])


        for year in years:
            self.conclude.loc[year]['retn'] = 100*self.pnl[str(year)].values.sum() / self.start_money
            self.conclude.loc[year]['sharpe'] = self.conclude.loc[year]['retn'] / (
                        100*(self.pnl[str(year)].values / self.start_money).std() * np.sqrt(len(self.pnl[str(year)])))
            self.conclude.loc[year]['maxdd'] =100*self.maxdrawdown_value[str(year)].max()/self.start_money
            self.conclude.loc[year]['tovr'] =100*self.turnover[str(year)].values.mean()
        self.conclude.loc['All']={  'retn':100*self.pnl.values.sum()/self.start_money/len(years),
                                    'sharpe':(self.pnl.values.sum()/self.start_money/len(years)) / (
                                             (self.pnl.values / self.start_money).std() * np.sqrt(len(self.pnl))/np.sqrt(len(years))),
                                    'maxdd': 100*self.maxdrawdown_value.values.max()/self.start_money,
                                    'tovr':100*self.turnover.values.mean()
                                    }


        self.ShowPerformance()
    def ShowPerformance(self):
        fig = plt.figure(figsize=(15,8))

        accumulate_pnl=fig.add_subplot(111)
        accumulate_pnl.plot(self.accumulate.index,self.accumulate.values/self.start_money+1,label='accumulate',color='Red')
        accumulate_pnl.set_ylabel('Account')
        mdd=accumulate_pnl.twinx()
        mdd.plot(self.maxdrawdown_value.index,self.maxdrawdown_value.values/self.start_money,label='maxdrawdown',color='Blue')
        mdd.set_ylabel('MaxDradown')



        plt.show()
        print(self.conclude)