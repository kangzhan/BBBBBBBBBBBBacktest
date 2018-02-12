from WindPy import *
import pandas as pd
import numpy as np
from Information import *

w.start()
print(w.isconnected())
w.start(waitTime=1000)
# "volume,amt,maxupordown,turn"
for data_name in Data_Set:
    temp_data=w.wsd(Column['Test'], data_name, "2010-01-01", "2018-02-10", "")
    temp_data = pd.DataFrame(np.array(temp_data.Data).T, index=temp_data.Times, columns=temp_data.Codes)
    temp_data.to_csv('Data\\'+data_name+'.csv')
# open=w.wsd(Column['Test'], "open", "2010-01-12", "2018-02-10", "")
# open=pd.DataFrame(np.array(open.Data).T,index=open.Times,columns=open.Codes)
# open.to_csv('open.csv')