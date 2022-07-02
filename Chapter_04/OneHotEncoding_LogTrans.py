import pandas as pd
import numpy as np
data = pd.read_csv("myColorset.csv")
#Dropping the outlier rows with standard deviation
print(0,data)

# #one hot encoding
# encoded_columns = pd.get_dummies(data['color'])
# data = data.join(encoded_columns).drop('color', axis=1)
# print(data)



#Log Transform Example
data = pd.DataFrame({'value':[57,-87,45,2,78,-78,57,23]})
data['log+1'] = (data['value']+1).transform(np.log)
#Negative Values Handling
#Note that the values are different
data['log'] = (data['value']-data['value'].min()+1) .transform(np.log)
print(data)