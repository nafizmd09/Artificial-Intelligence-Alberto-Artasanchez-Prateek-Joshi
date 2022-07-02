import pandas as pd
data = pd.read_csv("myDataset.csv")
#Dropping the outlier rows with standard deviation
print(0,data)
factor = 2


# #Dropping the outlier rows with standard deviation
# upper_lim = data['power'].mean () + data['power'].std() * factor
# lower_lim = data['power'].mean () - data['power'].std() * factor
# data = data[(data['power'] < upper_lim) & (data['power'] > lower_lim)]
# print(data)



# #Let's look at a Python example where we drop the top and bottom 1%:
# #Dropping the outlier rows with Percentiles
# upper_lim = data['power'].quantile(.99)
# lower_lim = data['power'].quantile(.01)
# data = data[(data['power'] < upper_lim) & (data['power'] > lower_lim)]
# print(data)



#Capping the outlier rows with percentiles
upper_lim = data['power'].quantile(.99)
lower_lim = data['power'].quantile(.01)
data.loc[(data['power'] > upper_lim), 'power'] = upper_lim
data.loc[(data['power'] < lower_lim), 'power'] = lower_lim
print(data)