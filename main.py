import pandas as pd

bikeRentalData = pd.read_csv("./bike_rental.csv")
df = bikeRentalData
df.drop_duplicated()
#print(bikeRentalData[bikeRentalData.isnull().any(axis=1)])
print(df.duplicated().sum())
