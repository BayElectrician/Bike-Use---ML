import pandas as pd

bikeRentalData = pd.read_csv("./bike_rental.csv")
global df
df = bikeRentalData
# Need to save the DataFrame as a new variable for drop to take place
df = df.drop_duplicates()

def fiveNumSummary():
    stats = pd.DataFrame({
        'min': round(df.min(numeric_only=True), 2),
        'max': round(df.max(numeric_only=True), 2),
        'mean': round(df.mean(numeric_only=True), 2),
        'median': round(df.median(numeric_only=True), 2),
        'std': round(df.std(numeric_only=True), 2)
    })
    print(stats)

def fillNull(df):
    values = {
        "temp": df['temp'].mean(),
        "atemp": df['atemp'].mean(),
        "humidity": df['humidity'].mean()
    }
    dfNullFilled=df.fillna(value=values)
    return dfNullFilled
    print("Null Values filled")

def fixOutliers(df):
    for x in ['temp', 'atemp', 'humidity', 'windspeed', 'count']:
        mean = df[x].mean()
        std = df[x].std()

        # Using 3stds to gather 99% of all average values
        lowerBound = mean - 3 * std
        if lowerBound < 0:
            lowerBound = 0
        upperBound = mean + 3 * std

        df = df[(df[x] >= lowerBound) & (df[x] <= upperBound)]
        print(df)
    return df


print(round((df.isnull().sum() / df.shape[0]) * 100, 2))
# df = fillNull(df) 
# print(df[df.isnull().any(axis=1)])
df = fixOutliers(df)
# Do it twice to remove all the outliers
df = fixOutliers(df)
print(fiveNumSummary())
