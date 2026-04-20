import pandas as pd

bikeRentalData = pd.read_csv("./bike_rental.csv")
global df
df = bikeRentalData
# Need to save the DataFrame as a new variable for drop to take place
df = df.drop_duplicates()
print(df.duplicated().sum())
print(' ')


def fiveNumSummary():
    stats = pd.DataFrame({
        'min': round(df.min(numeric_only=True), 2),
        'max': round(df.max(numeric_only=True), 2),
        'mean': round(df.mean(numeric_only=True), 2),
        'median': round(df.median(numeric_only=True), 2),
        'std': round(df.std(numeric_only=True), 2)
    })
    print(stats)

print(df[df.isnull().any(axis=1)])

def fillNull(df):
    values = {
        "temp": df['temp'].mean(),
        "atemp": df['atemp'].mean(),
        "humidity": df['humidity'].mean()
    }
    dfNullFilled=df.fillna(value=values)
    return dfNullFilled
    print("Null Values filled")

df = fillNull(df) 
print(df[df.isnull().any(axis=1)])
print(fiveNumSummary())
