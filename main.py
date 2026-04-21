import pandas as pd
import matplotlib.pyplot as plt

bikeRentalData = pd.read_csv("./bike_rental.csv")
# Need to save the DataFrame as a new variable for drop to take place

def fiveNumSummary(df):
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
        lowerBound = mean - 2.5 * std
        if lowerBound < 0:
            lowerBound = 0
        upperBound = mean + 2.5 * std

        df = df[(df[x] >= lowerBound) & (df[x] <= upperBound)]

    return df

def cleanData(df):
    df = df.drop_duplicates()
    df = fillNull(df)
    df = fixOutliers(df)
    df = fixOutliers(df)
    return df


def generateGraphs(df):
    # Count, Average Temp, Season
    for season in ['Summer', 'Autumn', 'Winter', 'Spring']:
        dfFiltered = df[df['season'] == season]
        x, y = dfFiltered['temp'], dfFiltered['count']
        plt.scatter(x, y, label=season, alpha=0.3, edgecolors='none')

    plt.legend()
    
    plt.xlabel('Temperature(C)')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

    for weather in ['Cloudy', 'Clear', 'Rain', 'Storm']:
        dfFiltered = df[df['weather'] == weather]
        x, y = dfFiltered['temp'], dfFiltered['count']
        plt.scatter(x, y, label=weather, alpha=0.3, edgecolors='none')

    plt.legend()
    
    plt.xlabel('Temperature(C)')
    plt.ylabel('Count')
    plt.grid()
    plt.show()


bikeRentalData = cleanData(bikeRentalData)
print(bikeRentalData)
print(round((bikeRentalData.isnull().sum() / bikeRentalData.shape[0]) * 100, 2))
print(fiveNumSummary(bikeRentalData))
generateGraphs(bikeRentalData)
