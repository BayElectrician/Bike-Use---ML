import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
# Making Graphs
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# One-Hot Encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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
    # Scatter - Count, Average Temp, Season
    for season in ['Summer', 'Autumn', 'Winter', 'Spring']:
        dfFiltered = df[df['season'] == season]
        x, y = dfFiltered['temp'], dfFiltered['count']
        plt.scatter(x, y, label=season, alpha=0.3, edgecolors='none')

    plt.legend()
    
    plt.xlabel('Temperature(C)')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

    # Scatter - Count, Average Temp, Weather
    for weather in ['Cloudy', 'Clear', 'Rain', 'Storm']:
        dfFiltered = df[df['weather'] == weather]
        x, y = dfFiltered['temp'], dfFiltered['count']
        plt.scatter(x, y, label=weather, alpha=0.3, edgecolors='none')

    plt.legend()
    plt.xlabel('Temperature(C)')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

    # Bar chart - Public Holiday count
    for publicDay in [0, 1]:
        dfFiltered = df[df['holiday'] == publicDay]
        x, y = dfFiltered['count'], dfFiltered['holiday']
        plt.scatter(x, y, label=publicDay, alpha=0.3)
    
    plt.yticks([0, 1], ['Normal', 'Public Holiday'])
    plt.xlabel('Count')
    plt.ylabel('Day Type')
    plt.legend()
    plt.grid(axis = 'x')
    plt.show()

    # Number of Public Holidays a Year
    x = ['Normal', 'Public Holiday']
    y = [(df['holiday'] == 0).sum(), (df['holiday'] == 1).sum()]
    plt.bar(x, y)
    plt.show()

    # Time Series
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    plt.figure(figsize=(12,4))
    sns.lineplot(data=df, x='date', y='count', errorbar=None)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))    # 1st of each month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))    # or '%b %Y' / '%Y-%m'
    plt.xticks(rotation=45)
    ax.set_xlim(df['date'].min(), df['date'].max())
    plt.show()
    # Time Series Resampling Plot
    # parse dates (day/month/year)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    failures = df[df['date'].isna() & df['date'].notna()].index
    # Display the failed features
    print("Failed rows:\n", failures)
    print("Deleting Row outside of date range")
    df.drop(failures, inplace=True)

    # convert numeric columns
    num_cols = ['temp','atemp','humidity','windspeed','count','holiday','workingday']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # set datetime index
    df = df.set_index('date').sort_index()

    # resample to monthly mean (for numeric cols)
    df_resampled = df.resample('ME').mean(numeric_only=True)
    sns.set(style="whitegrid") 

    plt.figure(figsize=(12, 4))  
    sns.lineplot(data=df_resampled, x=df_resampled.index, y='count', errorbar=None, color='blue')

    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('Monthly Resampling of Bicycle Rentals Over Time')

    plt.show()


def basicEDAGraphs(df):
    # Large values
    sns.boxplot(data=df['count'])
    plt.title('Count Column Variation')
    plt.show()

    # Big STD
    sns.boxplot(data=[df['temp'], df['atemp'], df['windspeed']])
    plt.show()


def linearRegression(df):
    # Parse with day-first, errors needed to work
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    failures = df[df['date'].isna() & df['date'].notna()].index
    # Display the failed features
    print("Failed rows:\n", failures)
    print("Deleting Row outside of date range")
    df.drop(failures, inplace=True)

    # Split Data
    X = df.drop(["date", "count"], axis=1)
    y = df["count"]
    categoricalFeatures = ['season', 'weather']

    # Making all Data numerical
    transformer = ColumnTransformer([("one_hot",
                                    OneHotEncoder(),
                                    categoricalFeatures)],
                                    remainder="passthrough")
    transformed_X = transformer.fit_transform(X)
    print(transformed_X[0])
    # Split Train and Test data
    X_train, X_test, y_train, y_test = train_test_split(transformed_X,
                                                        y,
                                                        test_size=0.3)
    # Train model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Evaluate
    y_pred = reg.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² score:", r2_score(y_test, y_pred))
    #print("Accuracy:", accuracy_score(y_test, y_pred))
    #print(classification_report(y_test, y_pred))


    sns.regplot(x=y_test, y=y_pred, ci=None, line_kws={"color":"red"})
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Comparing predicted vs actual Count Values")
    plt.show()

    return reg


def predictCount(model):
    season = input("What season is this in?(Summer/Autumn/Winter/Spring): ")
    holiday = input("Is this a public holiday day?(Y/n): ")
    if holiday == "Y":
        holiday = 1
    else:
        holiday = 0
    workDay = input("Is this a work day?(Y/n): ")
    if workDay == "Y":
        workDay = 1
    else:
        workDay = 0
    weather = input("What is the predicted weather?(Clear, Cloudy, Rain, Storm): ")
    temp = round(float(input("Input what will the temperature be?: ")), 1)
    atemp = round(float(input("What is the apparent temperature?: ")), 1)
    humidity = round(float(input("What is the humidity?: ")), 1)
    windSpeed = round(float(input("What is the windspeed in?(km/h): ")), 1)

    X = pd.DataFrame([{
        "season": season, 
        "holiday": holiday, 
        "workingday": workDay,
        "weather": weather,
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windSpeed
        }])
    # Making all Data numerical
    categoricalFeatures = ['season','weather', 'holiday', 'workingday']
    transformer = ColumnTransformer([("one_hot",
                                    OneHotEncoder(categories = [
                                        ["Summer", "Autumn", "Winter", "Spring"],
                                        ["Clear", "Cloudy", "Rain", "Storm"],
                                        [0, 1],
                                        [0, 1]]),
                                    categoricalFeatures)],
                                    remainder="passthrough")
    transformed_X = transformer.fit_transform(X)

    print(transformed_X)
    output = model.predict(transformed_X)
    print(output)


def startFunc(bikeRentalData):
    os.system('cls')
    print("The follow options can be done with the Dataset")
    print(' ' * 4, "1. 4 Number Summary")
    print(' ' * 4, "2. Basic EDA Graphs")
    print(' ' * 4, "3. Complex EDA Graphs")
    print(' ' * 4, "4. Data Cleanse")
    print(' ' * 4, "5. Machine Learning")
    print(' ' * 4, "6. Predict a Count")
    
    print(' ')
    num = 110
    while num not in range(0,7):
        print("To Exit Type 0")
        userInput = input("Which Question Would you like Answered?: ")
        try:
            num = int(userInput)
        except ValueError:
            print("Please enter an interger number between 0 and 9")

    print(' ')
    whichQuestion(num, bikeRentalData)

def whichQuestion(num, bikeRentalData):
    if num == 1:
        fiveNumSummary(bikeRentalData)
    elif num == 2:
        basicEDAGraphs(bikeRentalData)
    elif num == 3:
        generateGraphs(bikeRentalData)
    elif num == 4:
        bikeRentalData = cleanData(bikeRentalData)
    elif num == 5:
        global model
        model = linearRegression(bikeRentalData)
    elif num == 6:
        predictCount(model)
    else:
        print("Exiting Code Now")
        exit()

    print(' ')
    os.system("pause")
    startFunc(bikeRentalData)
    


bikeRentalData = pd.read_csv("./bike_rental.csv")
startFunc(bikeRentalData)
