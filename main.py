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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Descion Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
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
    print("Deleting Row outside of date Range")
    df.drop(failures, inplace=True)

    # Split Data
    X = df.drop(["date", "count"], axis=1)
    y = df["count"]
    categoricalFeatures = ['season', 'weather']
    #Making all Data numerical
    transformer = ColumnTransformer([("one_hot",
                                    OneHotEncoder(),
                                    categoricalFeatures)],
                                    remainder="passthrough"
                                    )
    transformed_X = transformer.fit_transform(X)

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

    sns.regplot(x=y_test, y=y_pred, ci=None, line_kws={"color":"red"})
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()
    #sns.plot(X, y_pred)
    '''
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
    '''

def startFunc(bikeRentalData):
    os.system('cls')
    print("The follow options can be done with the Dataset")
    print(' ' * 4, "1. 4 Number Summary")
    print(' ' * 4, "2. Basic EDA Graphs")
    print(' ' * 4, "3. Complex EDA Graphs")
    print(' ' * 4, "4. Data Cleanse")
    print(' ' * 4, "5. Machine Learning")
    
    print(' ')
    num = 110
    while num not in range(0,6):
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
        linearRegression(bikeRentalData)
    else:
        print("Exiting Code Now")
        exit()

    print(' ')
    os.system("pause")
    startFunc(bikeRentalData)
    


bikeRentalData = pd.read_csv("./bike_rental.csv")
startFunc(bikeRentalData)
