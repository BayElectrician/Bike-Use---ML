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
from sklearn.preprocessing import LabelEncoder
# One-Hot Encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def five_num_summary(df):
    stats = pd.DataFrame({
        'min': round(df.min(numeric_only=True), 2),
        'max': round(df.max(numeric_only=True), 2),
        'mean': round(df.mean(numeric_only=True), 2),
        'median': round(df.median(numeric_only=True), 2),
        'std': round(df.std(numeric_only=True), 2)
    })
    print(stats)

def fill_null(df):
    values = {
        "temp": df['temp'].mean(),
        "atemp": df['atemp'].mean(),
        "humidity": df['humidity'].mean()
    }
    df_null_filled=df.fillna(value=values)
    return df_null_filled
    print("Null Values filled")

def fix_outliers(df):
    for x in ['temp', 'atemp', 'humidity', 'windspeed', 'count']:
        mean = df[x].mean()
        std = df[x].std()

        # Using 3stds to gather 99% of all average values
        lower_bound = mean - 2.5 * std
        if lower_bound < 0:
            lower_bound = 0
        upper_bound = mean + 2.5 * std

        df = df[(df[x] >= lower_bound) & (df[x] <= upper_bound)]

    return df

def clean_data(df):
    df = df.drop_duplicates()
    df = fill_null(df)
    df = fix_outliers(df)
    df = fix_outliers(df)
    return df


def generate_graphs(df):
    # Scatter - Count, Average Temp, Season
    for season in ['Summer', 'Autumn', 'Winter', 'Spring']:
        df_filtered = df[df['season'] == season]
        x, y = df_filtered['temp'], df_filtered['count']
        plt.scatter(x, y, label=season, alpha=0.3, edgecolors='none')

    plt.legend()
    
    plt.xlabel('Temperature(C)')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

    # Scatter - Count, Average Temp, Weather
    for weather in ['Cloudy', 'Clear', 'Rain', 'Storm']:
        df_filtered = df[df['weather'] == weather]
        x, y = df_filtered['temp'], df_filtered['count']
        plt.scatter(x, y, label=weather, alpha=0.3, edgecolors='none')

    plt.legend()
    plt.xlabel('Temperature(C)')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

    # Bar chart - Public Holiday count
    for public_day in [0, 1]:
        df_filtered = df[df['holiday'] == public_day]
        x, y = df_filtered['count'], df_filtered['holiday']
        plt.scatter(x, y, label=public_day, alpha=0.3)
    
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


def basic_EDA_graphs(df):
    # Large values
    sns.boxplot(data=df['count'])
    plt.title('Count Column Variation')
    plt.show()

    # Big STD
    sns.boxplot(data=[df['temp'], df['atemp'], df['windspeed']])
    plt.show()


def linear_regression(df):
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
    categorical_features = ['season', 'weather']

    # Making all Data numerical
    transformer = ColumnTransformer([("one_hot",
                                    OneHotEncoder(),
                                    categorical_features)],
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
    plt.xlabel("Actual Count")
    plt.ylabel("Predicted Count")
    plt.title("Comparing Predicted vs Actual Count Values")
    plt.show()

    return reg


def predict_count(model):
    season = input("What season is this in?(Summer/Autumn/Winter/Spring): ")
    holiday = input("Is this a public holiday day?(Y/n): ")
    if holiday == "Y":
        holiday = 1
    else:
        holiday = 0
    work_day = input("Is this a work day?(Y/n): ")
    if work_day == "Y":
        work_day = 1
    else:
        work_day = 0
    weather = input("What is the predicted weather?(Clear, Cloudy, Rain, Storm): ")
    temp = round(float(input("Input what will the temperature be?: ")), 1)
    atemp = round(float(input("What is the apparent temperature?: ")), 1)
    humidity = round(float(input("What is the humidity?: ")), 1)
    wind_speed = round(float(input("What is the windspeed in?(km/h): ")), 1)

    X = pd.DataFrame([{
        "season": season, 
        "holiday": holiday, 
        "workingday": work_day,
        "weather": weather,
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": wind_speed
        }])
    # Making all Data numerical
    categorical_features = ['season','weather', 'holiday', 'workingday']
    transformer = ColumnTransformer([("one_hot",
                                    OneHotEncoder(categories = [
                                        ["Summer", "Autumn", "Winter", "Spring"],
                                        ["Clear", "Cloudy", "Rain", "Storm"],
                                        [0, 1],
                                        [0, 1]]),
                                    categorical_features)],
                                    remainder="passthrough")
    transformed_X = transformer.fit_transform(X)

    print(transformed_X)
    output = model.predict(transformed_X)
    print(output)


def start_func(bike_rental_data):
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
        user_input = input("Which Question Would you like Answered?: ")
        try:
            num = int(user_input)
        except ValueError:
            print("Please enter an interger number between 0 and 9")

    print(' ')
    which_question(num, bike_rental_data)

def which_question(num, bike_rental_data):
    if num == 1:
        five_num_summary(bike_rental_data)
    elif num == 2:
        basic_EDA_graphs(bike_rental_data)
    elif num == 3:
        generate_graphs(bike_rental_data)
    elif num == 4:
        bike_rental_data = clean_data(bike_rental_data)
    elif num == 5:
        global model
        model = linear_regression(bike_rental_data)
    elif num == 6:
        predict_count(model)
    else:
        print("Exiting Code Now")
        exit()

    print(' ')
    os.system("pause")
    start_func(bike_rental_data)
    


bike_rental_data = pd.read_csv("./bike_rental.csv")
start_func(bike_rental_data)
