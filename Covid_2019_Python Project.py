# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:36:59 2025

@author: deepankarg
"""

import pandas as pd

### Question 1: Data Loading

# Load the datasets
confirmed_cases = pd.read_csv('C:/Users/deepankarg/Downloads/covid_19_confirmed_v1.csv')
deaths = pd.read_csv('C:/Users/deepankarg/Downloads/covid_19_deaths_v1.csv')
recovered_cases = pd.read_csv('C:/Users/deepankarg/Downloads/covid_19_recovered_v1.csv')


# Export to excel

confirmed_cases.to_excel("C:/Users/deepankarg/Downloads/covid_19_confirmed_v1.xlsx", index = False)
deaths.to_excel("C:/Users/deepankarg/Downloads/covid_19_deaths_v1.xlsx", index = False)
recovered_cases.to_excel("C:/Users/deepankarg/Downloads/covid_19_recovered_v1.xlsx", index = False)

# Optional Part - in case of excel files

confirmed_cases = pd.read_excel('C:/Users/deepankarg/Downloads/covid_19_confirmed_v1.xlsx', sheet_name ='Sheet1')
deaths = pd.read_excel('C:/Users/deepankarg/Downloads/covid_19_deaths_v1.xlsx', header = 1)
recovered_cases = pd.read_excel('C:/Users/deepankarg/Downloads/covid_19_recovered_v1.xlsx', header = 1)

### Question 2: Data Exploration

# Q2.1: After loading the datasets, what is the structure of each dataset in terms of rows, 
# columns, and data types?

# Basic structure and data type information
print(confirmed_cases.info())
print(deaths.info())
print(recovered_cases.info())

# Q2.2: Generate plots of confirmed cases over time for top countries?

import matplotlib.pyplot as plt
# Visual exploration: Plot the total confirmed cases over time for the top three countries
top_countries = confirmed_cases.groupby('Country/Region').sum().nlargest(3, confirmed_cases.columns[-1]).index
plt.figure(figsize =(12,6)) # increasing plot size for better visibility
for country in top_countries:
    data = confirmed_cases[confirmed_cases['Country/Region'] == country].iloc[:, 4:].sum()  # Sum across all dates
    plt.plot(data.index, data.values, label=country)
plt.xticks(rotation=45)
plt.legend()
plt.title("Total Confirmed Cases Over Time for Top 3 Countries")
plt.xlabel("Date")
plt.ylabel("Cumulative Confirmed Cases")
plt.show()

# Q2.3: Generate plots of confirmed cases over time for China?

# Explore data by provinces for a specific country, such as China
china_data = confirmed_cases[confirmed_cases['Country/Region'] == 'China']
china_data.set_index('Province/State').iloc[:, 3:].T.plot()
plt.title("COVID-19 Confirmed Cases Over Time for Provinces in China")
plt.xlabel("Date")
plt.ylabel("Cumulative Confirmed Cases")
plt.legend(title="Province")
plt.xticks(rotation=45)
plt.show()

### Question 3: Handling Missing Data

# Q3.1: Identify these missing values and replace them using a suitable imputation 
# method such as forward filling for time-series data.

# Finding missing values in the datasets
import pandas as pd

# Check for any missing values across the datasets
deaths_missing = deaths.isnull().any().any()
recovered_missing = recovered_cases.isnull().any().any()
print("Are there any missing values in the deaths dataset?", deaths_missing)
print("Are there any missing values in the recovered_cases dataset?", recovered_missing)

# If missing values are found in deaths, identify where they are
# Convert empty strings to NaN
deaths.replace('', pd.NA, inplace=True)
missing_values_deaths_count =  deaths.isnull().sum()
missing_values_deaths_count.sort_values(ascending=False)

# If missing values are found in recovered_cases, identify where they are
# Convert empty strings to NaN
recovered_cases.replace('', pd.NA, inplace=True)
# Count missing values across columns
missing_values_recovered_count = recovered_cases.isnull().sum()
missing_values_recovered_count.sort_values(ascending=False)

# Impute missing values

mode_value = deaths['Province/State'].mode()[0]
deaths['Province/State'].fillna(mode_value, inplace=True)
deaths.fillna(method='ffill', inplace=True)  # Forward fill missing values

print("Any missing values left in deaths?", deaths.isnull().any().any())

mode_value = recovered_cases['Province/State'].mode()[0]
recovered_cases['Province/State'].fillna(mode_value, inplace=True)
recovered_cases.fillna(method='ffill', inplace=True)  # Forward fill missing values

print("Any missing values left in recovered?", recovered_cases.isnull().any().any())

# Alternate Approach

def impute_with_moving_average(df):
    # Specify columns 'Lat' and 'Long' for imputation
    columns_to_impute = ['Lat', 'Long','4/20/20']
    
    # Apply moving average imputation to specified columns
    for col in columns_to_impute:
        df[col] = df[col].fillna(df[col].rolling(window=3, min_periods=1, center=True).mean())
    
    return df

# Apply moving average imputation to the 'deaths' DataFrame
deaths = impute_with_moving_average(deaths)
# Apply moving average imputation to the 'recovered_cases' DataFrame
recovered_cases = impute_with_moving_average(recovered_cases)

### Question 4: Data Cleaning and Preparation

# Q4.1: Replace blank values in province column with “All Provinces”

confirmed_cases['Province/State'].fillna('All Province', inplace=True)
deaths['Province/State'].fillna('All Province', inplace=True)
recovered_cases['Province/State'].fillna('All Province', inplace=True)

### Question 5: Independent Dataset Analysis

# Q5.1: Analyze the peak number of daily new cases in Germany, 
# France, and Italy. Which country experienced the highest single-day surge, 
# and when did it occur? 

import pandas as pd
import matplotlib.pyplot as plt
# Calculate daily new cases for Germany, France, and Italy
countries = ['Germany', 'France', 'Italy']
daily_new_cases = {}
for country in countries:
    country_data = confirmed_cases[confirmed_cases['Country/Region'] == country]
#    print(country_data)
    summed_daily_cases = country_data.iloc[:, 4:].sum(axis=0).diff().fillna(0)  # Sum across all provinces/states
#    print(summed_daily_cases)
    daily_new_cases[country] = summed_daily_cases

peaks = {}
for country, data in daily_new_cases.items():
    peak_date = data.idxmax()
    peak_cases = data.max()
    peaks[country] = (peak_cases, peak_date)
print(peaks)

# Visualize the daily new cases for comparison
plt.figure(figsize =(12,6)) # increasing plot size for better visibility
for country, data in daily_new_cases.items():
    plt.plot(data.index, data.values, label=country)
plt.legend()
plt.title("Daily New COVID-19 Cases")
plt.xlabel("Date")
plt.ylabel("Number of New Cases")
plt.xticks(rotation=45)
plt.show()

# Q5.2: Compare the recovery rates (recoveries/confirmed cases) between 
# Canada and Australia as of December 31, 2020. Which country showed better 
# management of the pandemic according to this metric?

recovery_rates = {}
for country in ['Canada', 'Australia']:
    country_cases = confirmed_cases[confirmed_cases['Country/Region'] == country].iloc[:, 4:].sum(axis=0)
    country_recoveries = recovered_cases[recovered_cases['Country/Region'] == country].iloc[:, 4:].sum(axis=0)
    recovery_rate = country_recoveries['12/31/20'] / country_cases['12/31/20']
    recovery_rates[country] = recovery_rate
print(recovery_rates)

# Visualization
plt.bar(recovery_rates.keys(), recovery_rates.values(), color=['blue', 'green'])
plt.title('Recovery Rates Comparison on 2020-12-31')
plt.ylabel('Recovery Rate')
plt.show()

# Q5.3: What is the distribution of death rates (deaths/confirmed cases) 
# among provinces in Canada? Identify the province with the highest and 
# lowest death rate as of the latest data point.

canada_cases = confirmed_cases[confirmed_cases['Country/Region'] == 'Canada'].drop(columns=['Country/Region', 'Lat', 'Long'])
canada_deaths = deaths[deaths['Country/Region'] == 'Canada'].drop(columns=['Country/Region', 'Lat', 'Long'])
 
# Sum the cases and deaths for each province across all dates (assuming the first few columns are non-date columns)
canada_cases = canada_cases.groupby('Province/State').sum().iloc[:, -1]  # Sum only the last date column for latest data point
canada_deaths = canada_deaths.groupby('Province/State').sum().iloc[:, -1]  # Sum only the last date column for latest data point

# Calculate death rates
death_rates = pd.Series(0, index=canada_cases.index)

death_rates[canada_cases != 0] = canada_deaths.div(canada_cases[canada_cases != 0])

print(death_rates)

# Identify the province with the highest and lowest death rate
max_death_rate = death_rates.idxmax()
min_death_rate = death_rates.idxmin()
print(f"Highest death rate in Canada is in {max_death_rate} province.")
print(f"Lowest death rate in Canada is in {min_death_rate} province.")
# Visualization
death_rates.plot(kind='bar')
plt.title('Death Rates by Province in Canada')
plt.ylabel('Death Rate')
plt.xlabel('Province')
plt.show()


### Question 6: Data Transformation

# Q6.1: Transform the 'deaths' dataset from wide format 
# (where each column represents a date) to long format where each row 
# represents a single date and columns are now country names, ensuring 
# that the date column is in datetime format. How would this transformation be executed?

deaths_long = deaths.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Deaths")
print(deaths_long)
deaths_long.dtypes
deaths_long['Date'] = pd.to_datetime(deaths_long['Date'])
deaths_long.dtypes

# Q6.2: What is the total number of deaths reported per country up to the current date?

total_deaths_per_country = deaths_long.groupby('Country/Region')['Deaths'].sum()
# Display the result
print("Total deaths per country:")
print(total_deaths_per_country)

# Q6.3: What are the top 5 countries with the highest average daily deaths?

average_daily_deaths_per_country = deaths_long.groupby('Country/Region')['Deaths'].mean().nlargest(5)
# Display the result
print("Top 5 countries with the highest average daily deaths:")
print(average_daily_deaths_per_country)

# Q6.4: How have the total deaths evolved over time in the United States?

us_deaths_over_time = deaths_long[deaths_long['Country/Region'] == "US"].groupby('Date')['Deaths'].sum()
# Plotting the result
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(us_deaths_over_time.index, us_deaths_over_time.values)
plt.title('Total COVID-19 Deaths Over Time in the United States')
plt.xlabel('Date')
plt.ylabel('Total Deaths')
plt.grid(True)
plt.show()


### Question 7: Data Merging

# Q7.1: How would you merge the transformed datasets of confirmed cases, 
# deaths, and recoveries on the 'Country/Region' and 'Date' columns to create a 
# comprehensive view of the pandemic's impact?

confirmed_long = confirmed_cases.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Confirmed")
confirmed_long['Date'] = pd.to_datetime(confirmed_long['Date'])

recovered_long = recovered_cases.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Recovered")
recovered_long['Date'] = pd.to_datetime(recovered_long['Date'])

# Merging datasets on 'Country/Region' and 'Date'
# Merge confirmed cases with deaths
merged_data = pd.merge(confirmed_long, deaths_long, on=["Province/State", "Country/Region", "Lat", "Long", "Date"], how='outer')
merged_data.shape
merged_data.columns

confirmed_long.shape
deaths_long.shape

final_merged_data = pd.merge(merged_data, recovered_long, on=["Province/State", "Country/Region", "Lat", "Long", "Date"], how='outer')
# Display the head of the final merged DataFrame to verify it's correct
print(final_merged_data.head())

# Q7.2: Analyze the monthly sum of confirmed cases, deaths, and 
# recoveries for countries to understand the progression of the pandemic.

# Ensure 'Date' is in datetime format and create a 'YearMonth' column for grouping
final_merged_data['YearMonth'] = final_merged_data['Date'].dt.to_period('M')
# Create a pivot table to see the monthly maximum of Confirmed, Deaths, and Recoveries by Country
monthly_max_trends = pd.pivot_table(final_merged_data, values=['Confirmed', 'Deaths', 'Recovered'],
                                    index=['Country/Region', 'YearMonth'],
                                    aggfunc='max')
# Display the pivot table
print(monthly_max_trends.head(20))

# Q7.3: Redo the analysis in Question 7.2 for United States, Italy and Brazil.

# Filter data for specific countries
filtered_data = final_merged_data[final_merged_data['Country/Region'].isin(['US', 'Italy', 'Brazil'])]
# Create a pivot table for the filtered data
country_monthly_trends = pd.pivot_table(filtered_data, values=['Confirmed', 'Deaths', 'Recovered'],
                                        index=['Country/Region', 'YearMonth'],
                                        aggfunc='sum')
# Display the pivot table for these countries
print(country_monthly_trends)

### Question 8: Combined Data Analysis

# Q8.1: For the combined dataset, identify the three countries with the highest 
# average death rates (deaths/confirmed cases) throughout 2020. What might this 
# indicate about the pandemic's impact in these countries?

import matplotlib.pyplot as plt
# Calculate the average death rates by country
average_death_rates = (final_merged_data.groupby('Country/Region')['Deaths'].sum() / final_merged_data.groupby('Country/Region')['Confirmed'].sum()).sort_values(ascending=False).head(3)
# Visualization
average_death_rates.plot(kind='bar', color='red')
plt.title("Top 3 Countries with Highest Average Death Rates")
plt.ylabel("Death Rate")
plt.xlabel("Country")
plt.show()
# Output: Bar chart showing the three countries with the highest average death rates throughout 2020.

# Q8.2: Using the merged dataset, compare the total number of recoveries 
# to the total number of deaths in South Africa. What can this tell us about 
# the outcomes of COVID-19 cases in the country?

# Calculate total recoveries and deaths in South Africa
total_recoveries = final_merged_data [final_merged_data['Country/Region'] == 'South Africa']['Recovered'].sum()
total_deaths = final_merged_data[final_merged_data['Country/Region'] == 'South Africa']['Deaths'].sum()
# Visualization
plt.bar(['Total Recoveries', 'Total Deaths'], [total_recoveries, total_deaths], color=['green', 'red'])
plt.title('Total Recoveries vs. Total Deaths in South Africa')
plt.ylabel('Counts')
plt.show()

# Q8.3: Analyze the ratio of recoveries to confirmed cases for the United States 
# on a monthly basis from March 2020 to May 2021. Which month experienced the 
# highest recovery ratio, and what could be the potential reasons?

import matplotlib.pyplot as plt

# Ensure 'Date' is in datetime format
final_merged_data['Date'] = pd.to_datetime(final_merged_data['Date'])

# Filter data for the US
us_data = final_merged_data[final_merged_data['Country/Region'] == 'US']

# Set 'Date' as the index
us_data.set_index('Date', inplace=True)

# Select only numeric columns for resampling
numeric_columns = ['Confirmed', 'Recovered']
us_data_numeric = us_data[numeric_columns]

# Resample by month and sum the values
us_data_monthly = us_data_numeric.resample('M').sum()

# Calculate the recovery ratio
recovery_ratio_us = us_data_monthly['Recovered'] / us_data_monthly['Confirmed']

# Plot the recovery ratio
fig, ax = plt.subplots()

recovery_ratio_us.plot(kind='line', marker='o', linestyle='-', color='blue', ax=ax)
plt.title('Monthly Recovery Ratio in the United States')
plt.ylabel('Recovery Ratio')
plt.xlabel('Month')
# Set the x-ticks to show all month names
ax.set_xticks(recovery_ratio_us.index)
ax.set_xticklabels([date.strftime('%Y-%m') for date in recovery_ratio_us.index], rotation=45)

plt.grid(True)
plt.show()
