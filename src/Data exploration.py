#Importing needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")
import os


# get script location
script_dir = os.path.dirname(os.path.abspath(__file__))

print("WORKING DIR:", os.getcwd())

# Loading data
df_cars = pd.read_csv('../data/used_cars.csv')

#Checking basic info about the data
df_cars.info()

#Cleaning the integer column by removing uncessary signs as mi, ., $ and spaces
df_cars['milage'] = df_cars['milage'].astype(str).str.replace("mi.","")
df_cars['milage'] = df_cars['milage'].str.replace(",","").astype(int)

df_cars['price'] = df_cars['price'].str.replace("$","")
df_cars['price'] = df_cars['price'].str.replace(",","").astype(int)


#Data exploration 
sns.set_theme(style='whitegrid')
plt.figure(figsize=(12,8))
sns.barplot(df_cars['brand'].value_counts().nlargest(20).reset_index(), x='count', y='brand', orient='y', errorbar=None)
plt.title("Top 20 brand of cars count")
plt.show()


sns.set_theme(style='whitegrid')
plt.figure(figsize=(10,10))
sns.barplot(df_cars.groupby('brand')['price'].median().reset_index().sort_values('price'), x='price', y='brand', orient='y', errorbar=None)
plt.title("Median price of each car")
plt.ticklabel_format(style='plain', axis='x')
plt.show()


plt.figure(figsize=(10,8))
sns.boxplot(df_cars[df_cars['brand'].isin(df_cars['brand'].value_counts().nlargest(20).index)] , 
            x='price', 
            y='brand',
           order=df_cars[df_cars['brand'].isin(df_cars['brand'].value_counts().nlargest(10).index)].groupby(['brand'])['price'].median().sort_values(ascending=False).index)

plt.title("Price distribution of top 20 brand")
plt.ticklabel_format(style='plain', axis='x')
plt.show()


fig, axes = plt.subplots(1,2, figsize=(16, 6))
sns.lineplot(df_cars, x='model_year', y='price', estimator='median', errorbar=None, ax=axes[0])
axes[0].set_title('Median Price')
sns.scatterplot(df_cars, x='model_year', y='price', alpha=0.9, ax=axes[1])
axes[1].set_title('Individual car distribution')


plt.figure(figsize=(8,6))
sns.histplot(df_cars['price'], kde=True, bins=1000, color='blue')
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.xlim(0, 150000)
plt.title('Distribution of Price (Zoomed in < $150k)')
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(df_cars['price'], kde=True, log_scale=True) 
plt.title('Distribution of Price (Log Scale)')
plt.show()