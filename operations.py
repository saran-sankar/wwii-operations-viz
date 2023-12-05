# Aerial Bombing Operations in World War II
# Target, aircraft used, and bombs deployed for every mission in WWII
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
import numpy as np


# Function to create PrettyTable from DataFrame
def dataframe_to_pretty_table(dataframe):
    x = PrettyTable(list(dataframe.columns))
    for row in dataframe.itertuples(index=False):
        x.add_row(row)
    return x


# Function to create and save static plots
def create_and_save_plot(plot_function, *args, **kwargs):
    plot_function(*args, **kwargs)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_function.__name__}.png")
    plt.show()


# Load the data
df = pd.read_csv('operations.csv')
print(f'There are {len(df)} observations in the dataset')
print(df.head())


# Data preprocessing

# Drop useless columns
df.drop(columns=['Target ID', 'Source ID', 'Unit ID'], inplace=True)

# Filter the DataFrame to include only columns with more than 10000 values
useful_columns = df.columns[len(df) - df.isnull().sum() > 50000]
df = df[useful_columns]

# Fill missing values
numerical_features = df.select_dtypes(include='float64').columns
categorical_features = df.columns[~df.columns.isin(df.select_dtypes(include='float64').columns)]
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].mean())

# Filter the DataFrame to include only the top 10 target countries
top_target_countries = df['Target Country'].value_counts().nlargest(10).index
df_top_target_countries = df[df['Target Country'].isin(top_target_countries)]

# Filter the DataFrame to include only the top 4 theaters of ops
top_theaters_of_ops = df['Theater of Operations'].value_counts().nlargest(4).index
df_top_theaters_of_ops = df_top_target_countries[
    df_top_target_countries['Theater of Operations'].isin(top_theaters_of_ops)]

# Filter the DataFrame to include only the top 10 target types
top_target_types = df['Target Type'].value_counts().nlargest(10).index
df_top_target_types = df_top_theaters_of_ops[
    df_top_theaters_of_ops['Target Type'].isin(top_target_types)]

# Drop missing values
df_reduced = df_top_target_types.dropna(subset=categorical_features)


# Line plot:
# Trends over time
df['Mission Date'] = pd.to_datetime(df['Mission Date'])
df['Year'] = df['Mission Date'].dt.year
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='High Explosives Weight (Tons)', data=df)
plt.title = 'Trends in High Explosives Weight Over Time'
plt.xlabel('Year')
plt.ylabel('High Explosives Weight (Tons)')
plt.savefig(f"lineplot.png")
plt.show()


# Bar plot:
# Bombing operations for each country
top_countries = df['Country'].value_counts()
plt.figure(figsize=(12, 6))
top_countries.plot(kind='bar')
plt.title = 'Bombing Operations for Each Country'
plt.xlabel('Country')
plt.ylabel('Number of Operations')
plt.tight_layout()
plt.savefig(f"barplot_1.png")
plt.show()

# High explosives weight (tons) for each country
country_and_explosives_weight = df[['Country', 'High Explosives Weight (Tons)']].dropna()
average_explosive_weight = country_and_explosives_weight.groupby('Country').mean()
average_explosive_weight.plot(kind='bar', legend=False)
plt.title = 'High Explosives Weight (Tons) for Each Country'
plt.tight_layout()
plt.savefig(f"barplot_2.png")
plt.show()

# Group Bar plot
# High explosives weight (tons) for the USA and GB
create_and_save_plot(sns.barplot, x='Country', y='High Explosives Weight (Tons)',
                     hue='Theater of Operations',
                     data=df_top_theaters_of_ops[df_top_theaters_of_ops['Country'].isin(['USA', 'GREAT BRITAIN'])]
                     .groupby(['Country', 'Theater of Operations'])
                     .sum().reset_index(), errorbar=None)


# Countplot
create_and_save_plot(sns.countplot, x='Theater of Operations', data=df_top_theaters_of_ops)


# Piechart
create_and_save_plot(df_top_theaters_of_ops['Theater of Operations'].value_counts().plot.pie, autopct='%1.1f%%')


# Distplot
create_and_save_plot(sns.histplot, x='High Explosives Weight (Tons)', data=df_reduced, kde=True)


# Pairplot
create_and_save_plot(sns.pairplot,
                     df_reduced[['Total Weight (Tons)',
                         'Altitude (Hundreds of Feet)',
                         'High Explosives Weight (Tons)']])


# Heatmap with color bar
create_and_save_plot(sns.heatmap, df[numerical_features].corr(), annot=True, cmap='coolwarm')


# Histogram plot with KDE
create_and_save_plot(sns.histplot, x='Total Weight (Tons)', data=df_reduced, kde=True)


# QQ-plot
# create_and_save_plot(sns.qqplot, df['Total Weight (Tons)'], line='s')


# KDE plot
create_and_save_plot(sns.kdeplot, x='Total Weight (Tons)', data=df_reduced, fill=True, alpha=0.6, palette='viridis', linewidth=2)


# lm or reg plot with scatter representation and regression line
create_and_save_plot(sns.regplot, x='Altitude (Hundreds of Feet)', y='Total Weight (Tons)', data=df_reduced, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})


# Multivariate Box or Boxen plot
create_and_save_plot(sns.boxenplot, x='Theater of Operations', y='Total Weight (Tons)', hue='Target Country', data=df_reduced)


# Area plot
create_and_save_plot(df.groupby('Theater of Operations')['Total Weight (Tons)'].sum().plot.area)


# Violin plot
create_and_save_plot(sns.violinplot, x='Theater of Operations', y='Total Weight (Tons)', hue='Target Type', split=True, data=df_reduced)


# Joint plot with KDE and scatter representation
create_and_save_plot(sns.jointplot, x='Altitude (Hundreds of Feet)', y='Total Weight (Tons)', data=df_reduced, kind='kde')


# Rugplot
create_and_save_plot(sns.rugplot, x=df_reduced['Total Weight (Tons)'])
