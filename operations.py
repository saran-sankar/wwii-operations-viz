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
print(df.head())

# Filter the DataFrame to include only columns with more than 50000 values
print('Filtering the DataFrame to include only columns with more than 50000 values...')
useful_columns = df.columns[len(df) - df.isnull().sum() > 50000]
df = df[useful_columns]
print(f'Useful features: {useful_columns}')

# Fill missing values
numerical_features = df.select_dtypes(include='float64').columns
categorical_features = df.columns[~df.columns.isin(
    df.select_dtypes(include='float64').columns)]
df[numerical_features] = df[numerical_features].fillna(
    df[numerical_features].mean())
print(df.head())

# Filter the DataFrame to include only the top 10 target countries
top_target_countries = df[
    'Target Country'].value_counts().nlargest(10).index
df_top_target_countries = df[df['Target Country'].isin(
    top_target_countries)]
print('DataFrame that only includes the top 10 target countries:\n', df_top_target_countries)

# Filter the DataFrame to include only the top 4 theaters of ops
top_theaters_of_ops = df[
    'Theater of Operations'].value_counts().nlargest(4).index
df_top_theaters_of_ops = df_top_target_countries[
    df_top_target_countries['Theater of Operations'].isin(
        top_theaters_of_ops)]
print('DataFrame that further only includes the top 4 theaters of ops:\n', df_top_theaters_of_ops)

# Filter the DataFrame to include only the top 10 target types
top_target_types = df['Target Type'].value_counts().nlargest(10).index
df_top_target_types = df_top_theaters_of_ops[
    df_top_theaters_of_ops['Target Type'].isin(top_target_types)]
print('DataFrame that further only includes the top 10 target types:\n', df_top_target_types)

# Drop missing values for categorical features
# df_reduced = df.dropna(subset=categorical_features)
# df_top_theaters_of_ops_reduced = df_top_target_types.dropna(
#     subset=categorical_features)


# Outlier detection and removal (IQR method)
def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_outlier_removed = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return df_outlier_removed


df_total_weight_outlier_removed = remove_outliers(df, 'Total Weight (Tons)')
df_total_weight_and_altitude_outlier_removed = remove_outliers(
    df_total_weight_outlier_removed, 'Altitude (Hundreds of Feet)')
df_top_theaters_of_ops_total_weight_outlier_removed = remove_outliers(
    df_top_theaters_of_ops, 'Total Weight (Tons)')
df_high_explosives_weight_outlier_removed = remove_outliers(df, 'High Explosives Weight (Tons)')
df_high_explosives_weight_and_altitude_outlier_removed = remove_outliers(
    df_high_explosives_weight_outlier_removed, 'Altitude (Hundreds of Feet)')
df_all_outlier_removed = remove_outliers(
    df_high_explosives_weight_and_altitude_outlier_removed, 'Total Weight (Tons)')

# Line plot:
# Trends over time
df['Mission Date'] = pd.to_datetime(df['Mission Date'])
df['Year'] = df['Mission Date'].dt.year
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='High Explosives Weight (Tons)', data=df)
plt.title('Trends in High Explosives Weight Over Time')
plt.xlabel('Year')
plt.ylabel('High Explosives Weight (Tons)')
plt.savefig(f"lineplot.png")
plt.show()

# Bar plot 1:
# Bombing operations for each country
top_countries = df['Country'].value_counts()
plt.figure(figsize=(12, 6))
top_countries.plot(kind='bar')
plt.title('Bombing Operations for Each Country')
plt.xlabel('Country')
plt.ylabel('Number of Operations')
plt.tight_layout()
plt.savefig(f"barplot_1.png")
plt.show()

# Bar plot 2:
# High explosives weight (tons) for each country
country_and_explosives_weight = df[['Country', 'High Explosives Weight (Tons)', 'Total Weight (Tons)']].dropna()
average_explosive_weight = country_and_explosives_weight.groupby('Country').mean()
average_explosive_weight.plot(kind='bar',
                              label=['High Explosives Weight (Tons)', 'Total Weight (Tons)'], stacked=True)
plt.title('Average High Explosives Weight and Total Weight for Each Country')
plt.xlabel('Country')
plt.ylabel('Average Weight (Tons)')
plt.legend()
plt.tight_layout()
plt.savefig("stacked_barplot.png")
plt.show()

# Group Bar plot
# High explosives weight (tons) for the USA and GB
plt.title('High Explosives Weight for USA and GB in Top Theaters of Operations')
create_and_save_plot(sns.barplot, x='Country', y='High Explosives Weight (Tons)',
                     hue='Theater of Operations',
                     data=df_top_theaters_of_ops[df_top_theaters_of_ops['Country']
                     .isin(['USA', 'GREAT BRITAIN'])]
                     .groupby(['Country', 'Theater of Operations'])
                     .sum().reset_index(), errorbar=None)

# Count plot
plt.title('Number of Operations in Top Theaters of Operations')
create_and_save_plot(sns.countplot, x='Theater of Operations',
                     data=df_top_theaters_of_ops)

# Pie chart
plt.title('Distribution of Operations in Top Theaters of Operations')
create_and_save_plot(df_top_theaters_of_ops['Theater of Operations']
                     .value_counts().plot.pie, autopct='%1.1f%%')

# Dist plot
plt.title('Distribution of High Explosives Weight')
create_and_save_plot(sns.histplot, x='High Explosives Weight (Tons)',
                     data=df_high_explosives_weight_outlier_removed, kde=True)

# Pair plot
plt.title('Pair Plot of Total Weight, Altitude, and High Explosives Weight')
create_and_save_plot(sns.pairplot,
                     df_all_outlier_removed.sample(1000)[['Total Weight (Tons)',
                         'Altitude (Hundreds of Feet)',
                         'High Explosives Weight (Tons)']])

# Heatmap with color bar
plt.title('Correlation Heatmap of Numerical Features')
create_and_save_plot(sns.heatmap, df[numerical_features].corr(),
                     annot=True, cmap='coolwarm')

# Histogram plot with KDE
plt.title('Distribution of Total Weight')
create_and_save_plot(sns.histplot, x='Total Weight (Tons)',
                     data=df_total_weight_outlier_removed.sample(10000), kde=True)

# KDE plot
plt.title('Kernel Density Estimation of Total Weight')
create_and_save_plot(sns.kdeplot, x='Total Weight (Tons)', data=df_total_weight_outlier_removed,
                     fill=True, alpha=0.6, palette='viridis', linewidth=2)

# lm or reg plot with scatter representation and regression line
plt.title('Regression Plot of Altitude vs. Total Weight')
create_and_save_plot(sns.regplot, x='Altitude (Hundreds of Feet)', y='Total Weight (Tons)',
                     data=df_total_weight_and_altitude_outlier_removed.sample(n=1000), scatter_kws={'alpha': 0.6},
                     line_kws={'color': 'red'})

# Multivariate Box or Boxen plot
plt.title('Boxen Plot of Total Weight in Top Theaters of Operations')
create_and_save_plot(sns.boxenplot, x='Theater of Operations', y='Total Weight (Tons)', hue='Target Country',
                     data=df_top_theaters_of_ops_total_weight_outlier_removed.sample(10000))

# Area plot
plt.title('Total Weight Over Different Theaters of Operations')
create_and_save_plot(df.groupby('Theater of Operations')['Total Weight (Tons)'].sum().plot.area)

# Violin plot
plt.title('Violin Plot of Total Weight in Top Theaters of Operations')
create_and_save_plot(sns.violinplot, x='Theater of Operations', y='Total Weight (Tons)',
                     data=df_top_theaters_of_ops_total_weight_outlier_removed)

# Joint plot with KDE and scatter representation
plt.title('Joint Plot of Altitude vs. Total Weight')
create_and_save_plot(sns.jointplot, x='Altitude (Hundreds of Feet)', y='Total Weight (Tons)',
                     data=df_total_weight_and_altitude_outlier_removed.sample(n=10000), kind='kde')

# Rugplot
plt.title('Rug Plot of Total Weight')
sns.kdeplot(data=df_total_weight_outlier_removed.sample(10000, random_state=1),
            x='Total Weight (Tons)')
create_and_save_plot(sns.rugplot,
                     x=df_total_weight_outlier_removed[
                         'Total Weight (Tons)'].sample(10000, random_state=1))
