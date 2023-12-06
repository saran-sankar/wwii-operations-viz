# Aerial Bombing Operations in World War II
# Target, aircraft used, and bombs deployed for every mission in WWII
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


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

# Create features
df['Mission Date'] = pd.to_datetime(df['Mission Date'])
df['Year'] = df['Mission Date'].dt.year

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
df_top_target_types_total_weight_outlier_removed = remove_outliers(
    df_top_target_types, 'Total Weight (Tons)')
df_high_explosives_weight_outlier_removed = remove_outliers(df, 'High Explosives Weight (Tons)')
df_high_explosives_weight_and_altitude_outlier_removed = remove_outliers(
    df_high_explosives_weight_outlier_removed, 'Altitude (Hundreds of Feet)')
df_all_outlier_removed = remove_outliers(
    df_high_explosives_weight_and_altitude_outlier_removed, 'Total Weight (Tons)')


# Data Visualization
# Line plot:
# Trends over time
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

# 3D plot and contour plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_all_outlier_removed['Altitude (Hundreds of Feet)'],
           df_all_outlier_removed['Total Weight (Tons)'],
           df_all_outlier_removed['High Explosives Weight (Tons)'],
           c='r', marker='o')
ax.set_xlabel('Altitude (Hundreds of Feet)')
ax.set_ylabel('Total Weight (Tons)')
ax.set_zlabel('High Explosives Weight (Tons)')
ax.set_title('3D Plot with Altitude, Total Weight, and High Explosives Weight')
plt.savefig('3d_plot.png')
plt.show()

# Cluster map
plt.title('Cluster Map of Correlation Matrix')
create_and_save_plot(sns.clustermap, df[numerical_features].corr(), annot=True, cmap='coolwarm')

# Hexbin
plt.title('Hexbin Plot of Altitude vs Total Weight')
create_and_save_plot(df_total_weight_and_altitude_outlier_removed.plot.hexbin,
                     x='Altitude (Hundreds of Feet)',
                     y='Total Weight (Tons)', gridsize=15, cmap='Blues')

# Strip plot
plt.title('Strip Plot of Total Weight by Theater of Operations and Target Type')
create_and_save_plot(sns.stripplot,
                     x='Theater of Operations', y='Total Weight (Tons)',
                     hue='Target Type', data=df_top_target_types_total_weight_outlier_removed,
                     jitter=True, dodge=True)

# Swarm plot
plt.title('Swarm Plot of Total Weight by Theater of Operations and Target Type')
create_and_save_plot(sns.swarmplot,
                     x='Theater of Operations', y='Total Weight (Tons)', hue='Target Type',
                     data=df_top_target_types_total_weight_outlier_removed.sample(1000),
                     dodge=True)


# Set up subplots for Figure 1
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

# Subplot 1: Line plot - High Explosives Weight Over Time
sns.lineplot(x='Year', y='High Explosives Weight (Tons)', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Trends in High Explosives Weight Over Time')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('High Explosives Weight (Tons)')

# Subplot 2: Bar plot - Total Operations by Country
top_countries = df['Country'].value_counts().nlargest(10)
top_countries.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Top 10 Countries with the Most Bombing Operations')
axes[0, 1].set_xlabel('Country')
axes[0, 1].set_ylabel('Number of Operations')

# Subplot 3: Stacked Bar plot - Average Explosives Weight by Country
country_and_explosives_weight = df[['Country', 'High Explosives Weight (Tons)',
                                    'Total Weight (Tons)']].dropna()
average_explosive_weight = country_and_explosives_weight.groupby('Country').mean()
average_explosive_weight.plot(kind='bar', stacked=True, ax=axes[1, 0])
axes[1, 0].set_title('Average High Explosives Weight and Total Weight by Country')
axes[1, 0].set_xlabel('Country')
axes[1, 0].set_ylabel('Average Weight (Tons)')

# Subplot 4: Pie chart - Distribution of Operations in Top Theaters of Operations
axes[1, 1].pie(df_top_theaters_of_ops['Theater of Operations'].value_counts(),
               labels=df_top_theaters_of_ops['Theater of Operations'].value_counts().index,
               autopct='%1.1f%%')
axes[1, 1].set_title('Distribution of Operations in Top Theaters of Operations')

# Adjust layout
plt.tight_layout()
plt.savefig(f"subplot_1.png")
plt.show()


# Set up subplots for Figure 2
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

# Subplot 1: Group Bar plot - High Explosives Weight for USA and GB in Top Theaters of Operations
sns.barplot(x='Country', y='High Explosives Weight (Tons)', hue='Theater of Operations',
            data=df_top_theaters_of_ops[df_top_theaters_of_ops['Country'].isin(['USA', 'GREAT BRITAIN'])],
            ax=axes[0, 0])
axes[0, 0].set_title('High Explosives Weight for USA and GB in Top Theaters of Operations')
axes[0, 0].set_xlabel('Country')
axes[0, 0].set_ylabel('High Explosives Weight (Tons)')

# Subplot 2: Count plot - Number of Operations in Top Theaters of Operations
sns.countplot(x='Theater of Operations', data=df_top_theaters_of_ops, ax=axes[0, 1])
axes[0, 1].set_title('Number of Operations in Top Theaters of Operations')
axes[0, 1].set_xlabel('Theater of Operations')
axes[0, 1].set_ylabel('Number of Operations')

# Subplot 3: Boxen plot and Violin plot - Total Weight in Top Theaters of Operations
sns.boxenplot(x='Theater of Operations', y='Total Weight (Tons)', hue='Target Country',
              data=df_top_theaters_of_ops_total_weight_outlier_removed.sample(10000), ax=axes[1, 0])
axes[1, 0].set_title('Boxen Plot of Total Weight in Top Theaters of Operations')
axes[1, 0].set_xlabel('Theater of Operations')
axes[1, 0].set_ylabel('Total Weight (Tons)')

# Subplot 4: Violin plot - Total Weight in Top Theaters of Operations
sns.violinplot(x='Theater of Operations', y='Total Weight (Tons)',
               data=df_top_theaters_of_ops_total_weight_outlier_removed, ax=axes[1, 1])
axes[1, 1].set_title('Violin Plot of Total Weight in Top Theaters of Operations')
axes[1, 1].set_xlabel('Theater of Operations')
axes[1, 1].set_ylabel('Total Weight (Tons)')

# Adjust layout
plt.tight_layout()
plt.savefig(f"subplot_2.png")
plt.show()


# Set up subplots for Figure 3
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# Subplot 1: Regression plot - Altitude vs. Total Weight
sns.regplot(x='Altitude (Hundreds of Feet)', y='Total Weight (Tons)',
            data=df_total_weight_and_altitude_outlier_removed.sample(n=1000),
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}, ax=axes[0])
axes[0].set_title('Regression Plot of Altitude vs. Total Weight')
axes[0].set_xlabel('Altitude (Hundreds of Feet)')
axes[0].set_ylabel('Total Weight (Tons)')

# Subplot 2: Hexbin plot - Altitude vs Total Weight
df_total_weight_and_altitude_outlier_removed.plot.hexbin(x='Altitude (Hundreds of Feet)',
                                                         y='Total Weight (Tons)',
                                                         gridsize=15, cmap='Blues', ax=axes[1])
axes[1].set_title('Hexbin Plot of Altitude vs Total Weight')
axes[1].set_xlabel('Altitude (Hundreds of Feet)')
axes[1].set_ylabel('Total Weight (Tons)')

# Adjust layout
plt.tight_layout()
plt.savefig(f"subplot_3.png")
plt.show()


# Set up subplots for Figure 4
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

# Subplot 1: Histogram plot with KDE - Total Weight
sns.histplot(x='Total Weight (Tons)',
             data=df_total_weight_outlier_removed.sample(10000), kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Total Weight')
axes[0, 0].set_xlabel('Total Weight (Tons)')
axes[0, 0].set_ylabel('Frequency')

# Subplot 2: KDE plot - Total Weight
sns.kdeplot(x='Total Weight (Tons)', data=df_total_weight_outlier_removed,
            fill=True, alpha=0.6, palette='viridis', linewidth=2, ax=axes[0, 1])
axes[0, 1].set_title('Kernel Density Estimation of Total Weight')
axes[0, 1].set_xlabel('Total Weight (Tons)')
axes[0, 1].set_ylabel('Density')

# Subplot 3: Rug plot - Total Weight
sns.rugplot(x=df_total_weight_outlier_removed['Total Weight (Tons)'].sample(10000, random_state=1),
            ax=axes[1, 0])
axes[1, 0].set_title('Rug Plot of Total Weight')
axes[1, 0].set_xlabel('Total Weight (Tons)')
axes[1, 0].set_ylabel('Density')

# Subplot 4: Hexbin plot - Altitude vs Total Weight
df_total_weight_and_altitude_outlier_removed.plot.hexbin(x='Altitude (Hundreds of Feet)',
                                                         y='Total Weight (Tons)',
                                                         gridsize=15, cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('Hexbin Plot of Altitude vs Total Weight')
axes[1, 1].set_xlabel('Altitude (Hundreds of Feet)')
axes[1, 1].set_ylabel('Total Weight (Tons)')

# Adjust layout
plt.tight_layout()
plt.savefig(f"subplot_4.png")
plt.show()
