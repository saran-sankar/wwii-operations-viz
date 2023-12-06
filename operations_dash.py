import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('operations.csv')

# Data preprocessing
df.drop(columns=['Target ID', 'Source ID', 'Unit ID'], inplace=True)
df.dropna(subset=['Country'], inplace=True)
df['Mission Date'] = pd.to_datetime(df['Mission Date'])
df['Year'] = df['Mission Date'].dt.year
# Filter the DataFrame to include only columns with more than 50000 values
print('Filtering the DataFrame to include only columns with more than 50000 values...')
useful_columns = df.columns[len(df) - df.isnull().sum() > 50000]
df = df[useful_columns]

def remove_outliers(df, column_names):
    df_outlier_removed = df.copy()

    for column_name in column_names:
        Q1 = df_outlier_removed[column_name].quantile(0.25)
        Q3 = df_outlier_removed[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_outlier_removed = df_outlier_removed[
            (df_outlier_removed[column_name] >= lower_bound) & (df_outlier_removed[column_name] <= upper_bound)]

    return df_outlier_removed


# Specify the numerical features for outlier removal
numerical_features = df.select_dtypes(include='float64').columns

# Remove outliers for all numerical features
df_all_outlier_removed = remove_outliers(df, numerical_features)


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("World War II Aerial Bombing Operations Dashboard"),

    dcc.Tabs([
        dcc.Tab(label='Count Plots', children=[
            html.H2("Count Plots for Categorical Features"),

            html.H3("Select Feature for Count Plot"),
            # Dropdown for selecting features for count plots
            dcc.Dropdown(
                id='count-plot-features-dropdown',
                options=[{'label': feature, 'value': feature}
                         for feature in df.select_dtypes(include='object').columns],
                value=df.select_dtypes(include='object').columns[2],
                multi=False,
                style={'width': '50%'}
            ),

            html.Br(),

            html.Label("Filter by Year Range"),
            dcc.RangeSlider(
                id='year-slider',
                min=df['Year'].min(),
                max=df['Year'].max(),
                step=1,
                marks={str(year): str(year) for year in range(df['Year'].min(), df['Year'].max() + 1)},
                value=[df['Year'].min(), df['Year'].max()]
            ),
            html.Label(id='selected-year-label'),

            html.Br(),

            # Graph component to display the count plots
            dcc.Graph(id='count-plots')
        ]),

       dcc.Tab(label='Trends', children=[html.H2("Trends in Explosives Weights Over Time for Top Countries"),
            html.H3("Select Country"),
            # Dropdown for selecting a country
            dcc.RadioItems(
                id='country-dropdown',
                options=[{'label': country, 'value': country}
                         for country in ['USA', 'GREAT BRITAIN', 'NEW ZEALAND']],
                value=df['Country'].unique()[0],
                style={'width': '50%'}
            ),

            html.H3("Select Features"),
            # Checklist for selecting features to display
            dcc.Checklist(
                id='feature-checklist',
                options=[
                    {'label': 'Total Weight (Tons)',
                     'value': 'Total Weight (Tons)'},
                    {'label': 'High Explosives Weight (Tons)',
                     'value': 'High Explosives Weight (Tons)'}
                ],
                value=['Total Weight (Tons)'],
                inline=True
            ),

            html.Br(),

            # Graph component to display the selected data
            dcc.Graph(id='line-plot'),
            # Loading component
            dcc.Loading(id='loading-indicator', type='circle'),

            # Text area for displaying additional information
            dcc.Textarea(
                id='info-text',
                value='',
                readOnly=True,
                style={'width': '100%', 'height': '100px'}
            ),
      ]),

      dcc.Tab(label='Scatter Plots', children=[html.H2("Regression Plots for Top Countries"),
            html.H3("Select Countries"),
            dcc.Dropdown(
                id='country-dropdown-2',
                options=[{'label': country, 'value': country} for country in ['USA', 'GREAT BRITAIN', 'NEW ZEALAND']],
                value=df['Country'].unique()[0:2],
                multi=True,
                style={'width': '50%'}
            ),

            html.H3("Select Features for Scatter Plot"),
            # Dropdown for selecting X feature
            dcc.Dropdown(
                id='x-feature-dropdown',
                options=[{'label': feature, 'value': feature} for feature in numerical_features],
                value='Target Latitude',
                style={'width': '50%'}
            ),

            # Dropdown for selecting Y feature
            dcc.Dropdown(
                id='y-feature-dropdown',
                options=[{'label': feature, 'value': feature} for feature in numerical_features],
                value='Target Longitude',
                style={'width': '50%'}
            ),

            html.Br(),

            # Graph component to display the selected data
            dcc.Graph(id='scatter-plot')
        ]),
    ]),
])


# Define callback to update the count plots based on user inputs
@app.callback(
    Output('count-plots', 'figure'),
    [Input('count-plot-features-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_count_plots(selected_count_plot_feature, selected_year_range):
    # Filter the data based on the selected year range
    filtered_df = df_all_outlier_removed[
        (df_all_outlier_removed['Year'] >= selected_year_range[0]) &
        (df_all_outlier_removed['Year'] <= selected_year_range[1])
    ]

    # Create count plots using Plotly Express
    fig_count_plots = px.histogram(
        filtered_df,
        x=selected_count_plot_feature,
        title=f'Count Plots for {selected_count_plot_feature} ({selected_year_range[0]} - {selected_year_range[1]})',
        labels={'value': 'Count'},
        template='plotly_dark'
    )

    return fig_count_plots


# Define callback to update the graph based on user inputs
@app.callback(
    [Output('line-plot', 'figure'),
     Output('info-text', 'value')],
    [Input('country-dropdown', 'value'),
     Input('feature-checklist', 'value')]
)
def update_line_plot(selected_country, selected_features):
    # Filter data based on selected country
    filtered_df = df_all_outlier_removed[df_all_outlier_removed['Country'] == selected_country]

    # Group by 'Year' and calculate the mean for selected features
    grouped_df = filtered_df.groupby('Year')[selected_features].mean().reset_index()

    # Create line plot using Plotly Express
    fig = px.line(
        grouped_df,
        x='Year',
        y=selected_features,
        title=f'Mean {", ".join(selected_features)} for {selected_country}',
        labels={'Year': 'Year', 'value': f'Mean {", ".join(selected_features)}'},
        template='plotly_dark'
    )

    # Prepare information text
    info_text = f"Selected Country: {selected_country}\nSelected Features: {', '.join(selected_features)}"

    return fig, info_text


# Define callback to update the regression plot based on user inputs
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('country-dropdown-2', 'value'),
     Input('x-feature-dropdown', 'value'),
     Input('y-feature-dropdown', 'value')]
)
def update_scatter_plot(selected_countries, x_feature, y_feature):
    # Filter data based on selected countries
    filtered_df_scatter = df_all_outlier_removed[
        df_all_outlier_removed['Country'].isin(selected_countries)]

    # Create scatter plot using Plotly Express
    fig_scatter = px.scatter(
        filtered_df_scatter,
        x=x_feature,
        y=y_feature,
        color='Country',
        trendline="ols",
        title=f'Scatter Plot for {y_feature} vs {x_feature} for Selected Countries',
        template='plotly_dark'
    )

    return fig_scatter


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
