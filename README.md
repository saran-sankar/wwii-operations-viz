# World War II Aerial Bombing Operations Dashboard

This repository contains a Dash web application that visualizes data related to aerial bombing operations during World War II. The application allows users to explore various aspects of bombing missions, including count plots, trends in explosive weights, and regression analysis.

## Features

- **Count Plots**: Visualize categorical data with the option to filter by year range.
- **Trends**: Analyze trends in explosive weights over time for selected countries.
- **Scatter Plots**: Explore relationships between different numerical features with regression analysis.
- **Image Display**: View an illustrative image related to the topic.

## Getting Started

### Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.x
- pip (Python package installer)

### Installation

1. Clone the repository:  
   ```bash  
   git clone https://github.com/saran-sankar/wwii-operations-viz.git && cd wwii-operations-viz  
   ```

2. Install the required packages:  
   ```bash  
   pip install dash pandas plotly  
   ```

3. Ensure you have the dataset (`operations.csv`) in the repository's root directory along with an image file (`PhotoofdevastatedDresden.jpeg`) for display.

### Running the Application

To start the Dash application, execute the following command in your terminal:  
```bash  
python app.py  
```
Replace `app.py` with the name of the file where the main code resides if different. Once the server is running, open your web browser and navigate to `http://127.0.0.1:8050` to view the dashboard.

## Code Overview

- **Data Preprocessing**: The application loads and preprocesses the `operations.csv` dataset, removing unnecessary columns and handling missing values.
- **Outlier Removal**: Outliers in numerical features are removed for cleaner data visualization.
- **Dash Layout**: The dashboard is structured into tabs for count plots, trends, and scatter plots.
- **Callbacks**: Interactive callbacks update the plots based on user selections.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Thank you to all contributors and libraries used in this project.
