<!-- Write a read me file for the project -->

# Fitness App Data Analysis

## Introduction

This project performs a comprehensive analysis of fitness app data. It includes exploratory data analysis (EDA), user engagement analysis, clustering analysis, and regression analysis. The script generates various visualizations to help understand user behavior, activity patterns, and factors influencing calorie burn.

## Prerequisites

To run this project, you need Python 3.x installed on your system.

## Installation

1. Clone this repository to your local machine.
2. Create a virtual environment (recommended):
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```
     .venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source .venv/bin/activate
     ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Program

1. Ensure you have the dataset file `as2-dataset.csv` in the same directory as the script.
2. Run the main script:
   ```
   python da.py
   ```
3. The script will perform the following analyses:
   - Exploratory Data Analysis (EDA)
   - User Engagement Analysis
   - Clustering Analysis
   - Regression Analysis

4. The script will generate various visualizations and save them in a `figures` directory.

5. Numerical results will be printed to the console, including:
   - Correlation matrix
   - Engagement metrics by activity level and location
   - Cluster profiles
   - Regression results (R-squared and RMSE)
   - Feature importance for predicting calories burned

## Output

The script generates several visualizations, which are saved in the `figures` directory:

1. EDA visualizations:
   - App sessions distribution
   - Calories burned distribution
   - Age distribution
   - Correlation heatmap

2. User engagement analysis:
   - Metrics by activity level
   - Metrics by location

3. Clustering analysis:
   - Cluster scatter plot
   - Cluster distribution by activity level

4. Regression analysis:
   - Prediction vs actual values plot
   - Feature importance plot

## License

This project is open-source and available under the MIT License.
