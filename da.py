import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# Create a directory for saving figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Set style for better visualizations
# 'default'
# 'classic'
# 'ggplot'
# 'bmh'
# 'fivethirtyeight'
# 'seaborn-v0_8'
# 'seaborn-v0_8-darkgrid'
# 'tableau-colorblind10'

plt.style.use('default')
sns.set_theme()

# Load and prepare the data
df = pd.read_csv('./as2-dataset.csv')

# 1. Separated EDA Visualizations
def perform_eda(df):
    # 1.1 App Sessions Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='App Sessions', bins=30, ax=ax1)
    ax1.set_title('Distribution of App Sessions')
    plt.tight_layout()
    fig1.savefig('figures/1_1_app_sessions_dist.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 1.2 Calories Burned Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Calories Burned', bins=30, ax=ax2)
    ax2.set_title('Distribution of Calories Burned')
    plt.tight_layout()
    fig2.savefig('figures/1_2_calories_dist.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 1.3 Age Distribution
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Age', bins=30, ax=ax3)
    ax3.set_title('Age Distribution of Users')
    plt.tight_layout()
    fig3.savefig('figures/1_3_age_dist.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 1.4 Correlation Heatmap
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax4)
    ax4.set_title('Correlation Heatmap')
    plt.tight_layout()
    fig4.savefig('figures/1_4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    return correlation

# 2. Separated User Engagement Analysis
def analyze_user_engagement(df):
    # Calculate metrics
    engagement_by_activity = df.groupby('Activity Level').agg({
        'App Sessions': 'mean',
        'Distance Travelled (km)': 'mean',
        'Calories Burned': 'mean'
    }).round(2)
    
    engagement_by_location = df.groupby('Location').agg({
        'App Sessions': 'mean',
        'Distance Travelled (km)': 'mean',
        'Calories Burned': 'mean'
    }).round(2)
    
    # 2.1 Activity Level Metrics
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    engagement_by_activity.plot(kind='bar', ax=ax1)
    ax1.set_title('Metrics by Activity Level')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    fig1.savefig('figures/2_1_activity_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2.2 Location Metrics
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    engagement_by_location.plot(kind='bar', ax=ax2)
    ax2.set_title('Metrics by Location')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    fig2.savefig('figures/2_2_location_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    return engagement_by_activity, engagement_by_location

# 3. Separated Clustering Analysis
def perform_clustering(df):
    # Prepare data and perform clustering
    features = ['App Sessions', 'Distance Travelled (km)', 'Calories Burned']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 3.1 Cluster Scatter Plot
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='App Sessions', y='Calories Burned', 
                   hue='Cluster', palette='deep', ax=ax1)
    ax1.set_title('Clusters: App Sessions vs Calories Burned')
    plt.tight_layout()
    fig1.savefig('figures/3_1_cluster_scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 3.2 Cluster Distribution
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    cluster_activity = pd.crosstab(df['Cluster'], df['Activity Level'])
    cluster_activity.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_title('Cluster Distribution by Activity Level')
    ax2.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    fig2.savefig('figures/3_2_cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Calculate cluster profiles
    cluster_profiles = df.groupby('Cluster').agg({
        'App Sessions': 'mean',
        'Distance Travelled (km)': 'mean',
        'Calories Burned': 'mean'
    }).round(2)
    
    return cluster_profiles

# 4. Regression Analysis with Visualizations
def perform_regression(df):
    # Prepare features for regression
    le = LabelEncoder()
    df['Activity_Level_Encoded'] = le.fit_transform(df['Activity Level'])
    df['Location_Encoded'] = le.fit_transform(df['Location'])
    
    # Define features and target
    X = df[['Age', 'Activity_Level_Encoded', 'Location_Encoded', 'Distance Travelled (km)']]
    y = df['Calories Burned']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Create figure for predictions vs actual values
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Calories Burned')
    ax1.set_ylabel('Predicted Calories Burned')
    ax1.set_title('Prediction vs Actual Values')
    fig1.tight_layout()
    fig1.savefig('figures/4_regression_predictions.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Feature importance visualization
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': abs(model.coef_)
    }).sort_values('Importance', ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax2)
    ax2.set_title('Feature Importance in Predicting Calories Burned')
    fig2.tight_layout()
    fig2.savefig('figures/5_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    return r2, rmse, feature_importance

# Execute analyses and save visualizations
print("Performing Exploratory Data Analysis...")
correlation = perform_eda(df)

print("\nAnalyzing User Engagement...")
engagement_by_activity, engagement_by_location = analyze_user_engagement(df)

print("\nPerforming Clustering Analysis...")
cluster_profiles = perform_clustering(df)

print("\nPerforming Regression Analysis...")
r2, rmse, feature_importance = perform_regression(df)

# Print numerical results
print("\nCorrelation:")
print(correlation)

print("\nEngagement by Activity:")
print(engagement_by_activity)

print("\nEngagement by Location:")
print(engagement_by_location)

print("\nCluster Profiles:")
print(cluster_profiles)

print("\nRegression Results:")
print(f"R-squared Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

print("\nFeature Importance:")
print(feature_importance)

print("\nAll visualizations have been saved in the 'figures' directory.")
