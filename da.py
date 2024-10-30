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
    
    # 1.2 Activity Level Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='Activity Level', ax=ax2)
    ax2.set_title('Distribution of Activity Levels')
    plt.tight_layout()
    fig2.savefig('figures/1_2_activity_level_dist.png', dpi=300, bbox_inches='tight')
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
    # Convert Activity Level to numeric for correlation
    le = LabelEncoder()
    df_corr = df.copy()
    df_corr['Activity_Level_Encoded'] = le.fit_transform(df['Activity Level'])
    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
    correlation = df_corr[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax4)
    ax4.set_title('Correlation Heatmap')
    plt.tight_layout()
    fig4.savefig('figures/1_4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    return correlation

# 2. User Engagement Analysis
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
    
    # Activity Level vs App Sessions
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='Activity Level', y='App Sessions', ax=ax1)
    ax1.set_title('App Sessions by Activity Level')
    plt.tight_layout()
    fig1.savefig('figures/2_1_activity_sessions.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Activity Level vs Location
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    activity_location = pd.crosstab(df['Activity Level'], df['Location'], normalize='index') * 100
    activity_location.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_title('Activity Level Distribution by Location')
    ax2.set_ylabel('Percentage')
    plt.legend(title='Location', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    fig2.savefig('figures/2_2_activity_location.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    return engagement_by_activity, engagement_by_location

# 3. Clustering Analysis
def perform_clustering(df):
    # Prepare data for clustering
    features = ['App Sessions', 'Distance Travelled (km)', 'Calories Burned']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Cluster vs Activity Level
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    cluster_activity = pd.crosstab(df['Cluster'], df['Activity Level'])
    cluster_activity.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title('Activity Level Distribution by Cluster')
    ax1.legend(title='Activity Level', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    fig1.savefig('figures/3_1_cluster_activity.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Cluster Characteristics
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='App Sessions', y='Calories Burned', 
                   hue='Activity Level', style='Cluster', ax=ax2)
    ax2.set_title('User Segments: Activity Level and Cluster Distribution')
    plt.tight_layout()
    fig2.savefig('figures/3_2_segments.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    return kmeans.cluster_centers_

# 4. Regression Analysis
def perform_regression(df):
    # Prepare features for regression
    le = LabelEncoder()
    df['Activity_Level_Encoded'] = le.fit_transform(df['Activity Level'])
    df['Location_Encoded'] = le.fit_transform(df['Location'])
    
    # First Regression: Predict Calories Burned
    print("\nPredicting Calories Burned:")
    X_calories = df[['Age', 'Activity_Level_Encoded', 'Location_Encoded', 
                    'Distance Travelled (km)', 'App Sessions']]
    y_calories = df['Calories Burned']
    
    X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(
        X_calories, y_calories, test_size=0.2, random_state=42
    )
    
    model_calories = LinearRegression()
    model_calories.fit(X_train_cal, y_train_cal)
    
    y_pred_cal = model_calories.predict(X_test_cal)
    r2_calories = r2_score(y_test_cal, y_pred_cal)
    rmse_calories = np.sqrt(mean_squared_error(y_test_cal, y_pred_cal))
    
    # Feature importance for Calories Burned prediction
    feature_importance_calories = pd.DataFrame({
        'Feature': X_calories.columns,
        'Importance': abs(model_calories.coef_)
    }).sort_values('Importance', ascending=False)
    
    # Second Regression: Predict App Sessions
    print("\nPredicting App Sessions:")
    X_sessions = df[['Age', 'Activity_Level_Encoded', 'Location_Encoded', 
                    'Distance Travelled (km)', 'Calories Burned']]
    y_sessions = df['App Sessions']
    
    X_train_ses, X_test_ses, y_train_ses, y_test_ses = train_test_split(
        X_sessions, y_sessions, test_size=0.2, random_state=42
    )
    
    model_sessions = LinearRegression()
    model_sessions.fit(X_train_ses, y_train_ses)
    
    y_pred_ses = model_sessions.predict(X_test_ses)
    r2_sessions = r2_score(y_test_ses, y_pred_ses)
    rmse_sessions = np.sqrt(mean_squared_error(y_test_ses, y_pred_ses))
    
    # Feature importance for App Sessions prediction
    feature_importance_sessions = pd.DataFrame({
        'Feature': X_sessions.columns,
        'Importance': abs(model_sessions.coef_)
    }).sort_values('Importance', ascending=False)
    
    # Create subplot for both predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Calories Burned predictions
    ax1.scatter(y_test_cal, y_pred_cal, alpha=0.5)
    ax1.plot([y_test_cal.min(), y_test_cal.max()], 
             [y_test_cal.min(), y_test_cal.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Calories Burned')
    ax1.set_ylabel('Predicted Calories Burned')
    ax1.set_title('Calories Burned: Prediction vs Actual')
    
    # App Sessions predictions
    ax2.scatter(y_test_ses, y_pred_ses, alpha=0.5)
    ax2.plot([y_test_ses.min(), y_test_ses.max()], 
             [y_test_ses.min(), y_test_ses.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual App Sessions')
    ax2.set_ylabel('Predicted App Sessions')
    ax2.set_title('App Sessions: Prediction vs Actual')
    
    plt.tight_layout()
    fig.savefig('figures/4_1_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create subplot for feature importance comparison
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Calories Burned feature importance
    sns.barplot(data=feature_importance_calories, x='Importance', y='Feature', ax=ax3)
    ax3.set_title('Feature Importance: Calories Burned Prediction')
    
    # App Sessions feature importance
    sns.barplot(data=feature_importance_sessions, x='Importance', y='Feature', ax=ax4)
    ax4.set_title('Feature Importance: App Sessions Prediction')
    
    plt.tight_layout()
    fig.savefig('figures/4_2_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'calories_burned': {
            'r2': r2_calories,
            'rmse': rmse_calories,
            'feature_importance': feature_importance_calories
        },
        'app_sessions': {
            'r2': r2_sessions,
            'rmse': rmse_sessions,
            'feature_importance': feature_importance_sessions
        }
    }

# Execute analyses
print("Performing Exploratory Data Analysis...")
correlation = perform_eda(df)

print("\nAnalyzing User Engagement...")
engagement_by_activity, engagement_by_location = analyze_user_engagement(df)

print("\nPerforming Clustering Analysis...")
cluster_centers = perform_clustering(df)

print("\nPerforming Regression Analysis...")
# r2, rmse, feature_importance = perform_regression(df)

# Update the execution part:
print("\nPerforming Regression Analysis...")
regression_results = perform_regression(df)

# Print detailed results
print("\nCalories Burned Prediction Results:")
print(f"R-squared Score: {regression_results['calories_burned']['r2']:.3f}")
print(f"RMSE: {regression_results['calories_burned']['rmse']:.3f}")
print("\nFeature Importance for Calories Burned:")
print(regression_results['calories_burned']['feature_importance'])

print("\nApp Sessions Prediction Results:")
print(f"R-squared Score: {regression_results['app_sessions']['r2']:.3f}")
print(f"RMSE: {regression_results['app_sessions']['rmse']:.3f}")
print("\nFeature Importance for App Sessions:")
print(regression_results['app_sessions']['feature_importance'])

