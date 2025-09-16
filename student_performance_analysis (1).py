
"""Student Performance Analysis
- Reads the dataset 'student_performance_dataset.csv'
- Cleans and preprocesses data (handles missing values)
- Performs correlation analysis and basic modeling
- Generates plots and saves them to disk
Requirements: pandas, matplotlib, numpy, scikit-learn 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = "/mnt/data/student_performance_project/student_performance_dataset.csv"
OUTPUT_DIR = Path("/mnt/data/student_performance_project") / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    # Basic info
    print('Initial shape:', df.shape)
    # Fill numeric missing values using median (simple, robust)
    num_cols = ['study_hours_per_week','attendance_rate','math_score','physics_score','chemistry_score','english_score','computer_score']
    for c in num_cols:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med)
    # For categorical missingness (if any), fill with mode
    cat_cols = ['gender','parental_education','extracurricular']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    # Recompute derived columns
    df['average_score'] = df[['math_score','physics_score','chemistry_score','english_score','computer_score']].mean(axis=1).round(1)
    df['grade'] = pd.cut(df['average_score'], bins=[-1,32.9,50,65,80,100], labels=['F','D','C','B','A']).astype(str)
    return df

def correlation_analysis(df):
    num_df = df[['study_hours_per_week','attendance_rate','math_score','physics_score','chemistry_score','english_score','computer_score','average_score']]
    corr = num_df.corr()
    print('\nTop correlations with average_score:')
    print(corr['average_score'].sort_values(ascending=False).head(10))
    # Save heatmap
    plt.figure(figsize=(8,6))
    plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation matrix (numeric features)')
    plt.colorbar()
    plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', bbox_inches='tight')
    plt.close()
    return corr

def plots(df):
    # Histogram of average scores
    plt.figure(figsize=(8,5))
    plt.hist(df['average_score'].dropna(), bins=30)
    plt.title('Distribution of Average Scores')
    plt.xlabel('Average Score')
    plt.ylabel('Count')
    plt.savefig(OUTPUT_DIR / 'average_score_histogram.png', bbox_inches='tight')
    plt.close()

    # Scatter: study_hours vs average_score
    plt.figure(figsize=(8,5))
    plt.scatter(df['study_hours_per_week'], df['average_score'], alpha=0.4)
    plt.title('Study Hours per Week vs Average Score')
    plt.xlabel('Study Hours per Week')
    plt.ylabel('Average Score')
    plt.savefig(OUTPUT_DIR / 'study_vs_avg_scatter.png', bbox_inches='tight')
    plt.close()

    # Bar: mean average score by parental education
    edu_means = df.groupby('parental_education')['average_score'].mean().sort_values()
    plt.figure(figsize=(8,5))
    plt.bar(edu_means.index, edu_means.values)
    plt.title('Average Score by Parental Education')
    plt.xlabel('Parental Education')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45)
    plt.savefig(OUTPUT_DIR / 'avg_by_parental_edu.png', bbox_inches='tight')
    plt.close()

def summary_stats(df):
    print('\nSummary statistics:')
    print(df[['math_score','physics_score','chemistry_score','english_score','computer_score','average_score']].describe())

def main():
    df = load_and_clean()
    summary_stats(df)
    corr = correlation_analysis(df)
    plots(df)
    print('\nAnalysis outputs saved in:', OUTPUT_DIR)

if __name__ == '__main__':
    main()
