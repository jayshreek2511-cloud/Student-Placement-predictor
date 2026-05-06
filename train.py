import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

def train_model():
    # Load data
    df = pd.read_csv('train.csv')
    
    # Drop Student_ID as it's just an identifier
    if 'Student_ID' in df.columns:
        df = df.drop('Student_ID', axis=1)
        
    # Categorical columns to encode
    categorical_cols = ['Gender', 'Degree', 'Branch']
    
    # Dictionary to store encoders
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Target encoding
    target_le = LabelEncoder()
    df['Placement_Status'] = target_le.fit_transform(df['Placement_Status'])
    encoders['Placement_Status'] = target_le
    print(f"Encoded Placement_Status: {dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))}")

    # Features and Target
    X = df.drop('Placement_Status', axis=1)
    y = df['Placement_Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model — improved hyperparameters
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy
    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    # ── Feature Importance Chart ──────────────────────────────────
    importances = model.feature_importances_
    feature_labels = [f.replace('_', ' ') for f in X.columns]
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#020617')
    ax.set_facecolor('#0F172A')

    bars = ax.barh(range(len(indices)), importances[indices], color='#6366F1', edgecolor='#22D3EE', linewidth=0.5)

    # Add gradient effect via alpha
    for i, bar in enumerate(bars):
        bar.set_alpha(0.6 + 0.4 * (i / len(bars)))

    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_labels[i] for i in indices], fontsize=11, color='#CBD5E1')
    ax.set_xlabel('Importance', fontsize=12, color='#94A3B8')
    ax.set_title('Feature Importance (Random Forest)', fontsize=16, color='#F8FAFC', pad=15)
    ax.tick_params(axis='x', colors='#94A3B8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1E293B')
    ax.spines['left'].set_color('#1E293B')

    plt.tight_layout()
    plt.savefig('importance.png', dpi=150, bbox_inches='tight', facecolor='#020617')
    plt.close()
    print("Feature importance chart saved to importance.png")

    # ── Save model, encoders, and importance data ─────────────────
    model_data = {
        'model': model,
        'encoders': encoders,
        'feature_names': X.columns.tolist(),
        'feature_importances': dict(zip(X.columns.tolist(), importances.tolist()))
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model and encoders saved to model.pkl")

if __name__ == "__main__":
    train_model()
