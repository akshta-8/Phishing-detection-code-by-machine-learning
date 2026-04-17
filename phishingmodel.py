import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("dataset_phishing.csv")

# -----------------------------
# 2. DATA CLEANING
# -----------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df = df.select_dtypes(include=[np.number])

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

print("✅ Data cleaned")

# -----------------------------
# 3. SPLIT DATA
# -----------------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -----------------------------
# 4. SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 5. FEATURE SELECTION
# -----------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, y)

importance = np.abs(lr.coef_[0])

# 🔥 stricter selection (faster)
selected = importance > 0.05
X_selected = X_scaled[:, selected]

print("Selected features:", np.sum(selected))

# -----------------------------
# 6. MODEL TRAINING (FAST)
# -----------------------------
rf = RandomForestClassifier(n_estimators=50)  # reduced trees

rf_score = cross_val_score(rf, X_selected, y, cv=3).mean()

print("Random Forest Accuracy:", rf_score)

# Train final model
rf.fit(X_selected, y)

# -----------------------------
# 7. URL FEATURE EXTRACTION
# -----------------------------
def extract_features(url):
    features = []
    
    features.append(len(url))
    features.append(url.count('.'))
    features.append(1 if "https" in url else 0)
    features.append(len(urlparse(url).path))

    features.append(url.count('-'))   # hyphens
    features.append(url.count('@'))   # @ symbol
    features.append(url.count('/'))   # slashes
    
    while len(features) < X.shape[1]:
     features.append(0)
    
    return np.array(features).reshape(1, -1)

# -----------------------------
# 8. PREDICTION
# -----------------------------
url = input("Enter URL: ")

features = extract_features(url)

# Step 1: scale (86 features)
features_scaled = scaler.transform(features)

# Step 2: apply same feature selection
features_selected = features_scaled[:, selected]

# Step 3: predict
prediction = rf.predict(features_selected)

if prediction[0] == 1:
    print("⚠️ Phishing Website (Fake)")
else:
    print("✅ Legitimate Website (Real)")