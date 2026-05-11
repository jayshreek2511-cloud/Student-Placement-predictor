import pickle, pandas as pd, numpy as np

with open('model.pkl','rb') as f:
    md = pickle.load(f)

model = md['model']
enc = md['encoders']
fn = md['feature_names']

# These are the EXACT default values in the HTML form
form_defaults = {"Degree":"B.Tech","Branch":"CSE",
    "CGPA":"8.5","Internships":"1","Projects":"2","Coding_Skills":"7",
    "Communication_Skills":"8","Aptitude_Test_Score":"85",
    "Soft_Skills_Rating":"8","Certifications":"2","Backlogs":"0"}

df = pd.DataFrame([form_defaults])
for col, le in enc.items():
    if col in df.columns:
        try: df[col] = le.transform(df[col])
        except ValueError: df[col] = 0
df = df[fn]
for c in df.columns: df[c] = pd.to_numeric(df[c])
pred = model.predict(df)[0]
proba = model.predict_proba(df)[0]
placed_idx = list(enc['Placement_Status'].classes_).index('Placed')
print(f"Form defaults: {enc['Placement_Status'].inverse_transform([pred])[0]} | Prob: {proba[placed_idx]*100:.1f}%")

# Test borderline / medium-weak values user might try
border = {"Degree":"B.Tech","Branch":"CSE",
    "CGPA":"6.5","Internships":"1","Projects":"2","Coding_Skills":"5",
    "Communication_Skills":"5","Aptitude_Test_Score":"60",
    "Soft_Skills_Rating":"5","Certifications":"1","Backlogs":"1"}

df = pd.DataFrame([border])
for col, le in enc.items():
    if col in df.columns:
        try: df[col] = le.transform(df[col])
        except ValueError: df[col] = 0
df = df[fn]
for c in df.columns: df[c] = pd.to_numeric(df[c])
pred = model.predict(df)[0]
proba = model.predict_proba(df)[0]
print(f"Borderline: {enc['Placement_Status'].inverse_transform([pred])[0]} | Prob: {proba[placed_idx]*100:.1f}%")
print(f"  CGPA=6.5, Coding=5, Comm=5, Aptitude=60, Backlogs=1")

# Test clearly weak
weak = {"Degree":"B.Tech","Branch":"CSE",
    "CGPA":"5.5","Internships":"0","Projects":"1","Coding_Skills":"3",
    "Communication_Skills":"4","Aptitude_Test_Score":"40",
    "Soft_Skills_Rating":"4","Certifications":"0","Backlogs":"2"}

df = pd.DataFrame([weak])
for col, le in enc.items():
    if col in df.columns:
        try: df[col] = le.transform(df[col])
        except ValueError: df[col] = 0
df = df[fn]
for c in df.columns: df[c] = pd.to_numeric(df[c])
pred = model.predict(df)[0]
proba = model.predict_proba(df)[0]
print(f"Weak: {enc['Placement_Status'].inverse_transform([pred])[0]} | Prob: {proba[placed_idx]*100:.1f}%")
print(f"  CGPA=5.5, Coding=3, Comm=4, Aptitude=40, Backlogs=2")
