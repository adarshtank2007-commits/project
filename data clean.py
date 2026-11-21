import pandas as pd
import numpy as np
import re
import traceback
from sklearn.ensemble import RandomForestRegressor

print("Model train ho raha hai bhai... 15 sec ruk ja...")

# 1. File Load
try:
    df = pd.read_csv("movie_box_office_with_errors.csv")
    print(f"File mil gayi! Rows: {len(df)}")
except FileNotFoundError:
    print("ERROR: File nahi mili bhai!")
    print("Check kar - movie_box_office_with_errors.csv aur predictor.py ek hi folder me hai?")
    exit()

# 2. Cleaning - Short wala
df = df.drop_duplicates()
df['Box_Office_Crore_INR'] = pd.to_numeric(df['Box_Office_Crore_INR'], errors='coerce')
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce', dayfirst=True)
df['Release_Year'] = df['Release_Date'].dt.year
df['Release_Month'] = df['Release_Date'].dt.month

def clean_runtime(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s == 'two hours':
        return 120
    m = re.match(r'(\d+)h\s*(\d+)m', s)
    if m:
        return int(m.group(1))*60 + int(m.group(2))
    m = re.match(r'(\d+)\s*min', s)
    if m:
        return int(m.group(1))
    try:
        v = float(s)
        return v if 0 < v <= 300 else np.nan
    except:
        return np.nan

df['Runtime_Minutes'] = df['Runtime_Minutes'].apply(clean_runtime)
df['Genre'] = df['Genre'].astype(str).str.strip().str.lower()
df['Language'] = df['Language'].astype(str).str.strip().str.lower()
df['Certification'] = df['Certification'].astype(str).str.strip().str.upper().replace({'UA': 'U/A'})

# Genre ke typo ko theek karo - lowercase me
df['Genre'] = df['Genre'].replace({
    'sci-fi': 'Sci-Fi', 'sc-ifi': 'Sci-Fi', 'sci-if': 'Sci-Fi',
    'comedy': 'Comedy', 'comdey': 'Comedy',
    'adventure': 'Adventure', 'adventrue': 'Adventure',
    'family': 'Family', 'faimly': 'Family',
    'thriller': 'Thriller', 'thrilelr': 'Thriller',
    'action': 'Action', 'atcion': 'Action',
    'animation': 'Animation', 'drama': 'Drama',
    'romance': 'Romance', 'roamnce': 'Romance',
    'horror': 'Horror', 'horrro': 'Horror',
})
# Language bhi
df['Language'] = df['Language'].replace({
    'hindi': 'Hindi', 'telugu': 'Telugu', 'tamil': 'Tamil',
    'kannada': 'Kannada', 'kannaad': 'Kannada',
    'malayalam': 'Malayalam', 'english': 'English', 'bengali': 'Bengali',
})

df.loc[df['Budget_Crore_INR'] <= 0, 'Budget_Crore_INR'] = np.nan
df.loc[(df['IMDb_Rating'] <= 0) | (df['IMDb_Rating'] > 10), 'IMDb_Rating'] = np.nan
df.loc[df['Box_Office_Crore_INR'] <= 0, 'Box_Office_Crore_INR'] = np.nan

for col in ['Runtime_Minutes','Budget_Crore_INR','Marketing_Budget_Crore_INR','IMDb_Rating','Box_Office_Crore_INR','Release_Year','Release_Month']:
    df[col] = df[col].fillna(df[col].median())
for col in ['Genre','Language','Certification']:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in ['Budget_Crore_INR','Marketing_Budget_Crore_INR','Box_Office_Crore_INR']:
    df[col] = df[col].clip(upper=df[col].quantile(0.99))

df['Total_Cost'] = df['Budget_Crore_INR'] + df['Marketing_Budget_Crore_INR']
df['Log_BoxOffice'] = np.log1p(df['Box_Office_Crore_INR'])

# 3. Model Train
df_model = pd.get_dummies(df, columns=['Genre','Language','Certification'], drop_first=True, dtype=int)
X = df_model.drop(columns=['Movie_ID','Movie_Name','Release_Date','Box_Office_Crore_INR','Log_BoxOffice'])
y = df_model['Log_BoxOffice']
print(f"Training Columns: {list(X.columns)}")

model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X, y)
print("Model READY hai bhai! Ab puch sakta hai.")

# 4. Puchne wala Loop - FIXED
while True:
    print("\n========== NAYI MOVIE ==========")
    name = input("Movie ka naam (exit likh ke band kar): ").strip()
    if name.lower() in ['exit', 'quit', 'band']:
        print("Bye bhai!")
        break

    try:
        budget = float(input("Budget (Cr) - jaise 50: "))
        marketing = float(input("Marketing (Cr) - jaise 10: "))
        runtime = float(input("Runtime (min) - jaise 150: "))
        imdb = float(input("IMDb - jaise 7.5: "))
        year = int(input("Year - jaise 2025: "))
        genre = input("Genre - Action/Comedy/Drama/Romance/Thriller/Horror/Family/Animation/Sci-Fi/Adventure: ").strip()
        lang = input("Language - Hindi/Telugu/Tamil/Kannada/Malayalam/English/Bengali: ").strip()
        cert = input("Certification - U/U/A/A: ").strip().upper()
        if cert == 'UA':
            cert = 'U/A'

        # FIX: Genre ko sahi format me lao
        # Sci-Fi ko Sci-Fi hi rakho, baaki Title case
        if genre.lower() == 'sci-fi':
            genre = 'Sci-Fi'
        else:
            genre = genre.title()
        lang = lang.title()

        # Row banao - X ke columns se hi
        row_dict = {col: 0 for col in X.columns}
        row_dict['Runtime_Minutes'] = runtime
        row_dict['Budget_Crore_INR'] = budget
        row_dict['Marketing_Budget_Crore_INR'] = marketing
        row_dict['IMDb_Rating'] = imdb
        row_dict['Release_Year'] = year
        row_dict['Release_Month'] = 6
        row_dict['Total_Cost'] = budget + marketing

        # One-hot wale column ko 1 karo - agar column exist karta hai to
        for col_name in [f'Genre_{genre}', f'Language_{lang}', f'Certification_{cert}']:
            if col_name in row_dict:
                row_dict[col_name] = 1
            else:
                print(f"Note: {col_name} model me nahi mila, 0 hi rahega.")

        # DataFrame banao aur column order fix karo
        input_df = pd.DataFrame([row_dict])[X.columns]
        print("Predict kar raha hu...")
        pred_log = model.predict(input_df)[0]
        pred_cr = float(np.expm1(pred_log))
        total = budget + marketing

        print("\n---------- RESULT ----------")
        print(f"Movie: {name}")
        print(f"Total Cost: {total} Cr")
        print(f">>> Box Office Collection: {round(pred_cr, 2)} Cr <<<")
        if pred_cr > total:
            print(f"Profit: {round(pred_cr - total, 2)} Cr - HIT hai!")
        else:
            print(f"Loss: {round(total - pred_cr, 2)} Cr - Flop ho sakti hai.")
        print("--------------------------")

    except Exception as e:
        print("\nERROR AA GAYA BHAI, ye raha:")
        traceback.print_exc()
        print("Dobara try kar, number me sirf number likhna.")