"""Membuat data raw copy, data clean, data train, dan data test.
Jalankan dari folder root proyek: python src/make_dataset.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split

SRC = "data/raw/data_hotel_booking_demand.csv"
CONT = "data/external/continents2.csv"

# 1. Data asli (df.copy): salinan utuh sebelum pembersihan
df_raw = pd.read_csv(SRC)
df = df_raw.copy()
df.to_csv("data/raw/data_asli_df_copy.csv", index=False)

# 2. Pembersihan (sama dengan notebook)
df = df[df["market_segment"] != "Undefined"].reset_index(drop=True)

# 3. Feature engineering (sama dengan notebook)
df["is_domestic"] = (df["country"] == "PRT").astype(int)
df_country = pd.read_csv(CONT, keep_default_na=False, na_values=[""])
peta_region = dict(zip(df_country["alpha-3"], df_country["region"]))
peta_sub = dict(zip(df_country["alpha-3"], df_country["sub-region"]))
peta_region.update({"CN": "Asia", "TMP": "Asia"})
peta_sub.update({"CN": "Eastern Asia", "TMP": "South-eastern Asia"})
df["region"] = df["country"].map(peta_region).fillna("Unknown")
df["sub_region"] = df["country"].map(peta_sub).fillna("Unknown")
df.to_csv("data/interim/data_clean.csv", index=False)

# 4. Split train/test (sama dengan notebook: 80/20, stratify, random_state=42)
X = df.drop(columns="is_canceled")
y = df["is_canceled"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train = X_train.assign(is_canceled=y_train)
test = X_test.assign(is_canceled=y_test)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("raw/df.copy :", df_raw.shape)
print("clean       :", df.shape, "| batal", round(df.is_canceled.mean()*100, 2), "%")
print("train       :", train.shape, "| batal", round(train.is_canceled.mean()*100, 2), "%")
print("test        :", test.shape, "| batal", round(test.is_canceled.mean()*100, 2), "%")
print("region unknown:", (df.region=="Unknown").sum(), "| country NaN:", df.country.isna().sum())
