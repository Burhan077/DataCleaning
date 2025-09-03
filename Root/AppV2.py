import streamlit as st
import pandas as pd
import numpy as np

# Streamlit page config
st.set_page_config(page_title="Smart Data Cleaning Tool", layout="wide")
st.title("🔧 Smart Auto Data Cleaning & EDA Tool")
st.write("This tool automatically cleans your dataset step-by-step while preserving your original data.")
st.sidebar.header("📋 Cleaning Progress")

# File upload
uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 First 25 rows of the dataset")
    st.write(df.head(25))

    # Make a copy
    df_clean = df.copy()
    st.sidebar.write("📄 Made a copy of the dataset for cleaning (df_clean).")

    # 1️⃣ Clean column names
    st.sidebar.write("🧹 Cleaning column names...")
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(" ", "_")

    # 2️⃣ Drop or fill mostly empty columns
    threshold = 0.4  # if more than 40% NaN, consider dropping
    for col in df_clean.columns:
        missing_ratio = df_clean[col].isna().mean()
        if missing_ratio > threshold:
            if df_clean[col].dtype in ['int64', 'float64', 'object']:
                st.sidebar.write(f"⚠️ Column '{col}' has {missing_ratio:.0%} missing values — dropping.")
                df_clean.drop(columns=[col], inplace=True)
        else:
            if df_clean[col].dtype == 'object':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                st.sidebar.write(f"📋 Filled missing values in '{col}' with mode.")
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                st.sidebar.write(f"🔢 Filled missing values in '{col}' with median.")

    # 3️⃣ Remove duplicates
    duplicates_count = df_clean.duplicated().sum()
    df_clean.drop_duplicates(inplace=True)
    st.sidebar.write(f"🗑 Removed {duplicates_count} duplicate rows.")

    # 4️⃣ Handle outliers
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            # Here we cap outliers instead of dropping
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound,
                                     np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col]))
            st.sidebar.write(f"✂️ Capped {outlier_count} outliers in '{col}'.")

    # 5️⃣ Null values check
    null_counts = df_clean.isna().sum().sum()
    if null_counts == 0:
        st.sidebar.write("✅ No null values remain.")
    else:
        st.sidebar.write(f"⚠️ {null_counts} null values still present after cleaning.")

    st.sidebar.success("🎯 Cleaning complete!")
    st.subheader("✅ Cleaned Dataset Preview")
    st.write(df_clean.head(25))

    # Optionally: Let user download cleaned dataset
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Cleaned Data", csv, "cleaned_dataset.csv", "text/csv")
