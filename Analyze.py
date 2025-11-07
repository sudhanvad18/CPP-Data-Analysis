# ===============================================================
# ECE Corporate Partnerships Program: Analysis & Visualization 
# ===============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import numpy as np

# Suppress seaborn FutureWarnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------
file_path = "ECE Corporate Partnerships - Sheet1.csv"
df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")

# Remove leading/trailing spaces and line breaks from column names
df.columns = df.columns.str.strip().str.replace(r"[\n\r]+", " ", regex=True).str.replace(" ", "_")

# Clean all text values
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.replace(r"[\n\r]+", " ", regex=True).str.strip()

df = df.dropna(how="all")

print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
print("Columns:", list(df.columns))

# ---------------------------------------------------------------
# Expand multi-value partner and sector columns
# ---------------------------------------------------------------
if "Corporate_Partners" in df.columns:
    df_expanded = df.assign(Corporate_Partners=df["Corporate_Partners"].str.split(",")).explode("Corporate_Partners")
    df_expanded["Corporate_Partners"] = df_expanded["Corporate_Partners"].str.strip()
else:
    raise KeyError("The dataset does not have a 'Corporate_Partners' column. Check your CSV headers.")

# ---------------------------------------------------------------
# Partners per college
# ---------------------------------------------------------------
if "College" in df_expanded.columns:
    partners_per_college = (
        df_expanded.groupby("College")["Corporate_Partners"]
        .count()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Corporate_Partners": "Count"})
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(y="College", x="Count", data=partners_per_college, palette="Blues_r")
    plt.title("Number of Corporate Partners per ECE Program")
    plt.xlabel("Number of Partners")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("partners_per_college.png", dpi=300)
    plt.close()
else:
    print("Skipping partners-per-college chart — 'College' column not found.")

# ---------------------------------------------------------------
# Technology sector frequency
# ---------------------------------------------------------------
possible_sector_cols = [c for c in df_expanded.columns if "sector" in c.lower() or "industry" in c.lower()]

if possible_sector_cols:
    sector_col = possible_sector_cols[0]
    print(f"Using column '{sector_col}' for sector frequency analysis.")

    df_expanded[sector_col] = df_expanded[sector_col].fillna("")
    df_sectors = df_expanded.assign(
        sector_split=df_expanded[sector_col].astype(str).str.split(",")
    ).explode("sector_split")
    df_sectors["sector_split"] = df_sectors["sector_split"].str.strip()
    df_sectors = df_sectors[df_sectors["sector_split"] != ""]

    sector_freq = (
        df_sectors["sector_split"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Sector", "sector_split": "Count"})
    )

    sector_freq.columns = ["Sector", "Count"]

    print("Columns in sector_freq:", sector_freq.columns)
    print(sector_freq.head())

    if not sector_freq.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(y="Sector", x="Count", data=sector_freq.head(15), palette="viridis")
        plt.title("Top Technology Sectors Represented by Corporate Partners")
        plt.xlabel("Count of Appearances")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig("sector_frequency.png", dpi=300)
        plt.close()
    else:
        print("No data available to plot sector frequencies.")
else:
    print("Skipping sector analysis — no 'Sector' or 'Industry' column found.")


# ---------------------------------------------------------------
# Membership tiers
# ---------------------------------------------------------------
if "Tiers_of_Membership" in df.columns:
    def extract_num_tiers(value):
        if pd.isna(value):
            return np.nan
        text = str(value).lower()
        return len(text.split(","))
        

    df["Num_Tiers"] = df["Tiers_of_Membership"].apply(extract_num_tiers)

    # Count how many colleges have 1, 2, 3... tiers
    tier_counts = (
        df["Num_Tiers"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    tier_counts.columns = ["Num_Tiers", "Count"]

    plt.figure(figsize=(7, 5))
    sns.barplot(
        x="Num_Tiers",
        y="Count",
        data=tier_counts,
        palette="crest"
    )
    plt.title("Distribution of Number of Membership Tiers", fontsize=14)
    plt.xlabel("Number of Tiers in Program")
    plt.ylabel("Number of Colleges")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add labels on bars
    for index, row in tier_counts.iterrows():
        plt.text(index, row["Count"] + 0.3, str(int(row["Count"])), ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("membership_tier_count_distribution.png", dpi=300)
    plt.close()
    print("Saved: membership_tier_count_distribution.png")
else:
    print("Column 'Tiers_of_Membership' not found!")

# ---------------------------------------------------------------
# Membership fees
# ---------------------------------------------------------------
if "Fee_Info" in df.columns:
    fee_df = df[df["Fee_Info"].notna()].copy()
    def parse_fee(value):
        """Extract a reasonable numeric fee estimate from messy text."""
        if pd.isna(value):
            return np.nan

        value = str(value).replace(",", "")
        match = re.search(r"(\d{2,7})", value)  # capture up to millions safely
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return np.nan
        return np.nan

    fee_df["Fee"] = fee_df["Fee_Info"].apply(parse_fee)


    plt.figure(figsize=(8, 5))
    sns.boxplot(y="Fee", x="College", data=fee_df)
    plt.title("Distribution of Reported Membership Fees (USD)")
    plt.ylabel("Fee (approx.)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("membership_fees.png", dpi=300)
    plt.close()
else:
    print("Skipping fee analysis — 'Fee_Info' column not found.")

def parse_fee(value):
    """Extract first reasonable number from text"""
    if pd.isna(value):
        return np.nan
    value = str(value).replace(",", "")
    match = re.search(r"(\d{2,7})", value)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan

if "Fee_Info" in df.columns:
    fee_df = df.copy()
    fee_df["Fee"] = fee_df["Fee_Info"].apply(parse_fee)
    plt.figure(figsize=(8, 5))
    sns.histplot(fee_df["Fee"].dropna(), bins=15, kde=True)
    plt.title("Distribution of Membership Fees (USD, Approx.)", fontsize=14)
    plt.xlabel("Fee Amount (USD)")
    plt.ylabel("Number of Colleges")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("membership_fees_dist.png", dpi=300)
    plt.close()
    print("Saved: membership_fees_dist.png")

# ---------------------------------------------------------------
# Department vs College level partnership programs
# ---------------------------------------------------------------

if "Program_Level" in df.columns:
    def categorize_level(val):
        text = str(val).lower()
        
        if "college" in text:
            return "ECE Programs"
        else:
            return "Departmental / General College Programs"

    df["Level_Group"] = df["Program_Level"].apply(categorize_level)
    summary = df["Level_Group"].value_counts().reset_index()
    summary.columns = ["Program Type", "Count"]

    plt.figure(figsize=(6, 5))
    sns.barplot(
        x="Program Type",
        y="Count",
        data=summary,
        palette="flare"
    )
    plt.title("Comparison: Departmental vs ECE-Level Programs", fontsize=12)
    plt.xlabel("")
    plt.ylabel("Number of Partner Programs")
    plt.xticks(rotation=15, ha="right", fontsize=10)

    # Add data labels
    for i, row in summary.iterrows():
        plt.text(i, row["Count"] + 0.3, str(row["Count"]), ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("dept_vs_ece_comparison.png", dpi=300)
    plt.close()
    print("Saved: dept_vs_ece_comparison.png")
else:
    print("Column 'Program_Level' not found!")

# ---------------------------------------------------------------
# Frequency of Corporate Partner appearances
# ---------------------------------------------------------------

company_list = []

for partners in df["Corporate_Partners"].dropna():
    for company in str(partners).split(","):
        clean_name = company.strip().title()  # normalize capitalization
        if clean_name:
            company_list.append(clean_name)

company_freq = pd.Series(company_list).value_counts().reset_index()
company_freq.columns = ["Company", "Appearances"]

print("Unique companies found:", len(company_freq))
print("Top 10 most frequent companies:\n", company_freq.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(
    y="Company",
    x="Appearances",
    data=company_freq.head(15),  # top 15 most frequent
    palette="crest"
)

plt.figure(figsize=(8,5))
sns.histplot(company_freq["Appearances"], bins=range(1, company_freq["Appearances"].max() + 2), kde=False, color="skyblue")
plt.title("Distribution of Company Partnership Frequency Across ECE Programs", fontsize=14, weight="bold")
plt.xlabel("Number of Universities Partnered With")
plt.ylabel("Number of Companies")
plt.tight_layout()
plt.show()

plt.savefig("company_frequency_distribution.png", dpi=300)

# ---------------------------------------------------------------
# Export cleaned datasets
# ---------------------------------------------------------------
df_expanded.to_csv("ece_partnerships_cleaned.csv", index=False)
print("Cleaned dataset saved as 'ece_partnerships_cleaned.csv'")

