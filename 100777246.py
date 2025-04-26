"""
Credit Card Fraud Detection Dashboard
=====================================

This Streamlit web application provides interactive data analysis, anomaly detection, 
and forecasting based on the Credit Card Fraud Detection dataset from Kaggle (originally by ULB).

Key Features:
- Data filtering by transaction type, amount, and time of day
- Visualization of transaction behavior and fraud patterns
- Anomaly detection using Isolation Forest
- Feature importance analysis using Random Forest and SHAP explainability
- Fraud risk profiling across different transaction bins and times
- Time-series forecasting of fraud trends using Prophet
- Exporting filtered datasets for Tableau visualization

Author:
    100777246

Dataset:
    - Credit Card Fraud Detection dataset (Kaggle, ULB)
    - 284,807 transactions over 2 days (2013), anonymized with PCA

Modules Used:
    Streamlit, Pandas, Seaborn, Matplotlib, Scikit-learn, SHAP, Prophet, FPDF

"""

#  *************************************************************************
# 📦 Standard Library Imports
import os                         # To interact with the operating system (check for file existence).
import zipfile                    # Work with .zip compressed files (unzip).
import subprocess                 # To run external system commands (download data using Kaggle CLI).
from io import BytesIO             # To handle in-memory file-like objects for downloads (without saving to disk).
from textwrap import wrap          # To format long strings into wrapped text.
#  *************************************************************************

#  *************************************************************************
# 📚 Data Handling and Analysis
import pandas as pd                # To load, manipulate, and analyze tabular data (CSV files).
import numpy as np                 # For numerical operations and array management.
#  *************************************************************************

#  *************************************************************************
# 📊 Visualization Libraries
import seaborn as sns              # To create stylish statistical plots easily.
import matplotlib.pyplot as plt    # To build flexible visualizations and figures.
import matplotlib.ticker as mtick  # To format axes ticks like currency or percentage.
from mpl_toolkits.mplot3d import Axes3D  # To enable 3D plotting (although not heavily used here).
#  *************************************************************************

#  *************************************************************************
# 🧠 Machine Learning & Explainability
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets.
from sklearn.ensemble import RandomForestClassifier   # To train a Random Forest model for feature importance analysis.
from sklearn.ensemble import IsolationForest          # To detect anomalies (outlier transactions) without needing labels.
import shap                     # To compute SHAP values for model explainability (explain predictions).
from shap import plots           # To visualize SHAP value summaries.
#  *************************************************************************

#  *************************************************************************
# 📈 Dimensionality Reduction
from sklearn.manifold import TSNE # To perform dimensionality reduction for complex visualizations.
import umap                       # To perform alternative dimensionality reduction (similar to t-SNE).
#  *************************************************************************

#  *************************************************************************
# 🔮 Time-Series Forecasting
from prophet import Prophet       # To forecast time series trends in fraud cases.
#  *************************************************************************

#  *************************************************************************
# 🌐 Streamlit App Framework
import streamlit as st             # To create the interactive dashboard web application.
# import streamlit.components.v1 as components  # (currently unused)
#  *************************************************************************

#  *************************************************************************
# 📄 File Export and Reporting
from fpdf import FPDF              # To generate PDF reports from data and visualizations.
import io
from PIL import Image
import datetime
#  *************************************************************************


# 🎨 Global Seaborn Styling
sns.set_theme(
    style="whitegrid",
    palette="muted",
    font_scale=1.1,
    rc={
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": True,
        "axes.facecolor": "white",
        "grid.linestyle": "--",
        "grid.alpha": 0.3
    }
)

#  *************************************************************************
# Page Configuration and Title
#  *************************************************************************
st.set_page_config(page_title="Inside the Swipe: Tracking and Profiling Credit Card Fraud", layout="wide")
st.title("Inside the Swipe: Tracking and Profiling Credit Card Fraud")
st.caption("Uncovering hidden fraud patterns through behavioral analytics, machine learning, and explainable AI.")

#  *************************************************************************
# Custom Styling for Streamlit Elements
#  *************************************************************************
# Apply custom CSS styles to enhance the visual appearance of the dashboard:
# - Sidebar background color set to white.
# - Sidebar text color set to black with adjusted font size.
# - Dropdown menus styled with black background and rounded borders.
# - Expander headers styled to have bold font and blue color.
# Note: 'unsafe_allow_html=True' it allows inserting raw HTML and CSS into Streamlit.
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: white !important;
        }
        section[data-testid="stSidebar"] * {
            color: #000 !important;
            font-size: 1rem !important;
        }
        div[data-baseweb="select"] {
            background-color:#000 !important;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        div[data-baseweb="select"] * {
            color: white !important;
        }
        .st-expander > summary {
            font-weight: bold;
            color: #2e6da4;
        }
    </style>
""", unsafe_allow_html=True)

# Loading of dataset directy from Kaggle
def download_dataset():

    """
    Downloads the Credit Card Fraud Detection dataset from Kaggle if it is not already present.

    Checks if the file 'creditcardfraud.zip' exists in the working directory.
    If the file is missing, it triggers a download command using the Kaggle CLI.

    Important Note:
        If some forgot to install the Kaggle CLI or didn't configure API keys which will cause the Kaggle download command fails (subprocess.CalledProcessError)
    """

    if not os.path.exists("creditcardfraud.zip"):
        st.info("📥 Downloading dataset...") # This shows a Streamlit message saying "Downloading..." so users know what’s happening.
        subprocess.run(["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud"], check=True) # Use system command kaggle datasets download to download the dataset from Kaggle. check=True makes Python throw an error if the download fails.

def extract_dataset():

    """
    Extracts the Credit Card Fraud Detection dataset from the downloaded ZIP file.

    Checks if the extracted CSV file ('data/creditcard.csv') already exists.
    If not, it extracts all contents from 'creditcardfraud.zip' into the 'data/' directory.

    Important Note:
        If the creditcardfraud.zip is missing, Python will throw a FileNotFoundError.
        If the ZIP file is damaged, Python will throw a BadZipFile error.
    """

    if not os.path.exists("data/creditcard.csv"):
        with zipfile.ZipFile("creditcardfraud.zip", "r") as zip_ref: # Opens the downloaded zip file in "read" mode.
            zip_ref.extractall("data") # Extracts all files inside the ZIP into the data/ folder.

@st.cache_data # tells Streamlit to cache the result — meaning if users re-run the app, it won’t reload from CSV every time (faster performance).
def load_data():

    """
    Loads the Credit Card Fraud Detection dataset into a pandas DataFrame.

    This function:
    - Ensures the dataset is downloaded and extracted (calls 'download_dataset()' and 'extract_dataset()').
    - Reads the CSV file from the 'data/' directory.
    - Adds a new column 'Hour' by converting 'Time' in seconds to hours.
    - Ensures the 'Class' column is treated as integer type (0 = normal, 1 = fraud).
    - Caches the result for faster repeated access using Streamlit's cache mechanism.

    Returns:
        pd.DataFrame: The loaded and preprocessed transaction dataset.
    
    Important Note:
        This function relies on download_dataset() and extract_dataset() being successful.
        If the CSV is missing or broken, pd.read_csv() will throw an error — but that's okay for now because downloading and extraction happen first.
    """
        
    download_dataset() # Makes sure the dataset ZIP is downloaded.
    extract_dataset() # Makes sure the dataset is extracted to CSV.
    df = pd.read_csv("data/creditcard.csv") # Loads the extracted CSV file into a pandas DataFrame.
    df["Hour"] = (df["Time"] // 3600).astype(int) # Adds a new column Hour to represent the transaction time grouped into hours.
    df["Class"] = df["Class"].astype(int) # Makes sure the class labels are integers (good for ML models).
    return df # Returns the preprocessed DataFrame. This df becomes our main dataset used throughout the whole dashboard.

#  *************************************************************************
# Load and Prepare Main Dataset
# calls the load_data() function
# Loads the credit card transactions into a DataFrame.
# Ensures dataset is downloaded, extracted, and preprocessed.
# Adds 'Hour' and corrects 'Class' datatype.
# This DataFrame will be used for all filtering, analysis, and modeling.
# It is like setting up the foundation of the building. Without it, none of the visualizations or models would work because there would be no data loaded!
#  *************************************************************************

df = load_data() 

#  *************************************************************************
# Sidebar Setup: Introduction and Transaction Filters
# - Display welcome message and dashboard purpose in the sidebar.
# - Provide three user filters:
#   1. Transaction Type: All, Fraud, or Non-Fraud transactions.
#   2. Hour of Day: Range slider from 0 to 23 hours.
#   3. Amount (€): Range slider based on dataset minimum and maximum amounts.
# These filters dynamically control which transactions are shown and analyzed in the main dashboard.
#  *************************************************************************

with st.sidebar: # Creates a sidebar area for controls and information.
    st.markdown("""
    ## 👋 Welcome!

    Explore transaction patterns, detect fraudulent behavior, and uncover hidden risks through interactive visualizations.  
    Use the filters below to customize your analysis.
    """)

    st.markdown("## 🔎 Filter Transactions")

    class_filter = st.selectbox("Select Transaction Type", options=["All", "Fraud", "Non-Fraud"])
    hour_range = st.slider("Select Hour Range", 0, 23, (0, 23))
    amount_range = st.slider("Select Amount Range (€)", 0, 10000, (0, 10000))

    #  *************************************************************************
    # Dataset Filtering, Anomaly Detection, and Export Setup
    # - Create `filtered_df` as a working copy of the loaded dataset.
    # - Apply Isolation Forest to detect anomalies (outliers) and add 'Anomaly' column.
    # - Create transaction amount bins and add 'AmountBin' column for risk analysis.
    # - Prepare an in-memory CSV file (BytesIO) for exporting enriched data.
    # - Provide a download button in the sidebar for users to export the filtered dataset.
    # - Warn if `filtered_df` is not found (as a safeguard).
    #  *************************************************************************

    filtered_df = df.copy() # 	Make a safe working copy of the loaded dataset.

if 'filtered_df' in locals(): # Make sure filtered_df exists before proceeding (extra protection).

    export_df = filtered_df.copy() # Another copy, so you don't accidentally modify the sidebar view.

    # Detects outliers (anomalous transactions) and marks them with -1 (anomaly) and 1 (normal).
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42) 
    features_for_anomaly = export_df.drop(columns=["Time", "Class"], errors='ignore')
    export_df["Anomaly"] = iso.fit_predict(features_for_anomaly)

    # Groups transaction amounts into specific ranges (e.g., €0–10, €10–50).
    export_df["AmountBin"] = pd.cut( 
        export_df["Amount"],
        bins=[0, 10, 50, 100, 500, 1000, 5000, 10000],
        right=False,
        include_lowest=True
    )

    # Export for Tableau
    csv_buffer = BytesIO() # Prepares an in-memory file to download filtered data without saving to disk first.
    export_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    csv_buffer.name = "filtered_data_for_tableau.csv"

    # Lets users export the enriched dataset (with Anomaly and AmountBin) as a CSV.
    with st.sidebar:
        st.download_button(
            label="📥 Download Tableau CSV",
            data=csv_buffer,
            file_name="filtered_data_for_tableau.csv",
            mime="text/csv"
        )
        st.caption("📄 Download includes anomalies and amount bins for advanced analysis.")

        st.success("📁 Exported enriched filtered_df for Tableau as CSV")
else:
    st.warning("⚠️ filtered_df not found. Please make sure it's defined before exporting.")  # If something goes wrong (e.g., filtered_df missing), shows a Streamlit warning.

#  *************************************************************************
# Apply Sidebar Filters to Dataset
# - Filter by transaction type based on 'Class':
#     'Fraud' => keep only rows where Class == 1
#     'Non-Fraud' => keep only rows where Class == 0
#     'All' => no filtering applied
# - Filter by selected hour range (Hour of Day).
# - Filter by selected amount range (Transaction Amount).
# All your charts, models, and metrics after this are based on this dynamically filtered data.
#  *************************************************************************

if class_filter == "Fraud":
    filtered_df = filtered_df[filtered_df["Class"] == 1] # If user selects "Fraud", keep only fraudulent transactions (Class == 1).
elif class_filter == "Non-Fraud":
    filtered_df = filtered_df[filtered_df["Class"] == 0] # 	If user selects "Non-Fraud", keep only normal transactions (Class == 0).

filtered_df = filtered_df[
    (filtered_df["Hour"] >= hour_range[0]) & (filtered_df["Hour"] <= hour_range[1]) & # Keep only transactions that happened during the selected hour range.
    (filtered_df["Amount"] >= amount_range[0]) & (filtered_df["Amount"] <= amount_range[1]) # Keep only transactions within the selected amount (€) range.
]

all_figures_for_pdf = []

#  *************************************************************************
# Transaction Behavior Analysis (Transactions per Hour)
# - Create a new 'Hour' column by converting 'Time' in seconds to hour of the day (0–23).
# - Count the number of transactions occurring in each hour.
# - Plot a bar chart showing the number of transactions per hour.
# - Display an explanation interpreting customer behavior trends based on time.
#  *************************************************************************

with st.expander("⏰ Spending Through Time: Transaction Patterns by Hour", expanded=False): # Puts the entire analysis inside a collapsible box for clean layout.

    st.subheader("**Hourly Distribution of Transactions**")
    filtered_df["Hour"] = ((filtered_df["Time"] // 3600) % 24).astype(int) # Converts transaction time from seconds into 24-hour clock format.
    hourly_counts = filtered_df["Hour"].value_counts().sort_index() # Counts how many transactions happen at each hour (sorted nicely).

    # Plot fraud transaction counts by hour
    fig1, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, ax=ax, palette="Blues_d") # Creates a bar chart showing how transaction activity varies throughout the day.

    # Set title and axis labels
    ax.set_title("Hourly Distribution of Transactions") 
    ax.set_xlabel("Hour of Day") 
    ax.set_ylabel("Number of Transactions") 
    st.pyplot(fig1)

    # Provides an easy-to-understand explanation of what the chart means.
    st.markdown(""" 
    **Explanation:**  
     The bar chart shows how many transactions happen at different times of the day. 
     People don't really spend much early in the morning, between 1 AM and 6 AM. 
     Starting around 9 AM, transactions pick up and stay busy throughout the day and evening. 
     The busiest time is around 9 PM to 10 PM. After 11 PM, things quiet down again. 
     Knowing when people are most active can help businesses decide when to have more staff, when to watch more closely for fraud, or when to run marketing promotions.
    """)

    #  *************************************************************************
    # Distribution of Transaction Amounts
    #  *************************************************************************
    # - Plot a histogram of the transaction amounts.
    # - Use 100 bins for granularity and overlay a KDE (Kernel Density Estimate) for smoothness.
    # - Apply a logarithmic scale to the Y-axis to better visualize rare large transactions.
    # - Provide an interpretation explaining the common transaction sizes and risks associated with very large amounts.
    # Note: The original dataset 'df' is used (not filtered_df) to show overall distribution.
    # This has been wrapped inside st.expander for collapsibility.

    st.subheader("**Transaction Amount Size Distribution**")

    fig2, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Amount"], bins=100, kde=True, ax=ax) # Creates a histogram showing how transaction amounts are distributed, with a smooth curve over it.

    # Set title and axis labels
    ax.set_title("Transaction Amount Size Distribution") 
    ax.set_xlabel("Amount (€)") 
    ax.set_ylabel("Number of Transactions") 

    # The ax.set_yscale("log") because small transactions dominate and large ones get "flattened" and we can't easily see rare large transactions.
    # It stretches the small counts and compresses the very large counts. This way, both small everyday transactions and rare big ones become visible together.
    # It's critical for fraud analysis, because big transactions, even rare, could signal fraud.
    ax.set_yscale("log") # Changes the Y-axis to a logarithmic scale so that smaller and larger values are more evenly visible — helps see rare big transactions better.
    
    st.pyplot(fig2) # Displays the plot inside the Streamlit app.

    # Provides an easy-to-understand explanation of what the chart means.
    st.markdown("""
    **Explanation:**  
     The chart shows how much money people are usually moving around. Most transactions are small, between €0 and €10,000. 
     As the amounts get bigger, there are fewer and fewer transactions. The chart uses a special scale to help show this more clearly. 
     After €10,000, big transactions become very rare, and there are even some empty spaces where no transactions happened at all. 
     Because there are a few really large money movements, it might be a good idea to double-check those for anything suspicious or unusual.
    """)

    # *************************************************************************
    # Discovery of Fraud Pattern (Fraud vs Non-Fraud Comparison)
    # *************************************************************************
    # - Count the number of Fraud and Non-Fraud transactions using 'Class' column.
    # - Create a bar plot comparing Fraud and Non-Fraud counts.
    # - Set appropriate x-axis labels ('Non-Fraud', 'Fraud') for better readability.
    # - Provide an interpretation explaining class imbalance and its challenges for model training.
st.markdown("---")  # <-- divider here
with st.expander("🚨 Chasing Shadows: Fraud vs Non-Fraud Transactions", expanded=False):
    class_counts = df["Class"].value_counts() # Counts how many Non-Fraud (0) and Fraud (1) transactions exist.

    st.subheader("**Fraud vs Non-Fraud Transaction Counts**")
    fig3, ax = plt.subplots(figsize=(10, 5)) 
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=["#2ca02c", "#d62728"]) # 	Creates a side-by-side bar chart showing the counts.
    ax.set_xticklabels(["Non-Fraud", "Fraud"]) # Changes the x-axis labels from numeric (0,1) to descriptive words.
    ax.set_title("Fraud vs Non-Fraud Transaction Counts") # Adds a title and y-axis label for clarity.
    ax.set_ylabel("Transaction Count")
    fig3.tight_layout()
    st.pyplot(fig3)

    all_figures_for_pdf.append(fig3)

    # Provides an easy-to-understand explanation of what the chart means.
    st.markdown("""
    **Explanation:**  
     The bar chart illustrates the number of fraud cases relative to standard transactions. The number of normal transactions 
     far exceeds fraudulent cases because fraud occurs very infrequently. The substantial gap between normal and fraudulent transactions creates 
     difficulties for machine learning models to detect fraud. The accuracy metric alone is insufficient because a model could predict 
     normal cases for everything while maintaining a correct appearance. The model requires specific techniques such as data balancing and 
     rare case detection to properly identify fraud during training.
    """)

    st.markdown(""" 
     Because there are way fewer fraud cases than normal transactions, we need to balance the data better when training the model. 
     Techniques like SMOTE, ADASYN, or undersampling help by making the fraud cases more noticeable to the model. Also, instead of just checking if 
     the model gets a high accuracy score (which can be misleading), it’s better to use other measures like precision-recall, F1-score, and ROC-AUC. 
     These tell us if the model is actually good at catching fraud without making too many mistakes.
    """)

    st.markdown(""" 
      The development of fraud detection models requires specific techniques which excel at detecting rare instances of fraud. Ensemble  models such as 
      XGBoost and Random Forest represent good choices because they unite multiple small models to generate  improved decisions. Anomaly detection methods 
      serve as a good alternative because they detect unusual behavior regardless of its  infrequent occurrence. These methods enhance fraud detection 
      accuracy when dealing with highly unbalanced data.
    """)

    #  *************************************************************************
    # Distribution of Transaction Amounts by Class (Fraud vs Non-Fraud)
    # - Create a violin plot comparing the distribution of transaction amounts between fraud and non-fraud cases.
    # - Use color coding: green for Non-Fraud, red for Fraud.
    # - Label the x-axis categories manually for better readability.
    # - Provide an interpretation highlighting the behavior of fraud transactions vs normal transactions.
    # Note: A violin plot shows both the distribution and the density of data points, unlike a boxplot.
    #  *************************************************************************

    st.subheader("**Transaction Amounts by Fraud Status**")
    fig4, ax = plt.subplots(figsize=(10, 5))
    
    # Draws violin-shaped plots showing the distribution (density + range) of transaction amounts for Non-Fraud and Fraud.
    # Green color for Non-Fraud, red color for Fraud.
    sns.violinplot(data=df, x="Class", y="Amount", palette=["#2ca02c", "#d62728"], inner="quartile") 
    ax.set_xticklabels(["Non-Fraud", "Fraud"]) # Replaces 0 and 1 labels with more user-friendly text.

    # Set title and axis labels
    ax.set_title("Transaction Amounts by Fraud Status")
    ax.set_xlabel("Class")
    ax.set_ylabel("	Transaction Amount (€)")
    fig4.tight_layout()
    st.pyplot(fig4)

    all_figures_for_pdf.append(fig4)

    st.markdown("""
    **Explanation:**  
     The chart shows the typical financial amounts involved in fraudulent activities versus standard business transactions. 
     Most fraud cases  occur with amounts ranging from small to medium and remain confined to a narrow range. 
     Normal transactions occur across  a wide range of amounts which includes transactions exceeding €25,000. 
     Small transaction amounts do not guarantee  safety because fraudsters use minimal amounts to stay under the radar. 
    """)

    st.markdown(""" 
     The detection of fraud requires more than just monitoring transaction amounts because fraudsters use various methods to avoid detection. 
     The damage from fraudulent activities often comes from small amounts of money that fraudsters use to stay under the radar. 
     Models should analyze the entire transaction context which includes behavioral patterns and timing information and other abnormal activities 
     instead of focusing only on absolute values. Organizations should analyze behavioral patterns together with transaction amounts to develop 
     more effective fraud detection systems.
    """)

    st.markdown(""" 
      The development of a fraud detection model requires evaluation of multiple factors beyond transaction amounts. The model can  develop advanced features through 
      three types of checks: amount-to-average ratio analysis for large transactions, frequency anomaly  detection for purchase rate spikes and user spending deviation 
      analysis for sudden pattern changes. The model becomes better at  detecting fraud through these feature types which go beyond basic amount analysis.
    """)

    #  *************************************************************************
    # Fraud Frequency by Hour of Day
    # - Filter the dataset to include only fraudulent transactions (Class == 1).
    # - Calculate the number of frauds occurring in each hour (0–23).
    # - Create a bar chart showing the distribution of frauds across different hours.
    # - Use a consistent color (red) to highlight fraud counts.
    # - Provide an interpretation on how fraud activity varies during the day, identifying risk peaks.
    #  *************************************************************************
       
    st.subheader("**Fraud Frequency Throughout the Day**")
    frauds = df[df["Class"] == 1] # Selects only the rows where fraud occurred (Class = 1).
    frauds["Hour"] = ((frauds["Time"] // 3600) % 24).astype(int) # Converts transaction time from seconds to hour of day (mod 24).
    fraud_by_hour = frauds["Hour"].value_counts().sort_index() # Counts how many frauds happened per hour and sorts them nicely.
    
    fig5, ax = plt.subplots(figsize=(10, 5)) 
    sns.barplot(x=fraud_by_hour.index, y=fraud_by_hour.values, ax=ax, color="#d62728") # Draws a red-colored bar chart showing number of frauds each hour.

    # Set title and axis labels
    ax.set_title("Fraud Frequency Throughout the Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Fraud Count")
    fig5.tight_layout()
    st.pyplot(fig5)

    all_figures_for_pdf.append(fig5)

    st.markdown("""
    **Explanation:**  
     This bar chart shows when fraud happens most during the day. There are two big jumps—one around 2 AM and another around 11 AM, each with more than 50 fraud cases. 
     The 2 AM spike might happen because fewer people are watching for fraud at night. The 11 AM spike might be because lots of normal transactions happen then too, 
     making it easier for fraud to slip through unnoticed. Throughout the afternoon (2 PM to 6 PM), fraud stays at a steady level. Very few fraud cases happen really 
     early in the morning, like at midnight, 5 AM, or 6 AM. Knowing this can help fraud teams decide the best times to be extra alert or add more staff.
    """)

    st.markdown(""" 
     The management of fraud risks requires enhanced real-time monitoring during peak fraud windows which occur at 2  AM and 11 AM. 
     The fraud detection teams need to increase their alertness and immediately investigate all  suspicious activities during the high-risk times. 
     The implementation of adaptive alert thresholds which modify their settings according to  the current time period will improve detection accuracy. 
     Organizations should adjust their sensitivity levels according to established risk patterns  to detect potential fraud effectively while minimizing 
     unnecessary team alerts during periods of low risk.
    """)

#  *************************************************************************
# Feature Correlation Heatmap
# - Compute the correlation matrix for numeric features in the filtered dataset.
# - Create a heatmap to visualize pairwise correlations.
# - Use the 'coolwarm' color palette to show positive and negative correlations.
# - Help users understand how features are related to each other.
#  *************************************************************************

st.markdown("---")  # <-- divider here
with st.expander("🔗 What Connects What? Feature Correlation Mapping", expanded=False):

    st.subheader("**Feature Correlation Overview**")
    corr = filtered_df.corr(numeric_only=True) # Calculates how strongly each numeric feature is related to every other feature (correlation matrix).

    fig6, ax = plt.subplots(figsize=(10, 5))

    # Creates a colorful matrix to visualize these correlations.
    sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig6)

    all_figures_for_pdf.append(fig6)

    st.markdown("""
    **Explanation:**  
     This chart shows how different pieces of information (like time, amount, and other features called V1–V28) are related to each other and to whether a transaction is fraud. 
     Most features, especially V1–V28, don’t have a strong connection to fraud, meaning fraud might be hidden in small, complicated patterns instead of obvious ones. 
     A few features like V4, V10, V12, V14, and V17 have a small link to fraud, but even things like the transaction amount and time of day don’t have a strong connection. 
     Time and Hour are perfectly linked, which makes sense because Hour was taken from Time. Overall, there aren’t big groups of features that are too similar to each other, 
     which is good because it means the model won’t get confused with repeated information.
    """)

@st.cache_resource # Cached training function

def train_random_forest(df):

    """
    Trains a Random Forest classifier on a 10% sample of the provided dataset.

    - Random Forests are good because they handle noise, outliers, and imbalanced classes better than many other models.
    - Drops 'Class' and 'Time' columns from input features.
    - Splits the data into training and testing sets (80/20 split).
    - Trains a Random Forest model with 50 estimators.

    Args:
        df (pd.DataFrame): Input transaction dataset.

    Returns:
        tuple: Trained Random Forest model, training features, testing features, training labels, testing labels.
    """

    df_sampled = df.sample(frac=0.1, random_state=42)  # Only take 10% of the dataset for faster training.
    X = df_sampled.drop(["Class", "Time"], axis=1) # 'Class' is the label, and 'Time' is not useful for tree splits.
    y = df_sampled["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42) # Split into training (80%) and testing (20%) data.
    rf = RandomForestClassifier(n_estimators=50, random_state=42) # Builds 50 decision trees and combines them.
    rf.fit(X_train, y_train) # Trains the model on the training data.
    return rf, X_train, X_test, y_train, y_test

    #  *************************************************************************
    # Feature Importance Plot (Top 10 Features by Random Forest)
    # - Train the Random Forest model (if not cached already).
    # - Retrieve feature importances from the trained model.
    # - Create a DataFrame to store feature names and their importance scores.
    # - Sort features by importance and keep only the top 10.
    # - Plot a horizontal bar chart to visualize the most influential features.
    # This helps identify which features most impact fraud detection predictions.
    #  *************************************************************************

    st.subheader("Top Predictive Features for Fraud")
    with st.spinner("Training Random Forest on 10% of data..."):
        rf, X_train, X_test, y_train, y_test = train_random_forest(df) # Train the model if needed (cached otherwise).

    importances = rf.feature_importances_ # Get the importance score of each feature (how much it affects model decisions).
    features = X_train.columns
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False).head(10) # Sort features by importance and pick the top 10.

    # Make a nice horizontal bar plot to easily compare feature strengths.
    fig7, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=imp_df, x="Importance", y="Importance Score", palette="viridis", ax=ax)
    ax.set_title("Top Predictive Features for Fraud")
    fig7.tight_layout()
    st.pyplot(fig7)

    all_figures_for_pdf.append(fig7)

@st.cache_data

def get_fraud_correlation(data):

    """
    Computes the correlation matrix for only fraudulent transactions.

    Filters the dataset to include only rows where Class == 1 (fraud cases),
    then calculates the pairwise Pearson correlation between numeric features.

    Args:
        data (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: Correlation matrix for fraud transactions only.
    """

    return data[data["Class"] == 1].corr(numeric_only=True) # Selects only transactions that are labeled as fraud (Class == 1).

    #  *************************************************************************
    # Correlation Heatmap (Fraud Cases Only)
    # - Compute the correlation matrix using only fraudulent transactions.
    # - Plot a heatmap to visualize how features relate to each other within fraud cases.
    # - Help identify strong relationships unique to fraudulent behavior patterns.
    #  *************************************************************************

    st.subheader("**Fraud-Specific Feature Correlations**")
    corr_fraud = get_fraud_correlation(filtered_df) # Calculates Pearson correlation only for numerical features among fraud cases.

    fig8, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_fraud, cmap="coolwarm", annot=False, linewidths=0.5, ax=ax) # Visualizes how fraud-related features move together or in opposite directions.
    ax.set_title("Fraud-Specific Feature Correlations")
    st.pyplot(fig8) # Displays the heatmap inside the dashboard. This zooms in only on fraud cases to understand unique relationships that normal transactions might hide.

@st.cache_resource

def get_shap_values(_model, X):

    """
    Calculates SHAP values for a trained Random Forest model.

    Uses the TreeExplainer to compute feature contributions for each prediction,
    providing interpretability for the model's behavior.

    Args:
        _model (RandomForestClassifier): The trained Random Forest model.
        X (pd.DataFrame): The input features for which SHAP values are computed.

    Returns:
        list or np.ndarray: SHAP values for the dataset.
    """
    explainer = shap.TreeExplainer(_model, feature_perturbation="tree_path_dependent") # Creates an explainer designed for tree-based models.
    shap_values = explainer.shap_values(X) # Calculates SHAP values (how much each feature pushed prediction towards fraud or not fraud).
    return shap_values

    #  *************************************************************************
    # SHAP Summary Plot (Explainable AI for Feature Contributions)
    # - Compute SHAP values to understand how features contribute to fraud predictions.
    # - Use TreeExplainer specific for tree-based models (Random Forest).
    # - If SHAP returns a list (binary classification), select the second element for "fraud" class.
    # - Render the SHAP summary plot as a static PNG image to avoid interactive plot issues.
    # - Display the SHAP plot with a caption in the dashboard.
    #  *************************************************************************

    st.subheader("**Explaining Model Decisions with SHAP**")
    with st.spinner("Calculating SHAP values..."):
        shap_values = get_shap_values(rf, X_test)

    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1]
    else:
        shap_vals_to_plot = shap_values

    # Render SHAP to PNG image instead of figure
    buffer = BytesIO() # Saves the figure in memory (no need to save image to disk).
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_vals_to_plot, X_test, plot_type="bar", show=False) # Draws a plot showing which features are most influential across the entire dataset.
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    st.image(buffer, caption="SHAP Summary Plot: Top Features for Fraud", use_container_width=False) # 	Displays the SHAP plot nicely inside the Streamlit dashboard.

#  *************************************************************************
# Outlier Detection using Boxplot (Transaction Amounts)
# - Create a boxplot comparing transaction amount distributions for fraud and non-fraud cases.
# - Show individual outliers using 'showfliers=True'.
# - Highlight differences in transaction amount behaviors between fraud and normal cases.
# Boxplots help detect extreme values that may represent unusual or suspicious transactions.
#  *************************************************************************

st.markdown("---")  # <-- divider here
with st.expander("📦 Outliers Unboxed: Transaction Amount Anomalies", expanded=False):

    st.subheader("**Identifying Outlier Transactions**")
    fig9, ax = plt.subplots(figsize=(10, 5))

    # Draws a box showing the middle 50% of data, and dots for outliers (points far from the average).
    # 'showfliers=True' makes sure we actually see the individual extreme transactions.
    # 'pallete' is for setting colors for Non-Fraud as green and Fraud as red.
    sns.boxplot(data=filtered_df, x="Class", y="Amount", palette=["#2ca02c", "#d62728"], showfliers=True) 
    ax.set_xticklabels(["Non-Fraud", "Fraud"])
    ax.set_title("Identifying Outlier Transactions")
    ax.set_ylabel("Amount (€)")
    ax.set_xlabel("")
    fig9.tight_layout()
    st.pyplot(fig9)

    all_figures_for_pdf.append(fig9)

    #  *************************************************************************
    # Feature Outlier Analysis: V10 and V14
    # - Focus on two specific features ('V10' and 'V14') known to show strong fraud patterns.
    # - Melt the dataset to long-form format to compare distributions side-by-side.
    # - Create a boxplot comparing the distributions of V10 and V14 between fraud and non-fraud transactions.
    # - Helps visually detect outliers and anomalies in key features.
    #  *************************************************************************

    st.subheader("**Detailed Outlier Analysis on Key Features**")
    key_features = ["V10", "V14"] # Focus only on the two strongest fraud-related features.
    melted = filtered_df.melt(id_vars="Class", value_vars=key_features) # Reshape the dataset so you can compare features side-by-side in one plot.

    # Draws boxplots for V10 and V14 split by class (fraud vs non-fraud). This lets users zoom in on the most suspicious features and see their behavioral differences clearly.
    # 'hue' is to color the boxplots by transaction type (Non-Fraud = green, Fraud = red).
    fig10, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=melted, x="variable", y="value", hue="Class", palette=["#2ca02c", "#d62728"], ax=ax)
    ax.set_title("Detailed Outlier Analysis on Key Features")
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    fig10.tight_layout()
    st.pyplot(fig10)

    all_figures_for_pdf.append(fig10)

    #  *************************************************************************
    # Anomaly Detection using Isolation Forest
    # - Exclude 'Time' and 'Class' columns from features.
    # - Train an Isolation Forest model to detect anomalous transactions.
    # - Assign -1 for anomalies and 1 for normal points.
    # - Visualize anomalies vs normal points using scatterplot (V14 vs V10).
    # - Display counts of total transactions, anomalies detected, and actual frauds within anomalies.
    #  *************************************************************************

    st.info(
        "🤖 Isolation Forest detects anomalies without using fraud labels. "
        "It learns the 'normal' behavior and flags deviations automatically."
    )

    st.subheader("**Unsupervised Fraud Detection with Isolation Forest**")
    with st.spinner("Running Isolation Forest..."):

        # Select a subset of features (excluding Time and Class)
        anomaly_features = filtered_df.drop(columns=["Time", "Class"])

        # Train Isolation Forest - an unsupervised ML model that detects "weird" transactions.
        iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42) 
        preds = iso.fit_predict(anomaly_features)

        # Helps mark suspicious activity. -1 = anomaly, 1 = normal
        filtered_df["Anomaly"] = preds

        # Shows how anomalies differ from normal behavior visually.
        fig11, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            data=filtered_df,
            x="V14", y="V10",
            hue="Anomaly",
            palette={1: "#2ca02c", -1: "#d62728"},
            alpha=0.5
        )
        ax.set_title("Unsupervised Fraud Detection with Isolation Forest")
        ax.legend(title="Prediction", labels=["Normal", "Anomaly"])
        st.pyplot(fig11)

        all_figures_for_pdf.append(fig11)

        # Report how many anomalies were detected and how many were real frauds.
        total_anomalies = (filtered_df["Anomaly"] == -1).sum()
        total_transactions = len(filtered_df)
        actual_frauds_in_anomalies = filtered_df[(filtered_df["Anomaly"] == -1) & (filtered_df["Class"] == 1)].shape[0]

        st.markdown("### 🧾 Anomaly Detection Summary")
        st.subheader("**🔍 Total Transactions Analyzed: **")
        st.warning(f"🚨 Anomalies Detected: **{total_anomalies:,}**")
        st.error(f"⚠️ Actual Frauds within Anomalies: **{actual_frauds_in_anomalies:,}**")

        if total_anomalies > 0:
            fraud_ratio = actual_frauds_in_anomalies / total_anomalies * 100
            st.info(f"🎯 Fraud Hit Rate in Anomalies: **{fraud_ratio:.2f}%**")

#  *************************************************************************
# Risk Profiling Heatmap & Automated Risk Commentary
# - Create a 2D heatmap showing fraud rate by Hour of Day and Transaction Amount Range.
# - Use a threshold (>1% fraud rate) to automatically detect high-risk zones.
# - Generate human-readable commentary describing the high-risk periods and amount bins.
# Helps decision-makers quickly identify when and where fraud is most likely to occur,
# supporting proactive fraud prevention strategies.
#  *************************************************************************

st.markdown("---")  # <-- divider here
with st.expander("🌡️ Mapping the Risk: Fraud Profiling by Time and Amount", expanded=False):

    st.subheader("**Fraud Risk Distribution Across Amount and Time**")
    # Create bins
    filtered_df["AmountBin"] = pd.cut(filtered_df["Amount"], bins=[0, 10, 50, 100, 500, 1000, 5000, 10000], right=False)

    # Calculate fraud rate per bin
    amount_risk = filtered_df.groupby("AmountBin")["Class"].mean().reset_index() # Calculates the fraud rate (percentage) in each amount range.
    amount_risk["FraudRate"] = amount_risk["Class"] * 100

    if amount_risk["FraudRate"].sum() == 0 or amount_risk.empty:
        st.info("🔎 No fraud cases detected in the current amount ranges and filters.")
    else:
        fig12, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=amount_risk, x="AmountBin", y="FraudRate", palette="rocket", ax=ax)
        ax.set_title("Fraud Rate by Transaction Amount Bin")
        ax.set_ylabel("Fraud Rate (%)")
        ax.set_xlabel("Transaction Amount Range (€)")
        ax.tick_params(axis="x", rotation=45)
        fig12.tight_layout()
        st.pyplot(fig12)

        all_figures_for_pdf.append(fig12)

    hour_risk = filtered_df.groupby("Hour")["Class"].mean().reset_index()
    hour_risk["FraudRate"] = hour_risk["Class"] * 100

    # Shows what hours of the day have higher fraud rates.
    fig13, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=hour_risk, x="Hour", y="FraudRate", marker="o", ax=ax)
    ax.set_title("Fraud Rate by Hour of Day")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xlabel("Hour of Day")
    fig13.tight_layout()
    st.pyplot(fig13)

    all_figures_for_pdf.append(fig13)

    st.subheader("**High-Risk Zones Automatically Detected**")
    # Define amount bins
    filtered_df["AmountBin"] = pd.cut(
        filtered_df["Amount"],
        bins=[0, 10, 50, 100, 500, 1000, 5000, 10000],
        right=False
    )

    # Group by Hour and AmountBin, then compute fraud rate
    heatmap_data = filtered_df.groupby(["Hour", "AmountBin"])["Class"].mean().unstack().fillna(0) * 100

    if (heatmap_data.values > 0).sum() == 0:
        st.info("🔎 No fraud risk patterns detected for the current filters and amount bins.")
    else:
        fig14, ax = plt.subplots(figsize=(10, 5))

        # Creates a 2D color-coded heatmap showing fraud rates for each Hour × AmountBin combination.
        # This highlights exact risky combinations.
        sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.5, annot=True, fmt=".1f", ax=ax) # Find cells where fraud rate exceeds 1%.
        ax.set_title("Fraud Rate Heatmap by Hour and Amount Range")
        ax.set_xlabel("Transaction Amount Range (€)")
        ax.set_ylabel("Hour of Day")
        fig14.tight_layout()
        st.pyplot(fig14)

        all_figures_for_pdf.append(fig14)

        # Automatically writes insights when fraud exceeds 1% in any cell.
        high_risk = heatmap_data[heatmap_data > 1.0]  # >1% fraud rate threshold
        risky_slots = high_risk.stack().reset_index()
        risky_slots.columns = ["Hour", "AmountBin", "FraudRate"]

        st.markdown("### 🧠 Automated Risk Commentary")

        if not risky_slots.empty:
            for _, row in risky_slots.iterrows():
                st.markdown(
                    f"• Between **Hour {int(row['Hour'])}** and for transactions in **{row['AmountBin']}**, "
                    f"fraud rate is **{row['FraudRate']:.2f}%**."
                )
        else:
            st.info("✅ No time and amount combinations exceeded the risk threshold (>1%) in the current filters.")

#  *************************************************************************
# Summary & Insights
# - Display final key observations and takeaways from the fraud analysis.
# - Summarize patterns in transaction behavior, fraud timing, amount risks, and feature importance.
# Helps users quickly grasp the most critical findings without analyzing every chart individually.
#  *************************************************************************

st.markdown("---")  # <-- divider here
with st.expander("📋 Wrapping It Up: Key Metrics and Strategic Recommendations", expanded=False):
    st.subheader("**Executive Summary and Key Findings**")
    total_tx = len(filtered_df)
    total_fraud = (filtered_df["Class"] == 1).sum()
    fraud_percent = total_fraud / total_tx * 100 if total_tx > 0 else 0

    anomalies = filtered_df["Anomaly"].value_counts().to_dict()
    total_anomalies = anomalies.get(-1, 0)
    fraud_in_anomalies = filtered_df[(filtered_df["Class"] == 1) & (filtered_df["Anomaly"] == -1)].shape[0]

    st.metric("📄 Total Transactions", f"{total_tx:,}")
    st.metric("⚠️ Total Frauds", f"{total_fraud:,} ({fraud_percent:.2f}%)")
    st.metric("🔍 Anomalies Detected", f"{total_anomalies:,}")
    st.metric("🎯 Fraud Inside Anomalies", f"{fraud_in_anomalies:,}")

    st.markdown("---")
    st.markdown("### 🧠 Executive Summary and Key Findings")
    st.markdown("""
    - Fraud transactions represent a small percentage of total volume but are concentrated in specific hours and amount ranges.
    - SHAP analysis shows **V14**, **V10**, and **Amount** are the most influential features for fraud detection.
    - Isolation Forest flagged **~1%** of transactions as anomalies, with several overlaps with true fraud cases.
    - High-risk windows include early morning hours and low-to-mid transaction amounts (€10–€100).
    """)
st.markdown("---")  # <-- divider here
with st.expander("📈 Predicting Tomorrow’s Trouble: Fraud Volume Forecasting", expanded=False):

    st.subheader("**Forecasting Future Fraud Trends**")
    st.markdown("Using Prophet to forecast daily fraud trends for the next 30 days.")

    # Prepare datetime and resample fraud counts
    df_forecast = df.copy()
    df_forecast["Datetime"] = pd.to_datetime(df_forecast["Time"], unit="s", origin="unix")
    df_forecast.set_index("Datetime", inplace=True)

    # Resample fraud count per day
    daily_fraud = df_forecast[df_forecast["Class"] == 1].resample("D").size().rename("FraudCount").reset_index()

    # Prepare for Prophet
    df_prophet = daily_fraud.rename(columns={"Datetime": "ds", "FraudCount": "y"})

    # Fit Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Forecast next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plot forecast
    fig15 = model.plot(forecast)
    fig15.tight_layout()
    st.pyplot(fig15)

    all_figures_for_pdf.append(fig15)

    # Optional components plot
    fig16 = model.plot_components(forecast)
    fig16.tight_layout()
    st.pyplot(fig16)

    all_figures_for_pdf.append(fig16)

    # 🧠 Fraud Forecast Commentary
    recent_window = forecast[forecast["ds"] < df_prophet["ds"].max()].tail(7)
    future_window = forecast[forecast["ds"] > df_prophet["ds"].max()].head(7)

    recent_avg = recent_window["yhat"].mean()
    future_avg = future_window["yhat"].mean()
    change_pct = ((future_avg - recent_avg) / recent_avg) * 100 if recent_avg > 0 else 0

    st.markdown("### 🧠 Forecast Commentary")

    if change_pct > 10:
        st.error(f"🚨 Fraud volume is projected to **increase by {change_pct:.2f}%** over the next week.")
    elif change_pct < -10:
        st.success(f"✅ Fraud volume is projected to **decrease by {abs(change_pct):.2f}%** in the coming week.")
    else:
        st.info(f"ℹ️ Fraud volume is expected to **remain stable** with a change of {change_pct:.2f}%.")
st.markdown("---")  # <-- divider here
with st.expander("📜 Acknowledgment", expanded=False):
    st.markdown("""
    Throughout this project I am thankful to Professor Perakis for his valuable guidance and encouragement and patience. 
    
    I would also like to thank Mr. Rompas for his supportive advice and encouragement along the way.
    
    In addition, I would like to thank the use of the publicly available Credit Card Fraud Detection dataset from Kaggle and the open-source tools Streamlit, Seaborn, Prophet, and Scikit-learn which made the creation of this dashboard possible.
    """)

st.markdown("""
---
<center>
<p style="font-size: 12px;">
Created with ❤️ by <b>100777246</b><br>
Powered by <b>Python</b>, <b>Streamlit</b>, and <b>Open-Source Tools</b><br>
© 2025 All Rights Reserved
</p>
</center>
""", unsafe_allow_html=True)

#  *************************************************************************
# 📄 PDF Report Export Section
# 📚 Initialize an empty list at the beginning of app (top of script):
# all_figures_for_pdf = []
#  *************************************************************************

def generate_pdf_from_figures(figures):
    """Generates an in-memory PDF file containing selected figures."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Inside the Swipe: Credit Card Fraud Analysis", ln=True, align="C")
    pdf.ln(10)

    today = datetime.date.today().strftime("%B %d, %Y")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date: {today}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, "Summary of Key Visual Insights:", ln=True, align="C")
    pdf.ln(20)

    for idx, fig in enumerate(figures):
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", bbox_inches='tight')
        buf.seek(0)

        # Create unique temp filename for each figure
        image_path = f"/tmp/temp_plot_{idx}.png"

        image = Image.open(buf)
        image.save(image_path)

        # Add figure to PDF
        pdf.add_page()
        pdf.image(image_path, x=10, w=190)
    
    # Output final PDF into memory
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# 📄 PDF Report Export Interface
with st.sidebar:
    st.header("📄 Export PDF Report")

    if st.button("📄 Generate PDF Report"):
        if all_figures_for_pdf:
            with st.spinner("Generating your report..."):
                pdf_file = generate_pdf_from_figures(all_figures_for_pdf)
                st.success("✅ Report generated successfully!")

            st.download_button(
                label="📥 Download Fraud Dashboard Report",
                data=pdf_file,
                file_name="Inside_The_Swipe.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("⚠️ No figures available yet. Please explore the dashboard first.")



