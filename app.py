import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit.components.v1 as components

# Custom CSS to style the sidebar and buttons
st.markdown("""
    <style>
    /* Change the sidebar background color to white */
    .css-1d391kg {
        background-color: black;  /* White background */
    }

    /* Style the buttons in the sidebar */
    .stButton > button {
        background-color: white;  /* Navy Blue */
        color: #000080;  /* White Text */
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        width: 100%;  /* Full width button */
    }

    .stButton > button:hover {
        background-color: #000066;  /* Darker Navy Blue for hover effect */
    }

    /* Change the color of the sidebar text to black */
    .css-1p7z1f8 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# Set background image (customizable)
background_image_path = r"C:\Users\PMLS\Desktop\Airline-Passenger-Satisfaction\images\airplane.jpg"
background_html = """
    <style>
    body {
        background-image: url("{background_image_path}");  /* Relative path to the image */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #000000;  /* Ensure text color is black */
    }

    #main {
        background: rgba(255, 255, 255, 0.8);  /* White background with 80% opacity for content area */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);  /* Adding a slight shadow to make the content area stand out */
    }

    h1, h2, h3, h4, p {
        color: #003366;  /* Set text color for headings and paragraphs */
    }
    </style>
    """
# Inject custom HTML to apply the background image
components.html(background_html, height=0)

# Load the dataset
df = pd.read_csv('test.csv')
df.columns = df.columns.str.lower()  # Converts all column names to lowercase

# Custom Title and Styling
st.markdown("<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>Airline Passenger Satisfaction Prediction</h1>", unsafe_allow_html=True)

# Add a main container for content to ensure background overlay
st.markdown('<div id="main">', unsafe_allow_html=True)

# Create a sidebar for buttons and interactions

button_overview = st.sidebar.button("Dataset Overview")
button_summary = st.sidebar.button("Summary Statistics")

# Main Visualization Section with Nested Buttons
visualization_option = st.sidebar.radio("Choose Visualization", ["None", "Visualizations"])

if visualization_option == "Visualizations":
    # Create a set of nested buttons inside this "Visualizations" category
    visualization_choice = st.sidebar.radio("Choose a Visualization", [
        "Satisfaction Distribution (Bar Plot)",
        "Missing Values Heatmap",
        "Correlation Heatmap",
        "Feature Distribution by Satisfaction",
        "Satisfaction vs Age",
        "Satisfaction vs Flight Distance"
    ])

    # Displaying corresponding visual based on user selection
    if visualization_choice == "Satisfaction Distribution (Bar Plot)":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Satisfaction Distribution (Bar Plot)</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='satisfaction', data=df, ax=ax, palette='Set2')
        ax.set_title('Satisfaction Distribution', fontsize=22, fontweight='bold')
        st.pyplot(fig)

    elif visualization_choice == "Missing Values Heatmap":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Missing Values Heatmap</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

    elif visualization_choice == "Correlation Heatmap":
        # Perform one-hot encoding for any categorical columns if necessary
        df = pd.get_dummies(df, drop_first=True)
        
        # Generate the correlation heatmap
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Correlation Heatmap</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title("Feature Correlation", fontsize=22, fontweight='bold')
        st.pyplot(fig)

    elif visualization_choice == "Feature Distribution by Satisfaction":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Feature Distribution by Satisfaction</h4>", unsafe_allow_html=True)
        
        # Check for available columns in the dataframe
        st.write("Columns in dataset:", df.columns)

        # Check if 'age' and 'satisfaction' are in the dataframe
        if 'age' in df.columns and 'satisfaction' in df.columns:
            # Handle missing values in 'age' if necessary
            df['age'] = df['age'].fillna(df['age'].mean())  # Handle missing values, replace NaN with mean
            
            # Plot the feature distribution by satisfaction (boxplot)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x='satisfaction', y='age', data=df, palette='Set2', ax=ax)
            ax.set_title('Age Distribution by Satisfaction', fontsize=22, fontweight='bold')
            st.pyplot(fig)
        else:
            st.error("The columns 'age' or 'satisfaction' do not exist in the dataset.")


    # Example fix for the scatter plot involving Age and Satisfaction
    elif visualization_choice == "Satisfaction vs Age":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Satisfaction vs Age</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use correct column names 'Age' and 'satisfaction'
        sns.scatterplot(x='age', y='satisfaction', data=df, ax=ax)
        
        ax.set_title("Satisfaction vs Age", fontsize=22, fontweight='bold')
        ax.set_xlabel("Age", fontsize=18)
        ax.set_ylabel("Satisfaction", fontsize=18)
        st.pyplot(fig)

    # Example fix for the scatter plot involving Flight Distance and Satisfaction
    elif visualization_choice == "Satisfaction vs Flight Distance":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Satisfaction vs Flight Distance</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use correct column names 'Flight Distance' and 'satisfaction'
        sns.scatterplot(x='flight distance', y='satisfaction', data=df, ax=ax)
        
        ax.set_title("Satisfaction vs Flight Distance", fontsize=22, fontweight='bold')
        ax.set_xlabel("Flight Distance", fontsize=18)
        ax.set_ylabel("Satisfaction", fontsize=18)
        st.pyplot(fig)


# Display Dataset Overview Section
if button_overview:
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Dataset Overview</h4>", unsafe_allow_html=True)
    st.dataframe(df.head(10))  # Display first few rows of the dataset

# Display Summary Statistics Section
if button_summary:
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Summary Statistics</h4>", unsafe_allow_html=True)
    st.write(df.describe())  # Summary statistics for numerical features

# Model Evaluation Section
button_model = st.sidebar.button("Model Evaluation")

if button_model:
    # Convert 'satisfaction' to numeric before using get_dummies
    df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'dissatisfied': 0})
    
    # Handle missing values in 'satisfaction' (drop rows with NaN in 'satisfaction')
    df = df.dropna(subset=['satisfaction'])  # Drops rows where 'satisfaction' is NaN
    
    # Alternatively, you can fill NaN values with a default value (e.g., 0 for dissatisfied)
    # df['satisfaction'].fillna(0, inplace=True)  # 0 represents 'dissatisfied'
    
    # Now, apply get_dummies to other categorical columns (except 'satisfaction')
    df = pd.get_dummies(df, drop_first=True)
    
    # Separate features and target (assuming 'satisfaction' is the target)
    if 'satisfaction' in df.columns:  # Check if 'satisfaction' is in the dataset
        X = df.drop('satisfaction', axis=1)  # Features
        y = df['satisfaction']  # Target

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize RandomForestClassifier model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = model.predict(X_test)

        # Accuracy score and Classification Report
        st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
        st.write(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['dissatisfied', 'satisfied'], yticklabels=['dissatisfied', 'satisfied'])
        ax.set_title('Confusion Matrix', fontsize=22, fontweight='bold')
        st.pyplot(fig)
    else:
        st.error("The 'satisfaction' column is not found in the dataset after encoding.")

# Conclusion Section
button_conclusion = st.sidebar.button("Conclusion")

if button_conclusion:
    st.markdown("<h2 style='text-align: center; font-size: 32px; color: white;'>Conclusion: Key Takeaways from Airline Passenger Satisfaction Project</h2>", unsafe_allow_html=True)

    st.markdown("<h4 style='font-size: 24px; font-weight: bold; color: white;'>Data Exploration & Preprocessing:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - The dataset contains various features related to passenger demographics, flight details, and their satisfaction with the flight.
    - Preprocessing steps like encoding categorical variables and handling missing values were essential for building a reliable model.
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='font-size: 24px; font-weight: bold; color: white;'>Modeling & Evaluation:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **Random Forest Classifier**: The model was trained using the Random Forest algorithm, which was effective in predicting passenger satisfaction.
    - **Evaluation Metrics**: Accuracy, classification report, and confusion matrix were used to assess the model's performance, with a focus on ensuring that the model's predictions align with passenger satisfaction trends.
    """, unsafe_allow_html=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)
