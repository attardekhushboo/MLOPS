import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("CSV Data Visualization App")

# Upload the file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Read and display the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Show dataset structure
    st.write("### Data Information")
    st.write(df.info())  # Show column types and non-null values

    st.write("### Descriptive Statistics")
    st.write(df.describe())  # Statistics for numeric columns

    # Ensure only numeric columns are used for visualizations
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    st.write("### Choose a Chart Type")
    chart_type = st.selectbox("Chart Type", ["Line Chart", "Bar Chart", "Histogram"])

    if chart_type == "Line Chart":
        # Line chart typically used for time-series data, but here we can show a relationship between Survived and PassengerId (if meaningful)
        st.line_chart(df[numeric_columns])

    elif chart_type == "Bar Chart":
        # Select a column for the bar chart: Focus on 'Survived' (1 = survived, 0 = not survived)
        bar_column = st.selectbox("Select Column for Bar Chart", ["Survived"])  # Only 'Survived' is categorical
        if bar_column:
            bar_data = df[bar_column].value_counts()  # Count the number of survivors and non-survivors
            st.bar_chart(bar_data)

    elif chart_type == "Histogram":
        # Let the user select a numeric column for the histogram: 'PassengerId' or 'Survived' (if meaningful)
        column = st.selectbox("Select Numeric Column for Histogram", numeric_columns)
        if column:
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)
