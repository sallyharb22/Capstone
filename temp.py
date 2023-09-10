# Libraries
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as colors
from streamlit_lottie import st_lottie
from sklearn.metrics import mean_squared_error, r2_score
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import train_test_split
import re
import ssl
import matplotlib.colors as mcolors  # Import mcolors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Create an SSL context with certificate verification disabled
ssl._create_default_https_context = ssl._create_unverified_context

import sklearn.feature_extraction.text as text
from sklearn import model_selection, preprocessing, linear_model, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from io import StringIO
from collections import Counter

import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
import os

# Additional imports (if needed)
import hydralit_components as hc
import json

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define local file paths
Categories_file_path = 'Category.xlsx'
Areas_file_path = 'Areas.xlsx'
Receipts_file_path = 'Receipts.xlsx'
PredictiveAnalytics_file_path = 'PredictiveAnalytics.xlsx'

# Load data files
Categories = pd.read_excel(Categories_file_path)
Areas = pd.read_excel(Areas_file_path)
Receipts = pd.read_excel(Receipts_file_path)
PredictiveAnalytics = pd.read_excel(PredictiveAnalytics_file_path)

# Preprocessing Steps
# Standardize capitalization
Categories['Category'] = Categories['Category'].str.lower()

# Remove leading/trailing whitespaces
Categories['Category'] = Categories['Category'].str.strip()

# Handle misspellings and variations
Categories['Category'].replace({'canfy': 'candy'}, inplace=True)

# Update category labels in the dataset
Categories['Category'].replace({'chocolat': 'chocolate', 'deserts': 'desserts'}, inplace=True)

# Standardize capitalization
Categories['Subcategory'] = Categories['Subcategory'].str.lower()

# Remove leading/trailing whitespaces
Categories['Subcategory'] = Categories['Subcategory'].str.strip()

# Standardize capitalization
Receipts['Subcategories'] = Receipts['Subcategories'].str.lower()

# Remove leading/trailing whitespaces
Receipts['Subcategories'] = Receipts['Subcategories'].str.strip()

# Standardize capitalization
Areas['Subcategory'] = Areas['Subcategory'].str.lower()

# Remove leading/trailing whitespaces
Areas['Subcategory'] = Areas['Subcategory'].str.strip()

# Prepare your data as needed
X = PredictiveAnalytics[['Kb', 'PhiNorm']]
y = PredictiveAnalytics['Theta of Optimal Solution']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Create the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Perform GridSearchCV to find the best hyperparameters using cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model with tuned hyperparameters
best_rf_model = grid_search.best_estimator_


# Define a function to perform Market Basket Analysis
def perform_market_basket_analysis():
    st.write("")
    st.write("")
    st.title("Association Analysis: Lift, Confidence, and Support Metrics for Item Pair")

    # Add a dropdown to select Item 1
    item1 = st.selectbox("Select Item 1:", Receipts['Purchased Item'].unique())

    # Add a dropdown to select Item 2
    item2 = st.selectbox("Select Item 2:", Receipts['Purchased Item'].unique())
    
    # Display lift and confidence for user-selected items
    if item1 and item2:
           st.subheader(f"Support, Confidence, and Lift for '{item1}' and '{item2}'")
           
           # Perform Market Basket Analysis for item1 and item2
           itemset = {item1, item2}
           transactions = Receipts.groupby('Order Number')['Purchased Item'].apply(set)
           itemset_occurrences = transactions.apply(lambda x: itemset.issubset(x))
           
           # Calculate support, confidence, and lift
           support = itemset_occurrences.mean()
           
           item1_occurrences = transactions.apply(lambda x: item1 in x)
           confidence = itemset_occurrences.mean() / item1_occurrences.mean()
           
           lift = confidence / support
           
           st.write(f"Support: {support:.2f}")
           st.write(f"Confidence: {confidence:.2f}")
           st.write(f"Lift: {lift:.2f}")
    
    # Add some spacing between the second and third graphs
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    
    st.title("Confidence and Lift Analysis: Item Pair Metrics Revealed")
    # User input for lift and confidence thresholds
    min_lift = st.slider("Minimum Lift", min_value=0.0, max_value=1.0, step=0.05, value=0.5)
    min_confidence = st.slider("Minimum Confidence", min_value=0.0, max_value=1.0, step=0.05, value=0.5)

    # Perform Market Basket Analysis if the "Analyze" button is clicked
    if st.button("Analyze"):
           # Step 1: Prepare the data for market basket analysis
           transactions = Receipts.groupby('Order Number')['Purchased Item'].apply(list)

           # Step 2: Perform binary encoding on the transaction data
           encoder = TransactionEncoder()
           encoded_transactions = encoder.fit_transform(transactions)
           encoded_df = pd.DataFrame(encoded_transactions, columns=encoder.columns_)

           # Step 3: Generate frequent itemsets using the FP-Growth algorithm
           frequent_itemsets = fpgrowth(encoded_df, min_support=0.02, use_colnames=True)

           # Step 4: Generate association rules
           rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.0)

           # Filter rules based on user inputs
           filtered_rules = rules[
               (rules['lift'] >= min_lift) &
               (rules['confidence'] >= min_confidence)
           ]
           
           # Convert frozensets to strings
           filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(lambda x: ', '.join(map(str, x)))
           filtered_rules['consequents'] = filtered_rules['consequents'].apply(lambda x: ', '.join(map(str, x)))

           # Display the filtered rules
           st.subheader("Filtered Association Rules")
           st.write(filtered_rules)

# Define a function to perform Market Basket Analysis for Subcategories
def perform_market_basket_analysis_subcategories():
    st.write("")
    st.write("")
    st.title("Association Analysis: Lift, Confidence, and Support Metrics for Subcategories Pair")

    # Add a dropdown to select Subcategory 1
    subcategory1 = st.selectbox("Select Subcategory 1:", Receipts['Subcategories'].unique())

    # Add a dropdown to select Subcategory 2
    subcategory2 = st.selectbox("Select Subcategory 2:", Receipts['Subcategories'].unique())
    
    # Display lift and confidence for user-selected subcategories
    if subcategory1 and subcategory2:
        st.subheader(f"Support, Confidence, and Lift for '{subcategory1}' and '{subcategory2}'")
        
        # Perform Market Basket Analysis for subcategory1 and subcategory2
        subcategory_set = {subcategory1, subcategory2}
        transactions = Receipts.groupby('Order Number')['Subcategories'].apply(set)
        subcategory_set_occurrences = transactions.apply(lambda x: subcategory_set.issubset(x))
        
        # Calculate support, confidence, and lift
        support = subcategory_set_occurrences.mean()
        
        subcategory1_occurrences = transactions.apply(lambda x: subcategory1 in x)
        confidence = subcategory_set_occurrences.mean() / subcategory1_occurrences.mean()
        
        lift = confidence / support
        
        st.write(f"Support: {support:.2f}")
        st.write(f"Confidence: {confidence:.2f}")
        st.write(f"Lift: {lift:.2f}")

    # Add some spacing between the second and third sections
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.title("Confidence and Lift Analysis: Subcategory Pair Metrics Revealed")
    # User input for lift and confidence thresholds
    min_lift = st.slider("Minimum Lift", min_value=0.0, max_value=1.0, step=0.05, value=0.5, key="min_lift")
    min_confidence = st.slider("Minimum Confidence", min_value=0.0, max_value=1.0, step=0.05, value=0.5, key="min_confidence")

    # Perform Market Basket Analysis if the "Analyze" button is clicked
    if st.button("Analyze", key="analyze_button"):
        # Step 1: Prepare the data for market basket analysis (for Subcategories)
        sub_transactions = Receipts.groupby('Order Number')['Subcategories'].apply(list)

        # Step 2: Perform binary encoding on the transaction data (for Subcategories)
        sub_encoder = TransactionEncoder()
        sub_encoded_transactions = sub_encoder.fit_transform(sub_transactions)
        sub_encoded_df = pd.DataFrame(sub_encoded_transactions, columns=sub_encoder.columns_)

        # Step 3: Generate frequent itemsets using the FP-Growth algorithm (for Subcategories)
        sub_frequent_itemsets = fpgrowth(sub_encoded_df, min_support=0.02, use_colnames=True)

        # Step 4: Generate association rules (for Subcategories)
        sub_rules = association_rules(sub_frequent_itemsets, metric='lift', min_threshold=0.0)

        # Filter rules based on user inputs
        sub_filtered_rules = sub_rules[
            (sub_rules['lift'] >= min_lift) &
            (sub_rules['confidence'] >= min_confidence)
        ]
        
        # Convert frozensets to strings
        sub_filtered_rules['antecedents'] = sub_filtered_rules['antecedents'].apply(lambda x: ', '.join(map(str, x)))
        sub_filtered_rules['consequents'] = sub_filtered_rules['consequents'].apply(lambda x: ', '.join(map(str, x)))

        # Display the filtered rules
        st.subheader("Filtered Association Rules for Subcategories")
        st.write(sub_filtered_rules)


# Settings streamlit page configuration
st.set_page_config(layout="wide", page_title=None)

# Function for making spaces
def space(n, element=st):  # n: number of lines
    for i in range(n):
        element.write("")

menu_data = [
    {"label": "Home", "icon": "bi bi-house"},
    {"label": "Exploratory Data Analysis", "icon": "bi bi-search"},
    {"label": "Market Basket Analysis", "icon": "bi bi-basket"},
    {"label": "Predictive Model", "icon": "bi bi-bar-chart"}
]

menu_id = hc.nav_bar(
    menu_definition=menu_data, sticky_mode='sticky',
    override_theme={
        'txc_inactive': 'white',
        'menu_background': '#00008B;',
        'txc_active': '#0178e4',
        'option_active': 'white'
    }
)

indv = 30
comp = 40
ret = 64
theme_vacc = {'bgcolor': '#f6f6f6','title_color': '#2A4657','content_color': '#b0c4de','progress_color': '#b0c4de','icon_color': '#b0c4de'}


# Define a CSS style for centering content
center_style = """
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    height: 100%; /* Ensure content takes up the entire column height */
"""

# Home Menu
if menu_id == "Home":
    # Add more spacing
    st.write("")
    #st.write("")
    st.markdown("<h1 style='text-align: center; color: #00008B;'>Predictive Analytics for In-Store Traffic: A Beirut-Based Supermarket</h1>", unsafe_allow_html=True)
    # Add more spacing
    st.write("")
    #st.write("")
    
    # Display an image
    st.image("pngtree-electronic-consumer-studio-photography-keyboard-shopping-cart-online-shopping-concept-map-picture-image_1541676.png", use_column_width=True)
    
elif menu_id == "Market Basket Analysis":
    
        # Statistical Analysis Dashboard
        st.write("")
        # Center-align the title using Markdown
        st.markdown("<h1 style='text-align: center; color: #00008B; '>Market Basket Analysis</h1>", unsafe_allow_html=True)

        # Add more spacing
        st.write("")
        
        # Display Market Basket Analysis page content here
        perform_market_basket_analysis()
        perform_market_basket_analysis_subcategories()
        
# Predictive Model Menu
elif menu_id == "Predictive Model":
    
    st.write("")
    
    # Center-align the title using Markdown
    st.markdown("<h1 style='text-align: center; color: #00008B; '>Predictive Model</h1>", unsafe_allow_html=True)

    # Add more spacing
    st.write("")
    st.write("")

    # Create a section for user input
    st.header("Input Parameters")
    kb_input = st.number_input("Layout Component (kb)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    phi_input = st.number_input("Allocation Component (Φb)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    # Create a button to trigger predictions
    if st.button("Predict Traffic Density"):
        # Create a DataFrame with user inputs
        user_inputs = pd.DataFrame({
            'Kb': [kb_input],
            'PhiNorm': [phi_input]
        })

        # Predict using the best model
        prediction = best_rf_model.predict(user_inputs)

        # Display the prediction
        st.subheader("Predicted Traffic Density")
        st.write(prediction)
        
# EDA Menu
elif menu_id == "Exploratory Data Analysis":

    # Statistical Analysis Dashboard
    st.write("")
    # Center-align the title using Markdown
    st.markdown("<h1 style='text-align: center; color: #00008B; '>Exploring Our Data</h1>", unsafe_allow_html=True)

    # Add more spacing
    st.write("")
    st.write("")
    st.write("")
    col1, col2, col3 = st.columns(3)
    
    # Define the CSS style for the light blue box with the desired color
    st.markdown(
        """
        <style>
        .light-blue-box {
            background-color: #b0c4de;  /* Use the desired blue color */
            padding: 10px;  /* Padding around the content */
            border: 1px solid #b0c4de;  /* Border color */
            border-radius: 5px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Column 1
    with col1:
       st.markdown("<div class='light-blue-box'><h1>Number of E-Receipts</h1></div>", unsafe_allow_html=True)
       st.markdown("<div class='light-blue-box'><h1><b>37715</b></h1></div>", unsafe_allow_html=True)

       # Add more spacing
       st.write("")
       st.write("")
       st.write("")

    # Column 2
    with col2:
        st.markdown("<div class='light-blue-box'><h1>Number of Categories</h1></div>", unsafe_allow_html=True)
        st.markdown("<div class='light-blue-box'><h1><b>45</b></h1></div>", unsafe_allow_html=True)

        # Add more spacing
        st.write("")
        st.write("")
        st.write("")
        
    # Column 3
    with col3:
        st.markdown("<div class='light-blue-box'><h1>Number of Subcategories</h1></div>", unsafe_allow_html=True)
        st.markdown("<div class='light-blue-box'><h1><b>60</b></h1></div>", unsafe_allow_html=True)

        # Add more spacing
        st.write("")
        st.write("")
        st.write("")

    # Place the third graph (horizontal bar chart for subcategories) at the top
    with col2:
       # Number of purchased items per area
        items_per_area = Receipts.groupby('Area')['Purchased Item'].count()

        # Sort the data in descending order
        items_per_area_sorted = items_per_area.sort_values(ascending=False)
        
        # Set the Seaborn style to remove grid lines
        sns.set_style("ticks")

        # Create a horizontal bar plot
        plt.figure(figsize=(10, 10))
        sns.barplot(x=items_per_area_sorted.values, y=items_per_area_sorted.index, orient='h', palette=color_palette, order=items_per_area_sorted.index)

        # Add labels and title
        plt.xlabel('Count', fontsize=14)
        plt.ylabel('Area', fontsize=14)
        plt.title('Number of Purchased Items per Area', fontsize=16, fontweight='bold')
       
        # Adjust layout for better spacing
        plt.subplots_adjust(top=0.9)  # Add space between title and graph) 

        # Beautify the plot
        sns.despine()  # Remove top and right spines

        plt.tight_layout()
        
        # Display the plot using st.pyplot()
        st.pyplot(plt)
        
        # Add some spacing between the first and second graphs
        st.write("")
        
        # Calculate category counts
        category_counts = Categories['Category'].value_counts()

        # Set up the Seaborn color palette for the bar plot
        sns.set_palette(color_palette)
        
        # Set the Seaborn style to remove grid lines
        sns.set_style("ticks")

        # Define a custom color palette with "bordeaux"
        custom_palette1 = ["red"] * len(category_counts)
        
        # Create a bar plot
        plt.figure(figsize=(10, 10))
        sns.barplot(x=category_counts.values, y=category_counts.index, palette=color_palette)

        # Add labels and title
        plt.xlabel('Count', fontsize=14)
        plt.ylabel('Category', fontsize=14)
        plt.title('Number of Items Available per Category', fontsize=16, fontweight='bold')
       
        # Adjust layout for better spacing
        plt.subplots_adjust(top=0.9)  # Add space between title and graph

        # Beautify the plot
        sns.despine()  # Remove top and right spines

        plt.tight_layout()
        
        # Display the bar plot using st.pyplot()
        st.pyplot(plt)

        # Add some spacing between the second and third graphs
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Create a pie chart
        plt.figure(figsize=(10, 10))
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85, wedgeprops=dict(edgecolor='w'), colors=color_palette)

        # Add a circle in the center for a donut-like appearance
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis('equal')

        # Add some space between title and chart
        plt.title('Proportion of Products by Category', fontsize=16, pad=20, fontweight='bold')
       
        # Adjust layout for better spacing
        plt.subplots_adjust(top=0.9)  # Add space between title and graph) 

        # Display the pie chart using st.pyplot()
        st.pyplot(plt)
        
    # Place the horizontal bar plot in the first column
    with col1:
        
        # Calculate the number of items per order
        items_per_order = Receipts.groupby('Order Number')['Purchased Item'].count()

        # Set the Seaborn style to remove grid lines
        sns.set_style("ticks")

        # Create a figure
        plt.figure(figsize=(10, 10))
        plt.box(False)  # Remove the box border around the entire plot
        
        # Beautify the plot
        sns.despine()  # Remove top and right spines
        
        # Define a custom color palette with "bordeaux"
        custom_palette1 = ["red"] * len(category_counts)

        # Plot the count of order numbers for each number of items as a vertical bar chart
        # Create a bar plot
        plt.figure(figsize=(10, 10))
        sns.barplot(x=items_per_order.value_counts().sort_index().index,
                    y=items_per_order.value_counts().sort_index().values, palette=color_palette)

        plt.title('Count of Orders by Number of Items', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Items')
        plt.ylabel('Count of Orders')

        # Set x-axis limits to zoom in on the range of 1 to 40
        plt.xlim(1, 30)
        
        # Remove upper and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Adjust layout for better spacing
        plt.subplots_adjust(top=0.9)  # Add space between title and graph

        # Display the bar chart using st.pyplot()
        st.pyplot(plt)
        
        # Add some spacing between the first and second graphs
        st.write("")
        st.write("")
        st.write("")
        
        # Calculate the co-occurrence of categories and subcategories
        heatmap_data = Categories.groupby(['Category', 'Subcategory']).size().unstack().fillna(0)

        # Define a colormap to match the blue used in the pie chart
        blue_colormap = sns.color_palette(['#E0E0E0', '#0000FF'], as_cmap=True)

        # Increase the figure size
        plt.figure(figsize=(8,8))

        # Define the custom color palette and reverse it
        num_colors1 = len(subcategory_counts)  # Adjust this based on your data
        color_palette1 = sns.color_palette("Blues_r", n_colors=num_colors1)[::-1]

        # Create the heatmap using Seaborn's heatmap function without annotations
        sns.heatmap(heatmap_data, cmap=color_palette1, annot=False, cbar=True, cbar_kws={'label': 'Count'})

        # Set title and labels with increased font sizes
        plt.title('Co-occurrence of Categories and Subcategories', fontsize=16, fontweight='bold')
        # Adjust layout for better spacing
        plt.subplots_adjust(top=0.9)  # Add space between title and graph
        
        plt.xlabel('Subcategory', fontsize=12)
        plt.ylabel('Category', fontsize=12)

        # Adjust tick labels font size and rotation
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()

        # Display the heatmap plot using st.pyplot()
        st.pyplot(plt)


    # Place the horizontal bar chart in the third column (col3)
    with col3:
        # Calculate subcategory counts
        subcategory_counts = Receipts['Subcategories'].value_counts()

        # Increase the figure size
        plt.figure(figsize=(10, 10))

        # Create a horizontal bar chart using Seaborn
        sns.barplot(x=subcategory_counts.values, y=subcategory_counts.index, palette=color_palette)

        # Add labels and title
        plt.xlabel('Count', fontsize=14)
        plt.ylabel('Subcategory', fontsize=14)
        plt.title('Number of Purchased Items per Subcategory', fontsize=16, fontweight='bold')
       
        # Adjust layout for better spacing
        plt.subplots_adjust(top=0.9)  # Add space between title and graph) 

        # Beautify the plot
        sns.despine()  # Remove top and right spines

        plt.tight_layout()
        
        # Display the plot using st.pyplot()
        st.pyplot(plt)
        
        # Add some spacing between the first and second graphs
        st.write("")

        # Calculate category counts
        subcategory_counts = Categories['Subcategory'].value_counts()

        # Create a bar plot
        plt.figure(figsize=(10, 10))
        sns.barplot(x=subcategory_counts.values, y=subcategory_counts.index, palette=color_palette)

        # Add labels and title
        plt.xlabel('Count', fontsize=14)
        plt.ylabel('Subcategory', fontsize=14)
        plt.title('Number of Items Available per Subcategory', fontsize=16, fontweight='bold')

        # Beautify the plot
        sns.despine()  # Remove top and right spines

        plt.tight_layout()
        
        # Display the bar plot using st.pyplot()
        st.pyplot(plt)

        # Add some spacing between the second and third graphs
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Create a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(subcategory_counts.values, labels=subcategory_counts.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85, wedgeprops=dict(edgecolor='w'), colors=color_palette)

        # Add a circle in the center for a donut-like appearance
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis('equal')

        # Add some space between title and chart
        plt.title('Proportion of Products by Subcategory', fontsize=16, pad=20, fontweight='bold')  # pad adjusts the distance

        # Adjust layout to add space at the top
        plt.subplots_adjust(top=0.85)  # Adjust this value to control the amount of space

        # Display the pie chart using st.pyplot()
        st.pyplot(plt)


"""
Spyder Editor

This is a temporary script file.
"""

