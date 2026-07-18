# ==========================================
# Ford Car Price Prediction Web App
# ==========================================

# Q1. Import Required Libraries

# Streamlit is used to create the web application
import streamlit as st

# Pandas is used for data handling
import pandas as pd

# Joblib is used to load saved machine learning model
import joblib

# ==========================================
# Q2. Page Configuration
# ==========================================

st.set_page_config(
    page_title="Ford Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ==========================================
# Q3. Load Model and Preprocessing Objects
# ==========================================

try:
    model = joblib.load("LR_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoded_columns = joblib.load("columns.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("About")

st.sidebar.info("""
**Ford Car Price Prediction**

Machine Learning Model:
- Linear Regression

Framework:
- Streamlit

Developed By:
- Vaishnavi Chandak
""")

# ==========================================
# Q4. Title and Description
# ==========================================

st.title("🚗 Ford Car Price Predictor")

st.write("Enter the car details below to predict the estimated selling price of a Ford car.")

st.markdown("---")

# ==========================================
# Q5. Numerical Inputs
# ==========================================

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2030,
    value=2018
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=300000,
    value=25000
)

tax = st.number_input(
    "Road Tax",
    min_value=0,
    max_value=600,
    value=145
)

mpg = st.number_input(
    "Miles Per Gallon (MPG)",
    min_value=0.0,
    max_value=100.0,
    value=55.4
)

engineSize = st.number_input(
    "Engine Size (L)",
    min_value=0.8,
    max_value=6.0,
    value=1.5
)

# ==========================================
# Q6. Dropdown Inputs
# ==========================================

transmission = st.selectbox(
    "Transmission",
    [
        "Automatic",
        "Manual",
        "Semi-Auto"
    ]
)

fuelType = st.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "Hybrid",
        "Electric",
        "Other"
    ]
)

car_model = st.selectbox(
    "Car Model",
    [
        "Fiesta",
        "Focus",
        "Kuga",
        "EcoSport",
        "Mondeo",
        "Ka+",
        "Puma",
        "S-Max",
        "Galaxy",
        "B-Max",
        "Grand C-Max",
        "C-Max",
        "Edge",
        "Tourneo Custom",
        "Transit Tourneo",
        "Mustang"
    ]
)

# ==========================================
# Q7. Predict Button
# ==========================================

predict = st.button("Predict Price")

# ==========================================
# Q8 & Q9 Prediction
# ==========================================

if predict:

    try:

        # Create DataFrame

        input_df = pd.DataFrame({

            "model": [car_model],
            "year": [year],
            "transmission": [transmission],
            "mileage": [mileage],
            "fuelType": [fuelType],
            "tax": [tax],
            "mpg": [mpg],
            "engineSize": [engineSize]

        })

        # One-Hot Encoding

        input_df = pd.get_dummies(input_df)

        # Match Training Columns

        input_df = input_df.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        # Numerical Columns

        numerical_columns = [
            "year",
            "mileage",
            "tax",
            "mpg",
            "engineSize"
        ]

        # Feature Scaling

        input_df[numerical_columns] = scaler.transform(
            input_df[numerical_columns]
        )

        # Prediction

        prediction = model.predict(input_df)

        # Display Prediction

        st.balloons()

        st.success(
            f"💰 Predicted Selling Price: £{prediction[0]:,.2f}"
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.caption(
    "Developed by Vaishnavi Chandak | Streamlit | Machine Learning"
)