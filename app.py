# ============================================
# Employee Attrition Prediction
# Streamlit Application
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from feature_columns import FEATURE_COLUMNS

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🏢",
    layout="wide"
)

# ============================================
# Sidebar
# ============================================

st.sidebar.title("🏢 Employee Attrition")

st.sidebar.markdown("---")

st.sidebar.info(
    """
This application predicts whether an employee is likely to leave the company using Machine Learning.

### 🤖 Model
- Logistic Regression

### 👨‍💻 Developed By
- Meet Chovatiya
"""
)

st.sidebar.markdown("---")

st.sidebar.success("✅ End-to-End Machine Learning Project")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Features")

st.sidebar.write("✔ Employee Details")
st.sidebar.write("✔ Salary Information")
st.sidebar.write("✔ Job Information")
st.sidebar.write("✔ Work Experience")
st.sidebar.write("✔ Satisfaction Scores")

st.sidebar.markdown("---")

st.sidebar.caption("Version 1.0")

# ============================================
# Load Model
# ============================================

with open("employee_attrition_model.pkl", "rb") as file:
    model = pickle.load(file)

# ============================================
# Load Scaler
# ============================================

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ============================================
# Title
# ============================================

st.markdown(
    """
    <h1 style="text-align:center; color:#1F4E79;">
        🏢 Employee Attrition Prediction
    </h1>

    <p style="text-align:center; font-size:18px;">
        Predict whether an employee is likely to leave the company using
        <b>Machine Learning (Logistic Regression)</b>.
    </p>
    """,
    unsafe_allow_html=True
)


st.divider()

# ============================================
# Employee Details
# ============================================

st.header("👤 Employee Information")

col1, col2 = st.columns(2)

# =======================
# LEFT COLUMN
# =======================
with col1:

    age = st.number_input("Age", min_value=18, max_value=60, value=30)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    education = st.selectbox(
        "Education",
        [1,2,3,4,5]
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        value=5
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )

    department = st.selectbox(
        "Department",
        [
            "Human Resources",
            "Research & Development",
            "Sales"
        ]
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Human Resources",
            "Life Sciences",
            "Marketing",
            "Medical",
            "Other",
            "Technical Degree"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Divorced",
            "Married",
            "Single"
        ]
    )

# =======================
# RIGHT COLUMN
# =======================
with col2:

    job_role = st.selectbox(
        "Job Role",
        [
            "Healthcare Representative",
            "Human Resources",
            "Laboratory Technician",
            "Manager",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative"
        ]
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

    job_level = st.selectbox(
        "Job Level",
        [1,2,3,4,5]
    )

    job_involvement = st.selectbox(
        "Job Involvement",
        [1,2,3,4]
    )

    performance_rating = st.selectbox(
        "Performance Rating",
        [3,4]
    )

    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1,2,3,4]
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1,2,3,4]
    )

    relationship_satisfaction = st.selectbox(
        "Relationship Satisfaction",
        [1,2,3,4]
    )

st.header("💰 Salary & Experience")

col3, col4 = st.columns(2)

with col3:

    daily_rate = st.number_input(
        "Daily Rate",
        min_value=100,
        value=800
    )

    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=20,
        value=60
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        value=5000
    )

    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=1000,
        value=15000
    )

    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=10,
        value=15
    )

    stock_option_level = st.selectbox(
        "Stock Option Level",
        [0,1,2,3]
    )

    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        value=2
    )

with col4:

    training_times_last_year = st.number_input(
        "Training Times Last Year",
        min_value=0,
        value=2
    )

    work_life_balance = st.selectbox(
        "Work Life Balance",
        [1,2,3,4]
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        value=10
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        value=5
    )

    years_in_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        value=3
    )

    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        value=1
    )

    years_with_curr_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        value=3
    )
    st.divider()
# ============================================
# Predict Button
# ============================================

st.divider()

if st.button("Predict Attrition"):

    

    # ============================================
    # Binary Encoding
    # ============================================

    # Gender
    if gender == "Male":
        gender = 1
    else:
        gender = 0

    # OverTime
    if overtime == "Yes":
        overtime = 1
    else:
        overtime = 0


    # ============================================
    # Business Travel Encoding
    # ============================================

    BusinessTravel_Travel_Frequently = 0
    BusinessTravel_Travel_Rarely = 0

    if business_travel == "Travel_Frequently":
        BusinessTravel_Travel_Frequently = 1

    elif business_travel == "Travel_Rarely":
        BusinessTravel_Travel_Rarely = 1


    # ============================================
    # Department Encoding
    # ============================================

    Department_Research_Development = 0
    Department_Sales = 0

    if department == "Research & Development":
        Department_Research_Development = 1

    elif department == "Sales":
        Department_Sales = 1


        # ============================================
    # Education Field Encoding
    # ============================================

    EducationField_Life_Sciences = 0
    EducationField_Marketing = 0
    EducationField_Medical = 0
    EducationField_Other = 0
    EducationField_Technical_Degree = 0

    if education_field == "Life Sciences":
        EducationField_Life_Sciences = 1

    elif education_field == "Marketing":
        EducationField_Marketing = 1

    elif education_field == "Medical":
        EducationField_Medical = 1

    elif education_field == "Other":
        EducationField_Other = 1

    elif education_field == "Technical Degree":
        EducationField_Technical_Degree = 1

    
        # ============================================
    # Job Role Encoding
    # ============================================

    JobRole_Human_Resources = 0
    JobRole_Laboratory_Technician = 0
    JobRole_Manager = 0
    JobRole_Manufacturing_Director = 0
    JobRole_Research_Director = 0
    JobRole_Research_Scientist = 0
    JobRole_Sales_Executive = 0
    JobRole_Sales_Representative = 0

    if job_role == "Human Resources":
        JobRole_Human_Resources = 1

    elif job_role == "Laboratory Technician":
        JobRole_Laboratory_Technician = 1

    elif job_role == "Manager":
        JobRole_Manager = 1

    elif job_role == "Manufacturing Director":
        JobRole_Manufacturing_Director = 1

    elif job_role == "Research Director":
        JobRole_Research_Director = 1

    elif job_role == "Research Scientist":
        JobRole_Research_Scientist = 1

    elif job_role == "Sales Executive":
        JobRole_Sales_Executive = 1

    elif job_role == "Sales Representative":
        JobRole_Sales_Representative = 1

    
        # ============================================
    # Marital Status Encoding
    # ============================================

    MaritalStatus_Married = 0
    MaritalStatus_Single = 0

    if marital_status == "Married":
        MaritalStatus_Married = 1

    elif marital_status == "Single":
        MaritalStatus_Single = 1



        # ============================================
    # Create Feature Dictionary
    # ============================================

    features = {

    "Age": age,
    "DailyRate": daily_rate,
    "DistanceFromHome": distance_from_home,
    "Education": education,
    "EnvironmentSatisfaction": environment_satisfaction,
    "Gender": gender,
    "HourlyRate": hourly_rate,
    "JobInvolvement": job_involvement,
    "JobLevel": job_level,
    "JobSatisfaction": job_satisfaction,
    "MonthlyIncome": monthly_income,
    "MonthlyRate": monthly_rate,
    "NumCompaniesWorked": num_companies_worked,
    "OverTime": overtime,
    "PercentSalaryHike": percent_salary_hike,
    "PerformanceRating": performance_rating,
    "RelationshipSatisfaction": relationship_satisfaction,
    "StockOptionLevel": stock_option_level,
    "TotalWorkingYears": total_working_years,
    "TrainingTimesLastYear": training_times_last_year,
    "WorkLifeBalance": work_life_balance,
    "YearsAtCompany": years_at_company,
    "YearsInCurrentRole": years_in_current_role,
    "YearsSinceLastPromotion": years_since_last_promotion,
    "YearsWithCurrManager": years_with_curr_manager,

    "BusinessTravel_Travel_Frequently": BusinessTravel_Travel_Frequently,
    "BusinessTravel_Travel_Rarely": BusinessTravel_Travel_Rarely,

    "Department_Research & Development": Department_Research_Development,
    "Department_Sales": Department_Sales,

    "EducationField_Life Sciences": EducationField_Life_Sciences,
    "EducationField_Marketing": EducationField_Marketing,
    "EducationField_Medical": EducationField_Medical,
    "EducationField_Other": EducationField_Other,
    "EducationField_Technical Degree": EducationField_Technical_Degree,

    "JobRole_Human Resources": JobRole_Human_Resources,
    "JobRole_Laboratory Technician": JobRole_Laboratory_Technician,
    "JobRole_Manager": JobRole_Manager,
    "JobRole_Manufacturing Director": JobRole_Manufacturing_Director,
    "JobRole_Research Director": JobRole_Research_Director,
    "JobRole_Research Scientist": JobRole_Research_Scientist,
    "JobRole_Sales Executive": JobRole_Sales_Executive,
    "JobRole_Sales Representative": JobRole_Sales_Representative,

    "MaritalStatus_Married": MaritalStatus_Married,
    "MaritalStatus_Single": MaritalStatus_Single
}

        # ============================================
    # Convert Dictionary to DataFrame
    # ============================================

    input_df = pd.DataFrame([features])

    # Arrange columns in same order
    
    input_df = input_df[FEATURE_COLUMNS]

    # ============================================
    # Scale Input
    # ============================================

    input_scaled = scaler.transform(input_df)

    # ============================================
    # Prediction
    # ============================================

    prediction = model.predict(input_scaled)

    prediction_proba = model.predict_proba(input_scaled)

    # ============================================
    # Display Result
    # ============================================

    st.divider()

    st.header("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Employee is likely to leave the company.")
    else:
        st.success("✅ Employee is likely to stay with the company.")

    st.write(
        f"Probability of Staying : {prediction_proba[0][0]*100:.2f}%"
    )

    st.write(
        f"Probability of Leaving : {prediction_proba[0][1]*100:.2f}%"
    )

    st.success("All information collected successfully!")