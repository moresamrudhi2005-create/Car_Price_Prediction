import streamlit as st
import pandas as pd
import pickle as pkl
import math

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

.main{
    background-color:#F5F7FA;
}

h1{
    color:#0E4D92;
    text-align:center;
}

div[data-testid="stButton"] > button{
    background-color:#0E4D92;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
}

div[data-testid="stButton"] > button:hover{
    background-color:#1565C0;
}

.result{
    background-color:#d4edda;
    padding:20px;
    border-radius:10px;
    font-size:30px;
    text-align:center;
    color:green;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
model = pkl.load(open("CPP.pkl", "rb"))

df = pd.read_csv("cleaned_CarPrediction.csv")

# ---------------- Sidebar ----------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/744/744465.png",
    width=150
)

st.sidebar.title("🚗 Car Price Prediction")

st.sidebar.info("""
Predict the resale value of a used car using Machine Learning.

**Algorithm Used**
- Linear Regression

**Developer**
Samruddhi More
""")

# ---------------- Title ----------------
st.title("🚗 Car Price Prediction")
st.write("### Enter the details of the car")

# ---------------- Layout ----------------
col1, col2 = st.columns(2)

with col1:

    companies = sorted(df['company'].unique())
    company = st.selectbox("🏭 Select Company", companies)

    #Display the car name according to the Company name
    names = sorted(df['name'][df['company'] == company])
    name = st.selectbox("🚘 Select Model", names)

    year = st.number_input(
        "📅 Manufacturing Year",
        min_value=2000,
        max_value=2020,
        value=2015
    )

with col2:

    kms_driven = st.number_input(
        "🛣 Kilometers Driven",
        min_value=500,
        value=10000
    )
# display fuel_type according to the car name
    fuel_types = sorted(df['fuel_type'][df['name'] == name])
    fuel_type = st.selectbox("⛽ Fuel Type", fuel_types)

    st.write("")
    st.write("")

# ---------------- Prediction ----------------
if st.button("💰 Predict Car Price"):

    inputdata = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=[
            'name',
            'company',
            'year',
            'kms_driven',
            'fuel_type'
        ]
    )

    result = model.predict(inputdata)

    price = math.ceil(result[0][0])

    st.balloons()

    st.markdown(
        f"""
        <div class='result'>
        Estimated Price <br><br>
        ₹ {price:,}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ using Streamlit & Machine Learning</center>",
    unsafe_allow_html=True
)