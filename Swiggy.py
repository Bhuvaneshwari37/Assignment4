import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# 🎯 App Title
# -----------------------------
st.set_page_config(page_title="Swiggy Recommender", layout="centered")
st.title("🍽️ Restaurant Recommendation🍫")

# -----------------------------
# 📂 Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Swiggy_cleaned_data.csv")
    return df

data = load_data()

# -----------------------------
# 🧠 OneHotEncoder (fit fresh)
# -----------------------------
categorical_cols = ['name', 'city_names', 'location', 'cuisine']

@st.cache_resource
def fit_encoder(data):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(data[categorical_cols])
    return encoder

encoder = fit_encoder(data)

# -----------------------------
# 🔢 Scale numeric columns
# -----------------------------
numeric_cols = ['rating', 'cost']
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(data[numeric_cols])
scaled_df = pd.DataFrame(scaled_numeric, columns=numeric_cols)

# -----------------------------
# 📋 User Inputs (Sidebar)
# -----------------------------
st.sidebar.header("🔍 Your Preferences")
city = st.sidebar.selectbox("Select City", sorted(data['city_names'].unique()))
cuisine = st.sidebar.selectbox("Preferred Cuisine", sorted(data['cuisine'].unique()))
rating = st.sidebar.slider("Minimum Rating", 1.0, 5.0, 4.0, 0.1)
cost = st.sidebar.slider("Approx Cost for Two", 100, 5000, 1000, 100)

# -----------------------------
# 🔍 Apply hard filters before similarity
# -----------------------------
filtered_data = data[
    (data['city_names'] == city) &
    (data['cuisine'].str.contains(cuisine, case=False)) &
    (data['rating'] >= rating) &
    (data['cost'] <= cost)
]

if filtered_data.empty:
    st.warning("😕 No restaurants found matching your filters. Try relaxing them.")
    st.stop()

# -----------------------------
# 🧠 Encode & Scale Filtered Data
# -----------------------------
filtered_encoded = encoder.transform(filtered_data[categorical_cols])
filtered_scaled = scaler.transform(filtered_data[numeric_cols])
filtered_feature_set = np.concatenate([filtered_encoded, filtered_scaled], axis=1)

# -----------------------------
# 🧠 Encode & Scale User Input
# -----------------------------
input_df = pd.DataFrame([{
    'name': 'Dummy Name',
    'city_names': city,
    'location': 'Dummy Location',
    'cuisine': cuisine
}])

input_df = input_df[encoder.feature_names_in_]
encoded_input = encoder.transform(input_df)
user_scaled = scaler.transform([[rating, cost]])
user_vector = np.concatenate([encoded_input[0], user_scaled[0]]).reshape(1, -1)

# -----------------------------
# 🤖 Compute Similarity
# -----------------------------
similarities = cosine_similarity(user_vector, filtered_feature_set)
top_n = min(5, len(filtered_data))
top_indices = similarities[0].argsort()[-top_n:][::-1]

# -----------------------------
# 📢 Show Recommendations
# -----------------------------
st.subheader("🔝5 Top Recommendations for You")

for idx in top_indices:
    r = filtered_data.iloc[idx]
    st.markdown(f"""
    ### 🏨 {r['name']}
    🌆 **City:** {r['city_names']}  
    📌 **Location:** {r['location']}  
    🍽️ **Cuisine:** {r['cuisine']}  
    ⭐ **Rating:** {r['rating']}  
    💰 **Cost for Two:** ₹{r['cost']}  
    ---
    """)
