import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import io
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import re
import requests
import json
from fpdf import FPDF

# Display Images
# import Image from pillow to open images

img = Image.open("crop.png")
# display image using streamlit
# width is used to set the width of an image
st.image(img)
df= pd.read_csv('Crop_recommendation.csv')

#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
labels = df['label']

# Split the data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain,Ytrain)
predicted_values = RF.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)





def show_crop_image(crop_name):
    # Assuming we have a directory named 'crop_images' with images named as 'crop_name.jpg'
    image_path = os.path.join('crop_images', crop_name.lower()+'.jpg')
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Recommended crop: {crop_name}", use_column_width=True)
    else:
        st.error("Image not found for the predicted crop.")


import pickle

RF_pkl_filename = 'RF.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close 
RF_Model_pkl.close()

RF_Model_pkl=pickle.load(open('RF.pkl','rb'))

## Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    # # Making predictions using the model
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

models = ['Decision Tree', 'Naive Bayes', 'SVM', 'Logistic Regression', 'Random Forest', 'XGBoost', 'KNN']
accuracies = [0.82, 0.79, 0.85, 0.81, 0.91, 0.88, 0.80]  # Replace with your actual results

plt.figure(figsize=(10,5))
plt.bar(models, accuracies, color='green')
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison - Crop Recommendation")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

@st.cache_resource(show_spinner=True)

# Function to extract disease information from DISEASE-GUIDE.md
def get_disease_info(disease_name, guide_path='DISEASE-GUIDE.md'):
    if not os.path.exists(guide_path):
        return "⚠️ Disease guide not found.", ""

    with open(guide_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = content.split("### ")
    for section in sections:
        lines = section.strip().split("\n", 1)
        if not lines:
            continue
        raw_title = lines[0].strip()
        # Remove numeric prefix like "1. "
        title_clean = re.sub(r'^\d+\.\s*', '', raw_title)

       
        print(f"Matching predicted: '{disease_name}' ↔ in file: '{title_clean}'")

        if title_clean == disease_name:
            body = lines[1].strip() if len(lines) > 1 else ""
            return f"{title_clean}", body

    return "⚠️ Disease information not found.", ""
    
# Function to create a downloadable PDF
def generate_pdf(disease_title, body):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, disease_title)

    # Body
    pdf.set_font("Arial", "", 12)
    for line in body.split('\n'):
        pdf.multi_cell(0, 8, line)

    # Save PDF
    output_path = "disease_info.pdf"
    pdf.output(output_path)
    return output_path

# Function to display disease details and PDF button
def display_disease_details(predicted_disease):
    st.subheader("🩺 Disease Details")
    disease_title, body = get_disease_info(predicted_disease)

    if "⚠️" in disease_title:
        st.warning(disease_title)
    else:
        st.markdown(f"### {disease_title}")
        st.markdown(body, unsafe_allow_html=True)

        # Generate and download PDF
        pdf_path = generate_pdf(disease_title, body)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download Disease Info as PDF",
                data=f,
                file_name=f"{predicted_disease}.pdf",
                mime="application/pdf"
            )

@st.cache_resource
def load_and_train_fertilizer_model():
    df = pd.read_csv("Fertilizer_recommendation.csv")

    # Encode categorical variables
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
    df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])

    X = df.drop('Fertilizer Name', axis=1)
    y = df['Fertilizer Name']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, accuracy, le_soil, le_crop

# Fetch weather data function
def fetch_weather_forecast(city_name, api_key):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != "200":
            st.warning("⚠️ Unable to fetch forecast data. Please check city or API key.")
            return None
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def fetch_current_weather(city_name, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return None
        return data['main']['temp']
    except Exception as e:
        return None

def extract_unique_dates(forecast_data):
    dates = sorted(list(set([item['dt_txt'].split(' ')[0] for item in forecast_data['list']])))
    return dates

def suggest_crop(weather_condition, temp, humidity):
    condition = weather_condition.lower()
    if "rain" in condition:
        return "Rice or Maize"
    elif "clear" in condition and 20 <= temp <= 30 and 40 <= humidity <= 70:
        return "Wheat or Barley"
    elif temp > 30:
        return "Millets or Sorghum"
    elif humidity > 80:
        return "Jute or Sugarcane"
    else:
        return "Crops like Pulses or Oilseeds"

def display_weather_data(data, selected_date):
    filtered_data = [item for item in data['list'] if item['dt_txt'].startswith(selected_date)]

    if not filtered_data:
        st.info("No forecast data available for the selected date.")
        return

    st.subheader(f"Forecast for {selected_date}")
    crop_suggestion = None
    times = []
    temps = []

    # Display each time slot
    for idx, item in enumerate(filtered_data):
        time = item['dt_txt'].split(' ')[1][:5]
        temp = item['main']['temp']
        humidity = item['main']['humidity']
        pressure = item['main']['pressure']
        wind_speed = item['wind']['speed']
        description = item['weather'][0]['description'].title()
        icon_code = item['weather'][0]['icon']
        icon_url = f"http://openweathermap.org/img/wn/{icon_code}.png"

        if idx==0:
            crop_suggestion = suggest_crop(description, temp, humidity)

        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                st.image(icon_url, width=50)
            with cols[1]:
                st.markdown(f"**{time}** - {description}")
                st.markdown(f"🌡️ `{temp}°C`  |  💧 `{humidity}%`  |  🌬️ `{wind_speed} m/s`  |  🧭 `{pressure} hPa`")

        times.append(time)
        temps.append(temp)


    # line chart for temperature trend
    times = [item['dt_txt'].split(' ')[1][:5] for item in filtered_data]
    temps = [item['main']['temp'] for item in filtered_data]

    fig, ax = plt.subplots()
    ax.plot(times, temps, marker='o', linestyle='-', color='tab:blue')
    ax.set_title(f"Temperature Trend on {selected_date}", fontsize=12)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Temperature (°C)", fontsize=10)
    ax.grid(True)
    st.pyplot(fig)

    # Show crop suggestion
    if crop_suggestion:
        st.success(f"🌾 **Recommended Crop for {selected_date}:** {crop_suggestion}")



# Sidebar menu
st.sidebar.title("TerraGrow")
app_mode = st.sidebar.selectbox("Select ",["HOME","DISEASE RECOGNITION","CROP RECOMMENDATION","FERTILIZER RECOMMENDATION","LOCATION WEATHER"])

# Home page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        predicted_disease = class_name[result_index]
        
        # Show disease details (description, prevention, cure, links)
        display_disease_details(predicted_disease)

        # Section to display training history
    with st.expander("📊 Show Training History and Accuracy"):
        try:
            # Load training history
            with open("training_hist.json", "r") as f:
                history = json.load(f)

            # Final accuracy
            final_val_acc = history["val_accuracy"][-1]
            st.success(f"✅ Final Validation Accuracy: {final_val_acc * 100:.2f}%")

            # Convert to DataFrame
            history_df = pd.DataFrame(history)

            # Line chart of accuracy
            st.subheader("📈 Accuracy Over Epochs")
            st.line_chart(history_df[['accuracy', 'val_accuracy']])

            # Optional: Add loss chart
            st.subheader("📉 Loss Over Epochs")
            st.line_chart(history_df[['loss', 'val_loss']])

        except Exception as e:
            st.error(f"⚠️ Could not load accuracy history: {e}")

elif app_mode == "CROP RECOMMENDATION":
    st.markdown("<h1 style='text-align: center;'>Nurture With Intelligence</h1>", unsafe_allow_html=True)
    st.sidebar.header("Enter Crop Details")
    nitrogen = st.sidebar.number_input("Nitrogen", 0.0, 140.0, 0.0, 0.1)
    phosphorus = st.sidebar.number_input("Phosphorus", 0.0, 145.0, 0.0, 0.1)
    potassium = st.sidebar.number_input("Potassium", 0.0, 205.0, 0.0, 0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 51.0, 0.0, 0.1)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 0.0, 0.1)
    ph = st.sidebar.number_input("pH Level", 0.0, 14.0, 0.0, 0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 0.0, 0.1)
    if st.sidebar.button("Predict"):
        inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"The recommended crop is: {prediction[0]}")
            st.write(f"🌾 Crop Recommendation Model Accuracy: {x * 100:.2f}%")
            st.pyplot(plt)

elif app_mode == "FERTILIZER RECOMMENDATION":
    st.subheader("🌾 Fertilizer Recommendation System")

    model, accuracy, le_soil, le_crop = load_and_train_fertilizer_model()

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Temperature (°C)", 0, 100, 30)
        humidity = st.number_input("Humidity (%)", 0, 100, 50)
        moisture = st.number_input("Moisture (%)", 0, 100, 40)
        nitrogen = st.number_input("Nitrogen (N)", 0, 100, 20)

    with col2:
        potassium = st.number_input("Potassium (K)", 0, 100, 0)
        phosphorous = st.number_input("Phosphorous (P)", 0, 100, 0)
        soil_type = st.selectbox("Soil Type", le_soil.classes_)
        crop_type = st.selectbox("Crop Type", le_crop.classes_)

    if st.button("Predict Fertilizer"):
        soil_encoded = le_soil.transform([soil_type])[0]
        crop_encoded = le_crop.transform([crop_type])[0]
        features = [[temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorous]]
        prediction = model.predict(features)[0]

        st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
        st.info(f"✅ Model Accuracy: **{accuracy*100:.2f}%**")


elif app_mode == "LOCATION WEATHER":
    st.title("🌦️ Weather Forecast")
    st.markdown("View a **5-day weather forecast** by selecting a city and picking a date.")

    city_name = st.text_input("Enter your city name:")

    
    if city_name:
        api_key = "f3fa75d04d345e1e232de465a435970e"
        forecast_data = fetch_weather_forecast(city_name, api_key)

        if forecast_data:
            unique_dates = extract_unique_dates(forecast_data)
            selected_date = st.selectbox("Select a date to view forecast:", unique_dates)

            # Current temperature
            current_temp = fetch_current_weather(city_name, api_key)
            if current_temp is not None:
                st.markdown(f"### 🌡️ Current Temperature in {city_name.title()}: `{current_temp}°C`")

            if selected_date:
                display_weather_data(forecast_data, selected_date)

