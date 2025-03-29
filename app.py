'''
Author: Shifa

This Crime Analytics Dashboard provides law enforcement with interactive geospatial crime visualization (heatmaps, clustered markers) and AI-powered report analysis, featuring PDF data extraction, machine learning classification, and severity risk assessment. Built with Python's Streamlit, Folium, and Scikit-learn, it enables real-time crime pattern analysis and automated report processing for tactical response planning
'''

# Import libraries
import streamlit as st
import folium
import pandas as pd
import streamlit.components.v1 as components
import re
import os
import fitz  
import joblib
from folium.plugins import HeatMap, MarkerCluster


# CUSTOM STYLING
#####################################################
st.markdown("""
    <style>
    
        :root {
            --primary-text: #00008B;
            --secondary-text: #4B5563;
        }
        
        
        /* Global text */
        body, p, div, span, 
        .stTextInput, .stSelectbox, .stFileUploader,
        .stDataFrame, .stMarkdown {
            color: var(--primary-text) !important;
        }
        
        
        /* Main container styling */
        .block-container {
            margin: 20px !important;
            padding: 20px !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Column styling - enhanced spacing */
        div[data-testid="column"] {
            border: 1px solid #e1e4e8;
            border-radius: 10px;
            padding: 20px;
            background-color: #f8f9fa;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        
        /* Add space between columns */
        div[data-testid="column"]:first-child {
            margin-right: 15px;
        }
        
        div[data-testid="column"]:last-child {
            margin-left: 15px;
        }
        
        /* Hover effect for columns */
        div[data-testid="column"]:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        /* Title styling */
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #00008B;
            padding-bottom: 10px;
            margin-bottom: 25px;
        }
        
        /* Subheader styling */
        h2 {
            color: #2980b9;
            margin-top: 20px !important;
            margin-bottom: 15px !important;
        }
        
        /* Expander styling */
        .st-expander {
            margin-bottom: 15px;
            border-radius: 8px;
        }
        
        /* File uploader styling */
        .stFileUploader {
            margin-bottom: 15px;
        }
        
        /* Add subtle divider between columns */
        .column-divider {
            width: 1px;
            background: linear-gradient(to bottom, transparent, #e1e4e8, transparent);
            margin: 0 15px;
        }
    </style>
""", unsafe_allow_html=True)


# DATA LOADING FUNCTIONS
#####################################################
def load_data():
    # Load and preprocess crime dataset
    df = pd.read_csv("Competition_Dataset.csv", 
                    usecols=["Latitude (Y)", "Longitude (X)", "Category", "Descript"])
    df.dropna(subset=["Latitude (Y)", "Longitude (X)"], inplace=True)
    return df

def load_crime_model():
    #Load pre-trained crime classification model
    try:
        model = joblib.load('crime_classifier_model.joblib')
        #st.success("Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None


# REPORT PROCESSING FUNCTIONS
#####################################################
def extract_report_data(pdf_path):
    # Extract structured data from police report PDF
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()

        # Extract key fields using regex
        report_data = {
            "Report Number": re.search(r"Report Number:\s*(\d+-\d+)", text),
            "Date & Time": re.search(r"Date & Time:\s*([\d-]+\s[\d:]+)", text),
            "Reporting Officer": re.search(r"Reporting Officer:\s*([^\n]+)", text),
            "Incident Location": re.search(r"Incident Location:\s*([^\n]+)", text),
            "Coordinates": re.search(r"Coordinates:\s*\(([^)]+)\)", text),
            "Detailed Description": re.search(r"Detailed Description:\s*([\s\S]+?)\nPolice District", text),
            "Police District": re.search(r"Police District:\s*([^\n]+)", text),
            "Resolution": re.search(r"Resolution:\s*([^\n]+)", text),
            "Suspect Description": re.search(r"Suspect Description:\s*([^\n]+)", text),
            "Victim Information": re.search(r"Victim Information:\s*([^\n]+)", text),
        }

        # Extract and clean values
        for key, match in report_data.items():
            report_data[key] = match.group(1).strip() if match else "N/A"

        return report_data

    except Exception as e:
        st.error(f"Failed to extract data: {str(e)}")
        return None

# PREDICTION FUNCTIONS
#####################################################
def map_categories_to_names(y_pred, category_mapping):
    """Map predicted categories to crime names."""
    return [list(category_mapping.keys())[list(category_mapping.values()).index(label)] 
            for label in y_pred]

def map_severity_to_categories(predicted_categories, crime_severity_mapping):
    """Map predicted crime categories to severity levels."""
    return [crime_severity_mapping.get(crime, 'Unknown') 
            for crime in predicted_categories]

def predict_crime(report_info, model, category_mapping, crime_severity_mapping):
    """Predict crime category & severity from report."""
    if model is None:
        return {"Crime Category": "N/A", "Severity": "N/A"}

    # Convert report info into model input
    structured_input = [report_info["Detailed Description"]]


    # Make predictions
    y_pred = model.predict(structured_input)
    predicted_categories = map_categories_to_names(y_pred, category_mapping)
    predicted_severity = map_severity_to_categories(predicted_categories, crime_severity_mapping)

    return {
        "Crime Category": predicted_categories[0], 
        "Severity": predicted_severity[0]
    }


# MAP GENERATION FUNCTION
#####################################################
@st.cache_resource
def generate_map(filtered_data):
    # Generate interactive crime map with heatmap and markers
    crime_map = folium.Map(location=[37.7749, -122.4194], zoom_start=12)

    # Marker Cluster Layer
    marker_cluster = MarkerCluster(name="Crime Markers").add_to(crime_map)

    # HeatMap Layer
    heat_map_layer = folium.FeatureGroup(name="Crime HeatMap").add_to(crime_map)

    # Sample data if too large
    filtered_data_sample = filtered_data.sample(n=1000) if len(filtered_data) > 1000 else filtered_data

    # Add markers to cluster
    for idx, row in filtered_data_sample.iterrows():
        folium.Marker(
            [row["Longitude (X)"], row["Latitude (Y)"]], 
            popup=f"<b>Crime:</b> {row['Category']}<br><b>Details:</b> {row['Descript']}"
        ).add_to(marker_cluster)

    # Add heatmap data
    heat_data = [
        [row["Longitude (X)"], row["Latitude (Y)"]] 
        for _, row in filtered_data_sample.iterrows()
    ]
    HeatMap(heat_data).add_to(heat_map_layer)

    # Add layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(crime_map)

    return crime_map


# MAIN APP LAYOUT
#####################################################
st.title("Crime Data Visualization & Prediction Dashboard 🚨")

# Create two main columns with better spacing
col1, gap, col2 = st.columns([2, 0.1, 2])

# Add vertical divider between columns
with gap:
    st.markdown('<div class="column-divider"></div>', unsafe_allow_html=True)


# COLUMN 1: CRIME DATA VISUALIZATION
#####################################################
with col1:
    st.header("Crime Location Visualization 📌")
    
    # Load data
    df = load_data()
    
    # Category selection
    category = st.selectbox(
        "Select Crime Category", 
        df['Category'].unique(),
        help="Choose a crime category to visualize on the map"
    )
    
    # Filter data
    filtered_data = df[df['Category'] == category]
    
    # Generate and display map
    crime_map = generate_map(filtered_data)
    crime_map.save("crime_map.html")
    
    with st.expander("Interactive Crime Map", expanded=True):
        with open("crime_map.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=600)
        
        st.caption(f"Showing {len(filtered_data)} crime records for {category}")

# COLUMN 2: REPORT ANALYSIS & PREDICTION
#####################################################
with col2:
    st.header("Police Report Analysis 📈")
    
    # Load model
    model = load_crime_model()
    
    # Define category mappings
    category_mapping = {
        'WARRANTS': 0, 'OTHER OFFENSES': 1, 'LARCENY/THEFT': 2, 
        'VEHICLE THEFT': 3, 'VANDALISM': 4, 'NON-CRIMINAL': 5, 
        'ROBBERY': 6, 'WEAPON LAWS': 7, 'BURGLARY': 8, 
        'SUSPICIOUS OCC': 9, 'FORGERY/COUNTERFEITING': 10, 
        'DRUG/NARCOTIC': 11, 'TRESPASS': 12, 'MISSING PERSON': 13, 
        'KIDNAPPING': 14, 'RUNAWAY': 15, 'FRAUD': 16, 
        'DISORDERLY CONDUCT': 17, 'ARSON': 18, 'BRIBERY': 19, 
        'EMBEZZLEMENT': 20, 'EXTORTION': 21, 'BAD CHECKS': 22, 
        'STOLEN PROPERTY': 23, 'RECOVERED VEHICLE': 24
    }
    
    # Define crime_severity_mapping
    crime_severity_mapping = {
        'NON-CRIMINAL': 1, 'SUSPICIOUS OCCURRENCE': 1, 
        'MISSING PERSON': 1, 'RUNAWAY': 1, 'RECOVERED VEHICLE': 1,
        'WARRANTS': 2, 'OTHER OFFENSES': 2, 'VANDALISM': 2, 
        'TRESPASS': 2, 'DISORDERLY CONDUCT': 2, 'BAD CHECKS': 2, 
        'LARCENY/THEFT': 3, 'VEHICLE THEFT': 3, 'FORGERY/COUNTERFEITING': 3, 
        'DRUG/NARCOTIC': 3, 'STOLEN PROPERTY': 3, 'FRAUD': 3, 
        'BRIBERY': 3, 'EMBEZZLEMENT': 3, 'ROBBERY': 4, 
        'WEAPON LAWS': 4, 'BURGLARY': 4, 'EXTORTION': 4, 
        'KIDNAPPING': 5, 'ARSON': 5
    }
    
    # File uploader with better spacing
    with st.expander("📤 Upload Police Report", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type=["pdf"],
            help="Upload a police report PDF for analysis and prediction"
        )
    
    if uploaded_file is not None and model is not None:
        # Save uploaded file
        os.makedirs("static", exist_ok=True)
        pdf_path = f"static/{uploaded_file.name}"
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Extract report data
        report_info = extract_report_data(pdf_path)
        
        if report_info:
            # Show extracted data with better spacing
            with st.expander("📋 Extracted Report Data", expanded=True):
                report_df = pd.DataFrame([report_info])
                st.dataframe(report_df)
                st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
            
            # Make predictions
            predictions = predict_crime(
                report_info, 
                model, 
                category_mapping, 
                crime_severity_mapping
            )
            
            # Display predictions with better visual hierarchy
            with st.expander("🚨 Prediction Results", expanded=True):
                col_pred1, col_pred2 = st.columns(2)
                
                with col_pred1:
                    st.metric(
                        "Predicted Crime", 
                        predictions['Crime Category'],
                        help="The predicted category of crime based on the report"
                    )
                
                with col_pred2:
                    severity_level = predictions['Severity']
                    
                    severity_color = {
                        1: "#28a745",
                        2: "#007bff",
                        3: "#fd7e14",
                        4: "#dc3545",
                        5: "#8B0000"
                    }.get(predictions['Severity'], "#6c757d")
                    
                    # Use components.html instead of st.markdown
                    components.html(f"""
                        <div style="
                            border-left: 4px solid {severity_color};
                            padding-left: 12px;
                            margin: 8px 0;
                            all: initial; /* Nuclear reset */
                        ">
                            <div style="
                                font-size: 0.8rem;
                                color: #00008B !important;
                                margin-bottom: 4px;
                                font-family: inherit;
                            ">Severity Level </div>
                            <div style="
                                font-size: 2.5rem;
                                font-weight: bold;
                                color: {severity_color} !important;
                                line-height: 1;
                                font-family: inherit;
                            ">{severity_level}</div>
                        </div>
                    """, height=80)  # Adjust height as needed
                    
                # Add some vertical space
                st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
            
            # Show location if coordinates exist
            try:
                lat, lon = map(float, report_info["Coordinates"].split(", "))
                crime_map = folium.Map(location=[lat, lon], zoom_start=14)
                folium.Marker(
                    [lat, lon], 
                    popup=report_info["Detailed Description"],
                    icon=folium.Icon(color='red', icon='exclamation-triangle')
                ).add_to(crime_map)
                
                with st.expander("Incident Location 📌", expanded=True):
                    crime_map.save("incident_map.html")
                    with open("incident_map.html", "r", encoding="utf-8") as f:
                        components.html(f.read(), height=400)
            
            except Exception as e:
                st.warning("⚠️ Could not extract coordinates from the report.")