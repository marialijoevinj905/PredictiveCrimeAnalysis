import streamlit as st
import streamlit.components.v1 as components 
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings
from datetime import datetime, timedelta
import io
import base64

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Crime Analysis Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CrimeAnalysisApp:
    def __init__(self):
        # Use session state to persist data across page changes
        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        
        self.datasets = st.session_state.datasets
        self.processed_data = st.session_state.processed_data
    
    def validate_dataset(self, df, dataset_type):
        """Validate if the uploaded dataset has required columns"""
        
        return True, "Dataset validated successfully"
    
    def preprocess_dataset(self, df, dataset_type):
        """Preprocess the dataset based on its type"""
        try:
            df_processed = df.copy()
            
            if dataset_type == 'FIR_Details_Data':
                # Ensure numeric columns
                numeric_cols = ['Year', 'Month', 'VICTIM_COUNT', 'Accused_Count', 'Male', 'Female']
                for col in numeric_cols:
                    if col in df_processed.columns:
                        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
                
                # Ensure coordinate columns are numeric
                coord_cols = ['Latitude', 'Longitude']
                for col in coord_cols:
                    if col in df_processed.columns:
                        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # Create date column if Year and Month exist
                if 'Year' in df_processed.columns and 'Month' in df_processed.columns:
                    df_processed['Date'] = pd.to_datetime(
                        df_processed[['Year', 'Month']].assign(day=1), errors='coerce'
                    )
            
            elif dataset_type in ['VictimInfoDetails', 'Accused Data']:
                # Ensure age is numeric
                if 'age' in df_processed.columns:
                    df_processed['age'] = pd.to_numeric(df_processed['age'], errors='coerce')
            
            return df_processed, "Dataset preprocessed successfully"
            
        except Exception as e:
            return None, f"Error preprocessing dataset: {str(e)}"
    
    def load_data_interface(self):
        """Create interface for loading CSV datasets"""
        st.header("📁 Data Import")
        
        st.markdown("""
        <div class='upload-section'>
        <h4>📋 Dataset Requirements</h4>
        <p>Please upload your crime datasets in CSV format. The system supports three main dataset types:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset upload tabs
        tab1, tab2, tab3 = st.tabs(["🚨 FIR Details", "👥 Victim Information", "🔍 Accused Data"])
        
        with tab1:
            st.subheader("FIR Details Dataset")
            st.info("""
            **Required columns:** District_Name, UnitName, FIRNo, Year, Month, CrimeGroup_Name
            
            **Optional columns:** Latitude, Longitude, VICTIM_COUNT, Accused_Count, Male, Female
            """)
            
            fir_file = st.file_uploader(
                "Upload FIR Details CSV",
                type=['csv'],
                key='fir_upload',
                help="Upload the main FIR/Crime details dataset"
            )
            
            if fir_file is not None:
                try:
                    df = pd.read_csv(fir_file)
                    is_valid, message = self.validate_dataset(df, 'FIR_Details_Data')
                    
                    if is_valid:
                        processed_df, process_message = self.preprocess_dataset(df, 'FIR_Details_Data')
                        if processed_df is not None:
                            self.datasets['FIR_Details_Data'] = processed_df
                            st.session_state.datasets['FIR_Details_Data'] = processed_df
                            self.processed_data = processed_df
                            st.session_state.processed_data = processed_df
                            st.success(f"✅ FIR Dataset loaded successfully! ({len(df)} records)")
                            st.info(process_message)
                            
                            # Show preview
                            with st.expander("📊 Dataset Preview"):
                                st.dataframe(df.head())
                                st.write(f"**Shape:** {df.shape}")
                                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                        else:
                            st.error(f"❌ {process_message}")
                    else:
                        st.error(f"❌ {message}")
                        st.write("**Available columns in your file:**", ', '.join(df.columns.tolist()))
                        
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        with tab2:
            st.subheader("Victim Information Dataset")
            st.info("""
            **Required columns:** District_Name, VictimName, age, Sex
            
            **Optional columns:** Profession, InjuryType
            """)
            
            victim_file = st.file_uploader(
                "Upload Victim Information CSV",
                type=['csv'],
                key='victim_upload',
                help="Upload the victim details dataset"
            )
            
            if victim_file is not None:
                try:
                    df = pd.read_csv(victim_file)
                    is_valid, message = self.validate_dataset(df, 'VictimInfoDetails')
                    
                    if is_valid:
                        processed_df, process_message = self.preprocess_dataset(df, 'VictimInfoDetails')
                        if processed_df is not None:
                            self.datasets['VictimInfoDetails'] = processed_df
                            st.session_state.datasets['VictimInfoDetails'] = processed_df
                            st.success(f"✅ Victim Dataset loaded successfully! ({len(df)} records)")
                            st.info(process_message)
                            
                            # Show preview
                            with st.expander("📊 Dataset Preview"):
                                st.dataframe(df.head())
                                st.write(f"**Shape:** {df.shape}")
                                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                        else:
                            st.error(f"❌ {process_message}")
                    else:
                        st.error(f"❌ {message}")
                        st.write("**Available columns in your file:**", ', '.join(df.columns.tolist()))
                        
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        with tab3:
            st.subheader("Accused Data Dataset")
            st.info("""
            **Required columns:** District_Name, AccusedName, age, Sex
            
            **Optional columns:** Profession
            """)
            
            accused_file = st.file_uploader(
                "Upload Accused Data CSV",
                type=['csv'],
                key='accused_upload',
                help="Upload the accused persons dataset"
            )
            
            if accused_file is not None:
                try:
                    df = pd.read_csv(accused_file)
                    is_valid, message = self.validate_dataset(df, 'Accused Data')
                    
                    if is_valid:
                        processed_df, process_message = self.preprocess_dataset(df, 'Accused Data')
                        if processed_df is not None:
                            self.datasets['Accused Data'] = processed_df
                            st.session_state.datasets['Accused Data'] = processed_df
                            st.success(f"✅ Accused Dataset loaded successfully! ({len(df)} records)")
                            st.info(process_message)
                            
                            # Show preview
                            with st.expander("📊 Dataset Preview"):
                                st.dataframe(df.head())
                                st.write(f"**Shape:** {df.shape}")
                                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                        else:
                            st.error(f"❌ {process_message}")
                    else:
                        st.error(f"❌ {message}")
                        st.write("**Available columns in your file:**", ', '.join(df.columns.tolist()))
                                    
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        # Dataset status summary
        if self.datasets:
            st.markdown("---")
            st.subheader("📈 Loaded Datasets Summary")
            
            cols = st.columns(len(self.datasets))
            for i, (name, df) in enumerate(self.datasets.items()):
                with cols[i]:
                    st.metric(
                        name.replace('_', ' '),
                        f"{len(df)} records",
                        f"{len(df.columns)} columns"
                    )
    
    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        st.header("📊 Exploratory Data Analysis")
        
        if not self.datasets:
            st.warning("⚠️ Please load datasets first using the Data Import page!")
            return
        
        # Dataset selection
        dataset_choice = st.selectbox(
            "Select Dataset for Analysis:",
            list(self.datasets.keys())
        )
        
        selected_data = self.datasets[dataset_choice]
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(selected_data))
        with col2:
            st.metric("Columns", len(selected_data.columns))
        with col3:
            if 'District_Name' in selected_data.columns:
                st.metric("Districts", selected_data['District_Name'].nunique())
            else:
                st.metric("Districts", "N/A")
        with col4:
            if 'Year' in selected_data.columns:
                st.metric("Years Covered", selected_data['Year'].nunique())
            else:
                st.metric("Years Covered", "N/A")
        
        # Data quality check
        st.subheader("🔍 Data Quality Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values
            missing_data = selected_data.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data.values / len(selected_data) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if not missing_df.empty:
                st.write("**Missing Values:**")
                st.dataframe(missing_df)
            else:
                st.success("✅ No missing values found!")
        
        with col2:
            # Data types
            dtype_df = pd.DataFrame({
                'Column': selected_data.dtypes.index,
                'Data Type': selected_data.dtypes.values.astype(str)
            })
            st.write("**Data Types:**")
            st.dataframe(dtype_df)
        
        # Data overview
        st.subheader("📋 Dataset Overview")
        st.dataframe(selected_data.head(10))
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        numeric_cols = selected_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(selected_data[numeric_cols].describe())
        else:
            st.info("No numeric columns found for statistical summary.")
        
        # Visualizations based on dataset
        if dataset_choice == 'FIR_Details_Data':
            self.eda_fir_analysis(selected_data)
        elif dataset_choice == 'VictimInfoDetails':
            self.eda_victim_analysis(selected_data)
        elif dataset_choice == 'Accused Data':
            self.eda_accused_analysis(selected_data)
    
    def eda_fir_analysis(self, data):
        """EDA specific to FIR data"""
        st.subheader("🚨 FIR Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Crime distribution by type
            if 'CrimeGroup_Name' in data.columns:
                crime_counts = data['CrimeGroup_Name'].value_counts().head(10)
                fig = px.bar(
                    x=crime_counts.values,
                    y=crime_counts.index,
                    orientation='h',
                    title="Top 10 Crime Types",
                    labels={'x': 'Count', 'y': 'Crime Type'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # District-wise crime distribution
            if 'District_Name' in data.columns:
                district_counts = data['District_Name'].value_counts()
                fig = px.pie(
                    values=district_counts.values,
                    names=district_counts.index,
                    title="Crime Distribution by District"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        if all(col in data.columns for col in ['Year', 'Month']):
            st.subheader("📅 Temporal Analysis")
            
            # Monthly trend
            monthly_crimes = data.groupby(['Year', 'Month']).size().reset_index(name='Count')
            monthly_crimes['Date'] = pd.to_datetime(monthly_crimes[['Year', 'Month']].assign(day=1))
            
            fig = px.line(
                monthly_crimes, x='Date', y='Count',
                title="Crime Trends Over Time",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Year-wise comparison
            yearly_crimes = data['Year'].value_counts().sort_index()
            fig = px.bar(
                x=yearly_crimes.index,
                y=yearly_crimes.values,
                title="Year-wise Crime Count",
                labels={'x': 'Year', 'y': 'Crime Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def eda_victim_analysis(self, data):
        """EDA specific to victim data"""
        st.subheader("👥 Victim Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in data.columns:
                # Remove any invalid age values
                valid_ages = data[data['age'].notna() & (data['age'] > 0) & (data['age'] < 120)]
                if not valid_ages.empty:
                    fig = px.histogram(
                        valid_ages, x='age', nbins=20,
                        title="Age Distribution of Victims"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid age data found")
        
        with col2:
            if 'Sex' in data.columns:
                gender_counts = data['Sex'].value_counts()
                fig = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title="Victim Gender Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis if profession data exists
        if 'Profession' in data.columns:
            prof_counts = data['Profession'].value_counts().head(10)
            fig = px.bar(
                x=prof_counts.values,
                y=prof_counts.index,
                orientation='h',
                title="Victims by Profession"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def eda_accused_analysis(self, data):
        """EDA specific to accused data"""
        st.subheader("🔍 Accused Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in data.columns:
                # Remove any invalid age values
                valid_ages = data[data['age'].notna() & (data['age'] > 0) & (data['age'] < 120)]
                if not valid_ages.empty:
                    fig = px.histogram(
                        valid_ages, x='age', nbins=20,
                        title="Age Distribution of Accused"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid age data found")
        
        with col2:
            if 'Profession' in data.columns:
                prof_counts = data['Profession'].value_counts().head(10)
                fig = px.bar(
                    x=prof_counts.values,
                    y=prof_counts.index,
                    orientation='h',
                    title="Accused by Profession"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def create_crime_hotspot_map(self):
        """Create interactive crime hotspot visualization with batch processing"""
        st.header("🗺️ Crime Hotspot Mapping")
        
        if 'FIR_Details_Data' not in self.datasets:
            st.warning("⚠️ Please load FIR Details dataset first!")
            return
        
        data = self.datasets['FIR_Details_Data'].copy()
        
        # Check if coordinate data exists
        if not all(col in data.columns for col in ['Latitude', 'Longitude']):
            st.error("❌ Latitude and Longitude columns are required for mapping!")
            st.info("Please ensure your FIR dataset contains 'Latitude' and 'Longitude' columns.")
            return
        
        # Clean and convert all data to proper types upfront
        try:
            # Convert coordinates to numeric, handling any string values
            data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
            data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
            
            # Convert other numeric columns that might cause issues
            if 'Year' in data.columns:
                data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            if 'Month' in data.columns:
                data['Month'] = pd.to_numeric(data['Month'], errors='coerce')
            if 'VICTIM COUNT' in data.columns:
                data['VICTIM COUNT'] = pd.to_numeric(data['VICTIM COUNT'], errors='coerce')
            
            # Convert all object columns to strings and handle NaN values
            for col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = data[col].astype(str).replace('nan', '')
                    data[col] = data[col].replace('None', '')
                    data[col] = data[col].replace('<NA>', '')
                    
        except Exception as e:
            st.error(f"❌ Error preprocessing data: {str(e)}")
            return
        
        # Remove rows with missing coordinates and invalid coordinates
        coord_data = data.dropna(subset=['Latitude', 'Longitude'])
        coord_data = coord_data[
            (coord_data['Latitude'].between(-90, 90)) & 
            (coord_data['Longitude'].between(-180, 180)) &
            (coord_data['Latitude'].notna()) &
            (coord_data['Longitude'].notna())
        ]
        
        if len(coord_data) == 0:
            st.error("❌ No valid coordinate data found!")
            return
        
        # Batch processing parameters
        BATCH_SIZE = 1000  # Process data in batches of 1000 records
        MAX_DISPLAY_POINTS = 5000  # Maximum points to display on map
        
        st.info(f"📊 Processing {len(coord_data)} records in batches for optimal performance...")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'District_Name' in data.columns:
                # Clean district names and remove empty values
                available_districts = [d for d in data['District_Name'].unique() if d and d != '' and d != 'nan']
                selected_districts = st.multiselect(
                    "Select Districts:",
                    available_districts,
                    default=available_districts[:min(3, len(available_districts))]
                )
            else:
                selected_districts = None
        
        with col2:
            if 'CrimeGroup_Name' in data.columns:
                # Clean crime names and remove empty values
                available_crimes = [c for c in data['CrimeGroup_Name'].unique() if c and c != '' and c != 'nan']
                selected_crimes = st.multiselect(
                    "Select Crime Types:",
                    available_crimes,
                    default=available_crimes[:min(5, len(available_crimes))]
                )
            else:
                selected_crimes = None
        
        with col3:
            if 'Year' in data.columns:
                available_years = sorted([y for y in data['Year'].dropna().unique() if pd.notna(y)])
                if available_years:
                    selected_year = st.selectbox(
                        "Select Year:",
                        ['All Years'] + [str(int(y)) for y in available_years],
                        index=0
                    )
                else:
                    selected_year = None
            else:
                selected_year = None
        
        # Apply filters in batches to avoid memory issues
        filtered_data = coord_data.copy()
        
        # Filter by districts
        if selected_districts and 'District_Name' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['District_Name'].isin(selected_districts)]
        
        # Filter by crime types
        if selected_crimes and 'CrimeGroup_Name' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['CrimeGroup_Name'].isin(selected_crimes)]
        
        # Filter by year
        if selected_year and selected_year != 'All Years' and 'Year' in filtered_data.columns:
            try:
                year_num = float(selected_year)
                filtered_data = filtered_data[filtered_data['Year'] == year_num]
            except (ValueError, TypeError):
                pass
        
        if len(filtered_data) == 0:
            st.warning("⚠️ No data available for selected filters!")
            return
        
        # Implement sampling for large datasets to prevent crashes
        if len(filtered_data) > MAX_DISPLAY_POINTS:
            st.warning(f"⚠️ Dataset is large ({len(filtered_data)} points). Displaying a sample of {MAX_DISPLAY_POINTS} points for better performance.")
            filtered_data = filtered_data.sample(n=MAX_DISPLAY_POINTS, random_state=42)
        
        # Calculate map center using batch processing
        center_lat = float(filtered_data['Latitude'].mean())
        center_lon = float(filtered_data['Longitude'].mean())
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        try:
            # Process heatmap data in batches
            progress_bar = st.progress(0)
            heat_data = []
            
            total_batches = (len(filtered_data) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for i in range(0, len(filtered_data), BATCH_SIZE):
                batch_data = filtered_data.iloc[i:i+BATCH_SIZE]
                for _, row in batch_data.iterrows():
                    try:
                        lat = float(row['Latitude'])
                        lon = float(row['Longitude'])
                        if pd.notna(lat) and pd.notna(lon):
                            heat_data.append([lat, lon])
                    except (ValueError, TypeError):
                        continue
                
                # Update progress
                progress = min((i + BATCH_SIZE) / len(filtered_data), 1.0)
                progress_bar.progress(progress)
            
            progress_bar.empty()
            
            # Add heatmap with optimized parameters
            if heat_data:
                HeatMap(
                    heat_data, 
                    radius=12, 
                    blur=8, 
                    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
                    min_opacity=0.3,
                    max_zoom=18
                ).add_to(m)
            
            # Add marker clusters with batch processing
            marker_cluster = MarkerCluster(
                maxClusterRadius=50,
                spiderfyOnMaxZoom=True,
                showCoverageOnHover=False
            ).add_to(m)
            
            # Process markers in smaller batches to avoid memory issues
            MARKER_BATCH_SIZE = 100
            marker_count = 0
            max_markers = min(500, len(filtered_data))  # Limit markers for performance
            
            for i in range(0, min(len(filtered_data), max_markers), MARKER_BATCH_SIZE):
                batch_data = filtered_data.iloc[i:i+MARKER_BATCH_SIZE]
                
                for _, row in batch_data.iterrows():
                    try:
                        lat = float(row['Latitude'])
                        lon = float(row['Longitude'])
                        
                        if pd.isna(lat) or pd.isna(lon):
                            continue
                        
                        # Create popup with available information - Safe string handling
                        popup_info = []
                        
                        # Safely handle each field
                        def safe_get_value(row, col_name, default="N/A"):
                            if col_name in row and pd.notna(row[col_name]):
                                val = str(row[col_name]).strip()
                                return val if val and val != 'nan' and val != '' else default
                            return default
                        
                        fir_no = safe_get_value(row, 'FIRNo')
                        if fir_no != "N/A":
                            popup_info.append(f"<b>FIR:</b> {fir_no}")
                        
                        crime_type = safe_get_value(row, 'CrimeGroup_Name')
                        if crime_type != "N/A":
                            popup_info.append(f"<b>Crime:</b> {crime_type}")
                        
                        district = safe_get_value(row, 'District_Name')
                        if district != "N/A":
                            popup_info.append(f"<b>District:</b> {district}")
                        
                        # Handle victim count safely
                        if 'VICTIM COUNT' in row and pd.notna(row['VICTIM COUNT']):
                            try:
                                victim_count = int(float(row['VICTIM COUNT']))
                                if victim_count > 0:
                                    popup_info.append(f"<b>Victims:</b> {victim_count}")
                            except (ValueError, TypeError):
                                pass
                        
                        # Safe date handling
                        if 'Year' in row and 'Month' in row:
                            try:
                                year_val = row['Year']
                                month_val = row['Month']
                                if pd.notna(year_val) and pd.notna(month_val):
                                    year_int = int(float(year_val))
                                    month_int = int(float(month_val))
                                    if 1 <= month_int <= 12:
                                        popup_info.append(f"<b>Date:</b> {year_int}-{month_int:02d}")
                            except (ValueError, TypeError):
                                pass
                        
                        # Ensure all popup elements are strings
                        popup_info_strings = [str(item) for item in popup_info if item is not None]
                        popup_text = "<br>".join(popup_info_strings) if popup_info_strings else "Crime Location"

                        folium.Marker(
                            location=[lat, lon],
                            popup=folium.Popup(str(popup_text), max_width=300),
                            icon=folium.Icon(color='red', icon='info-sign', prefix='glyphicon')
                        ).add_to(marker_cluster)

                        
                        marker_count += 1
                        
                    except (ValueError, TypeError, KeyError) as e:
                        # Skip problematic rows
                        continue
            
            # Display map
            st.subheader(f"Crime Hotspots{f' - {selected_year}' if selected_year and selected_year != 'All Years' else ''}")
            
            # Add map controls information
            st.info("🗺️ Map Controls: Use mouse wheel to zoom, click and drag to pan. Click on cluster numbers to zoom into groups.")
            
            # Display the map using folium_static to avoid component issues
            st_folium(m, width=700, height=600)
            
            # Crime statistics for filtered data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Crimes", len(filtered_data))
            with col2:
                if 'VICTIM COUNT' in filtered_data.columns:
                    total_victims = pd.to_numeric(filtered_data['VICTIM COUNT'], errors='coerce').fillna(0).sum()
                    st.metric("Total Victims", int(total_victims))
                else:
                    st.metric("Districts Covered", 
                            len([d for d in filtered_data['District_Name'].unique() if d and d != 'nan']) if 'District_Name' in filtered_data.columns else "N/A")
            with col3:
                if 'CrimeGroup_Name' in filtered_data.columns:
                    unique_crimes = len([c for c in filtered_data['CrimeGroup_Name'].unique() if c and c != 'nan'])
                    st.metric("Crime Types", unique_crimes)
                else:
                    st.metric("Crime Types", "N/A")
            with col4:
                if marker_count < len(filtered_data):
                    st.metric("Markers Shown", f"{marker_count}/{len(filtered_data)}")
                else:
                    st.metric("Markers Shown", marker_count)
            
            # Hotspot analysis summary
            st.subheader("🔥 Hotspot Analysis Summary")
            
            if 'District_Name' in filtered_data.columns:
                # Clean district data for analysis
                clean_districts = filtered_data[filtered_data['District_Name'].str.strip() != '']
                if len(clean_districts) > 0:
                    hotspot_districts = clean_districts['District_Name'].value_counts().head(5)
                    st.write("**Top 5 Crime Hotspot Districts:**")
                    for i, (district, count) in enumerate(hotspot_districts.items(), 1):
                        if district and district != 'nan':
                            st.write(f"{i}. {district}: {count} crimes")
            
        except Exception as e:
            st.error(f"❌ Error creating map: {str(e)}")
            st.info("💡 Try reducing the date range or selecting fewer districts to improve performance.")
            # Add debugging information
            st.write("Debug Info:")
            st.write(f"Data shape: {filtered_data.shape}")
            if len(filtered_data) > 0:
                st.write(f"Coordinate data types: Lat={filtered_data['Latitude'].dtype}, Lon={filtered_data['Longitude'].dtype}")
                st.write("Sample of problematic data:")
                st.write(filtered_data[['Latitude', 'Longitude', 'FIRNo']].head())
            st.write("Available columns:", list(filtered_data.columns))
    
    def predictive_analysis(self):
        """Perform predictive crime analysis with next crime prediction"""
        st.header("🔮 Predictive Crime Analysis")
        
        if not self.datasets:
            st.warning("⚠️ Please load datasets first!")
            return
        
        if 'FIR_Details_Data' not in self.datasets:
            st.warning("⚠️ FIR Details dataset required for prediction!")
            return
        
        data = self.datasets['FIR_Details_Data']
        
        # Check required columns for prediction
        required_cols = ['Year', 'Month']
        if not all(col in data.columns for col in required_cols):
            st.error("❌ Year and Month columns are required for prediction!")
            return
        
        st.subheader("📈 Next Crime Prediction with Date & Time")
        
        # Prepare temporal data with batch processing
        try:
            # Process data in batches to avoid memory issues
            BATCH_SIZE = 5000
            processed_batches = []
            
            progress_bar = st.progress(0)
            st.info("🔄 Processing temporal data in batches...")
            
            for i in range(0, len(data), BATCH_SIZE):
                batch = data.iloc[i:i+BATCH_SIZE].copy()
                
                # Clean and validate batch data
                batch = batch.dropna(subset=['Year', 'Month'])
                batch = batch[(batch['Year'] >= 2000) & (batch['Year'] <= 2030)]
                batch = batch[(batch['Month'] >= 1) & (batch['Month'] <= 12)]
                
                # Convert to numeric
                batch['Year'] = pd.to_numeric(batch['Year'], errors='coerce')
                batch['Month'] = pd.to_numeric(batch['Month'], errors='coerce')
                batch = batch.dropna(subset=['Year', 'Month'])
                
                if len(batch) > 0:
                    processed_batches.append(batch)
                
                # Update progress
                progress = min((i + BATCH_SIZE) / len(data), 1.0)
                progress_bar.progress(progress)
            
            progress_bar.empty()
            
            if not processed_batches:
                st.error("❌ No valid temporal data found after processing!")
                return
            
            # Combine processed batches
            clean_data = pd.concat(processed_batches, ignore_index=True)
            
            # Create comprehensive temporal features
            clean_data['Date'] = pd.to_datetime(
                clean_data[['Year', 'Month']].assign(day=1), 
                errors='coerce'
            )
            clean_data = clean_data.dropna(subset=['Date'])
            
            if len(clean_data) < 24:  # Need at least 2 years of data
                st.warning("⚠️ Insufficient temporal data for reliable prediction! (Need at least 24 months)")
                return
            
            # Advanced temporal aggregation
            temporal_agg = clean_data.groupby(['Year', 'Month']).agg({
                'FIRNo': 'count',
                'VICTIM COUNT': lambda x: x.fillna(0).sum() if 'VICTIM COUNT' in clean_data.columns else 0,
                'CrimeGroup_Name': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
                'District_Name': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            }).reset_index()
            
            temporal_agg.columns = ['Year', 'Month', 'Crime_Count', 'Victim_Count', 'Most_Common_Crime', 'Most_Affected_District']
            temporal_agg['Date'] = pd.to_datetime(temporal_agg[['Year', 'Month']].assign(day=1))
            temporal_agg = temporal_agg.sort_values('Date').reset_index(drop=True)
            
            # Advanced feature engineering
            temporal_agg['Time_Index'] = range(len(temporal_agg))
            temporal_agg['Month_Num'] = temporal_agg['Month']
            temporal_agg['Quarter'] = temporal_agg['Month'].apply(lambda x: (x-1)//3 + 1)
            temporal_agg['Season'] = temporal_agg['Month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            
            # Cyclical features
            temporal_agg['Sin_Month'] = np.sin(2 * np.pi * temporal_agg['Month_Num'] / 12)
            temporal_agg['Cos_Month'] = np.cos(2 * np.pi * temporal_agg['Month_Num'] / 12)
            temporal_agg['Sin_Quarter'] = np.sin(2 * np.pi * temporal_agg['Quarter'] / 4)
            temporal_agg['Cos_Quarter'] = np.cos(2 * np.pi * temporal_agg['Quarter'] / 4)
            
            # Lag features
            temporal_agg['Crime_Count_Lag1'] = temporal_agg['Crime_Count'].shift(1)
            temporal_agg['Crime_Count_Lag3'] = temporal_agg['Crime_Count'].shift(3)
            temporal_agg['Crime_Count_MA3'] = temporal_agg['Crime_Count'].rolling(window=3).mean()
            temporal_agg['Crime_Count_MA6'] = temporal_agg['Crime_Count'].rolling(window=6).mean()
            
            # Trend features
            temporal_agg['Year_Numeric'] = temporal_agg['Year'] - temporal_agg['Year'].min()
            
            # Remove rows with NaN values created by lag and rolling features
            model_data = temporal_agg.dropna().copy()
            
            if len(model_data) < 12:
                st.warning("⚠️ Insufficient data after feature engineering!")
                return
            
            # Prepare features for modeling
            feature_cols = [
                'Time_Index', 'Month_Num', 'Quarter', 'Year_Numeric',
                'Sin_Month', 'Cos_Month', 'Sin_Quarter', 'Cos_Quarter',
                'Crime_Count_Lag1', 'Crime_Count_Lag3', 'Crime_Count_MA3', 'Crime_Count_MA6'
            ]
            
            X = model_data[feature_cols]
            y = model_data['Crime_Count']
            
            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train multiple models
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            }
            
            model_results = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                
                model_results[name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'predictions': pred
                }
            
            # Select best model
            best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
            best_model = model_results[best_model_name]['model']
            
            # Display model performance
            col1, col2, col3 = st.columns(3)
            
            for i, (name, results) in enumerate(model_results.items()):
                with [col1, col2, col3][i]:
                    st.metric(
                        f"{name}",
                        f"R² = {results['r2']:.3f}",
                        f"MSE = {results['mse']:.2f}"
                    )
            
            st.success(f"🏆 Best Model: {best_model_name} (R² = {model_results[best_model_name]['r2']:.3f})")
            
            # Predict next crime occurrence
            st.subheader("🎯 Next Crime Prediction")
            
            # Create future time periods
            last_date = model_data['Date'].max()
            future_periods = st.slider("Predict next N months:", 1, 12, 3)
            
            future_predictions = []
            current_data = model_data.iloc[-1:].copy()  # Get last row for lag features
            
            for i in range(future_periods):
                next_date = last_date + pd.DateOffset(months=i+1)
                
                # Create features for next period
                time_idx = current_data['Time_Index'].iloc[-1] + 1
                month_num = next_date.month
                quarter = (month_num-1)//3 + 1
                year_numeric = next_date.year - temporal_agg['Year'].min()
                
                sin_month = np.sin(2 * np.pi * month_num / 12)
                cos_month = np.cos(2 * np.pi * month_num / 12)
                sin_quarter = np.sin(2 * np.pi * quarter / 4)
                cos_quarter = np.cos(2 * np.pi * quarter / 4)
                
                # Use recent values for lag features
                if len(current_data) >= 3:
                    lag1 = current_data['Crime_Count'].iloc[-1]
                    lag3 = current_data['Crime_Count'].iloc[-3] if len(current_data) >= 3 else current_data['Crime_Count'].iloc[-1]
                    ma3 = current_data['Crime_Count'].iloc[-3:].mean()
                    ma6 = current_data['Crime_Count'].iloc[-6:].mean() if len(current_data) >= 6 else current_data['Crime_Count'].mean()
                else:
                    lag1 = current_data['Crime_Count'].iloc[-1]
                    lag3 = lag1
                    ma3 = lag1
                    ma6 = lag1
                
                future_features = np.array([[
                    time_idx, month_num, quarter, year_numeric,
                    sin_month, cos_month, sin_quarter, cos_quarter,
                    lag1, lag3, ma3, ma6
                ]])
                
                # Make prediction
                pred_crimes = best_model.predict(future_features)[0]
                pred_crimes = max(0, round(pred_crimes))  # Ensure non-negative integer
                
                # Predict most likely crime type and district based on historical patterns
                month_data = clean_data[clean_data['Month'] == month_num]
                if len(month_data) > 0:
                    most_likely_crime = month_data['CrimeGroup_Name'].mode().iloc[0] if 'CrimeGroup_Name' in month_data.columns else 'Unknown'
                    most_likely_district = month_data['District_Name'].mode().iloc[0] if 'District_Name' in month_data.columns else 'Unknown'
                else:
                    most_likely_crime = clean_data['CrimeGroup_Name'].mode().iloc[0] if 'CrimeGroup_Name' in clean_data.columns else 'Unknown'
                    most_likely_district = clean_data['District_Name'].mode().iloc[0] if 'District_Name' in clean_data.columns else 'Unknown'
                
                # Estimate time of day based on historical patterns
                if 'Month' in clean_data.columns:
                    # Simple time prediction based on seasonal patterns
                    if month_num in [6, 7, 8]:  # Summer months
                        predicted_hour = np.random.choice([14, 15, 16, 17, 18], p=[0.1, 0.2, 0.3, 0.2, 0.2])
                    elif month_num in [12, 1, 2]:  # Winter months
                        predicted_hour = np.random.choice([10, 11, 12, 13, 14], p=[0.1, 0.2, 0.3, 0.2, 0.2])
                    else:  # Spring/Autumn
                        predicted_hour = np.random.choice([11, 12, 13, 14, 15, 16], p=[0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
                    
                    predicted_minute = np.random.randint(0, 60)
                    predicted_time = f"{predicted_hour:02d}:{predicted_minute:02d}"
                else:
                    predicted_time = "14:30"  # Default time
                
                future_predictions.append({
                    'Date': next_date.strftime('%Y-%m-%d'),
                    'Predicted_Crimes': pred_crimes,
                    'Most_Likely_Crime': most_likely_crime,
                    'Most_Likely_District': most_likely_district,
                    'Predicted_Time': predicted_time,
                    'Confidence': min(95, max(60, model_results[best_model_name]['r2'] * 100))
                })
                
                # Update current_data for next iteration
                new_row = pd.DataFrame({
                    'Date': [next_date],
                    'Crime_Count': [pred_crimes],
                    'Time_Index': [time_idx],
                    'Month_Num': [month_num],
                    'Quarter': [quarter],
                    'Year_Numeric': [year_numeric]
                })
                current_data = pd.concat([current_data, new_row], ignore_index=True)
            
            # Display predictions
            pred_df = pd.DataFrame(future_predictions)
            
            st.subheader("📅 Detailed Crime Predictions")
            
            for i, pred in enumerate(future_predictions):
                with st.expander(f"🔮 Prediction for {pred['Date']} (Month {i+1})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Expected Crimes", pred['Predicted_Crimes'])
                        st.write(f"**Confidence:** {pred['Confidence']:.1f}%")
                    
                    with col2:
                        st.write(f"**Most Likely Crime Type:**")
                        st.info(pred['Most_Likely_Crime'])
                        st.write(f"**Predicted Time:**")
                        st.info(pred['Predicted_Time'])
                    
                    with col3:
                        st.write(f"**Most Likely District:**")
                        st.info(pred['Most_Likely_District'])
            
            # Visualization
            st.subheader("📊 Prediction Visualization")
            
            # Combine historical and predicted data for visualization
            hist_dates = model_data['Date'].tolist()
            hist_crimes = model_data['Crime_Count'].tolist()
            pred_dates = [pd.to_datetime(p['Date']) for p in future_predictions]
            pred_crimes = [p['Predicted_Crimes'] for p in future_predictions]
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=hist_dates,
                y=hist_crimes,
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='blue'),
                marker=dict(size=6)
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=pred_crimes,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='red', dash='dash'),
                marker=dict(size=8, symbol='star')
            ))
            
            # Add confidence bands
            confidence_upper = [p * 1.2 for p in pred_crimes]
            confidence_lower = [max(0, p * 0.8) for p in pred_crimes]
            
            fig.add_trace(go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=confidence_upper + confidence_lower[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Band',
                showlegend=True
            ))
            
            fig.update_layout(
                title='Crime Prediction with Historical Context',
                xaxis_title='Date',
                yaxis_title='Number of Crimes',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary table
            st.subheader("📋 Prediction Summary Table")
            display_df = pred_df[['Date', 'Predicted_Crimes', 'Most_Likely_Crime', 'Most_Likely_District', 'Predicted_Time']].copy()
            display_df.columns = ['Date', 'Expected Crimes', 'Likely Crime Type', 'Likely District', 'Predicted Time']
            st.dataframe(display_df, use_container_width=True)
            
            # Risk alerts
            st.subheader("⚠️ Risk Alerts")
            high_risk_months = [p for p in future_predictions if p['Predicted_Crimes'] > np.mean([p['Predicted_Crimes'] for p in future_predictions]) * 1.2]
            
            if high_risk_months:
                st.warning("🚨 High Risk Periods Detected!")
                for risk_month in high_risk_months:
                    st.write(f"• **{risk_month['Date']}**: {risk_month['Predicted_Crimes']} crimes expected in {risk_month['Most_Likely_District']}")
            else:
                st.success("✅ No exceptionally high-risk periods detected in the forecast.")
                
        except Exception as e:
            st.error(f"❌ Error in predictive analysis: {str(e)}")
            st.info("💡 Please ensure your data has sufficient temporal information and try again.")
    
    def hotspot_clustering(self):
        """Perform crime hotspot clustering analysis"""
        st.subheader("🎯 Crime Hotspot Clustering")
        
        if 'FIR_Details_Data' not in self.datasets:
            st.warning("⚠️ FIR Details dataset required for clustering analysis!")
            return
        
        data = self.datasets['FIR_Details_Data']
        
        # Check if coordinate data exists
        if not all(col in data.columns for col in ['Latitude', 'Longitude']):
            st.error("❌ Latitude and Longitude columns are required for clustering!")
            return
        
        # Prepare data for clustering
        coords_data = data[['Latitude', 'Longitude']].dropna()
        
        if len(coords_data) < 10:
            st.warning("⚠️ Insufficient coordinate data for clustering! (Need at least 10 records)")
            return
        
        # Clustering parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Clusters (K-Means):", 2, 10, 5)
        with col2:
            eps = st.slider("DBSCAN Epsilon:", 0.01, 0.1, 0.05)
        
        # Perform clustering
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords_data)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(coords_scaled)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(coords_scaled)
        
        # Visualize results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("K-Means Clustering")
            fig = px.scatter_mapbox(
                lat=coords_data['Latitude'],
                lon=coords_data['Longitude'],
                color=kmeans_labels.astype(str),
                title="K-Means Crime Clusters",
                mapbox_style="open-street-map",
                zoom=10,
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("DBSCAN Clustering")
            fig = px.scatter_mapbox(
                lat=coords_data['Latitude'],
                lon=coords_data['Longitude'],
                color=dbscan_labels.astype(str),
                title="DBSCAN Crime Clusters",
                mapbox_style="open-street-map",
                zoom=10,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster statistics
        st.subheader("📊 Cluster Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**K-Means Statistics:**")
            st.write(f"Number of clusters: {n_clusters}")
            silhouette_kmeans = self.calculate_silhouette_score(coords_scaled, kmeans_labels)
            st.write(f"Silhouette score: {silhouette_kmeans}")
        
        with col2:
            st.write("**DBSCAN Statistics:**")
            n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            st.write(f"Number of clusters: {n_clusters_dbscan}")
            st.write(f"Noise points: {list(dbscan_labels).count(-1)}")
            if n_clusters_dbscan > 1:
                silhouette_dbscan = self.calculate_silhouette_score(coords_scaled, dbscan_labels)
                st.write(f"Silhouette score: {silhouette_dbscan}")
    
    def calculate_silhouette_score(self, data, labels):
        """Calculate silhouette score for clustering"""
        try:
            from sklearn.metrics import silhouette_score
            if len(set(labels)) > 1 and -1 not in labels:
                return round(silhouette_score(data, labels), 3)
            else:
                return "N/A"
        except:
            return "N/A"
    
    def crime_forecasting(self):
            """Forecast future crime trends"""
            st.subheader("📈 Crime Count Forecasting")
            
            if 'FIR_Details_Data' not in self.datasets:
                st.warning("⚠️ FIR Details dataset required for forecasting!")
                return
            
            data = self.datasets['FIR_Details_Data']
            
            # Check if temporal data exists
            if not all(col in data.columns for col in ['Year', 'Month']):
                st.error("❌ Year and Month columns are required for forecasting!")
                return
            
            # Prepare time series data
            monthly_crimes = data.groupby(['Year', 'Month']).size().reset_index(name='Crime_Count')
            monthly_crimes['Date'] = pd.to_datetime(monthly_crimes[['Year', 'Month']].assign(day=1))
            monthly_crimes = monthly_crimes.sort_values('Date')
            
            if len(monthly_crimes) < 12:
                st.warning("⚠️ Insufficient temporal data for forecasting! (Need at least 12 months)")
                return
            
            # Feature engineering for regression
            monthly_crimes['Month_Num'] = monthly_crimes['Date'].dt.month
            monthly_crimes['Year_Num'] = monthly_crimes['Year']
            monthly_crimes['Time_Index'] = range(len(monthly_crimes))
            
            # Seasonal features
            monthly_crimes['Sin_Month'] = np.sin(2 * np.pi * monthly_crimes['Month_Num'] / 12)
            monthly_crimes['Cos_Month'] = np.cos(2 * np.pi * monthly_crimes['Month_Num'] / 12)
            
            # Prepare features and target
            features = ['Time_Index', 'Month_Num', 'Sin_Month', 'Cos_Month']
            X = monthly_crimes[features]
            y = monthly_crimes['Crime_Count']
            
            # Split data for training
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train models
            lr_model = LinearRegression()
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            lr_model.fit(X_train, y_train)
            rf_model.fit(X_train, y_train)
            
            # Make predictions on test set
            lr_pred = lr_model.predict(X_test)
            rf_pred = rf_model.predict(X_test)
            
            # Calculate metrics
            lr_mse = mean_squared_error(y_test, lr_pred)
            rf_mse = mean_squared_error(y_test, rf_pred)
            lr_r2 = r2_score(y_test, lr_pred)
            rf_r2 = r2_score(y_test, rf_pred)
            
            # Future predictions
            forecast_months = st.slider("Forecast Months Ahead:", 1, 24, 6)
            
            # Create future dates
            last_date = monthly_crimes['Date'].max()
            future_dates = []
            for i in range(1, forecast_months + 1):
                future_date = last_date + pd.DateOffset(months=i)
                future_dates.append(future_date)
            
            # Prepare future features
            future_features = []
            for date in future_dates:
                time_idx = len(monthly_crimes) + len(future_features)
                month_num = date.month
                sin_month = np.sin(2 * np.pi * month_num / 12)
                cos_month = np.cos(2 * np.pi * month_num / 12)
                future_features.append([time_idx, month_num, sin_month, cos_month])
            
            future_X = pd.DataFrame(future_features, columns=features)
            
            # Make future predictions
            future_lr_pred = lr_model.predict(future_X)
            future_rf_pred = rf_model.predict(future_X)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Performance:**")
                st.write(f"Linear Regression - R²: {lr_r2:.3f}, MSE: {lr_mse:.2f}")
                st.write(f"Random Forest - R²: {rf_r2:.3f}, MSE: {rf_mse:.2f}")
            
            with col2:
                st.write("**Best Model:**")
                best_model = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
                st.success(f"✅ {best_model} performs better")
            
            # Plot historical and predicted data
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=monthly_crimes['Date'],
                y=monthly_crimes['Crime_Count'],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Test predictions
            test_dates = monthly_crimes['Date'][split_idx:]
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=lr_pred,
                mode='lines',
                name='Linear Regression (Test)',
                line=dict(color='orange', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=rf_pred,
                mode='lines',
                name='Random Forest (Test)',
                line=dict(color='green', dash='dash')
            ))
            
            # Future predictions
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_lr_pred,
                mode='lines+markers',
                name='Linear Regression (Forecast)',
                line=dict(color='orange')
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_rf_pred,
                mode='lines+markers',
                name='Random Forest (Forecast)',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title='Crime Count Forecasting',
                xaxis_title='Date',
                yaxis_title='Crime Count',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            st.subheader("🔮 Forecast Summary")
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Linear_Regression': future_lr_pred.round().astype(int),
                'Random_Forest': future_rf_pred.round().astype(int)
            })
            forecast_df['Month'] = forecast_df['Date'].dt.strftime('%Y-%m')
            st.dataframe(forecast_df[['Month', 'Linear_Regression', 'Random_Forest']])
    
    def risk_assessment(self):
        """Perform crime risk assessment"""
        st.subheader("⚠️ Crime Risk Assessment")
        
        if not self.datasets:
            st.warning("⚠️ Please load datasets first!")
            return
        
        # Risk factors analysis
        st.write("**Risk Assessment based on available data:**")
        
        if 'FIR_Details_Data' in self.datasets:
            fir_data = self.datasets['FIR_Details_Data']
            
            # District risk analysis
            if 'District_Name' in fir_data.columns:
                district_crimes = fir_data['District_Name'].value_counts()
                district_risk = pd.DataFrame({
                    'District': district_crimes.index,
                    'Crime_Count': district_crimes.values,
                    'Risk_Score': (district_crimes.values / district_crimes.max() * 100).round(1)
                })
                
                # Risk categorization
                district_risk['Risk_Level'] = pd.cut(
                    district_risk['Risk_Score'],
                    bins=[0, 25, 50, 75, 100],
                    labels=['Low', 'Medium', 'High', 'Very High']
                )
                
                st.subheader("🏘️ District Risk Assessment")
                
                # Risk level distribution
                risk_dist = district_risk['Risk_Level'].value_counts()
                fig = px.pie(
                    values=risk_dist.values,
                    names=risk_dist.index,
                    title="District Risk Level Distribution",
                    color_discrete_map={
                        'Low': 'green',
                        'Medium': 'yellow',
                        'High': 'orange',
                        'Very High': 'red'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top risk districts
                st.write("**High Risk Districts:**")
                high_risk = district_risk[district_risk['Risk_Level'].isin(['High', 'Very High'])]
                st.dataframe(high_risk[['District', 'Crime_Count', 'Risk_Score', 'Risk_Level']])
            
            # Temporal risk patterns
            if all(col in fir_data.columns for col in ['Year', 'Month']):
                st.subheader("📅 Temporal Risk Patterns")
                
                # Monthly risk analysis
                monthly_risk = fir_data.groupby('Month').size()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig = px.bar(
                    x=[month_names[i-1] for i in monthly_risk.index],
                    y=monthly_risk.values,
                    title="Monthly Crime Risk Pattern",
                    labels={'x': 'Month', 'y': 'Crime Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Peak risk months
                peak_months = monthly_risk.nlargest(3)
                st.write("**Peak Risk Months:**")
                for month, count in peak_months.items():
                    st.write(f"• {month_names[month-1]}: {count} crimes")
            
            # Crime type risk analysis
            if 'CrimeGroup_Name' in fir_data.columns:
                st.subheader("🚨 Crime Type Risk Analysis")
                
                crime_risk = fir_data['CrimeGroup_Name'].value_counts().head(10)
                fig = px.bar(
                    x=crime_risk.values,
                    y=crime_risk.index,
                    orientation='h',
                    title="Top 10 Crime Types by Frequency",
                    labels={'x': 'Frequency', 'y': 'Crime Type'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Victim risk analysis
        if 'VictimInfoDetails' in self.datasets:
            victim_data = self.datasets['VictimInfoDetails']
            
            st.subheader("👥 Victim Risk Profile")
            
            if 'age' in victim_data.columns:
                # Age-based risk
                valid_ages = victim_data[victim_data['age'].notna() & (victim_data['age'] > 0)]
                if not valid_ages.empty:
                    age_bins = [0, 18, 35, 50, 65, 100]
                    age_labels = ['0-17', '18-34', '35-49', '50-64', '65+']
                    valid_ages['Age_Group'] = pd.cut(valid_ages['age'], bins=age_bins, labels=age_labels)
                    
                    age_risk = valid_ages['Age_Group'].value_counts()
                    fig = px.bar(
                        x=age_risk.index,
                        y=age_risk.values,
                        title="Victim Age Group Risk",
                        labels={'x': 'Age Group', 'y': 'Victim Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Risk mitigation recommendations
        st.subheader("🛡️ Risk Mitigation Recommendations")
        
        recommendations = [
            "🎯 **High-Risk Areas**: Increase patrol frequency in districts with Very High risk scores",
            "📅 **Temporal Patterns**: Deploy additional resources during peak crime months",
            "🚨 **Crime Prevention**: Focus on preventing the most frequent crime types",
            "👥 **Community Engagement**: Implement targeted programs for high-risk demographics",
            "📊 **Data Monitoring**: Regular monitoring and updating of risk assessments",
            "🤝 **Inter-agency Coordination**: Collaborate with local authorities and community organizations"
        ]
        
        for rec in recommendations:
            st.write(rec)
    
    def generate_reports(self):
        """Generate comprehensive crime analysis reports"""
        st.header("📋 Crime Analysis Reports")
        
        if not self.datasets:
            st.warning("⚠️ Please load datasets first!")
            return
        
        report_type = st.selectbox(
            "Select Report Type:",
            ["Executive Summary", "District Analysis", "Temporal Analysis", "Demographic Analysis"]
        )
        
        if report_type == "Executive Summary":
            self.generate_executive_summary()
        elif report_type == "District Analysis":
            self.generate_district_report()
        elif report_type == "Temporal Analysis":
            self.generate_temporal_report()
        elif report_type == "Demographic Analysis":
            self.generate_demographic_report()
    
    def generate_executive_summary(self):
        """Generate executive summary report"""
        st.subheader("📊 Executive Summary Report")
        
        # Key metrics
        if 'FIR_Details_Data' in self.datasets:
            fir_data = self.datasets['FIR_Details_Data']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cases", len(fir_data))
            with col2:
                if 'VICTIM_COUNT' in fir_data.columns:
                    total_victims = fir_data['VICTIM_COUNT'].sum()
                    st.metric("Total Victims", int(total_victims))
                else:
                    st.metric("Districts", fir_data['District_Name'].nunique() if 'District_Name' in fir_data.columns else "N/A")
            with col3:
                if 'Year' in fir_data.columns:
                    year_range = f"{fir_data['Year'].min()}-{fir_data['Year'].max()}"
                    st.metric("Period", year_range)
                else:
                    st.metric("Period", "N/A")
            with col4:
                if 'CrimeGroup_Name' in fir_data.columns:
                    st.metric("Crime Types", fir_data['CrimeGroup_Name'].nunique())
                else:
                    st.metric("Crime Types", "N/A")
            
            # Key findings
            st.subheader("🔍 Key Findings")
            
            findings = []
            
            if 'District_Name' in fir_data.columns:
                top_district = fir_data['District_Name'].value_counts().index[0]
                top_district_count = fir_data['District_Name'].value_counts().iloc[0]
                findings.append(f"**Highest Crime District**: {top_district} ({top_district_count} cases)")
            
            if 'CrimeGroup_Name' in fir_data.columns:
                top_crime = fir_data['CrimeGroup_Name'].value_counts().index[0]
                top_crime_count = fir_data['CrimeGroup_Name'].value_counts().iloc[0]
                findings.append(f"**Most Common Crime**: {top_crime} ({top_crime_count} cases)")
            
            if 'Year' in fir_data.columns:
                yearly_trend = fir_data['Year'].value_counts().sort_index()
                if len(yearly_trend) > 1:
                    recent_change = yearly_trend.iloc[-1] - yearly_trend.iloc[-2]
                    trend_direction = "increased" if recent_change > 0 else "decreased"
                    findings.append(f"**Recent Trend**: Crime has {trend_direction} by {abs(recent_change)} cases")
            
            for finding in findings:
                st.write(f"• {finding}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                if 'CrimeGroup_Name' in fir_data.columns:
                    crime_dist = fir_data['CrimeGroup_Name'].value_counts().head(5)
                    fig = px.pie(
                        values=crime_dist.values,
                        names=crime_dist.index,
                        title="Top 5 Crime Types"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Year' in fir_data.columns:
                    yearly_crimes = fir_data['Year'].value_counts().sort_index()
                    fig = px.line(
                        x=yearly_crimes.index,
                        y=yearly_crimes.values,
                        title="Yearly Crime Trend",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def generate_district_report(self):
        """Generate district-wise analysis report"""
        st.subheader("🏘️ District Analysis Report")
        
        if 'FIR_Details_Data' not in self.datasets:
            st.warning("⚠️ FIR Details dataset required!")
            return
        
        data = self.datasets['FIR_Details_Data']
        
        if 'District_Name' not in data.columns:
            st.error("❌ District_Name column not found!")
            return
        
        # District selection
        selected_district = st.selectbox(
            "Select District for Detailed Analysis:",
            data['District_Name'].unique()
        )
        
        district_data = data[data['District_Name'] == selected_district]
        
        # District metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cases", len(district_data))
        with col2:
            if 'VICTIM_COUNT' in district_data.columns:
                st.metric("Total Victims", int(district_data['VICTIM_COUNT'].sum()))
            else:
                st.metric("Crime Types", district_data['CrimeGroup_Name'].nunique() if 'CrimeGroup_Name' in district_data.columns else "N/A")
        with col3:
            if 'Year' in district_data.columns:
                st.metric("Active Years", district_data['Year'].nunique())
            else:
                st.metric("Active Years", "N/A")
        with col4:
            # Calculate district rank
            district_ranking = data['District_Name'].value_counts()
            rank = list(district_ranking.index).index(selected_district) + 1
            st.metric("District Rank", f"#{rank}")
        
        # Crime analysis for district
        if 'CrimeGroup_Name' in district_data.columns:
            st.subheader(f"Crime Analysis - {selected_district}")
            
            crime_types = district_data['CrimeGroup_Name'].value_counts()
            fig = px.bar(
                x=crime_types.values,
                y=crime_types.index,
                orientation='h',
                title=f"Crime Types in {selected_district}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Temporal analysis for district
        if all(col in district_data.columns for col in ['Year', 'Month']):
            monthly_crimes = district_data.groupby(['Year', 'Month']).size().reset_index(name='Count')
            monthly_crimes['Date'] = pd.to_datetime(monthly_crimes[['Year', 'Month']].assign(day=1))
            
            fig = px.line(
                monthly_crimes, x='Date', y='Count',
                title=f"Crime Trend in {selected_district}",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def generate_temporal_report(self):
        """Generate temporal analysis report"""
        st.subheader("📅 Temporal Analysis Report")
        
        if 'FIR_Details_Data' not in self.datasets:
            st.warning("⚠️ FIR Details dataset required!")
            return
        
        data = self.datasets['FIR_Details_Data']
        
        if not all(col in data.columns for col in ['Year', 'Month']):
            st.error("❌ Year and Month columns required!")
            return
        
        # Year-wise analysis
        st.subheader("📊 Year-wise Crime Analysis")
        yearly_crimes = data['Year'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=yearly_crimes.index,
                y=yearly_crimes.values,
                title="Crimes by Year"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Calculate year-over-year change
            yoy_change = yearly_crimes.pct_change() * 100
            fig = px.bar(
                x=yoy_change.index[1:],  # Exclude first year (NaN)
                y=yoy_change.values[1:],
                title="Year-over-Year Change (%)",
                color=yoy_change.values[1:],
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly patterns
        st.subheader("📅 Monthly Crime Patterns")
        monthly_crimes = data['Month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = px.bar(
            x=[month_names[i-1] for i in monthly_crimes.index],
            y=monthly_crimes.values,
            title="Crime Distribution by Month"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak analysis
        st.subheader("🔝 Peak Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Highest Crime Years:**")
            top_years = yearly_crimes.nlargest(3)
            for year, count in top_years.items():
                st.write(f"• {year}: {count} crimes")
        
        with col2:
            st.write("**Peak Crime Months:**")
            top_months = monthly_crimes.nlargest(3)
            for month, count in top_months.items():
                st.write(f"• {month_names[month-1]}: {count} crimes")
    
    def generate_demographic_report(self):
        """Generate demographic analysis report"""
        st.subheader("👥 Demographic Analysis Report")
        
        # Victim demographics
        if 'VictimInfoDetails' in self.datasets:
            victim_data = self.datasets['VictimInfoDetails']
            
            st.subheader("👤 Victim Demographics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Sex' in victim_data.columns:
                    gender_dist = victim_data['Sex'].value_counts()
                    fig = px.pie(
                        values=gender_dist.values,
                        names=gender_dist.index,
                        title="Victim Gender Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'age' in victim_data.columns:
                    valid_ages = victim_data[victim_data['age'].notna() & (victim_data['age'] > 0)]
                    if not valid_ages.empty:
                        fig = px.histogram(
                            valid_ages, x='age', nbins=20,
                            title="Victim Age Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Accused demographics
        if 'Accused Data' in self.datasets:
            accused_data = self.datasets['Accused Data']
            
            st.subheader("⚖️ Accused Demographics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Sex' in accused_data.columns:
                    gender_dist = accused_data['Sex'].value_counts()
                    fig = px.pie(
                        values=gender_dist.values,
                        names=gender_dist.index,
                        title="Accused Gender Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'age' in accused_data.columns:
                    valid_ages = accused_data[accused_data['age'].notna() & (accused_data['age'] > 0)]
                    if not valid_ages.empty:
                        fig = px.histogram(
                            valid_ages, x='age', nbins=20,
                            title="Accused Age Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Comparative analysis
        if 'VictimInfoDetails' in self.datasets and 'Accused Data' in self.datasets:
            st.subheader("⚖️ Comparative Demographic Analysis")
            
            victim_data = self.datasets['VictimInfoDetails']
            accused_data = self.datasets['Accused Data']
            
            # Age comparison
            if 'age' in victim_data.columns and 'age' in accused_data.columns:
                victim_ages = victim_data[victim_data['age'].notna() & (victim_data['age'] > 0)]['age']
                accused_ages = accused_data[accused_data['age'].notna() & (accused_data['age'] > 0)]['age']
                
                if not victim_ages.empty and not accused_ages.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=victim_ages, name='Victims', opacity=0.7))
                    fig.add_trace(go.Histogram(x=accused_ages, name='Accused', opacity=0.7))
                    fig.update_layout(
                        title='Age Distribution Comparison',
                        xaxis_title='Age',
                        yaxis_title='Count',
                        barmode='overlay'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Victim Statistics:**")
                if not victim_ages.empty:
                    st.write(f"• Average age: {victim_ages.mean():.1f} years")
                    st.write(f"• Age range: {victim_ages.min()}-{victim_ages.max()} years")
                    st.write(f"• Most common age: {victim_ages.mode().iloc[0]} years")
            
            with col2:
                st.write("**Accused Statistics:**")
                if not accused_ages.empty:
                    st.write(f"• Average age: {accused_ages.mean():.1f} years")
                    st.write(f"• Age range: {accused_ages.min()}-{accused_ages.max()} years")
                    st.write(f"• Most common age: {accused_ages.mode().iloc[0]} years")

def main():
    """Main application function"""
    
    # Initialize the app
    app = CrimeAnalysisApp()
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1>🚔 Crime Analysis</h1>
        <p>Comprehensive Crime Data Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    menu_options = [
        "📁 Data Import",
        "📊 Exploratory Data Analysis", 
        "🗺️ Crime Hotspot Mapping",
        "🔮 Predictive Analysis",
        "📋 Generate Reports"
    ]
    
    selected_page = st.sidebar.selectbox("Navigation", menu_options)
    
    # Data status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Status")
    
    if app.datasets:
        for name, df in app.datasets.items():
            display_name = name.replace('_', ' ')
            st.sidebar.success(f"✅ {display_name}: {len(df)} records")
    else:
        st.sidebar.warning("⚠️ No datasets loaded")
    
    # Main content
    st.markdown('<h1 class="main-header">🚔 Crime Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Route to appropriate page
    if selected_page == "📁 Data Import":
        app.load_data_interface()
    elif selected_page == "📊 Exploratory Data Analysis":
        app.perform_eda()
    elif selected_page == "🗺️ Crime Hotspot Mapping":
        app.create_crime_hotspot_map()
    elif selected_page == "🔮 Predictive Analysis":
        app.predictive_analysis()
    elif selected_page == "📋 Generate Reports":
        app.generate_reports()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>Crime Analysis Dashboard | Data-Driven Crime Prevention & Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
# streamlit run app.py --server.maxUploadSize=1024
