import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, FastMarkerCluster
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DWLR Pune Groundwater Monitor",
    page_icon="💧",
    layout="wide",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1e3c72;
    }
    .status-online {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .status-offline {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .status-warning {
        background-color: #ffc107;
        color: black;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffa726, #ff9800);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .alert-success {
        background: linear-gradient(135deg, #66bb6a, #4caf50);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        # Load datasets
        sites = pd.read_csv('gw_sites.csv')
        daily_data = pd.read_csv('gw_rain_merged_daily.csv')
        trends = pd.read_csv('mk_trend_by_well.csv')
        recharge = pd.read_csv('recharge_monthly_wtf.csv')
        lag = pd.read_csv('lag_by_well.csv')

        # Process daily data
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data = daily_data.merge(sites, on='gw_code', how='left')

        return sites, daily_data, trends, recharge, lag

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

def simulate_real_time_readings(daily_data):
    """Generate realistic readings"""
    latest = daily_data.groupby('gw_code').agg({
        'gw_name': 'first',
        'gw_lat': 'first',
        'gw_lon': 'first',
        'gw_tehsil': 'first',
        'gw_block': 'first',
        'bgl_m': 'last',
        'rain_mm': 'last',
        'date': 'max'
    }).reset_index()

    np.random.seed(int(datetime.now().timestamp()) // 30)
    daily_variation = np.random.normal(0, 0.05, len(latest))
    seasonal_factor = np.sin(datetime.now().timetuple().tm_yday / 365.25 * 2 * np.pi) * 0.2
    latest['bgl_m_live'] = (latest['bgl_m'] + daily_variation + seasonal_factor).clip(lower=0.1)
    latest['status'] = np.random.choice(['Online', 'Delayed', 'Offline'], len(latest), p=[0.93, 0.05, 0.02])

    conditions = []
    for depth in latest['bgl_m_live']:
        if depth < 3: conditions.append('Excellent')
        elif depth < 6: conditions.append('Good')
        elif depth < 10: conditions.append('Fair')
        elif depth < 15: conditions.append('Poor')
        else: conditions.append('Critical')
    latest['condition'] = conditions
    latest['last_update'] = datetime.now()

    return latest

def create_header():
    """Create monitoring header"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div class="main-header">
        <h1>🌊 DWLR Groundwater Monitoring System</h1>
        <h3>Pune Region - Network Status</h3>
        <p>Last Updated: {current_time}</p>
    </div>
    """, unsafe_allow_html=True)

def display_system_overview(latest_data):
    """Enhanced system overview with KPIs"""
    st.subheader("📊 Network Overview")
    total_wells = len(latest_data)
    online_wells = len(latest_data[latest_data['status'] == 'Online'])
    avg_depth = latest_data['bgl_m_live'].mean()
    critical_wells = len(latest_data[latest_data['condition'] == 'Critical'])
    excellent_wells = len(latest_data[latest_data['condition'] == 'Excellent'])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🏛️ Total DWLR Stations", f"{total_wells}", "Pune Region")
    with col2:
        connectivity_pct = (online_wells / total_wells) * 100
        st.metric("🌐 Network Connectivity", f"{connectivity_pct:.1f}%", f"{online_wells}/{total_wells} online")
    with col3:
        st.metric("📏 Average Water Depth", f"{avg_depth:.2f}m", "Below Ground Level")
    with col4:
        st.metric("⚠️ Critical Wells", f"{critical_wells}", "Require Attention", delta_color="inverse" if critical_wells > 0 else "normal")
    with col5:
        st.metric("✅ Excellent Condition", f"{excellent_wells}", "Optimal Status")

def create_status_alerts(latest_data):
    """Create intelligent alert system"""
    st.subheader("🚨 System Alerts & Notifications")
    critical_wells = latest_data[latest_data['condition'] == 'Critical']
    offline_wells = latest_data[latest_data['status'] == 'Offline']

    col1, col2 = st.columns(2)
    with col1:
        if not critical_wells.empty:
            st.markdown(f'<div class="alert-critical"><strong>🔴 CRITICAL ALERT</strong><br>{len(critical_wells)} wells showing critical water levels (>15m depth)</div>', unsafe_allow_html=True)
            with st.expander("View Critical Wells"):
                for _, well in critical_wells.iterrows():
                    st.write(f"📍 **{well['gw_name']}** ({well['gw_tehsil']}): {well['bgl_m_live']:.2f}m depth")
        if not offline_wells.empty:
            st.markdown(f'<div class="alert-warning"><strong>🟡 MAINTENANCE ALERT</strong><br>{len(offline_wells)} DWLR stations offline</div>', unsafe_allow_html=True)
    with col2:
        excellent_wells = latest_data[latest_data['condition'] == 'Excellent']
        if not excellent_wells.empty:
            st.markdown(f'<div class="alert-success"><strong>✅ SYSTEM HEALTHY</strong><br>{len(excellent_wells)} wells in excellent condition</div>', unsafe_allow_html=True)
        st.info("Recent Activity:\n- Network uptime: 98.5%\n- Data quality: 99.2%\n- Last system maintenance: 2 days ago")

def create_interactive_map(latest_data):
    """Enhanced interactive map with clustering and canvas rendering for performance"""
    st.subheader("🗺️ Station Map")
    center_lat, center_lon = latest_data['gw_lat'].mean(), latest_data['gw_lon'].mean()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='CartoDB positron',
        control_scale=True,
        prefer_canvas=True
    )

    condition_colors = {
        'Excellent': 'green',
        'Good': 'lightgreen',
        'Fair': 'orange',
        'Poor': 'red',
        'Critical': 'darkred'
    }

    num_points = len(latest_data)
    # Use faster clustering for large datasets
    if num_points > 600:
        coords = latest_data[['gw_lat', 'gw_lon']].values.tolist()
        FastMarkerCluster(data=coords, name='Stations').add_to(m)
    else:
        cluster = MarkerCluster(name='Stations').add_to(m)
        for _, well in latest_data.iterrows():
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4 style=\"color: #1e3c72;\">{well['gw_name']}</h4><hr>
                <p><strong>📍 Location:</strong> {well['gw_tehsil']}, {well['gw_block']}</p>
                <p><strong>💧 Current Depth:</strong> {well['bgl_m_live']:.2f}m BGL</p>
                <p><strong>🎯 Condition:</strong> <span style=\"color: {condition_colors.get(well['condition'], 'blue')}; font-weight: bold;\">{well['condition']}</span></p>
                <p><strong>🔗 Status:</strong> {well['status']}</p>
                <p><strong>⏰ Updated:</strong> {well['last_update'].strftime('%H:%M:%S')}</p>
            </div>"""
            folium.Marker(
                location=[well['gw_lat'], well['gw_lon']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{well['gw_name']} - {well['condition']}",
                icon=folium.Icon(color=condition_colors.get(well['condition'], 'blue'), icon='tint', prefix='fa')
            ).add_to(cluster)

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, width=700, height=500)

def create_time_series_analysis(daily_data):
    """Advanced time series visualization with custom date and location filters"""
    st.subheader("📈 Historical Trend Analysis")

    # --- Location Filters ---
    st.markdown("#### Select Location")
    col1, col2 = st.columns(2)

    # Tehsil/Village Filter
    available_tehsils = ['All'] + sorted(daily_data['gw_tehsil'].dropna().unique().tolist())
    selected_tehsil = col1.selectbox("Select Tehsil/Village:", options=available_tehsils)

    # Filter data based on selected tehsil
    if selected_tehsil != 'All':
        location_filtered_data = daily_data[daily_data['gw_tehsil'] == selected_tehsil]
    else:
        location_filtered_data = daily_data

    # Well/Station Filter based on Tehsil selection
    available_wells = sorted(location_filtered_data['gw_name'].dropna().unique())
    if not available_wells:
        st.warning("No wells found for the selected Tehsil.")
        return

    selected_wells = col2.multiselect(
        "Select Station(s):",
        options=available_wells,
        default=available_wells[:1] # Default to the first well in the list
    )

    # --- Time Period Filters ---
    st.markdown("#### Select Time Period")
    col_time1, col_time2, col_time3 = st.columns(3)

    date_range_options = ["Last 30 Days", "Last 90 Days", "Last Year", "All Data", "Custom Range"]
    date_range = col_time1.selectbox("Time Period:", date_range_options, index=3) # Default to All Data

    start_date = None
    end_date = None

    min_date = daily_data['date'].min().date()
    max_date = daily_data['date'].max().date()

    if date_range == "Custom Range":
        start_date = col_time2.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = col_time3.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        if start_date > end_date:
            st.error("Error: Start date must be before end date.")
            return

    if selected_wells:
        # Filter data by selected wells
        filtered_data = location_filtered_data[
            (location_filtered_data['gw_name'].isin(selected_wells)) &
            (location_filtered_data['qc_keep'] == True)
        ].copy()

        # Apply date filter
        final_filtered_data = pd.DataFrame() # Initialize empty dataframe
        if date_range != "Custom Range":
            current_end_date = filtered_data['date'].max()
            if date_range == "Last 30 Days":
                current_start_date = current_end_date - timedelta(days=30)
            elif date_range == "Last 90 Days":
                current_start_date = current_end_date - timedelta(days=90)
            elif date_range == "Last Year":
                current_start_date = current_end_date - timedelta(days=365)
            else: # All Data
                current_start_date = filtered_data['date'].min()
            
            final_filtered_data = filtered_data[(filtered_data['date'] >= current_start_date) & (filtered_data['date'] <= current_end_date)]
        else: # Custom Range
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            final_filtered_data = filtered_data[(filtered_data['date'] >= start_date_dt) & (filtered_data['date'] <= end_date_dt)]

        if final_filtered_data.empty:
            st.warning("No data available for the selected criteria.")
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Water Level Trends', 'Rainfall Pattern'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )

        # Water level trends
        for well in selected_wells:
            well_data = final_filtered_data[final_filtered_data['gw_name'] == well]
            fig.add_trace(
                go.Scatter(
                    x=well_data['date'],
                    y=well_data['bgl_m'],
                    mode='lines+markers',
                    name=well,
                    line=dict(width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )

        # Rainfall data
        rainfall_data = final_filtered_data.groupby('date')['rain_mm'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=rainfall_data['date'],
                y=rainfall_data['rain_mm'],
                name='Rainfall (mm)',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=600,
            title_text="Multi-Well Analysis Dashboard",
            showlegend=True,
            hovermode='x unified'
        )
        fig.update_yaxes(title_text="Water Depth (m BGL)", autorange="reversed", row=1, col=1)
        fig.update_yaxes(title_text="Rainfall (mm)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Statistical summary
        st.subheader("📊 Statistical Summary")
        stats_df = final_filtered_data.groupby('gw_name')['bgl_m'].agg([
            ('Mean Depth (m)', 'mean'),
            ('Min Depth (m)', 'min'),
            ('Max Depth (m)', 'max'),
            ('Std Dev (m)', 'std'),
            ('Data Points', 'count')
        ]).round(2)
        st.dataframe(stats_df, use_container_width=True)


def create_recharge_analysis(recharge):
    """Advanced recharge analysis"""
    st.subheader("💧 Groundwater Recharge Analysis")
    recharge['date'] = pd.to_datetime(recharge[['year', 'month']].assign(day=1))
    tab1, tab2, tab3 = st.tabs(["📊 Annual Trends", "📅 Monthly Patterns", "🎯 Scenario Comparison"])
    
    with tab1:
        annual_recharge = recharge.groupby('year')[['recharge_mm_sy02', 'recharge_mm_sy05', 'recharge_mm_sy10']].sum().reset_index()
        fig = go.Figure(data=[
            go.Bar(name='Conservative (Sy=0.02)', x=annual_recharge['year'], y=annual_recharge['recharge_mm_sy02']),
            go.Bar(name='Moderate (Sy=0.05)', x=annual_recharge['year'], y=annual_recharge['recharge_mm_sy05']),
            go.Bar(name='Liberal (Sy=0.10)', x=annual_recharge['year'], y=annual_recharge['recharge_mm_sy10'])
        ])
        fig.update_layout(title='Annual Groundwater Recharge', xaxis_title='Year', yaxis_title='Recharge (mm)', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        monthly_avg = recharge.groupby('month')[['recharge_mm_sy02', 'recharge_mm_sy05', 'recharge_mm_sy10']].mean().reset_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: month_names[x-1])
        fig = px.line(monthly_avg, x='month_name', y=['recharge_mm_sy02', 'recharge_mm_sy05', 'recharge_mm_sy10'], title='Average Monthly Recharge Patterns')
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        total_recharge = recharge.groupby('gw_code')[['recharge_mm_sy02', 'recharge_mm_sy05']].sum().reset_index()
        fig = px.scatter(total_recharge, x='recharge_mm_sy02', y='recharge_mm_sy05', title='Recharge Scenario Comparison', labels={'recharge_mm_sy02': 'Conservative', 'recharge_mm_sy05': 'Moderate'})
        st.plotly_chart(fig, use_container_width=True)

def create_data_export_section(latest_data, daily_data):
    """Enhanced data export functionality"""
    st.subheader("📥 Data Export & Reports")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button("📊 Current Status Report", latest_data.to_csv(index=False), f"dwlr_current_status_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    with col2:
        st.download_button("📈 Historical Data", daily_data.to_csv(index=False), f"dwlr_historical_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    with col3:
        if st.button("📋 Generate Summary Report"):
            # This part is simplified, in a real app you might generate a PDF
            st.success("Report generation logic would go here.")

def main():
    """Main application"""
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        st.image("logo.png", width=200)
    create_header()

    with st.spinner("🔄 Loading groundwater data..."):
        data = load_data()
        if data[0] is not None:
            sites, daily_data, trends, recharge, lag = data
            latest_data = simulate_real_time_readings(daily_data)

            display_system_overview(latest_data)
            create_status_alerts(latest_data)

            col1, col2 = st.columns([1, 1])
            with col1:
                create_interactive_map(latest_data)
            with col2:
                st.subheader("📋 Live Station Data")
                condition_filter = st.multiselect("Filter by condition:", options=latest_data['condition'].unique(), default=latest_data['condition'].unique())
                filtered_latest = latest_data[latest_data['condition'].isin(condition_filter)]
                st.dataframe(filtered_latest[['gw_name', 'gw_tehsil', 'bgl_m_live', 'condition', 'status']], hide_index=True, use_container_width=True)

            create_time_series_analysis(daily_data)
            create_recharge_analysis(recharge)
            create_data_export_section(latest_data, daily_data)

            # Footer with Support and Contact
            st.markdown("---")
            st.subheader("📞 Support & Contact")
            col1, col2 = st.columns(2)
            with col1:
                st.info("""
                **Technical Support:**
                Email: support@dwlr-monitor.gov.in
                Phone: +91-11-2958-XXXX
                """)
            with col2:
                st.info("""
                **Emergency Hotline:**
                24/7: +91-11-EMERGENCY
                """)
        else:
            st.error("❌ Failed to load groundwater monitoring data. Please ensure CSV files are present.")

if __name__ == "__main__":
    main()

