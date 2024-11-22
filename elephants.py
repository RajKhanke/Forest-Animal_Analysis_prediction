import pandas as pd
import folium
import plotly.express as px
from prophet import Prophet
import streamlit as st

def elephants_page():
    st.markdown(
        '<h1 style="text-align: center; color: lightgreen; font-weight: bold;">Elephant Monitoring in India</h1>',
        unsafe_allow_html=True)

    df = pd.read_csv('elephant_historic_data.csv')

    # Update years list
    years = ['1993', '1997', '2002', '2007', '2012', '2017']

    # Add a select box for year selection
    selected_year = st.selectbox('Select Year', years)

    # Create a map centered around India
    india_map = folium.Map(location=[17.5937, 82.9629], zoom_start=5, width=1300, height=800)

    # Add bubble markers for each state based on the selected year
    for i, row in df.iterrows():
        state = row['State']
        location = {
            'Andhra Pradesh': [15.9129, 79.7400],
            'Arunachal Pradesh': [27.1137, 93.6054],
            'Assam': [26.2006, 92.9376],
            'Bihar': [25.0961, 85.3131],
            'Chhattisgarh': [21.2787, 81.8661],
            'Goa': [15.2993, 74.1240],
            'Gujarat': [22.2587, 71.1924],
            'Jharkhand': [23.6102, 85.2799],
            'Karnataka': [15.3173, 75.7139],
            'Kerala': [10.8505, 76.2711],
            'Madhya Pradesh': [23.4734, 77.9479],
            'Maharashtra': [19.6633, 75.3202],
            'Meghalaya': [25.4670, 91.3662],
            'Mizoram': [23.1645, 92.9376],
            'Nagaland': [26.1584, 94.5624],
            'Odisha': [20.9517, 85.0985],
            'Rajasthan': [27.0238, 74.2176],
            'Tamil Nadu': [11.1271, 78.6569],
            'Telangana': [17.0220, 78.3555],
            'Tripura': [23.9408, 91.9882],
            'Uttar Pradesh': [27.2599, 79.4126],
            'Uttarakhand ': [30.0668, 79.0193],
            'West Bengal ': [22.9868, 87.8550],
            'Sikkim': [27.5330, 88.6139],
            'Jammu & Kashmir': [33.2778, 76.5765],
            'Punjab': [31.1471, 75.3412],
            'Haryana': [29.0588, 76.0856],
            'Himachal Pradesh': [31.1048, 77.1734],
            'Manipur': [24.6637, 93.9063],
            'Andaman & Nicobar Islands ': [11.7401, 92.6586]
        }.get(state, [20.5937, 78.9629])

        elephant_count = row[f'Elephants in {selected_year}']

        if elephant_count <= 0:
            continue

        if elephant_count > 200:
            color = 'pink'
        elif elephant_count > 50:
            color = 'orange'
        else:
            color = 'yellow'

        bubble_size = 20

        folium.CircleMarker(
            location=location,
            radius=bubble_size,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(india_map)

        folium.map.Marker(
            location,
            icon=folium.DivIcon(html=f'''
                <div style="text-align: left; font-size: 8.5pt; font-weight: bold; width:{bubble_size * 0.2}px; color: black;">
                    {elephant_count:.0f}
                </div>''')
        ).add_to(india_map)

    title_html = f'''
        <h3 align="center" style="font-size:20px"><b>Elephant Count in India ({selected_year})</b></h3>
    '''
    india_map.get_root().html.add_child(folium.Element(title_html))

    legend_html = '''
    <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 150px; height: 110px; 
         border:2px solid grey; z-index:9999; font-size:12px;
         background-color:white; padding: 10px;">
         <b>Elephant Frequency</b><br>
         <i class="fa fa-circle" style="color:pink"></i> > 200 Elephants<br>
         <i class="fa fa-circle" style="color:orange"></i> 50-200 Elephants<br>
         <i class="fa fa-circle" style="color:yellow"></i> < 50 Elephants<br>
    </div>
    '''
    india_map.get_root().html.add_child(folium.Element(legend_html))

    st.components.v1.html(india_map._repr_html_(), height=700)

    selected_state = st.selectbox('Select a State', df['State'].unique())

    state_data = df[df['State'] == selected_state]

    # Reshape the DataFrame
    state_data = state_data.melt(id_vars=['State'], var_name='Year', value_name='Elephant Count')
    state_data['Year'] = state_data['Year'].str.extract(r'(\d{4})').astype(int)

    line_fig = px.line(state_data, x='Year', y='Elephant Count', title=f'Elephant Population Over Time in {selected_state}')
    bar_fig = px.bar(state_data, x='Year', y='Elephant Count', title=f'Elephant Population Count in {selected_state}')

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(line_fig)

    with col2:
        st.plotly_chart(bar_fig)

    st.subheader('Predict Future Elephant Counts')
    future_year = st.number_input('Enter Future Year (2025 and onwards)', min_value=2025, step=1)

    # Add a button to start the prediction
    if st.button('Start Prediction'):
        if future_year >= 2025:
            future_data = []
            for state in df['State'].unique():
                state_df = df[df['State'] == state]
                state_df = state_df.melt(id_vars=['State'], var_name='Year', value_name='Elephant Count')
                state_df['Year'] = state_df['Year'].str.extract(r'(\d{4})').astype(int)
                state_df = state_df.rename(columns={'Year': 'ds', 'Elephant Count': 'y'})
                state_df['ds'] = pd.to_datetime(state_df['ds'], format='%Y')  # Ensure correct date format

                # Use Prophet for forecasting
                model = Prophet()
                state_df = state_df[['ds', 'y']]
                model.fit(state_df)

                future = pd.DataFrame({'ds': pd.date_range(start='2023-01-01', periods=future_year - 2022, freq='Y')})
                forecast = model.predict(future)

                # Ensure forecast is not empty
                if forecast['yhat'].size > 0:
                    future_elephant_count = abs(
                        forecast['yhat'].iloc[-1])  # Use absolute value to handle negative predictions
                else:
                    future_elephant_count = 0  # Default value if forecast is empty

                # Only append data if the elephant count is greater than zero
                if future_elephant_count > 0:
                    future_data.append(
                        {'State': state, 'Year': future_year, 'Predicted Elephant Count': future_elephant_count})

            future_df = pd.DataFrame(future_data)

            # Map for predicted elephant counts
            pred_map = folium.Map(location=[17.5937, 82.9629], zoom_start=5, width=1300, height=800)

            for i, row in future_df.iterrows():
                state = row['State']
                location = {
                    'Andhra Pradesh': [15.9129, 79.7400],
                    'Arunachal Pradesh': [27.1137, 93.6054],
                    'Assam': [26.2006, 92.9376],
                    'Bihar': [25.0961, 85.3131],
                    'Chhattisgarh': [21.2787, 81.8661],
                    'Goa': [15.2993, 74.1240],
                    'Gujarat': [22.2587, 71.1924],
                    'Jharkhand': [23.6102, 85.2799],
                    'Karnataka': [15.3173, 75.7139],
                    'Kerala': [10.8505, 76.2711],
                    'Madhya Pradesh': [23.4734, 77.9479],
                    'Maharashtra': [19.6633, 75.3202],
                    'Meghalaya': [25.4670, 91.3662],
                    'Mizoram': [23.1645, 92.9376],
                    'Nagaland': [26.1584, 94.5624],
                    'Odisha': [20.9517, 85.0985],
                    'Rajasthan': [27.0238, 74.2176],
                    'Tamil Nadu': [11.1271, 78.6569],
                    'Telangana': [17.0220, 78.3555],
                    'Tripura': [23.9408, 91.9882],
                    'Uttar Pradesh': [27.2599, 79.4126],
                    'Uttarakhand ': [30.0668, 79.0193],
                    'West Bengal ': [22.9868, 87.8550],
                    'Sikkim': [27.5330, 88.6139],
                    'Jammu & Kashmir': [33.2778, 76.5765],
                    'Punjab': [31.1471, 75.3412],
                    'Haryana': [29.0588, 76.0856],
                    'Himachal Pradesh': [31.1048, 77.1734],
                    'Manipur': [24.6637, 93.9063],
                    'Andaman & Nicobar Islands ': [11.7401, 92.6586]
                }.get(state, [20.5937, 78.9629])

                predicted_count = row['Predicted Elephant Count']

                if predicted_count > 200:
                    color = 'pink'
                elif predicted_count > 50:
                    color = 'orange'
                else:
                    color = 'yellow'

                bubble_size = 20

                folium.CircleMarker(
                    location=location,
                    radius=bubble_size,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8
                ).add_to(pred_map)

                folium.map.Marker(
                    location,
                    icon=folium.DivIcon(html=f'''
                        <div style="text-align: left; font-size: 8.5pt; font-weight: bold; width:{bubble_size * 0.2}px; color: black;">
                            {predicted_count:.0f}
                        </div>''')
                ).add_to(pred_map)

            pred_title_html = f'''
                <h3 align="center" style="font-size:20px"><b>Predicted Elephant Count in {future_year}</b></h3>
            '''
            pred_map.get_root().html.add_child(folium.Element(pred_title_html))

            pred_legend_html = '''
            <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 150px; height: 110px; 
                 border:2px solid grey; z-index:9999; font-size:12px;
                 background-color:white; padding: 10px;">
                 <b>Predicted Elephant Frequency</b><br>
                 <i class="fa fa-circle" style="color:pink"></i> > 200 Elephants<br>
                 <i class="fa fa-circle" style="color:orange"></i> 50-200 Elephants<br>
                 <i class="fa fa-circle" style="color:yellow"></i> < 50 Elephants<br>
            </div>
            '''
            pred_map.get_root().html.add_child(folium.Element(pred_legend_html))

            st.components.v1.html(pred_map._repr_html_(), height=700)

# Add your Streamlit app configuration and page registration
if __name__ == '__main__':
    st.set_page_config(page_title='Elephant Monitoring', layout='wide')
    elephants_page()
