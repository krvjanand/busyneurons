import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Updated synthetic data with diversity across all parameters
data = {
    'specialty': ['Cardiac', 'Orthopedic', 'Neuro', 'Cardiac', 'Neuro', 'Orthopedic', 'Cardiac', 'Pulmonary',
                  'Gastroenterology', 'Urology', 'Dermatology', 'Neurology', 'Pulmonary', 'Cardiac', 'Urology'],
    'age_group': ['40-50', '50-60', '40-50', '60-70', '50-60', '60-70', '50-60', '40-50', '60-70', '50-60',
                  '60-70', '40-50', '50-60', '40-50', '60-70'],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male',
               'Female', 'Male', 'Female', 'Male', 'Male'],
    'income_level': ['High', 'Medium', 'Medium', 'Low', 'High', 'Low', 'Medium', 'High', 'Low', 'High',
                     'Medium', 'Low', 'High', 'Medium', 'Low'],
    'comorbid_conditions': ['Diabetes', 'None', 'Hypertension', 'None', 'Diabetes', 'Hypertension', 'None',
                            'Hypertension', 'Diabetes', 'None', 'Hypertension', 'None', 'Diabetes', 'None', 'None'],
    'ethnicity': ['Arab', 'Asian', 'South Asian', 'Caucasian', 'African', 'Hispanic', 'European', 'Arab', 'Asian',
                  'South Asian', 'Caucasian', 'African', 'Hispanic', 'European', 'Arab'],
    'geography': ['Abu Dhabi', 'Dubai', 'Sharjah', 'Abu Dhabi', 'Dubai', 'Sharjah', 'Abu Dhabi', 'Ajman',
                  'Fujairah', 'Ras Al Khaimah', 'Abu Dhabi', 'Dubai', 'Sharjah', 'Abu Dhabi', 'Dubai'],
    'infection': [10, 5, 12, 3, 8, 15, 7, 6, 10, 11, 14, 2, 10, 8, 5],  # Number of infection cases
    'complication': [2, 3, 4, 1, 6, 4, 5, 2, 4, 3, 6, 4, 5, 3, 2],  # Number of complications
    'time_to_ambulation': [24, 36, 48, 30, 40, 20, 25, 35, 30, 45, 40, 30, 24, 48, 36],  # in hours
    'length_of_stay': [7, 10, 5, 8, 6, 12, 14, 9, 7, 10, 8, 6, 7, 5, 9],  # in days
    'readmission_30days': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 1: readmitted, 0: not readmitted
    'mortality': [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'return_to_surgery': [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # unplanned return to surgery
    'antibiotic_usage': [5, 6, 4, 3, 8, 5, 9, 2, 6, 7, 6, 5, 7, 8, 9],  # Number of cases using antibiotics
    'cost_of_care': [15000, 20000, 12000, 25000, 13000, 22000, 18000, 21000, 17000, 26000, 23000, 24000,
                     15000, 20000, 21000],  # cost in USD
    'patient_satisfaction': [90, 80, 85, 92, 75, 88, 91, 78, 85, 90, 82, 87, 80, 90, 88],  # satisfaction score out of 100
}

df = pd.DataFrame(data)

# Lat/long for UAE regions (Abu Dhabi, Dubai, etc.)
uae_locations = {
    'Abu Dhabi': [24.4539, 54.3773],
    'Dubai': [25.276987, 55.296249],
    'Sharjah': [25.3463, 55.4209],
    'Ajman': [25.4052165, 55.5136433],
    'Fujairah': [25.128809, 56.326485],
    'Ras Al Khaimah': [25.8006926, 55.9762449]
}

# Append lat/lon values to the dataframe
df['lat'] = df['geography'].map(lambda x: uae_locations[x][0] if x in uae_locations else None)
df['lon'] = df['geography'].map(lambda x: uae_locations[x][1] if x in uae_locations else None)

# Parameters for segmentation
parameters = ['specialty', 'age_group', 'gender', 'income_level', 'comorbid_conditions', 'ethnicity', 'geography']

# Define layout generation for each KPI
def kpi_page(title, y_col, data_df):
    graphs = []

    # Loop through segmentation parameters and create charts
    for i, param in enumerate(parameters):
        if param in data_df.columns:
            if param == 'gender':  # Change gender segmentation to a bar chart with aggregated value and data labels
                fig = px.bar(data_df.groupby(param).agg({y_col: 'sum'}).reset_index(),
                             x=param, y=y_col, title=f"{title} by {param.capitalize()} (Aggregated)", text_auto=True)
                fig.update_traces(texttemplate='%{y}', textposition='outside')  # Adding data labels
            elif param == 'age_group':  # Aggregated totals for age group segmentation
                total_value = data_df.groupby(param)[y_col].sum().reset_index()
                fig = px.bar(total_value, x=param, y=y_col, title=f"{title} by {param.capitalize()} (Total)", text_auto=True)
            elif param == 'specialty':  # Specialty-wise segmentation with aggregated value
                fig = px.bar(data_df.groupby(param).agg({y_col: 'sum'}).reset_index(),
                             x=param, y=y_col, title=f"{title} by {param.capitalize()} (Aggregated)", text_auto=True)
            elif param == 'income_level':  # Income level segmentation
                fig = px.pie(data_df, names=param, values=y_col, title=f"{title} by {param.capitalize()}")
            elif param == 'ethnicity':  # Ethnicity-wise segmentation
                fig = px.bar(data_df.groupby(param).agg({y_col: 'sum'}).reset_index(),
                             x=param, y=y_col, title=f"{title} by {param.capitalize()} (Aggregated)", text_auto=True)
            elif param == 'geography':  # Geography-wise segmentation (UAE Map using Lat/Long)
                fig = px.scatter_mapbox(data_df, lat='lat', lon='lon', color=y_col,
                                        size=y_col, size_max=30, zoom=6,
                                        title=f"{title} by {param.capitalize()} (UAE only)",
                                        mapbox_style="carto-positron")
            elif param == 'comorbid_conditions':  # Comorbid conditions with single aggregated x-axis value and data labels
                fig = px.bar(data_df.groupby(param).agg({y_col: 'sum'}).reset_index(),
                             x=y_col, y=param, orientation='h', title=f"{title} by {param.capitalize()} (Aggregated)", text_auto=True)
                fig.update_traces(texttemplate='%{x}', textposition='auto')  # Adding data labels

            graphs.append(dcc.Graph(figure=fig))

    # Ensure 10 charts are populated
    while len(graphs) < 10:
        graphs.append(html.Div())

    layout = html.Div([
        html.H2(f"{title} Dashboard"),
        dbc.Row([dbc.Col(graphs[i], width=6) for i in range(0, 5)]),
        dbc.Row([dbc.Col(graphs[i], width=6) for i in range(5, 10)])
    ])
    return layout

# Sidebar with KPI navigation
sidebar = html.Div(
    [
        html.H4("KPI Dashboard"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Infection Rate", href="/infection-rate", active="exact"),
                dbc.NavLink("Complication Rate", href="/complication-rate", active="exact"),
                dbc.NavLink("Readmission Rate", href="/readmission-rate", active="exact"),
                dbc.NavLink("Mortality Rate", href="/mortality-rate", active="exact"),
                dbc.NavLink("Return to Surgery", href="/return-to-surgery", active="exact"),
                dbc.NavLink("Length of Stay", href="/length-of-stay", active="exact"),
                dbc.NavLink("Cost of Care", href="/cost-of-care", active="exact"),
                dbc.NavLink("Patient Satisfaction", href="/satisfaction-rate", active="exact"),
                dbc.NavLink("Antibiotic Usage", href="/antibiotic-usage", active="exact"),
                dbc.NavLink("Time to Ambulation", href="/time-to-ambulation", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "20%", "padding": "20px 10px", "background-color": "#f8f9fa"},
)

# Main content area for KPI pages
content = html.Div(id="page-content", style={"margin-left": "20%"})

# App layout
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# Callback for routing and loading KPI pages
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/infection-rate":
        return kpi_page("Infection Rate", 'infection', df)
    elif pathname == "/complication-rate":
        return kpi_page("Complication Rate", 'complication', df)
    elif pathname == "/readmission-rate":
        return kpi_page("Readmission Rate", 'readmission_30days', df)
    elif pathname == "/mortality-rate":
        return kpi_page("Mortality Rate", 'mortality', df)
    elif pathname == "/return-to-surgery":
        return kpi_page("Return to Surgery", 'return_to_surgery', df)
    elif pathname == "/length-of-stay":
        return kpi_page("Length of Stay", 'length_of_stay', df)
    elif pathname == "/cost-of-care":
        return kpi_page("Cost of Care", 'cost_of_care', df)
    elif pathname == "/satisfaction-rate":
        return kpi_page("Patient Satisfaction", 'patient_satisfaction', df)
    elif pathname == "/antibiotic-usage":
        return kpi_page("Antibiotic Usage", 'antibiotic_usage', df)
    elif pathname == "/time-to-ambulation":
        return kpi_page("Time to Ambulation", 'time_to_ambulation', df)
    else:
        return html.Div("Welcome to the Hospital KPI Dashboard")

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
