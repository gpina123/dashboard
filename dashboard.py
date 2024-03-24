import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from datetime import datetime,date
import pytz

import dash
from dash import Dash, dcc, html, Input, Output, callback, dash_table, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_daq as daq

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn import  metrics


####libraries for feature selection
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

#stores the data from 2019, which includes power consumption and meteorological data.
df_raw_2019 = pd.read_csv('testData_2019_Central.csv')
df_clean_2019=df_raw_2019.copy()
df_raw_2019=df_raw_2019.rename(columns={"Central (kWh)":"Power_kW"})
df_raw_2019["Date"]=pd.to_datetime(df_raw_2019["Date"],format="%Y-%m-%d %H:%M:%S")
df_raw_2019=df_raw_2019.set_index('Date', drop=True)

#df_clean_2019= df_raw_2019.drop(columns=["temp_C","HR","windSpeed_m/s","windGust_m/s","pres_mbar","rain_mm/h","rain_day"])
df_clean_2019=df_clean_2019.rename(columns={"Central (kWh)":"Power_kW"})
df_clean_2019["Date"]=pd.to_datetime(df_clean_2019["Date"],format="%Y-%m-%d %H:%M:%S")
df_clean_2019=df_clean_2019.set_index('Date', drop=True)


#preparing data for training
df_raw_2017 = pd.read_csv('IST_Central_Pav_2017.csv') # loads a csv file into a dataframe
df_raw_2018 = pd.read_csv('IST_Central_Pav_2018.csv')
df_meteo = pd.read_csv('IST_meteo_data_2017_2018_2019.csv')
df_power_consuption = pd.concat([df_raw_2017,df_raw_2018])
df_power_consuption["Date_start"]=pd.to_datetime(df_power_consuption["Date_start"],format="%d-%m-%Y %H:%M")
df_power_consuption=df_power_consuption.rename(columns={"Date_start":"Date"})
df_meteo=df_meteo.rename(columns={"yyyy-mm-dd hh:mm:ss":"Date"})
df_meteo["Date"]=pd.to_datetime(df_meteo["Date"],format="%Y-%m-%d %H:%M:%S")
df_pwer_consuption=df_power_consuption.set_index('Date', drop=True)
df_meteo = df_meteo.set_index('Date', drop=True)
df_meteo=df_meteo.resample('h').mean()
df_all_data= pd.merge(df_power_consuption,df_meteo, on="Date")
df_all_data = df_all_data.set_index('Date', drop=True)

df_clean = df_all_data[~df_all_data.index.duplicated(keep='first')]; #keeps the first occurence and drops the duplicated one

df_missing_data=pd.read_csv("dados_em_falta.csv")
df_missing_data["datetime"]=pd.to_datetime(df_missing_data["datetime"],format="%Y-%m-%d %H:%M:%S")
df_missing_data=df_missing_data.rename(columns={"datetime":"Date","temp":"temp_C","humidity":"HR","solarradiation":"solarRad_W/m2","sealevelpressure":"pres_mbar","precip":"rain_mm/h"})
df_missing_data = df_missing_data.set_index("Date", drop=True)
df_missing_data = df_missing_data[~df_missing_data.index.duplicated(keep='first')]; #also removing the duplicate rows.
df_clean.update(df_missing_data,overwrite=False) #here I update the database with the new data
df_clean=df_clean.dropna()

import holidays
pt_holidays = holidays.PT() #stores all the holidays in Portugal
#This command line adds all the time periods where the students are not having classes
vacations = ((df_clean.index> "2016-12-31") & (df_clean.index < "2017-01-07")) | ((df_clean.index> "2017-02-04") & (df_clean.index < "2017-02-20")) | ((df_clean.index> "2017-02-26") & (df_clean.index <= "2017-03-01")) | ((df_clean.index> "2017-04-09") & (df_clean.index < "2017-04-15")) |((df_clean.index> "2017-06-04") & (df_clean.index < "2017-06-10"))|((df_clean.index> "2017-07-08") & (df_clean.index < "2017-09-11"))|((df_clean.index> "2017-12-22") & (df_clean.index < "2018-01-06"))| ((df_clean.index> "2018-02-03") & (df_clean.index < "2018-02-19")) | ((df_clean.index> "2018-03-25") & (df_clean.index < "2018-03-31")) | ((df_clean.index> "2018-06-03") & (df_clean.index < "2018-06-09")) | ((df_clean.index> "2018-07-07") & (df_clean.index < "2018-09-10")) | (df_clean.index> "2018-12-21")
#creates new column with the information about the weekends, holidays and vacations.
df_clean["weekend_holiday"] = ((df_clean.index.weekday > 4).astype(int) | (df_clean.index.map(lambda d: d in pt_holidays)).astype(int) | vacations.astype(int))

df_clean['Power-1_kW']=df_clean['Power_kW'].shift(1) #Previous hour consumption
df_clean=df_clean.dropna() #drops the nan row
df_clean['Power-1week_kW']=df_clean['Power_kW'].shift(168)# the number 168 comes from 7weeks*24hours
df_clean=df_clean.dropna() #drops the nan rows
df_clean["Power_1stderiv_kW"]=df_clean["Power_kW"]-df_clean["Power-1_kW"]

############################


import holidays
pt_holidays = holidays.PT() #stores all the holidays in Portugal

#This command line adds all the time periods where the students are not having classes
vacations = ((df_clean_2019.index> "2018-12-31") & (df_clean_2019.index < "2019-01-04")) | ((df_clean_2019.index> "2019-02-05") & (df_clean_2019.index < "2019-02-18")) | ((df_clean_2019.index> "2019-03-03") & (df_clean_2019.index < "2019-03-06")) | ((df_clean_2019.index> "2019-04-14") & (df_clean_2019.index < "2019-04-19"))
#creates new column with the information about the weekends, holidays and vacations.
df_clean_2019["weekend_holiday"] = ((df_clean_2019.index.weekday > 4).astype(int) | (df_clean_2019.index.map(lambda d: d in pt_holidays)).astype(int) | vacations.astype(int))
df_clean_2019['Power-1_kW']=df_clean_2019['Power_kW'].shift(1) #Previous hour consumption
#df_clean_2019=df_clean_2019.iloc[:, [0,3,1,2]] #puts both columns of power consuption side by side
df_clean_2019=df_clean_2019.dropna() #drops the nan row
df_clean_2019['Power-1week_kW']=df_clean_2019['Power_kW'].shift(168)# the number 168 comes from 7weeks*24hours
#df_clean_2019=df_clean_2019.iloc[:, [0,1,4,2,3]] #puts both columns of power consuption side by side
df_clean_2019=df_clean_2019.dropna() #drops the nan rows

df_clean_2019["Power_1stderiv_kW"]=df_clean_2019["Power_kW"]-df_clean_2019["Power-1_kW"]
#df_clean_2019=df_clean_2019.iloc[:, [0,1,2,5,3,4]]

fig_raw = px.line(df_clean_2019,x=df_clean_2019.index,y="Power_kW")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL,dbc.themes.BOOTSTRAP,'styles.css'],suppress_callback_exceptions=True,prevent_initial_callbacks='initial_duplicate')
server=app.server

sticky= {
        "position": "fixed",
        "width": "100%",
        "background-color": "#f1f1f1",
        #"padding": "10px",
        "z-index": "1000", #Ensure it's above other content
        "box-shadow": "0px 2px 5px rgba(0, 0, 0, 0.1)", #Optional: Add shadow for a better look
        "margin-bottom": "200px",
        "margin-left":"25px",
        "margin-right":"25px",
        "margin-top":"0px"
    }
    
custom_date= {
    "color": "white",
    "text-align": "center",
    "font-size": "4px",
    "height": "25px",
    "width": "100%",
    "padding-left": "3px",
    "padding-right": "3px",
}

style_tab_cima={"position":"fixed","z-index": "1000",
                #"margin-left":"-25px","margin-right":"50px",
                #"left":"0px","right":"0px",
                #"padding-left":"0px","padding-right":"0px",
                "background-color": "#f1f1f1"}

app.layout = dbc.Container([
    dbc.Container([
        dbc.Tabs(children=[
            dbc.Tab(label='Raw Data from 2019', tab_id='tab-1-example-graph'),
            dbc.Tab(label='Feature Selection', tab_id='feature_selection_tab'),
            dbc.Tab(label='Predict Consumption', tab_id='tab-2-example-graph'),
        ], id="tabs", active_tab="tab-1-example-graph"),
        dbc.Row(html.H1('Energy Services Dashboard'),style={"padding-top": "5px"}),
        dbc.Row(dcc.Markdown('''
                    Gonçalo Almeida 100308''')),
    ],
        style={"background-color": "#f1f1f1","box-shadow": "0px 2px 5px rgba(0, 0, 0, 0.1)"},
    ),
    dbc.Container(id='tabs-content-example-graph',style={"margin-top":"10px"})
],
)


@callback(Output('tabs-content-example-graph', 'children'),
        Input("tabs","active_tab"))
              #Input('tabs-example-graph', 'value'))
def render_content(tabs):
    if tabs=="tab-1-example-graph":
        return dbc.Container([
            dbc.Row(dcc.Markdown("### Exploratory Data Analysis\n"+"In this section, you can visualize the raw data from 2019, "+
                    "which includes the power consumption from the central building at IST and the meteorological data. "+
                    "Here, you are allowed to select which variable you want to display (one at a time) and how it is displayed. "+
                    "You may select the date range and the typle of plot."
                    )),
            dbc.Row([
                dbc.Col([
                dbc.Card([
                    dcc.Markdown("Select variable to display."),
                    dcc.Dropdown(
                        id='y-values-dropdown',
                        options=[
                            {'label': 'Power Consumption', 'value': 'Power_kW'},
                            {'label': 'Temperature', 'value': 'temp_C'},
                            {'label': 'Relative Humidity', 'value': 'HR'},
                            {'label': 'Wind speed', 'value': 'windSpeed_m/s'},
                            {'label': 'Wind gust', 'value': 'windGust_m/s'},
                            {'label': 'Pressure', 'value': 'pres_mbar'},
                            {'label': 'Solar irradiation', 'value': 'solarRad_W/m2'},
                            {'label': 'Rain', 'value': 'rain_mm/h'},
                            {'label': 'Rain day', 'value': 'rain_day'}
                        ],
                        value='Power_kW')
                    ],
                    body=True,
                    #className="card text-white bg-info mb-3",
                    style={"margin-top": "25px"},
                ),
                dbc.Card([
                    dcc.Markdown("Select date range."),
                    dbc.Row([
                        dbc.Col(dcc.DatePickerRange(
                            id='date-picker-range',
                            display_format='MMM Do, YYYY',
                            min_date_allowed=df_raw_2019.index.min(),
                            max_date_allowed=df_raw_2019.index.max(),
                            initial_visible_month=df_raw_2019.index.min(),
                            start_date=df_raw_2019.index.min(),
                            end_date=df_raw_2019.index.max()
                        )),
                        dbc.Col(html.Button('Reset Date Range', id='reset-button', n_clicks=0,
                                    className="btn btn-primary"))
                    ])
                    ],
                        body=True,
                        #className="card text-white bg-info mb-3",
                        style={"margin-top": "10px"}
                ),
                dbc.Card([
                    dcc.Markdown("Select type of plot."),
                    dbc.RadioItems(
                        id="radio_items",
                        inline=True,
                        options=[
                            {'label': 'Time Series', 'value': 'time_series'},
                            {'label': 'Boxplot', 'value': 'boxplot'},
                            {'label': 'Histogram', 'value': 'histogram'},
                        ],
                        value='time_series',
                        )
                    
                ],
                        body=True,
                        #className="card text-white bg-info mb-3",
                        style={"margin-top": "10px"})
                ],
                    width=5
                    
                ),
            
            dbc.Col([dcc.Graph(id='raw_data',className='journal-theme-graph',)]
                     #style={'width': '50%', 'display': 'inline-block','background-color': "#D3D3D3",
                           #"margin":"auto"}
            ,width=7)
            ]),
            #className="bg-primary"
            ]),
            
    

    elif tabs=="feature_selection_tab":
        return dbc.Container([
            dbc.Row(dcc.Markdown("### Feature Selection"+
                                 "\nBefore performing the prediction of the power consuption, it is wise to test which features "+
                                 "are relevant for building the model."+
                                 "\n\nThe feature selection uses the data from 2017 and 2018. Depending on the feature selection method, the number of features can be asked.")),
            dbc.Row([
            dbc.Col([
                dbc.Card([
                dcc.Markdown("Select feature selection method."),
                dcc.Dropdown(
                        id='feature_selection_method_dropdown',
                        options=[
                            {'label': 'SelectKBest', 'value': 'SK'},
                            {'label': 'Recursive Feature Elimination', 'value':'RFE'},
                            {'label': 'Random Forest Regressor', 'value':'RF'}
                        ],
                        value='SK',  # Default selected options
                        style={"margin-bottom":"5px"}
                ),
                daq.NumericInput(
                    id="numeric_input",
                    min=1,
                    max=len(df_clean.columns)-1,
                    label='Number of features:',
                    labelPosition='top',
                    value=4,
                    style={"display":"Block"}
                )],
                    body=True,
                    #className="card text-white bg-info mb-3",
                    style={"margin-top": "10px"})
                ],width=4),
            dbc.Col([
                dbc.Card([
                    dbc.Row([    
                        dbc.Col(html.Button('Apply feature selection method', id='apply_feature_method', n_clicks=0,style={"width":"100%"}),width=9),
                        dbc.Col(dbc.Button(
                                    [dbc.Spinner(size="sm"), " Loading..."],
                                    color="primary",
                                    disabled=True,
                                    id="loading_button_feature",
                                    style={"display":"None"}
                                )
                            ,width=3)])],
                            body=True,
                            style={"margin-top":"10px","margin-bottom":"10px"}),
                html.Div(id="bar_graph_features")],width=8)
            ])
        ]),
        
    elif tabs=="tab-2-example-graph":
        return dbc.Container([
            dbc.Row(dcc.Markdown("### Test Regression\n"+
                                 "In order to predict the power consumption of 2019, we need to first train a regression model based on pre-selected features. "+
                                 "Here, you can train a regression model of your choice and select the features you think that suit best. The pre-selected options were used in my first project." +
                                 "\n\nThe data used to train is correspondent to 2017 and 2018, and is selected randomly. "+
                                 "\nPress the button \"Apply method\" to get your results. Two plots comparing the predicted and real data will appear, as well as a table with the metrics that evaluate the fit.")),
            dbc.Row([
                dbc.Col([
                dbc.Card([
                    dcc.Markdown("Select features to train the model."),
                    dcc.Dropdown(
                        id='features_variables',
                        options=[
                            {'label': 'Power Consumption 1 day before', 'value': 'Power-1_kW'},
                            {'label': 'Power Consumption 1 week before', 'value': 'Power-1week_kW'},
                            {'label': 'First derivative', 'value': 'Power_1stderiv_kW'},
                            {'label': 'Weekend/holiday', 'value':'weekend_holiday'},
                            {'label': 'Temperature', 'value': 'temp_C'},
                            {'label': 'Relative Humidity', 'value': 'HR'},
                            {'label': 'Wind speed', 'value': 'windSpeed_m/s'},
                            {'label': 'Wind gust', 'value': 'windGust_m/s'},
                            {'label': 'Pressure', 'value': 'pres_mbar'},
                            {'label': 'Solar irradiation', 'value': 'solarRad_W/m2'},
                            {'label': 'Rain', 'value': 'rain_mm/h'},
                            {'label': 'Rain day', 'value': 'rain_day'}
                        ],
                        multi=True,
                        value=['Power-1_kW','Power-1week_kW','weekend_holiday',
                            "Power_1stderiv_kW",'solarRad_W/m2']  # Default selected options
                )],
                        body=True,
                        style={"margin-top": "25px"}),
                dbc.Card([
                    dcc.Markdown("Select regression model."),
                    dcc.Dropdown(
                        id='method',
                        options=[
                            {'label': 'Gradient Boosting', 'value': 'GB'},
                            {'label': 'Random Forest', 'value':'RF'},
                            {"label": "Support Vector", "value":"SV"}
                        ],
                        value='GB'  # Default selected options
                )],
                        body=True,
                        style={"margin-top": "10px","margin-bottom":"10px"}),
                html.Div(id="alert",style={"display":"None"}),
        ],width=4),
            dbc.Col([
                dbc.Card([
                    dbc.Row([    
                        dbc.Col(html.Button('Apply Method', id='apply_method', n_clicks=0,style={"width":"100%"}),width=9),
                        dbc.Col(dbc.Button(
                                    [dbc.Spinner(size="sm"), " Loading..."],
                                    color="primary",
                                    disabled=True,
                                    id="loading_button",
                                    style={"display":"None"}
                                )
                            ,width=3)])],
                            body=True,
                            style={"margin-top": "25px","margin-bottom":"25px"}),
                
                dbc.Row(dcc.Graph(id='graph_predict',style={'display': 'none'})),
                dbc.Row(dcc.Graph(id='comparison_predict',style={'display': 'none'})),
                dbc.Row(dash_table.DataTable(id="metrics_first",style_cell={'textAlign': 'center'},
                                             )),
                dbc.Row(dash_table.DataTable(id="metrics_second",style_cell={'textAlign': 'center'},
                                            ))
        ], width=8)])
        ])
        

#df_metrics.to_dict('records'),[{"name": i, "id": i} for i in df_metrics.columns]

'''
@app.callback(Output("bar_graph_features","figure"),
              Input("feature_selection_method_dropdown","value"))
def feature_methods(feature_method):
    Z=df_clean_2019.values
    Y=Z[:,0] #stores the power consumption separately
    X=Z[:,1:] #stores the other columns
    
    traces=[]
    
    if feature_method=="SK":
        features=SelectKBest(k=4,score_func=f_regression) # Test different k number of features. 
        fit=features.fit(X,Y) #calculates the scores using the score_function f_regression of the features
        #print(df_clean.columns[1:])
        print(fit.scores_)
        #print("Features selected: ")
        print(fit.get_feature_names_out(input_features=df_clean_2019.columns[1:]))
        #features_results=fit.transform(X)
        #print(features_results)
        traces.append(go.Bar(x=df_clean_2019.columns[1:],
                             y=fit.scores_))
    

    layout = go.Layout(title="SelectKBest method using f_regression",
                       yaxis_title="Score",
                       xaxis_title="Feature")
    
    return {'data': traces, 'layout': layout}
'''

@app.callback(Output("apply_method","disabled"),
              Input("features_variables","value"))
def turn_off(value):
    if value is None or len(value) == 0:
        return True
    else:
        return False

@app.callback(Output("numeric_input","style"),
              Input("feature_selection_method_dropdown","value"))
def numeric_input(value):
    if value in ["SK","RFE"]:
        return {"display":"Block"}
    elif value in ["RF"]:
        return {"display":"None"}
                    
                    
#Define callback to build the feature selection results
@app.callback([Output("bar_graph_features","children"),
                Output("loading_button_feature","style")],
              [Input("apply_feature_method","n_clicks")],
              [State("feature_selection_method_dropdown","value"),
              State("numeric_input","value")],
              prevent_initial_call=True)
def feature_methods(n_clicks,feature_method, k_number):
    Z = df_clean.values
    Y = Z[:, 0]  # Stores the power consumption separately
    X = Z[:, 1:]  # Stores the other columns
    
    if feature_method == "SK":
        features = SelectKBest(k=k_number, score_func=f_regression)  # Test different k number of features
        fit = features.fit(X, Y)  # Calculates the scores using the score_function f_regression of the features
        # Get the indices of the top k features
        top_indices = np.argsort(fit.scores_)[-k_number:]
        
        # Get the feature names and scores for the top k features
        top_feature_names = df_clean.columns[1:][top_indices]
        top_scores = fit.scores_[top_indices]
        
        # Sort the columns based on their index
        column_indices = [list(df_clean.columns[1:]).index(feature) for feature in df_clean.columns[1:]]
        sorted_indices = np.argsort(column_indices)
        sorted_feature_names = df_clean.columns[1:][sorted_indices]
        sorted_scores = fit.scores_[sorted_indices]
        
        # Create traces for top features with red color
        top_traces = []
        for feature, score in zip(sorted_feature_names, sorted_scores):
            if feature in top_feature_names:
                top_traces.append(go.Bar(x=[feature], y=[score], text=[f'{score:.2f}'], name=feature, marker=dict(color='red')))
            else:
                top_traces.append(go.Bar(x=[feature], y=[score], text=[f'{score:.2f}'], name=feature, marker=dict(color='blue')))

        traces = top_traces
        
        layout = go.Layout(title="SelectKBest method using f_regression",
                    yaxis_title="Score",
                    xaxis_title="Feature")

        return dcc.Graph(figure={'data': traces, 'layout': layout}),{"display":"None"}
    
    
    elif feature_method=="RFE":
        model=LinearRegression() # LinearRegression Model as Estimator
        rfe=RFE(model,n_features_to_select=k_number)
        fit=rfe.fit(X,Y)
        
        return f"The best {k_number} features to be used together are {fit.get_feature_names_out(input_features=df_clean.columns[1:])}",{"display":"None"}
    
    elif feature_method == "RF":
        model_rf = RandomForestRegressor()
        model_rf.fit(X, Y)
        
        # Get the feature names and importances
        feature_importances = model_rf.feature_importances_
        feature_names = df_clean.columns[1:]
        
        # Create bar traces with feature importances
        traces = [go.Bar(x=feature_names, y=feature_importances, text=[f'{imp:.6f}' for imp in feature_importances], 
                        name="Feature Importances", marker=dict(color='blue'))]

        layout = go.Layout(title="Random Forest Regressor",
                        yaxis_title="Importance Score",
                        xaxis_title="Feature")

        return dcc.Graph(figure={'data': traces, 'layout': layout}),{"display":"None"}
 
#Define callback to create loading button for tab2
@app.callback(Output("loading_button_feature","style",allow_duplicate=True),
              Input("apply_feature_method", "n_clicks"),
              prevent_initial_call=True)
def appear_loading_button_feature(n_clicks):
    if n_clicks:
        return {}       


#Define callback to create loading button for tab3
@app.callback(Output("loading_button","style",allow_duplicate=True),
              Input("apply_method", "n_clicks"),
              prevent_initial_call=True)
def appear_loading_button(n_clicks):
    if n_clicks:
        return {}

# Define callback to update the graph based on selected options
@app.callback(
    Output('raw_data', 'figure'),
    [Input('y-values-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input("radio_items","value")]
)
def update_graph(selected_options,start_date, end_date,type_graph):
    traces=[]
    if "time_series" in type_graph:
        traces.append(go.Scatter(
            x=df_raw_2019.loc[start_date:end_date].index,
            y=df_raw_2019.loc[start_date:end_date][selected_options],
            mode='lines',
            #name=selected_options
        ))
    elif "boxplot" in type_graph:
        traces.append(go.Box(
            name=selected_options,
            y=df_raw_2019.loc[start_date:end_date][selected_options],
        ))
    elif "histogram" in type_graph:
        traces.append(go.Histogram(
            x=df_raw_2019.loc[start_date:end_date][selected_options],
            nbinsx=50,
        ))
    

    if selected_options=="Power_kW":
        layout = go.Layout(title="Power Consumption in the Central Building (kW)")
    elif selected_options== "temp_C":
        layout = go.Layout(title='Power data from 2019',
                       #xaxis={'title': 'X Axis'},
                       #yaxis={'title': 'T'}
                       )
    elif selected_options=="HR":
        layout = go.Layout(title="Relative Humidity")
    elif selected_options=="windSpeed_m/s":
        layout = go.Layout(title="Wind speed (m/s)")
    elif selected_options=="windGust_m/s":
        layout = go.Layout(title="Wind gust (m/s)")
    elif selected_options=="pres_mbar":
        layout = go.Layout(title="Pressure (mbar)")
    elif selected_options=="solarRad_W/m2":
        layout = go.Layout(title="Solar Irradiation (W/m2)")
    elif selected_options=="rain_mm/h":
        layout = go.Layout(title="Rain (mm/h)")
    else:
        layout = go.Layout(title="Rain day")
        
    return {'data': traces, 'layout': layout}

'''
def update_graph(selected_options,start_date, end_date):
    traces = []
    for option in selected_options:
        traces.append(go.Scatter(
            x=df_raw_2019.loc[start_date:end_date].index,
            y=df_raw_2019.loc[start_date:end_date][option],
            mode='lines',
            name=option
        ))

    layout = go.Layout(title='Power data from 2019',
                       #xaxis={'title': 'X Axis'},
                       yaxis={'title': 'Power (kW)'})

    return {'data': traces, 'layout': layout}
'''
# Define callback to reset the date range when the button is clicked
@app.callback(
    [Output('date-picker-range', 'start_date'),
     Output('date-picker-range', 'end_date')],
    [Input('reset-button', 'n_clicks')],
    prevent_initial_call=True
)
def reset_date_range(n_clicks):
    start_date = df_raw_2019.index.min()
    end_date = df_raw_2019.index.max()

    return start_date, end_date


#Define callback to make the prediction of the power consumption
@app.callback(
    [Output('graph_predict', 'figure'),
    Output('metrics_first','data'),
    Output('metrics_second','data'),
    Output('graph_predict',"style",allow_duplicate=True),
    Output("loading_button","style"),
    Output("comparison_predict","figure"),
    Output("comparison_predict","style"),
    Output("alert","children"),
    Output("alert","style")],
    [Input('apply_method', 'n_clicks')],
    [State('features_variables','value'),
     State('method','value')],
    prevent_initial_call=True,
)
def make_prediction(n_clicks,selected_features,method_selected):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    Y=df_clean["Power_kW"]
    X=df_clean[selected_features]
    
    Y_2019=df_clean_2019["Power_kW"]
    X_2019=df_clean_2019[selected_features]
    X_2019=X_2019.values
    
    Y=Y.values
    X=X.values
    
    X_train, X_test, y_train, y_test = train_test_split(X,Y) #splitting the data into train and test
    
    traces=[]
    traces_scatter=[]
    
    if method_selected=="GB":
        GB_model= GradientBoostingRegressor()
        GB_model.fit(X_train, y_train)
        y_pred = GB_model.predict(X_2019)
        layout = go.Layout(title="Gradient Boosting regression")
        
    elif method_selected=="RF":
        parameters = {'bootstrap': True,
                'min_samples_leaf': 3,
                'n_estimators': 200, 
                'min_samples_split': 15,
                'max_features': 'sqrt',
                'max_depth': 20,
                'max_leaf_nodes': None}
        RF_model = RandomForestRegressor(**parameters)
        RF_model.fit(X_train, y_train)
        y_pred = RF_model.predict(X_2019)
        layout = go.Layout(title="Random Forest regression")
    
    elif method_selected=="SV":
        SVR_model= SVR(kernel='rbf')
        SVR_model.fit(X_train,y_train)
        y_pred= SVR_model.predict(X_test)
        layout = go.Layout(title="Support Vector regression")
        
        
    traces.append(go.Scatter(
        x=df_clean_2019.index,
        y=Y_2019,
        mode='lines',
        name="real data"
    ))
    traces.append(go.Scatter(
        x=df_clean_2019.index,
        y=y_pred,
        mode='lines',
        name="predicted data"
    ))
    
    
    traces_scatter.append(go.Scatter(
        x=df_clean_2019["Power_kW"],
        y=y_pred,
        mode="markers"
    ))
    layout_scatter = go.Layout(title="Scatter plot of the predicted and real data for the Power consumption",
                            xaxis_title="Real data",
                            yaxis_title="Predicted data",)
    
    MAE=metrics.mean_absolute_error(Y_2019,y_pred) 
    MBE=np.mean(Y_2019-y_pred)
    MSE=metrics.mean_squared_error(Y_2019,y_pred)  
    RMSE= np.sqrt(metrics.mean_squared_error(Y_2019,y_pred))
    cvRMSE=RMSE/np.mean(Y_2019)
    NMBE=MBE/np.mean(Y_2019)
    
    if (np.abs(cvRMSE*100)<20) & (np.abs(NMBE)*100<5): 
        alert=dbc.Alert([
                    html.H4("Well done!", className="alert-heading"),
                    html.P("You achieved good results!"),
                    html.P(f"|cvRMSE| should be <20%. You got {np.abs(cvRMSE*100):.2f}%"),
                    html.P(f"|NMBE| should be <5%. You got {np.abs(NMBE*100):.2f}%",style={"margin-top":"5px"})                    
        ])
    else:
        alert=dbc.Alert([
                    html.H4("It should be better", className="alert-heading"),
                    html.P(f"\n|cvRMSE| should be <20%. You got {np.abs(cvRMSE*100):.2f}%"),
                    html.P(f"\n|NMBE| should be <5%. You got {np.abs(NMBE*100):.2f}%",style={"margin-top":"5px"}),
                    html.Hr(),
                    html.P(
                        "Try different features and/or methods.",
                    )
        ],color="danger")
        
    d1={"MAE":[round(MAE, 4)],"MBE":[round(MBE, 4)],"MSE":[round(MSE, 4)]}
    d2={"RMSE":[round(RMSE, 4)],"cvRMSE":[round(cvRMSE, 4)],"NMBE":[round(NMBE, 4)]}

    df_metrics_1 =pd.DataFrame(data=d1)
    df_metrics_2 =pd.DataFrame(data=d2)
    
    metrics_data_1 = df_metrics_1.to_dict('records')
    metrics_data_2 = df_metrics_2.to_dict('records')
        
    return {'data': traces, 'layout': layout}, metrics_data_1,metrics_data_2,{}, {"display":"None"},{'data': traces_scatter, 'layout': layout_scatter},{},alert,{}

if __name__ == '__main__':
    app.run(debug=True,port=8050)
