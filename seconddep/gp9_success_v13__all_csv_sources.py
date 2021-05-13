import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.offline as pyo
import pandas as pd
import plotly.express as px
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

############https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

# import the xlsx file and covert to df
radardata = pd.read_csv('https://raw.githubusercontent.com/FYP2021g9/firstattempt/main/SportsData.csv')


# Read MC980output202011101547[2175].csv
mc980 = pd.read_csv('https://raw.githubusercontent.com/homanchau/FYP/main/MC980output202103191751.csv')


colors = {
    'background': '#192444',
    'text': 'white'
}

###############Stephen radar

new_radardata = radardata.drop(0)
Banding_table = {'Band': [1, 2, 3, 4, 5]}
name_radar = ['Agility', 'Power', 'Coordination', 'Flexibility', 'Endurance']

lastrow = new_radardata.shape[0]
for i in range(1,lastrow+1):
    new_radardata['Agility'][i] = float(new_radardata['Agility'][i])
    new_radardata['Power'][i] = float(new_radardata['Power'][i])
    new_radardata['Coordination'][i] = float(new_radardata['Coordination'][i])
    new_radardata['Flexibility'][i] = int(new_radardata['Flexibility'][i])
    new_radardata['Endurance'][i] = int(new_radardata['Endurance'][i])

for xName in name_radar:
     
    # parse band 1-5 to the bandwidth object and round to 3 decimals
    bandwidth = round((new_radardata[xName].max() - new_radardata[xName].min())/5,3)

    x = new_radardata[xName].min()
    y = bandwidth
    
    Banding_table[xName] = [round(x,3), round(x+y,2), round(x+2*y,2), round(x+3*y,2), round(x+4*y,2)]

# change data type from list to dataframe
df_Banding_table = pd.DataFrame(Banding_table)

# create new dataframe based on old one
df_radardata = new_radardata.copy()

# "df_Banding_table" is created!!

# column "Endurance" name is duplicated in the last column in the "new_radardata" 
# change the duplicated name to "Endurance.1"

new_radardata.columns = [*new_radardata.columns[:-1], 'Endurance.1']
df_radardata = df_radardata.rename(columns={'Student No.': 'ID No.'})

# new_radardata has renamed the last column name from Endurance to Endurance.1

# find the lastrow 
last_row = df_radardata.shape[0] 

for xName in name_radar:
    
    for rowNum in range(0,last_row):

        rowNum = rowNum + 1

        for numB in range(0,5):

            pass_band = df_Banding_table.loc[numB,xName]

            if df_radardata.loc[rowNum, xName] >= pass_band:
                
                df_radardata.loc[rowNum, xName + '.1'] = numB + 1

########################################


# remove the second row beacuse can not generate the min or max number
new_radardata = radardata.drop(0)

# find out the body data
date1 = mc980.columns[:1]
mc980_confirmed = mc980.melt(id_vars=['Date', 'Time', 'ID No.', 'Gender', 'Age', 'Height', 'Weight', 'BMI'],
                             value_vars=date1, var_name='Mem', value_name='mc980')

# Remove the duplicated data and keep the most updated data base on ID.NO
mc = mc980_confirmed.drop_duplicates(subset='ID No.', keep='last')

# remove unnecessary columns and replace Gender to M or F table
mc = mc.drop(columns=['Mem', 'mc980'])
mc['Gender'] = mc['Gender'].replace([[1], [2]], ['M', 'F'])

# find out the body data
date2 = mc980.columns[:2]
mc980_confirmed2 = mc980.melt(id_vars=['Date', 'Time', 'ID No.', 'Fat mass', 'Fat %', 'Muscle mass'], value_vars=date1,
                              var_name='Mem', value_name='mc980')

# Remove the duplicated data and keep the most updated data base on ID.NO
mc2 = mc980_confirmed2.drop_duplicates(subset='ID No.', keep='last')

# Table with "Fat mass"	"Fat %"	"Muscle mass" only
#mc2 = mc2.drop(columns=['Mem', 'mc980'])
mc2 = mc2.drop(columns=['Mem','mc980','Date', 'Time'])


# visited time table
visit_time = mc980[["Date", "Time"]]


######## merge mc and sportData

#change a column's format first
df_radardata= df_radardata.astype({"ID No.": 'int64'})

mcData = pd.merge(mc, df_radardata, on="ID No.")
#mcData = pd.concat([mc, df_radardata])


mc3 = pd.merge(mcData, mc2, on='ID No.')

## First main df~!
mcData = mc3



lastDate = mcData['Date'].tail(1)
lastTime = mcData['Time'].tail(1)
# height groups (6 groups)

h1 = len(mc[(mc['Height'] < 150)])  # shorter than 150cm
h2 = len(mc[(mc['Height'] >= 150) & (mc['Height'] < 160)])  # 150cm to 159cm
h3 = len(mc[(mc['Height'] >= 160) & (mc['Height'] < 170)])  # 160cm to 169cm
h4 = len(mc[(mc['Height'] >= 170) & (mc['Height'] < 180)])  # 170cm to 179cm
h5 = len(mc[(mc['Height'] >= 180) & (mc['Height'] < 190)])  # 180cm to 189cm
h6 = len(mc[(mc['Height'] > 190)])  # higher than 190cm

StudentHeight = {
    'Height Group': ['<150cm', '150cm to 159cm', '160cm to 169cm', '170cm to 179cm', '180cm to 189cm', '>190cm'],
    'Number of Student': [h1, h2, h3, h4, h5, h6]
}

SHeight = pd.DataFrame(StudentHeight, columns=['Height Group', 'Number of Student'])

mcIdNo = mc['ID No.']
mcBmi = mc['BMI']
countId = []
countId = mcIdNo
len(countId)

# Weight groups (6 groups)

w1 = len(mc[(mc['Weight'] < 40)])  # lighter than 40kg
w2 = len(mc[(mc['Weight'] >= 40) & (mc['Weight'] < 49)])  # 40kg-49kg
w3 = len(mc[(mc['Weight'] >= 50) & (mc['Weight'] < 59)])  # 50kg-59kg
w4 = len(mc[(mc['Weight'] >= 60) & (mc['Weight'] < 69)])  # 60kg-69kg
w5 = len(mc[(mc['Weight'] >= 70) & (mc['Weight'] < 79)])  # 70kg-79kg
w6 = len(mc[(mc['Weight'] >= 80) & (mc['Weight'] < 89)])  # 80kg-89kg
w7 = len(mc[(mc['Weight'] > 90)])  # heavier than 90kg
StudentWeight = {
    'Weight Group': ['<40kg', '40kg-49kg', '50kg-59kg', '60kg-69kg', '70kg-79kg', '80kg-89kg', '>90kg'],
    'Number of Student': [w1, w2, w3, w4, w5, w6, w7]
}

SWeight = pd.DataFrame(StudentWeight, columns=['Weight Group', 'Number of Student'])

### using head function(list out first 5 data) to create variables for plotting the bar chart ###
mcData_Agility = mcData['Agility'].sort_values(ascending=True).head()
mcData_Power = mcData['Power'].sort_values(ascending=False).head()
mcData_Coordination = mcData['Coordination'].sort_values(ascending=True).head()
mcData_Flexibility = mcData['Flexibility'].sort_values(ascending=True).head()
mcData_Endurance = mcData['Endurance'].sort_values(ascending=False).head()

## FINAL FYP: new added dataframe

# 先搵出AA-BZ的COLUMN名 type=list

more_radar_chart_columns_name = list(mc980)[26:79]

# 將AA-BZ的COLUMN 和 ID.NO 合成一個DF

newdf = mc980[['ID No.']].copy()

for columnName in more_radar_chart_columns_name:
    newdf = pd.concat([newdf, pd.DataFrame(mc980[[columnName]])], axis=1)

##SECOND main df~!
new_mcData = pd.merge(mcData, newdf, on='ID No.')

###########################fat and muscle radar chart
radar_mcData = new_mcData

selected_columns = ['Trunk Muscle score', 'LA Fat % score', 'LL Muscle score', 'RL Muscle score', 'RA Muscle score',
                    'Trunk Fat % score', 'LA Fat % score', 'LL Fat % score', 'RL Fat % score', 'RA Fat % score']

for selectItem in selected_columns:

    for rowNum01 in range(0, last_row):

        if int(radar_mcData[selectItem][rowNum01]) > 1:
            radar_mcData.loc[rowNum01, selectItem] = 3

        elif int(radar_mcData[selectItem][rowNum01]) > -2:
            radar_mcData.loc[rowNum01, selectItem] = 2

        else:
            radar_mcData.loc[rowNum01, selectItem] = 1

mcData = radar_mcData


fatp_metabolicAge = mcData[['ID No.','Fat %','Metabolic Age']]
fatp_metabolicAge


#######################ML section
#def update_prediction(X1, X2, X3):
mc980.corr()['Metabolic Age'].sort_values() 

# two features TBW% and Trunk Fat %
MAfeatures = mc980[['Metabolic Age','Fat %','TBW %','BMI']]

x = MAfeatures[['Fat %','TBW %','BMI']].values
y = MAfeatures[['Metabolic Age']].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

later_model = load_model('predict.v1')

########################















########################### Start to create dashboard

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])


server = app.server



################Scatter chart


#fig = px.scatter(fatp_metabolicAge, x="Metabolic Age", y="Fat %")
scatter_plot = go.Figure(data=go.Scatter(x=fatp_metabolicAge['Metabolic Age'],
                                y=fatp_metabolicAge['Fat %'],
                                mode='markers',
                
                                text=fatp_metabolicAge['ID No.']))
scatter_plot.update_layout(
    title= 'Scatter_Plot',
    paper_bgcolor = '#262529',
    #plot_bgcolor = '#262529',
    font =  dict(family='sans-serif',
                             color='white',
                             size=15),
    titlefont = {'color': 'white',
                              'size': 12},
            

)


##################

### Title
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    html.Div([
            html.H6(children='Welcome!', 
            style={
                'font-size': '70px',
                'color': colors['text']})
            
    ]),
    html.Br(),
    html.Div([
            html.H6(children='Page1 is some ive student health graph', 
            style={
                'font-size': '35px',
                'color': colors['text']})
            
    ]),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    html.Div([
            html.H6(children='Page2 is predict your Metabolic Age', 
            style={
                'font-size': '35px',
                'color': colors['text']})
            
    ]),
    dcc.Link('Go to Page 2', href='/page-2'),
    
],style={
        'background-image':"/assets/tyschool.jpg",
        'background-repeat': 'no-repeat',
	    'background-position': 'center',
	    'background-size': 'cover',
	    'position': 'fixed',
	    'min-height': '100%',
	    'min-width': '100%'
    })


page_1_layout = html.Div([
    ## BEGIN of Content
    html.Div([
        # Topic
        html.Div([
            html.H3('Health and Sports Passport', style={'margin-bottom': '8px', 'color': 'white'})

        ], className='one-third column', id='title1'),

        ##########################end of title

        ###########################start create bar chart (Agaility)

        # SECOND ROW LAYOUT
        html.Div([

            # "CONTAINER BOX1 ON 2nd ROW"
            html.Div([

                html.Div([
                    html.Div([
                        html.H6(children='The Last One enter (Date)',
                                style={'textAlign': ' center',
                                       'color': '#ee82ee'}),
                        html.P(mcData['Date'].iloc[-1],
                               style={'textAlign': 'center',
                                      'color': 'white',
                                      'fontSize': 40})
                    ]),
                    html.Div([
                        html.H6(children='The Last One enter (Time)',
                                style={'textAlign': ' center',
                                       'color': '#ee82ee'}),
                        html.P(mcData['Time'].iloc[-1],
                               style={'textAlign': 'center',
                                      'color': 'white',
                                      'fontSize': 40})
                    ])

                ])

            ], className='create_container2 two columns'),
            # END of "CONTAINER BOX1 ON 2nd ROW"

            # "CONTAINER BOX2 ON 2nd ROW"
            html.Div([
                dcc.RadioItems(id='radio_items',
                               labelStyle={'display': 'inline-block'},
                               value='Agility',
                               options=[{'label': 'Agility', 'value': 'Agility'},
                                        {'label': 'Power', 'value': 'Power'},
                                        {'label': 'Coordination', 'value': 'Coordination'},
                                        {'label': 'Flexibility', 'value': 'Flexibility'},
                                        {'label': 'Endurance', 'value': 'Endurance'}
                                        ],
                               style={'text-align': 'center', 'color': 'white'},
                               className='dcc_compon'),

                dcc.Graph(id='cx1', config={'displayModeBar': 'hover'},
                          style={'height': '350px'})

            ], className='create_container2 three columns', style={'height': '450px', 'width': '450px'}),
            # END of "CONTAINER BOX2 ON 2nd ROW"

            # "CONTAINER BOX3 ON 2nd ROW"
            ## create body result panel
            html.Div([
                html.H5(children='Body measurement results',
                        style={'color': '#ee82ee'}),

                # create input box for target user
                html.Div([dcc.Input(id='idno_inputed', type='number', value=mcData['ID No.'][0],
                                    placeholder='Input your Id Number')]),

                html.Br(),
                html.Div(id='bodyResult',
                         style={
                             'font-size': '20px',
                             # define the global color setting
                             'color': colors['text'],

                         }),
                # style={'height': '350px'})

            ], className='create_container2 four columns', style={'height': '400px', 'weight': '350px'}),
            # END of "CONTAINER BOX3 ON 2nd ROW"


        ], className='row flex-display'),
        # END of 2nd ROW

        # THIRD ROW LAYOUT
        html.Div([

            # "CONTAINER BOX1 ON 3rd ROW"
            html.Div([
                ###create radar chart 1
                html.Div([

                    html.H6('Radar diagram for 5 of your test data: ', style={'color': '#ee82ee'}),

                    dcc.Graph(id='radar_chart',
                              config={'displayModeBar': 'hover',
                                      },
                              style={'height': '350px'})
                ])
            ], className='create_container2 three columns', style={'height': '450px', 'width': '450px'}),
            # END of "CONTAINER BOX1 ON 3rd ROW"

            # "CONTAINER BOX2 ON 3rd ROW"
            html.Div([
                ###create radar chart 2
                html.Div([
                    html.H6('Your MUSCLE distribution: ', style={'color': '#ee82ee'}),
                    dcc.Graph(id='muscle_radar_chart',
                              config={'displayModeBar': 'hover',
                                      },
                              style={'height': '350px'})
                ])

            ], className='create_container2 three columns', style={'height': '450px', 'width': '450px'}),
            # END OF "CONTAINER BOX2 ON 3rd ROW"

            # "CONTAINER BOX3 ON 3rd ROW"
            html.Div([
                ###create radar chart 3
                html.Div([
                    html.H6('Your FAT distribution: ', style={'color': '#ee82ee'}),
                    dcc.Graph(id='fat_radar_chart',
                              config={'displayModeBar': 'hover',
                                      },
                              style={'height': '350px'})
                ])

            ], className='create_container2 three columns', style={'height': '450px', 'width': '450px'}),
            # END OF "CONTAINER BOX3 ON 3rd ROW"

        ], className='row flex-display'),
        # END of 3nd ROW
            html.Div([
        ##########creeate Student HEIGHT distribution
                html.Div([

                    html.Div([
                        dcc.Graph(
                            id='SHeight',
                                figure={
                                        'data': [
                                        {'x': SHeight['Height Group'],
                                        'y': SHeight['Number of Student'],
                                        'type': 'bar',
                                        'marker': {'color': '#ee82ee'},
                                        },
                                         ],
                                    'layout':{
                                        
                                    'title': 'Student Height Distribution',
                                    'color': 'white',
                                    'plot_bgcolor': '#262529',
                                    'paper_bgcolor': '#262529',
                                    'titlefont': {'color': 'white',
                                                'size': 12},
                                    'font': dict(family='sans-serif',
                                                color='white',
                                                size=15),
                                    'legend': {'orientation': 'h',
                                            'bgcolor': '#262529',
                                            'xanchor': 'center', 'x': 0.5, 'y': -0.7},
                                    'xaxis': dict(title='<b></b>',
                                                color='white',
                                                showline=True,
                                                showgrid=True,
                                                showticklabels=True,
                                                
                                                linecolor='white',
                                                linewidth=1,
                                                ticks='outside',
                                                tickfont=dict(
                                                    family='Aerial',
                                                    color='white',
                                                    size=12
                                                )),
                                    'yaxis': dict(title='number of Student',
                                                color='white',

                                                showline=False,
                                                showgrid=False,
                                                showticklabels=True,
                                                linecolor='white',
                                                linewidth=1,
                                                ticks='outside',
                                                tickfont=dict(
                                                    family='Aerial',
                                                    color='white',
                                                    size=12
                                                )
                                                )
                                }
                                    
                                   
                                },
                              config={'displayModeBar': 'hover',
                                      },
                              style={'height': '350px'})
                    ])
                 ], className='create_container2 three columns', style={'height': '450px', 'width': '350px'}),


        ##########END OF Student HEIGHT distribution

        ##########creeate Student WEIGHT distribution

                html.Div([

                    html.Div([
                        dcc.Graph(
                            id='SWeight',
                                figure={
                                        'data': [
                                        {'x': SWeight['Weight Group'],
                                        'y': SWeight['Number of Student'],
                                        'type': 'bar',
                                        'marker': {'color': '#ee82ee'},
                                        },
                                         ],
                                    'layout':{
                                        
                                    'title': 'Student Weight Distribution',
                                    'color': 'white',
                                    'plot_bgcolor': '#262529',
                                    'paper_bgcolor': '#262529',
                                    'titlefont': {'color': 'white',
                                                'size': 12},
                                    'font': dict(family='sans-serif',
                                                color='white',
                                                size=15),
                                    'legend': {'orientation': 'h',
                                            'bgcolor': '#262529',
                                            'xanchor': 'center', 'x': 0.5, 'y': -0.7},
                                    'xaxis': dict(title='<b></b>',
                                                color='white',
                                                showline=True,
                                                showgrid=True,
                                                showticklabels=True,
                                                
                                                linecolor='white',
                                                linewidth=1,
                                                ticks='outside',
                                                tickfont=dict(
                                                    family='Aerial',
                                                    color='white',
                                                    size=12
                                                )),
                                    'yaxis': dict(title='number of Student',
                                                color='white',

                                                showline=False,
                                                showgrid=False,
                                                showticklabels=True,
                                                linecolor='white',
                                                linewidth=1,
                                                ticks='outside',
                                                tickfont=dict(
                                                    family='Aerial',
                                                    color='white',
                                                    size=12
                                                )
                                                )
                                }
                                    
                                   
                                },

                              config={'displayModeBar': 'hover',
                                      },
                              style={'height': '350px'})
                    ])
                 ], className='create_container2 three columns', style={'height': '450px', 'width': '350px'}),


                 html.Div([
                ###create radar chart 3
                html.Div([
                    html.H6('Metabolic Age & Fat %', style={'color': '#ee82ee'}),
                    dcc.Graph(
                        id = 'scatter_plot',
                        
                        figure = px.scatter(fatp_metabolicAge, x="Metabolic Age", y="Fat %", 
                                        template='plotly_dark').update_layout({
                                                 'plot_bgcolor': '#262529',
                                                'paper_bgcolor': '#262529',
                                        }),
                              config={'displayModeBar': 'hover',
                                      },
                              style={'height': '500px', })
                ])

            ], className='create_container2 three columns', style={'height': '500px', 'width': '650px'})



                ], className='row flex-display')

        ##########END OF Student WEIGHT distribution


    ], id='mainContainer', style={'display': 'flex', 'flex-direction': 'column'}),
    ## END of Content
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
])


@app.callback(dash.dependencies.Output('page-1-content', 'children'),
              [dash.dependencies.Input('page-1-dropdown', 'value')])
def page_1_dropdown(value):
    return 'You have selected "{}"'.format(value)


##################################Page 2 Section
page_2_layout = html.Div([
    html.Div([
            html.H6(children='Page 2', 
            style={
                'font-size': '70px',
                'color': colors['text']})
    ]),
    html.Div([
            html.H6(children='Predict your Metabolic Age', 
            style={
                'font-size': '35px',
                'color': colors['text']})
    ]),
    html.Div([
        html.Div(dcc.Input(id='Fat_input', type='text')),
        html.Br(),
        html.Div(dcc.Input(id='TBW_input', type='text')),
        html.Br(),
        html.Div(dcc.Input(id='BMI_input', type='text')),
        html.Br(),
        html.Button('Submit', id='submit-val',
        style={'color':'white'}),
        html.Br(),
        html.Div(id = "model",style={'color': colors['text']})
    ]),
    html.Div(id='page-2-content'),
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

@app.callback(dash.dependencies.Output('page-2-content', 'children'),
              [dash.dependencies.Input('page-2-radios', 'value')])
def page_2_radios(value):
    return 'You have selected "{}"'.format(value)

# Update the index

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page

#############################################End of page 2 section

#######################call back function of input box
@app.callback(
    dash.dependencies.Output('model', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State("Fat_input", "value")],
    [dash.dependencies.State("TBW_input", "value")],
    [dash.dependencies.State("BMI_input", "value")],

)
def update_output(n_clicks, Fat_input, TBW_input, BMI_input):
    new_data= []
    new_Fat_input = int(Fat_input)
    new_TBW_input = int(TBW_input)
    new_BMI_input = int(BMI_input)
    new_data = [[new_Fat_input, new_TBW_input, new_BMI_input]]
    new_data = scaler.transform(new_data)

    result = later_model.predict(new_data)
    ss = result[0]
    dd = ss[0]
    ff = int(dd)
    return "Your Metabolic Age is :  {}".format(ff) 
    

#######################call back function of input box

##################call back function of bar chart
@app.callback(Output('cx1', 'figure'),
              [Input('radio_items', 'value')])
def update_Graph(radio_items):
    if radio_items == 'Agility':
        return {
            'data': [
                {'x': str(mcData['ID No.']),
                 'y': mcData_Agility,
                 'type': 'bar',
                 'text': mcData['ID No.'],
                 'name': 'Agility',
                 'marker': {'color': '#ee82ee'}
                 },
            ],
            'layout': {
                'title': 'Agility Ranking ( Top5 )',
                'color': 'white',
                'plot_bgcolor': '#262529',
                'paper_bgcolor': '#262529',
                'titlefont': {'color': 'white',
                              'size': 12},
                'font': dict(family='sans-serif',
                             color='white',
                             size=15),
                'legend': {'orientation': 'h',
                           'bgcolor': '#262529',
                           'xanchor': 'center', 'x': 0.5, 'y': -0.7},
                'xaxis': dict(title='<b></b>',
                              color='white',
                              showline=True,
                              showgrid=True,
                              showticklabels=True,
                              #  # in asc order
                              # categoryorder= 'total ascending',
                              linecolor='white',
                              linewidth=1,
                              ticks='outside',
                              tickfont=dict(
                                  family='Aerial',
                                  color='white',
                                  size=12
                              )),
                'yaxis': dict(title='in seconds',
                              color='white',

                              showline=False,
                              showgrid=False,
                              showticklabels=True,
                              linecolor='white',
                              linewidth=1,
                              ticks='outside',
                              tickfont=dict(
                                  family='Aerial',
                                  color='white',
                                  size=12
                              )
                              )
            }
        }

    if radio_items == 'Power':
        return {
            'data': [
                {'x': str(mcData['ID No.']),
                 'y': mcData_Power,
                 'type': 'bar',
                 'name': 'Power',
                 'text': mcData['ID No.'],
                 'marker': {'color': '#ee82ee'}
                 },

            ],
            'layout': {
                'title': 'Power Ranking  ( Top5 )',
                'color': 'white',
                'plot_bgcolor': '#262529',
                'paper_bgcolor': '#262529',
                'titlefont': {'color': 'white',
                              'size': 12},
                'font': dict(family='sans-serif',
                             color='white',
                             size=15),
                'legend': {'orientation': 'h',
                           'bgcolor': '#262529',
                           'xanchor': 'center', 'x': 0.5, 'y': -0.7},
                'xaxis': dict(title='<b></b>',
                              color='white',
                              showline=True,
                              showgrid=True,
                              showticklabels=True,
                              linecolor='white',
                              linewidth=1,
                              ticks='outside',
                              tickfont=dict(
                                  family='Aerial',
                                  color='white',
                                  size=12
                              )),
                'yaxis': dict(title='<b></b>',
                              color='white',

                              showline=False,
                              showgrid=False,
                              showticklabels=True,
                              linecolor='white',
                              linewidth=1,
                              ticks='outside',
                              tickfont=dict(
                                  family='Aerial',
                                  color='white',
                                  size=12
                              )
                              )
            }
        }
    if radio_items == 'Coordination':
        return {
            'data': [
                {'x': str(mcData['ID No.']),
                 'y': mcData_Coordination,
                 'type': 'bar',
                 'name': 'Power',
                 'text': mcData['ID No.'],
                 'marker': {'color': '#ee82ee'},
                 },
            ],
            'layout': {
                'title': 'Coordination Ranking ( Top5 )',
                'color': 'white',
                'plot_bgcolor': '#262529',
                'paper_bgcolor': '#262529',
                'titlefont': {'color': 'white',
                              'size': 12},
                'font': dict(family='sans-serif',
                             color='white',
                             size=15),
                'legend': {'orientation': 'h',
                           'bgcolor': '#262529',
                           'xanchor': 'center', 'x': 0.5, 'y': -0.7},
                'xaxis': dict(title='<b></b>',
                              color='white',
                              showline=True,
                              showgrid=True,
                              showticklabels=True,
                              linecolor='white',
                              linewidth=1,
                              ticks='outside',
                              tickfont=dict(
                                  family='Aerial',
                                  color='white',
                                  size=12
                              )),
                'yaxis': dict(title='<b></b>',
                              color='white',

                              showline=False,
                              showgrid=False,
                              showticklabels=True,
                              linecolor='white',
                              linewidth=1,
                              ticks='outside',
                              tickfont=dict(
                                  family='Aerial',
                                  color='white',
                                  size=12
                              )
                              )
            }
        }
    if radio_items == 'Flexibility':
        return {
            'data': [
                {'x': str(mcData['ID No.']),
                 'y': mcData_Flexibility,
                 'type': 'bar',
                 'name': 'Power',
                 'text': mcData['ID No.'],
                 'marker': {'color': '#ee82ee'},
                 },
            ],
            'layout': {
                'title': 'Flexibility Ranking ( Top5 )',
                'color': 'white',
                'plot_bgcolor': '#262529',
                'paper_bgcolor': '#262529',
                'titlefont': {'color': 'white',
                              'size': 12},
                'font': dict(family='sans-serif',
                             color='white',
                             size=15),
                'legend': {'orientation': 'h',
                           'bgcolor': '#262529',
                           'xanchor': 'center', 'x': 0.5, 'y': -0.7},
                'xaxis': dict(title='<b></b>',
                              color='white',
                              showline=True,
                              showgrid=True,
                              showticklabels=True,
                              linecolor='white',
                              linewidth=1,
                              ticks='outside',
                              tickfont=dict(
                                  family='Aerial',
                                  color='white',
                                  size=12
                              )),
                'yaxis': dict(title='<b></b>',
                              color='white',

                              showline=False,
                              showgrid=False,
                              showticklabels=True,
                              linecolor='white',
                              linewidth=1,
                              ticks='outside',
                              tickfont=dict(
                                  family='Aerial',
                                  color='white',
                                  size=12
                              )
                              )
            }
        }
    if radio_items == 'Endurance':
        return {
            'data': [
                {'x': str(mcData['ID No.']),
                 'y': mcData_Endurance,
                 'type': 'bar',
                 'name': 'Power',
                 'text': mcData['ID No.'],
                 'marker': {'color': '#ee82ee'},
                 },
            ],
            'layout': {
                'title': 'Endurance Ranking ( Top5 )',
                'color': 'white',
                'plot_bgcolor': '#262529',
                'paper_bgcolor': '#262529',
                'titlefont': {'color': 'white',
                              'size': 12},
                'font': dict(family='sans-serif',
                             color='white',
                             size=15),
                'legend': {'orientation': 'h',
                           'bgcolor': '#262529',
                           'xanchor': 'center', 'x': 0.5, 'y': -0.7},
                'xaxis': dict(title='<b></b>',
                              color='white',
                              showline=True,
                              showgrid=True,
                              showticklabels=True,
                              linecolor='white',
                              linewidth=1,
                              ticks='outside',
                              tickfont=dict(
                                  family='Aerial',
                                  color='white',
                                  size=12
                              )),
                'yaxis': dict(title='<b></b>',
                              color='white',

                              showline=False,
                              showgrid=False,
                              showticklabels=True,
                              linecolor='white',
                              linewidth=1,
                              ticks='outside',
                              tickfont=dict(
                                  family='Aerial',
                                  color='white',
                                  size=12
                              )
                              )
            }
        }


################## end of call back function (bar chart)


###########call back function of radar chart


@app.callback(Output('radar_chart', 'figure'),
            [Input('idno_inputed','value')])

def update_figure(idno_inputed):
        new_column = mcData.loc[mcData['ID No.'] == idno_inputed]
        newAgility = new_column['Agility.1'].tolist()
        newPower = new_column['Power.1'].tolist()
        newCoordination = new_column['Coordination.1'].tolist()
        newFlexibility = new_column['Flexibility.1'].tolist()
        newEndurance = new_column['Endurance.1'].tolist()
        r=[newAgility[0],
        newPower[0], 
        newCoordination[0], 
        newFlexibility[0], 
        newEndurance[0]]
        theta=['Agility',
        'Power',
        'Coordination',
        'Flexibility',
        'Endurance']
        d = {'r':r,'theta':theta}
        ssdf = pd.DataFrame(d)

        radar_chart1 = px.line_polar(ssdf, r= 'r', theta='theta',line_close=True)
        radar_chart1.update_traces(fill='toself')
        radar_chart1.update_layout(
            paper_bgcolor = '#262529',
            font_color = '#ee82ee',
            polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 5]
            )
        ))
        
        return radar_chart1


@app.callback(Output('bodyResult', 'children'),
            [Input('idno_inputed','value')])

def update_output(idno_inputed):
    new_column = mcData.loc[mcData['ID No.'] == idno_inputed]

    newHeight = new_column['Height'].tolist()
    newWeight = new_column['Weight'].tolist()
    newFatper = new_column['Fat %'].tolist()
    newFatmass = new_column['Fat mass'].tolist()
    newMusclemass = new_column['Muscle mass'].tolist()
    newBMI = new_column['BMI'].tolist()
    return html.H6("Your Weight:{}".format(newWeight[0]) + " kg"), html.H6("Your Height:{}".format(newHeight[0]) + " cm"), html.H6("Your Fat :{}".format(newFatper[0]) + " %"), html.H6("Your Fat mass:{}".format(newFatmass[0])), html.H6("Your Muscle mass:{}".format(newMusclemass[0])),html.H6("Your BMI:{}".format(newBMI[0]))

@app.callback(Output('muscle_radar_chart', 'figure'),
            [Input('idno_inputed','value')])

def update_figure(inputed_items):
        new_column = mcData.loc[mcData['ID No.'] == inputed_items]
        new_Trunk_Muscle_score = new_column['Trunk Muscle score'].tolist()
        new_LA_Muscle_score = new_column['LA Muscle score'].tolist()
        new_LL_Muscle_score = new_column['LL Muscle score'].tolist()
        new_RL_Muscle_score = new_column['RL Muscle score'].tolist()
        new_RA_Muscle_score = new_column['RA Muscle score'].tolist()
        r=[new_Trunk_Muscle_score[0],
        new_LA_Muscle_score[0], 
        new_LL_Muscle_score[0], 
        new_RL_Muscle_score[0], 
        new_RA_Muscle_score[0]]
        theta=['Trunk Muscle mass',
        'LA Muscle mass',
        'LL Muscle mass',
        'RL Muscle mass',
        'RA Muscle mass']
        d = {'r':r,'theta':theta}
        ssdf = pd.DataFrame(d)

        radar_chart2 = px.line_polar(ssdf, r= 'r', theta='theta',line_close=True)
        radar_chart2.update_traces(fill='toself')
        radar_chart2.update_layout(
            paper_bgcolor = '#262529',
            font_color = '#ee82ee',
            polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 3]
            )
        ))
            
        
        return radar_chart2


@app.callback(Output('fat_radar_chart', 'figure'),
            [Input('idno_inputed','value')])

def update_figure(inputed_items):
        new_column = mcData.loc[mcData['ID No.'] == inputed_items]
        new_Trunk_Fat_score = new_column['Trunk Fat % score'].tolist()
        new_LA_Fat_score = new_column['LA Fat % score'].tolist()
        new_LL_Fat_score = new_column['LL Fat % score'].tolist()
        new_RL_Fat_score = new_column['RL Fat % score'].tolist()
        new_RA_Fat_score = new_column['RA Fat % score'].tolist()
        r=[
            int(new_Trunk_Fat_score[0]),
            int(new_LA_Fat_score[0]), 
            int(new_LL_Fat_score[0]), 
            int(new_RL_Fat_score[0]), 
            int(new_RA_Fat_score[0])]
        theta=['Trunk Fat mass',
        'LA Fat mass',
        'LL Fat mass',
        'RL Fat mass',
        'RA Fat mass']
        d = {'r':r,'theta':theta}
        ssdf = pd.DataFrame(d)

        radar_chart3 = px.line_polar(ssdf, r= 'r', theta='theta',line_close=True)
        radar_chart3.update_traces(fill='toself')
        radar_chart3.update_layout(
            paper_bgcolor = '#262529',
            font_color = '#ee82ee',
            polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0,3]
            )
        ))
            
        
        return radar_chart3


if __name__ == '__main__':
    app.run_server(debug=False)