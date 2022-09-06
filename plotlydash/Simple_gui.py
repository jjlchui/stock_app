from dash import dcc, html,dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash
import datetime
import os
import pandas as pd
import pandas_ta as ta
from dash.exceptions import PreventUpdate
import numpy as np
from decimal import Decimal
from scipy.stats import linregress
import statsmodels.api as sm



def slope20(ser, n=20):
    #"function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n, len(ser)+1):
        y=ser[i-n:i]
        x=np.array(range(n))
        y_scales = (y-y.min())/(y.max()-y.min())
        x_scales = (x-x.min())/(x.max()-x.min())
        x_scales = sm.add_constant(x_scales)
        model = sm.OLS(y_scales, x_scales)
        results = model.fit()
        
        slopes.append(results.params[-1])
        #results.summary()
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

    
def ma_strategy(data):
    df = data.copy()
    
    
    df['old_slope20'] = slope20(df.Close)
    df['old_ma10_50'] = slope20(df.MA50)
    
    slopes = df['Close'].rolling(10).apply(lambda s: linregress(s.reset_index())[0])
    df['slopes'] = (np.rad2deg(np.arctan(np.array(slopes))))
    
   
   
    df['old_slope20s1'] = df['old_slope20'].shift(1)
    df['slope_chg_dn'] = np.where(((df.old_slope20 < 0) & (df.old_slope20s1 > 0)),"y", "NaN")
    df['slope_chg_up'] = np.where(((df.old_slope20 > 0) & (df.old_slope20s1 < 0)),"y", "NaN")
    
    #df['slopess1p'] = df['slopes'].shift(1)
    #df['slope_chg_dn'] = np.where(((df.slopes < 0) & (df.slopess1p > 0)),"y", "NaN")
    #df['slope_chg_up'] = np.where(((df.slopes > 0) & (df.slopess1p < 0)),"y", "NaN")
    
    df['slope_chg_up_s1'] = df['slope_chg_up'].shift(1)
    df['slope_chg_dn_s1'] = df['slope_chg_dn'].shift(1)
    
    
    df['ma1s1'] = df['MA10'].shift(-1)
    df['ma1s2'] = df['MA10'].shift(-2)
    df['ma2s1'] = df['MA20'].shift(-1)
    df['ma2s2'] = df['MA20'].shift(-2)
    
    df['buy_pt'] = np.where((((df.MA10 <= df.MA20) & (df.ma1s1 > df.ma2s1)) | 
                ((df.MA10 >= df.MA20) & (df.ma1s1 < df.ma2s1))),
                "y", "NaN")
    df['sell_pt'] =  np.where((((df.MA10 <= df.MA20) & (df.ma1s1 > df.ma2s1)) | 
                ((df.MA10 >= df.MA20) & (df.ma1s1 < df.ma2s1))),       
                "y", "NaN")
    
    df['buy_pt_s1'] = df['buy_pt'].shift(1)
    df['sell_pt_s1'] = df['sell_pt'].shift(1)

    #......BUY......
    
    Conditional_Buy = np.where(
                                ((df.MA10 >= df.MA20) & (df.MA20 >= df.MA50)) &
                                (df['old_ma10_50'] > 0) 
                                ,df['Close'], "NaN"
                                )
    Buy = np.where(
                (df['buy_pt'] == "y") &
                (df['old_slope20'] > 0) &
                ((df.MA10 >= df.MA20) | (df.MA20 >= df.MA50))
                #(Conditional_Buy == df.Close)
                 ,df['Close'], "NaN")
    

    
    # for calculate unique sell
    # sell compare with the buy price
    buy_price=np.where(Buy =='NaN', "NaN", Buy)
    #buy_price = buy_price.T
    df.reset_index()
    df['buy_price'] = buy_price
    df['buy_price'].fillna(method='ffill', inplace=True)
    
    # total Buy/Buy count
    Buy_list = list(Buy)
    df_Buy = pd.DataFrame(Buy_list, columns=['Buy'])
    #df_Buy = df_Buy.T     
    df = pd.concat([df, df_Buy], axis=1)
    df['Buy_zero'] = np.where(df.Buy == "NaN", 0, df.Buy)
    df['tot_buy'] = np.add.accumulate(df['Buy_zero'].astype(float))
    
    df['Buy_count_tmp'] = np.where(df['Buy'] == 'NaN', 0, 1)
    df['Buy_count'] = np.add.accumulate(df['Buy_count_tmp'])
            
    

    #......SELL......
              
    Sell = np.where(
                    (df['sell_pt'] == "y") & 
                    (df['old_slope20'] < 0)  &
                    ((df.MA10 <= df.MA20) | (df.MA20 <= df.MA50))
                      ,df.Close, "NaN")
    
    
    # avoid SELL without BUY
    """
    df['Sell_count_rev'] = np.where((df.Sell != "NaN")  , -1, 0)
    df['Buy_count_rev'] = np.where(df['Sell_count_rev'] !=0, df['Sell_count_rev'], df['Buy_count_rev'])

    
    df['Buy-Sellff']=np.where(df['Buy_count_rev'] == 0, np.NaN, df['Buy_count_rev'])
    df['Buy-Sellff'].fillna(method='ffill', inplace=True)
    df['Buy-Sellffs1'] = df['Buy-Sellff'].shift(1)
    df['Sell_single'] = np.where((df.Buy.astype(float) != df.Close.astype(float)) & 
                                 (df['Buy-Sellff'] + df['Buy-Sellffs1'] == 0), 
                                 df.Close, "NaN")
    
    # total SELL/count
    df['Sell_zero'] = np.where(df['Sell_single'] == "NaN", 0, df['Sell_single'])
    df['tot_sell'] = np.add.accumulate(df['Sell_zero'].astype(float)) 
    df['Sell_count_tmp'] = np.where((df['Sell_single'] != "NaN"), 1, 0)
    df['Sell_count'] = np.add.accumulate(df['Sell_count_tmp']) 
    """
    
    Sell_list = list(Sell)
    df_Sell = pd.DataFrame(Sell_list, columns=['Sell'])
    #df_Sell = df_Sell.T
    df = pd.concat([df, df_Sell], axis=1)
    df['Sell_zero'] = np.where(df.Sell == "NaN", 0, df.Sell)
    df['tot_sell'] = np.add.accumulate(df['Sell_zero'].astype(float))
    
    
    df['Sell_count_tmp'] = np.where(df['Sell'] == 'NaN', 0, 1)
    df['Sell_count'] = np.add.accumulate(df['Sell_count_tmp'])
    
      
    df.Sell = pd.to_numeric(df.Sell, errors='coerce') 
    
    # total SELL dep on BUY/count

    Sell_count = abs(df['Sell_count'] )
    Buy_count = abs(df['Buy_count'])
    df.to_csv("ma_new.csv")
    return Buy, Sell, df.tot_buy, df.tot_sell, Buy_count, Sell_count
        
 



def create_dash(flask_app):
    
        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/ma/"), prevent_initial_callbacks=True)

        GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)
        
        
        app.layout = html.Div([
            
            ##### Store data
                dcc.Store(id='df_value'),
                dcc.Store(id='buy'),
                dcc.Store(id='sell'),
                dcc.Store(id='buy_nplist'),
                dcc.Store(id='sell_nplist'),

              #### Header 
                 html.Div([
                 html.Div([
                     html.Img(src = app.get_asset_url('logo.jpeg'),
                              style = {'height': '30px'},
                              className = 'title_image'),
                     html.H6('Have Fun !!! with stock ...',
                             style = {'color': 'white'},
                             className = 'title'),
         
                 ], className = 'logo_title'),
                 
                 html.Div([
                   html.Div([
                       html.Div([
                           html.P('Buy:',
                                   style = {'color': 'white', 'font-size':'12px'},
                                   className = 'stock_label'),
                           html.Div(id = 'tot_buy',
                                   style = {'color': 'white', },
                                   className = 'stock_score'),
                       ],className = 'stock_score_label'),
                       html.P(['  /  '], className = 'stock_score'),
                       html.Div(id = 'buy_count',
                               style = {'color': 'white', },
                               className = 'stock_score'),
                       html.P([' ('], className = 'stock_score'),
                       html.Div(id = 'tot_profit',
                               style = {'color': 'white', },
                               className = 'stock_ave'),
                       html.P([')'], className = 'stock_score'),
                        
                   ], className = 'buy_sell_b'),
                    
                   html.Div([
                   html.Div([
                       html.P('Sell:',
                               style = {'color': 'white', 'font-size':'12px'},
                               className = 'stock_label'),
                       html.Div(id = 'tot_sell',
                               style = {'color': 'white'},
                               className = 'stock_score'),
                       ],className = 'stock_score_label'),
                       html.P(['  /  '], className = 'stock_score'),
                       html.Div(id = 'sell_count',
                               style = {'color': 'white', },
                               className = 'stock_score'),
                       html.P([' ('], className = 'stock_score'),
                       html.Div(id = 'tot_long',
                               style = {'color': 'white', },
                               className = 'stock_ave'),
                       html.P([')'], className = 'stock_score'),
                   ], className = 'buy_sell_s'),
                  
                 html.Div([
                   html.Div([
                       html.P('Profit:',
                               style = {'color': 'white', 'font-size':'12px'},
                               className = 'stock_label'),
                       html.Div(id = 'profit'),
                       ],className = 'stock_score_label'),
                   html.Div([
                       html.Div(id = 'Img'),
                       ],className = 'profit_img'),
                   ], className = 'buy_sell_p'),
                ], className = 'stock_score_container'),
                                               
                 
                 html.H6(id = 'get_date_time',
                         style = {'color': 'white'},
                         className = 'adjust_date_time'),
                 ], className = 'title_date_time_container'),
         
                 html.Div([
                 dcc.Interval(id = 'update_date_time',
                              interval = 1000,
                              n_intervals = 0)
                 ]),
            ##### button   
        
            
        
            
            ##### Graph Display
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(id = 'price_candlesticker', animate=False, 
                                    style={'width': 'auto', 
                                           'height': '60vh',
                                           'border': '1px #5c5c5c solid',
                                           'margin-top' : "40px"},
                                    config = {'displayModeBar': True, 'responsive': True},
                                    className = 'chart_width'),
                                
                            ]),
                        ]),
                    
                        html.Div([
                            html.Title('MACD',
                            style = {'color': 'black'},
                            ),
                            
                            dcc.Graph(id = 'price_macd', animate=False, 
                                    style={'width': 'auto', 
                                           'height': '20vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),
                        
                        html.Div([
                            html.Title('Volume',
                            style = {'color': 'black'},
                            ),
                            
                            dcc.Graph(id = 'price_vol', animate=False, 
                                    style={'width': 'auto', 
                                           'height': '20vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),
                        
                        html.Div([
                            html.Title('RSI',
                            style = {'color': 'black'},
                            ),
                            
                            dcc.Graph(id = 'price_rsi', animate=False, 
                                    style={'width': 'auto', 
                                           'height': '10vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),
                        
                        html.Div([
                            html.Title('profitchart',
                            style = {'color': 'black'},
                            ),
                            html.Div(id='table-container', className ='table_style',

                                    ),
                            ]),

        
                        dcc.Interval(id = 'update_value',
                                         interval = int(GRAPH_INTERVAL) ,
                                         n_intervals = 0)
                    ]),
        
            
                
        ])
        
        @app.callback(Output('df_value', 'data'), Input('update_value', 'n_intervals'))
        
        def update_df(n_intervals): 
                if n_intervals == 0:
                    raise PreventUpdate
                else:
                    
                    p_filename = "_out_stock_data.csv"
                    
                    #time_stamp = datetime.datetime.now() - datetime.timedelta(hours=13)
                    time_stamp =  datetime.datetime.now().strftime('%Y-%m-%d')
                    #time_stamp = datetime.now().strftime('%Y-%m-%d')
                    filename = time_stamp+" NQ=F USTime"+p_filename
                    
                    #filename = "2022-08-31 NQ=F USTime_out_stock_data.csv"
                    cwd = os.getcwd()
                    path = os.path.dirname(cwd)
                
                    file_path = path + "\\stock\\data\\"
                    file = os.path.join(file_path, filename)
                    #df = pd.read_csv(os.path.basename(filename))
                    df = pd.read_csv(file)
                    df.columns =['Datetime','Open','High','Low','Close', 'Volume']
                    df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
                    df.ta.rsi(close=df['Close'], length=14, append=True, signal_indicators=True, xa=70, xb=30)
                    
        
                return df.to_dict('records')
        

        @app.callback(Output('get_date_time', 'children'),
                      [Input('update_date_time', 'n_intervals')])
        def live_date_time(n_intervals):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                now = datetime.datetime.now()
                dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        
            return [
                html.Div(dt_string)
            ]
        
        @app.callback([
                Output('price_candlesticker', 'figure'),
                Output('tot_buy', 'children'),
                Output('tot_sell', 'children'),
                Output('buy_count', 'children'),
                Output('sell_count', 'children'),
                Output('buy', 'children'),
                Output('sell', 'children'),
                ],
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])
        
        def update_graph(n_intervals, data):
        
            if n_intervals == 0:
                raise PreventUpdate
            else:
                
                df = pd.DataFrame(data)
                
                df['MA10'] = df.Close.rolling(10).mean()
                df['MA20'] = df.Close.rolling(20).mean()
                df['MA50'] = df.Close.rolling(50).mean()
                
                max = df.Close.max()
                max_ind = df[['Close']].idxmax()
                min = df.Close.min()
                min_ind = df[['Close']].idxmin()
                max_20 = df.Close.tail(20).max()
                max_20_ind = df.Close.tail(20).idxmax()
                min_20 = df.Close.tail(20).min()
                min_20_ind = df.Close.tail(20).idxmin()
                
                
                buy, sell, tot_buy, tot_sell, buy_count, sell_count = ma_strategy(df)
                
                tt_buy = tot_buy.iloc[-1]
                tt_sell =  tot_sell.iloc[-1]
                bbuy_count =  buy_count.iloc[-1]
                ssell_count =  sell_count.iloc[-1]
                
                ave_b = tt_buy / bbuy_count
                ave_b = np.around(ave_b, 2)


                ave_s = tt_sell / ssell_count
                ave_s = np.around(ave_s, 2)
                
                
                buyitem = np.array(buy).tolist()
                        
                sellitem = np.array(sell).tolist()
                
 
                
                figure = go.Figure(
                    data = [go.Scatter(x=df.index, y=df.Close, line=dict(color='#fc0080', width=2), 
                            name = 'Close',
                            hoverinfo = 'text',
                            hovertext =
                            '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                            '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.Close] + '<br>'),
                    
                            go.Candlestick(x=df.index,
                                open=df.Open,
                                high=df.High,
                                low=df.Low,
                                close=df.Close),  
                            
                            go.Scatter(x=df.index, y=df.MA10, line=dict(color='#AA76DB', width=1), 
                                name = 'MA10',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA10] + '<br>'),
                            go.Scatter(x=df.index, y=df.MA20, line=dict(color='#2ed9ff', width=1),
                                name = 'MA20',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA20] + '<br>'),
                            go.Scatter(x=df.index, y=df.MA50, line=dict(color='#b6e880', width=1),
                                name = 'MA50',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA50] + '<br>'),
                        
            
                go.Scatter(x=[0, len(df)], 
                         y=[min,min], name='min',
                         line=dict(color='rgba(152,78,163,0.5)', width=1, dash='dash'),
                         ),
            
                go.Scatter(x=[0, len(df)], 
                         y=[max,max], name='max',
                         line=dict(color='rgba(152,78,163,0.5)', width=1, dash='dash'),
                         ),
            
                go.Scatter(x=[0, len(df)], 
                         y=[min_20,min_20], name='min20',
                         line=dict(color='rgba(124,124,124,0.5)', width=1, dash='dash'),
                         ),
            
                go.Scatter(x=[0, len(df)], 
                         y=[max_20,max_20], name='max20',
                         line=dict(color='rgba(124,124,124,0.5)', width=1, dash='dash'),
                         ),

                go.Scatter(x=df.index, y=buyitem, name="UP", mode="markers",
                              marker=dict(
                              symbol="5" ,
                              color="#FDDE00",
                              size=12)), 

                go.Scatter(x=df.index, y=sellitem, name="DOWN", mode="markers",
                              marker=dict(
                              symbol="6" ,
                              color="#76C7DB",
                              size=12)),

                                            
                    ],
                    
                    
                    layout = go.Layout(
                                   xaxis_rangeslider_visible=False,
                                   hovermode = 'closest',
                                   uirevision = 'dataset',
                                   margin = dict(t = 35  , r = 0, l = 60, b=20),
                                   xaxis = dict(autorange= True,
                                                color = 'black',
                                                matches = 'x',
                                                showspikes = True,
                                                showline = True,
                                                showgrid = True,
                                                linecolor = 'black',
                                                linewidth = 1,
                                                ticks = 'outside',
                                                tickfont = dict(
                                                    family = 'Arial',
                                                    size = 10,
                                                    color = 'black'
                                                )),
                                   yaxis = dict(autorange= True,
                                                showspikes = True,
                                                showline = True,
                                                showgrid = False,
                                                linecolor = 'black',
                                                linewidth = 1,
                                                ticks = 'outside',
                                                tickfont = dict(
                                                    family = 'Arial',
                                                    size = 10,
                                                    color = 'black'
                                       )),
        
                                   font = dict(
                                       family = 'sans-serif',
                                       size = 10,
                                       color = 'black'
                                   )
                                   
                                   )
                 )
                return [figure, tt_buy, tt_sell, bbuy_count, ssell_count, buy, sell] 
            
            


        @app.callback(
                    Output('price_macd', 'figure'),
                    [Input('update_value', 'n_intervals'),
                     Input('df_value', 'data')])
        
        def update_macd(n_intervals, data): 
                if n_intervals == 0:
                    raise PreventUpdate
                else:
                    df = pd.DataFrame(data)
                    
                    return{
                        "data" : [
                                    go.Scatter(
                                            x=df.index,
                                            y=df['MACD_12_26_9'],
                                            line=dict(color='#ff9900', width=1),
                                            name='macd',
                                            # showlegend=False,
                                            legendgroup='2',),
        
                                    go.Scatter(
                                            x=df.index,
                                            y=df['MACDs_12_26_9'],
                                            line=dict(color='#000000', width=1),
                                            # showlegend=False,
                                            legendgroup='2',
                                            name='signal'),
                                    go.Bar(
                                            x=df.index,
                                            y=df['MACDh_12_26_9'],
                                        marker_color=np.where(df['MACDh_12_26_9'] < 0, '#000', '#ff9900'),
                                        name='bar'),
                                  
                                    go.Scatter(x=[0, len(df)], 
                                         y=[-5,-5], showlegend=False,
                                         line=dict(color='#000000', width=1, dash='dash'),
                                 ),
        
        
                                 ],
        
                        "layout" : go.Layout(
                                   hovermode = 'x unified',
                                   uirevision = 'dataset',
                                   margin = dict(t = 0  , r = 0, l = 60, b=0),
                        )
                              
                    }
                    
                
        @app.callback(
                Output('price_rsi', 'figure'),
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])
        
        def update_rsi(n_intervals, data):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                
                df = pd.DataFrame(data)

                return {'data': [go.Scatter(x=df.index, y=df.RSI_14, name='RSI',
                                 line=dict(color='#000000', width=1),
                                 # showlegend=False,
                                 legendgroup='3'),
                                 
                                 go.Scatter(x=[0, len(df)], 
                                 y=[20,20], name='OB(20)',
                                 line=dict(color='#f705c3', width=2, dash='dash'),
                                 ),
                        
                                 go.Scatter(x=[0, len(df)], 
                                 y=[80,80], name='OS(80)',
                                 line=dict(color='#f705c3', width=2, dash='dash'),
                                 ),
                                 
                                 go.Scatter(x=[0, len(df)], 
                                     y=[50,50], showlegend=False,
                                     line=dict(color='#000000', width=1, dash='dash')),
                                 ],
                                
                         'layout': go.Layout(
                                            hovermode = 'x unified',
                                            uirevision = 'dataset', 
                                            margin = dict(t = 0  , r = 0, l = 60, b=0),
        
                            )
                                
                       }
            
        @app.callback(
                Output('price_vol', 'figure'),
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])
        
        def update_vol(n_intervals, data):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                
                df = pd.DataFrame(data)
                delta_vol = df.Volume - df.Volume.shift(1)

                return {'data': [go.Bar(x=df.index, y=delta_vol, name='volume',
                                  showlegend=True,
                                 legendgroup='4'),
                            
                                 ],
                                
                         'layout': go.Layout(
                                            hovermode = 'x unified',
                                            uirevision = 'dataset', 
                                            margin = dict(t = 0  , r = 0, l = 60, b=0),
        
                            )
                                
                       }
            
        @app.callback(
                [
                 Output('tot_profit', 'children'),
                 Output('tot_long', 'children'),
                 Output('buy_nplist', 'children'),
                 Output('sell_nplist', 'children'),
                ],
                [Input('update_value', 'n_intervals'),
                State('buy', 'children'),
                State('sell', 'children'),
                 ])
            
        def update_profitchart(n_intervals, buy, sell):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                
                buyitem = np.array(buy).tolist()                    
                sellitem = np.array(sell).tolist()
                
                ###### Calculate profit START
     
                buy_list = []
                buy_idx = []
                for idx, xs in enumerate(buyitem):
                    if xs != 'NaN':
                        buy_list.append(xs)
                        buy_idx.append(idx)
                
                sell_list = []
                sell_idx = []
                for idx, xs in enumerate(sellitem):
                    if xs != 'NaN':
                        sell_list.append(xs)
                        sell_idx.append(idx)
                
                buy_nplist = np.column_stack((buy_list, buy_idx))
                sell_nplist =  np.column_stack((sell_list, sell_idx))
                
                min_len = np.minimum(len(buy_nplist), len(sell_nplist))
                
                
                tot_profit = 0 
                buy_profit = 0
                tot_long = 0
                long_profit = 0
                

                for i in range (min_len):
                    #buy-> sell

                    if(np.minimum(int(buy_nplist[i,1]), int(sell_nplist[i,1])) == int(buy_nplist[i,1])):
                        buy_profit = float(sell_nplist[i,0]) - float(buy_nplist[i,0])
                        tot_profit += buy_profit
                        print("buy_profit_idx",i," ",(buy_nplist[i,1]),"<", sell_nplist[i,1],buy_profit)
                        print("buy_profit",i," ",(buy_nplist[i,0]),"<", sell_nplist[i,0],buy_profit, tot_profit)
                    else:
                        long_profit = float(sell_nplist[i,0]) - float(buy_nplist[i,0])
                        tot_long +=  long_profit
                        print("long_profit_idx",i," ",buy_nplist[i,1],">", (sell_nplist[i,1]), long_profit)
                        print("long_profit",i," ",buy_nplist[i,0],">", (sell_nplist[i,0]), long_profit, tot_long)
                        
                    #sell -> buy

                    print("total", tot_profit, tot_long)
                    
                
                ###### Calculate profit FINISH            
 
                return [tot_profit, tot_long, buy_nplist, sell_nplist]

        @app.callback([Output('profit', 'children'),
                       Output('Img', 'children'),
                       ],
                      [Input('update_value', 'n_intervals'),
                       Input('tot_profit', 'children'),
                       Input('tot_long', 'children'),
                       ])
        
        def profit(n_intervals, tot_profit, tot_long):
            
            if n_intervals == 0:
                raise PreventUpdate
            else:
                sell_gain = Decimal(str(tot_long)).quantize(Decimal('0.01'))
                buy_gain = Decimal(str(tot_profit)).quantize(Decimal('0.01'))
                profit = buy_gain + sell_gain
                
                
                if (profit > 0):
                        return [html.H6('${0:,.2f}'.format(profit), style = {'color': '#84de02', 'fontSize' : 17}),
                                html.Img(id = "Img",src = app.get_asset_url('money-bag.png'),
                                     style = {'height': '30px'},
                                     className = 'coin'), ]
                    
                else:
                        return [
                            html.H6('${0:,.2f}'.format(profit), style = {'color': '#f20540', 'fontSize' : 17}),
                            html.Img(id = "Img",src = app.get_asset_url('cry.png'),
                                     style = {'height': '30px'},
                                     className = 'coin'), ]
                    
                    
        @app.callback(
                Output('table-container', 'children'),
                [Input('update_value', 'n_intervals'),
                 Input('buy_nplist', 'children'),
                 Input('sell_nplist', 'children'),
                 ])
        
        def update_table(n_intervals, buy_nplist, sell_nplist):
            
            if n_intervals == 0:
                raise PreventUpdate
            else:
                df_b = pd.DataFrame(buy_nplist, columns=['buy', 'b_idx'])
                df_s = pd.DataFrame(sell_nplist, columns=['sell', 's_idx'])

                df_c = pd.concat([df_b, df_s], axis=1)
                df_c = df_c.fillna(0)
                df_c.b_idx = df_c.b_idx.astype(int)
                df_c.s_idx = df_c.s_idx.astype(int)
                df_c.sell = df_c.sell.astype(float)
                df_c.buy = df_c.buy.astype(float)
                
                profit = np.where(df_c.b_idx > df_c.s_idx, 
                                      df_c.sell - df_c.buy,
                                      df_c.sell - df_c.buy, 
                                       )
                df_c['profit'] = profit
                  
                

                
                
                
                
                df_c['profit'] = profit
                

                if len(df_s) > len(df_b):
                    styles = [
                              {'if': {'column_id': 'profit', 'filter_query': '{buy} != 0 && {b_idx} > {s_idx}'}, 'color': 'tomato','fontWeight':'normal'},
                              {'if': {'column_id': 'profit', 'filter_query': '{buy} != 0 && {b_idx} < {s_idx}'}, 'color': '#39CCCC','fontWeight':'normal'},
                              {"if": {'column_id': 'buy', "filter_query": "{buy} != 0 && {b_idx} > {s_idx}" }, "backgroundColor": "yellow", 'color': 'black',},
                              {"if": {'column_id': 'sell', "filter_query": "{buy} != 0 && {b_idx} < {s_idx}" }, "backgroundColor": "B10DC9",'color': 'black',},
                              ]
                else:
                    styles = [
                              {'if': {'column_id': 'profit', 'filter_query': '{sell} != 0 && {b_idx} > {s_idx}'}, 'color': 'tomato','fontWeight':'normal'},
                              {'if': {'column_id': 'profit', 'filter_query': '{sell} != 0 && {b_idx} < {s_idx}'}, 'color': '#39CCCC','fontWeight':'normal'},
                              {"if": {'column_id': 'buy', "filter_query": "{sell} != 0 && {b_idx} > {s_idx}" }, "backgroundColor": "yellow", 'color': 'black',},
                              {"if": {'column_id': 'sell', "filter_query": "{sell} != 0 && {b_idx} < {s_idx}" }, "backgroundColor": "B10DC9",'color': 'black',},
                              ]

                
                
                layout = html.Div([dash_table.DataTable(
                        columns=[{'name': i, 'id': i} for i in df_c.columns],
                        data=df_c.to_dict('records'),
                        style_cell={'font-family': 'sans-serif'},
                        
                        style_header={
                            'backgroundColor': 'rgb(10, 10, 10)',
                            'border': '1px solid black',
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        style_data={
                            'backgroundColor': 'rgb(60, 60, 60)',
                            'border': '1px solid grey', 
                            'color': 'rgb(255,255,255, 0.5)',
                            'fontWeight': 'normal'
                        },                        
                        style_data_conditional=styles,
                        )])
                                
                return layout       

 
        return(app)

"""
if __name__ == '__main__':
    app.run_server()
    
"""
