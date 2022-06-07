from flawsleuth.ai import predict, kalman_forecast
from flawsleuth.timeseries import fetch_data_frame, fetch_entity_series
import streamlit as st
import numpy as np
import time
from pdb import set_trace
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import count
# from data_process import Preprocessing
import pandas as pd
import plotly.express as px


def run():
    st.set_page_config ( layout="wide" )  # setting the display in the

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown ( hide_menu_style, unsafe_allow_html=True )
    st.title ( "SADS: Shop floor Anomaly Detection Service" )

    ########################## PREDICTION FORM #######################################
    SADA_settings = st.sidebar.form ( "SADS" )
    SADA_settings.title ( "SADS settings" )

    with st.sidebar.form ( "Models" ):
        st.title ( "SADS models" )
        train_model = st.form_submit_button ( "train model" )
        show_validation = st.checkbox ( 'show train and validation error' )

    max_x = SADA_settings.slider ( "Max length of the scatter", min_value=112, max_value=10 * 448, step=112,
                                   key='max_length' )


    model_choice = SADA_settings.selectbox (
        "Choose the model",
        ("Random Forest", "XGboost", "SVM", "Thresholding") )

    stop_bt, rerun_bt, show_bt = SADA_settings.columns ( (1, 1, 2) )
    SADS_submit = show_bt.form_submit_button ( "Predict" )
    stop_submit = stop_bt.form_submit_button ( "Stop" )
    rerun_submit = rerun_bt.form_submit_button ( "Rerun" )

    if rerun_submit:
        st.experimental_rerun ()
    if stop_submit:
        st.stop ()

    SADS_info = SADA_settings.expander ( "See explanation" )
    SADS_info.markdown ( """
    - Max length of the scatter: Maximum length of the plot
    - Stop: Stop the simulation
    -
    - Choose the model :
        - ***Repeat***: Model based on the repeated labeling method
            - Labeling strategy provided by the WAM technik
        - ***Iforest***: Model base on Isolation forest labeling method
            - Unsupervised labeling mechanisme for anomalies. A clustering based method

            """ )

    counter = count ( 0 )

    progress_bar = st.empty ()
    plot1, plot2 = st.columns ( (2, 1) )

    title_main = st.empty ()
    title_left, title_right = title_main.columns ( 2 )
    Main = st.empty ()
    day_left, time_right = Main.columns ( 2 )
    good_weld = day_left.empty ()
    bad_weld = time_right.empty ()
    weld = st.empty ()
    anomaly_plot = plot1.empty ()
    forecast_plot = st.empty ()
    py_chart = plot2.empty ()
    colum = ['Output Joules', 'Charge (v)', 'Residue (v)', 'Force L N', 'Force L N_1', 'anomaly']
    df = pd.DataFrame ( columns=colum )

    new_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> Bad Welding points </p> </h2></center>'
    title_right.markdown ( new_title, unsafe_allow_html=True )

    new_title = '<center> <h2> <p style="font-family:fantasy; color:#184aa1; font-size: 24px;"> Good Welding points </p> </h2></center>'
    title_left.markdown ( new_title, unsafe_allow_html=True )

    df_chart = pd.DataFrame ( columns=colum )

    # fig, ax = plt.subplots()
    fig = make_subplots ( rows=3, cols=1 )
    # fig  = go.Figure()

    if SADS_submit:
        if model_choice == 'Random Forest':
            model_type = 1
        elif model_choice == "XGboost":
            model_type = 2
        else :
            model_type = 0
        while True:  # stop_forecast == 'continue':
            fig = make_subplots ( rows=4, cols=1 )
            rr = fetch_entity_series ()
            answer = predict ( rr , model_type = model_type)
            time_count = next ( counter )
            if time_count == 0:
                df_chart = pd.DataFrame(rr.dict())
                df_chart['anomaly'] = answer.Label.value
            else:
                df_new = pd.DataFrame( rr.dict ())
                df_new['anomaly'] = answer.Label.value
                df_chart = pd.concat ( [df_chart, df_new], ignore_index=True )

            test = df_chart[-max_x:]
            good_weld.dataframe ( df_chart[df_chart['anomaly'] == False] )
            bad_weld.dataframe ( df_chart[df_chart['anomaly'] == True] )
            # set_trace()
            haha = test.copy ()
            haha['anomaly'] = haha['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )

            fig_px = px.scatter ( haha, y='joules', color='anomaly',
                                  color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
            fig_px.update_layout ( width=1500, height=250, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
            progress_bar.plotly_chart ( fig_px )


            fig.append_trace ( go.Scatter ( y=test['force_n'], x=test.index, mode='lines', name='Force_N',
                                            line_color='blue' if True else 'red' ), row=1, col=1, )
            fig.append_trace ( go.Scatter ( y=test['force_n_1'], x=test.index, mode='lines', name='Force_N_1',
                                            line_color='blue' if True else 'red' ), row=2, col=1, )
            fig.append_trace ( go.Scatter ( y=test['charge'], x=test.index, mode='lines', name='Charge',
                                            line_color='blue' if True else 'red' ), row=3, col=1, )
            fig.append_trace ( go.Scatter ( y=test['residue'], x=test.index, mode='lines', name='Residue',
                                            line_color='blue' if True else 'red' ), row=4, col=1, )
            fig.update_layout ( plot_bgcolor='rgb(206, 237, 240)' )

            anomaly_plot.plotly_chart ( fig, use_container_width=True )

            haha_pie = df_chart.copy ()
            haha_pie['anomaly'] = haha_pie['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )

            fig_pi = px.pie ( haha_pie, values='joules', hover_name='anomaly' , names='anomaly', title='the ratio of anomaly vs normal',
                              hole=.3, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) #, color_discrete_sequence=px.colors.sequential.RdBu)
            py_chart.plotly_chart ( fig_pi, use_container_width=True )

            # colors = ['red','blue']

            # fig11 = go.Figure ( data=[go.Pie ( labels= haha_pie.anomaly.values,
            #                                  values=  haha_pie.joules.values )] )
            # fig11.update_traces ( hoverinfo='label+percent', textinfo='value', textfont_size=20,
            #                     marker=dict ( colors=colors, line=dict ( color='#000000', width=2 ) ) )
            # py_chart.plotly_chart(fig11)
            del fig

            # #########################  TIME SERIE FORECASTING PLACEHOLDER   ###############
            ########### KALMAN FILTERING
            ########### DEEP LEARNING MODEL ( DNN , CNN, RNN )


            mu, sigma =  kalman_forecast (rr, forecast_steps=112)
            result = np.hstack([mu[:,4].reshape(-1,1), sigma[:,0,4].reshape(-1,1)])
            print(f"result { result.shape} -- {mu.shape} ---sigma {sigma.shape}")

            if time_count == 0:
                df_fif = pd.DataFrame(data=result, columns=['mu', 'std'])
            else:
                df_new = pd.DataFrame ( data=result, columns=['mu', 'std'] )
                df_fif = pd.concat([df_fif, df_new], ignore_index=True)
            df_forecate = df_fif[:max_x]
            # bad_weld.dataframe ( df_forecate)

            fig_forecast = go.Figure([
                    go.Scatter(
                        name='Forecast',
                        # x=df_fif['Time'],
                        y=df_forecate['mu'],
                        mode='lines',
                        marker=dict(color='red', size=2),
                        showlegend=True
                    ),
                    go.Scatter(
                        name='Measurement',
                        # x=df_forecate['Time'],
                        y=haha['joules'],
                        mode='lines',
                        marker=dict(color='green', size=2),
                        showlegend=True
                    ),
                    go.Scatter(
                        name='Upper Bound',
                        # x=df_forecate['Time'],
                        y=df_forecate['mu']+df_forecate['std'],
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=1),
                        showlegend=False
                    ),
                    go.Scatter(
                        name='Lower Bound',
                        # x=df_forecate['Time'],
                        y=df_forecate['mu']-df_forecate['std'],
                        marker=dict(color="#444"),
                        line=dict(width=1),
                        mode='lines',
                        fillcolor='rgba(68, 68, 68, 0.3)',
                        fill='tonexty',
                        showlegend=False
                    )
                ])
            # fig.update_layout(
            #     yaxis_title='Wind speed (m/s)',
            #     title='Continuous, variable value error bars',
            #     hovermode="x"
            # )
            # fig.show()
            forecast_plot.plotly_chart(fig_forecast, use_container_width=True)
            time.sleep ( 1 )