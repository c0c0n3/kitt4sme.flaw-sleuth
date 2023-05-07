# from flawsleuth.ai import predict, kalman_forecast
# from flawsleuth.timeseries import fetch_data_frame, fetch_entity_series
# import streamlit as st
# import numpy as np
# import time
# from pdb import set_trace
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from itertools import count
# # from data_process import Preprocessing
# import pandas as pd
# import plotly.express as px


# def run():
#     st.set_page_config ( layout="wide" )  # setting the display in the

#     hide_menu_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
#     st.markdown ( hide_menu_style, unsafe_allow_html=True )
#     st.title ( "SADS: Shop floor Anomaly Detection Service" )

#     ########################## PREDICTION FORM #######################################
#     SADA_settings = st.sidebar.form ( "SADS" )
#     SADA_settings.title ( "SADS settings" )

#     with st.sidebar.form ( "Models" ):
#         st.title ( "SADS models" )
#         train_model = st.form_submit_button ( "train model" )
#         show_validation = st.checkbox ( 'show train and validation error' )

#     max_x = SADA_settings.slider ( "Max length of the scatter", min_value=112, max_value=10 * 448, step=112,
#                                    key='max_length' )


#     model_choice = SADA_settings.selectbox (
#         "Choose the model",
#         ("Random Forest", "XGboost", "SVM", "Thresholding") )

#     stop_bt, rerun_bt, show_bt = SADA_settings.columns ( (1, 1, 2) )
#     SADS_submit = show_bt.form_submit_button ( "Predict" )
#     stop_submit = stop_bt.form_submit_button ( "Stop" )
#     rerun_submit = rerun_bt.form_submit_button ( "Rerun" )

#     if rerun_submit:
#         st.experimental_rerun ()
#     if stop_submit:
#         st.stop ()

#     SADS_info = SADA_settings.expander ( "See explanation" )
#     SADS_info.markdown ( """
#     - Max length of the scatter: Maximum length of the plot
#     - Stop: Stop the simulation
#     -
#     - Choose the model :
#         - ***Repeat***: Model based on the repeated labeling method
#             - Labeling strategy provided by the WAM technik
#         - ***Iforest***: Model base on Isolation forest labeling method
#             - Unsupervised labeling mechanisme for anomalies. A clustering based method

#             """ )

#     counter = count ( 0 )

#     progress_bar = st.empty ()
#     plot1, plot2 = st.columns ( (2, 1) )

#     title_main = st.empty ()
#     title_left, title_right = title_main.columns ( 2 )
#     Main = st.empty ()
#     day_left, time_right = Main.columns ( 2 )
#     good_weld = day_left.empty ()
#     bad_weld = time_right.empty ()
#     weld = st.empty ()
#     anomaly_plot = plot1.empty ()
#     forecast_plot = st.empty ()
#     py_chart = plot2.empty ()
#     colum = ['Output Joules', 'Charge (v)', 'Residue (v)', 'Force L N', 'Force L N_1', 'anomaly']
#     df = pd.DataFrame ( columns=colum )

#     new_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> Bad Welding points </p> </h2></center>'
#     title_right.markdown ( new_title, unsafe_allow_html=True )

#     new_title = '<center> <h2> <p style="font-family:fantasy; color:#184aa1; font-size: 24px;"> Good Welding points </p> </h2></center>'
#     title_left.markdown ( new_title, unsafe_allow_html=True )

#     df_chart = pd.DataFrame ( columns=colum )

#     # fig, ax = plt.subplots()
#     fig = make_subplots ( rows=3, cols=1 )
#     # fig  = go.Figure()

#     if SADS_submit:
#         if model_choice == 'Random Forest':
#             model_type = 1
#         elif model_choice == "XGboost":
#             model_type = 2
#         else :
#             model_type = 0
#         while True:  # stop_forecast == 'continue':
#             fig = make_subplots ( rows=4, cols=1 )
#             rr = fetch_entity_series ()
#             answer = predict ( rr , model_type = model_type)
#             time_count = next ( counter )
#             if time_count == 0:
#                 df_chart = pd.DataFrame(rr.dict())
#                 df_chart['anomaly'] = answer.Label.value
#             else:
#                 df_new = pd.DataFrame( rr.dict ())
#                 df_new['anomaly'] = answer.Label.value
#                 df_chart = pd.concat ( [df_chart, df_new], ignore_index=True )

#             test = df_chart[-max_x:]
#             good_weld.dataframe ( df_chart[df_chart['anomaly'] == False] )
#             bad_weld.dataframe ( df_chart[df_chart['anomaly'] == True] )
#             # set_trace()
#             haha = test.copy ()
#             haha['anomaly'] = haha['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )

#             fig_px = px.scatter ( haha, y='joules', color='anomaly',
#                                   color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
#             fig_px.update_layout ( width=1500, height=250, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
#             progress_bar.plotly_chart ( fig_px )


#             fig.append_trace ( go.Scatter ( y=test['force_n'], x=test.index, mode='lines', name='Force_N',
#                                             line_color='blue' if True else 'red' ), row=1, col=1, )
#             fig.append_trace ( go.Scatter ( y=test['force_n_1'], x=test.index, mode='lines', name='Force_N_1',
#                                             line_color='blue' if True else 'red' ), row=2, col=1, )
#             fig.append_trace ( go.Scatter ( y=test['charge'], x=test.index, mode='lines', name='Charge',
#                                             line_color='blue' if True else 'red' ), row=3, col=1, )
#             fig.append_trace ( go.Scatter ( y=test['residue'], x=test.index, mode='lines', name='Residue',
#                                             line_color='blue' if True else 'red' ), row=4, col=1, )
#             fig.update_layout ( plot_bgcolor='rgb(206, 237, 240)' )

#             anomaly_plot.plotly_chart ( fig, use_container_width=True )

#             haha_pie = df_chart.copy ()
#             haha_pie['anomaly'] = haha_pie['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )

#             fig_pi = px.pie ( haha_pie, values='joules', hover_name='anomaly' , names='anomaly', title='the ratio of anomaly vs normal',
#                               hole=.3, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) #, color_discrete_sequence=px.colors.sequential.RdBu)
#             py_chart.plotly_chart ( fig_pi, use_container_width=True )

#             # colors = ['red','blue']

#             # fig11 = go.Figure ( data=[go.Pie ( labels= haha_pie.anomaly.values,
#             #                                  values=  haha_pie.joules.values )] )
#             # fig11.update_traces ( hoverinfo='label+percent', textinfo='value', textfont_size=20,
#             #                     marker=dict ( colors=colors, line=dict ( color='#000000', width=2 ) ) )
#             # py_chart.plotly_chart(fig11)
#             del fig

#             # #########################  TIME SERIE FORECASTING PLACEHOLDER   ###############
#             ########### KALMAN FILTERING
#             ########### DEEP LEARNING MODEL ( DNN , CNN, RNN )


#             mu, sigma =  kalman_forecast (rr, forecast_steps=112)
#             result = np.hstack([mu[:,4].reshape(-1,1), sigma[:,0,4].reshape(-1,1)])
#             print(f"result { result.shape} -- {mu.shape} ---sigma {sigma.shape}")

#             if time_count == 0:
#                 df_fif = pd.DataFrame(data=result, columns=['mu', 'std'])
#             else:
#                 df_new = pd.DataFrame ( data=result, columns=['mu', 'std'] )
#                 df_fif = pd.concat([df_fif, df_new], ignore_index=True)
#             df_forecate = df_fif[:max_x]
#             # bad_weld.dataframe ( df_forecate)

#             fig_forecast = go.Figure([
#                     go.Scatter(
#                         name='Forecast',
#                         # x=df_fif['Time'],
#                         y=df_forecate['mu'],
#                         mode='lines',
#                         marker=dict(color='red', size=2),
#                         showlegend=True
#                     ),
#                     go.Scatter(
#                         name='Measurement',
#                         # x=df_forecate['Time'],
#                         y=haha['joules'],
#                         mode='lines',
#                         marker=dict(color='green', size=2),
#                         showlegend=True
#                     ),
#                     go.Scatter(
#                         name='Upper Bound',
#                         # x=df_forecate['Time'],
#                         y=df_forecate['mu']+df_forecate['std'],
#                         mode='lines',
#                         marker=dict(color="#444"),
#                         line=dict(width=1),
#                         showlegend=False
#                     ),
#                     go.Scatter(
#                         name='Lower Bound',
#                         # x=df_forecate['Time'],
#                         y=df_forecate['mu']-df_forecate['std'],
#                         marker=dict(color="#444"),
#                         line=dict(width=1),
#                         mode='lines',
#                         fillcolor='rgba(68, 68, 68, 0.3)',
#                         fill='tonexty',
#                         showlegend=False
#                     )
#                 ])
#             # fig.update_layout(
#             #     yaxis_title='Wind speed (m/s)',
#             #     title='Continuous, variable value error bars',
#             #     hovermode="x"
#             # )
#             # fig.show()
#             forecast_plot.plotly_chart(fig_forecast, use_container_width=True)
#             time.sleep ( 1 )


from flawsleuth.ai import predict, kalman_forecast
from flawsleuth.timeseries import fetch_data_frame, fetch_entity_series
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import streamlit as st
import numpy as np
import time
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import count
# from data_process import Preprocessing
import pandas as pd
import plotly.express as px
# from batch_preprocess import Preprocessing
from joblib import dump, load
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from flawsleuth.utils.kalman import DiscreteKalmanFilter
from collections import defaultdict, deque
from flawsleuth import utils


st.set_page_config ( layout="wide" )

def run():
    # st.set_page_config ( layout="wide" )  # setting the display in the
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        button[data-baseweb="tab"] {font-size: 30px;}
        </style>
        """
    st.markdown ( hide_menu_style, unsafe_allow_html=True )

    new_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> SADS(TEST): Shop-floor Anomaly Detection Service: Online mode </p> </h2></center>'
    st.markdown(new_title, unsafe_allow_html=True)

    info = st.empty()
    info_train = st.empty()
    pack_info = st.empty()
    pack_test = st.empty()
    SMALL_SIZE = 5
    MEDIUM_SIZE =5
    BIGGER_SIZE = 5
    # plt.rcParams['figure.figsize'] = (5, 10)
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE, dpi=600)  # fontsize of the figure title
    plt.style.context('bmh')

    WINDOW_SIZE = 10

    # st.title ( "SADS: Shop floor Anomaly Detection Service" )


    dataPath = 'data/final.csv'
    modelPath = 'data/kalman_update.pf'
    # modelPath = 'dashboard/kalman_update.pf'
    clsPath = 'data/cls_new.joblib'
    clsPath_ifor = 'data/ifor_cls_new.joblib'
    scalerPath = 'data/scaler_new.joblib'

    # @st.cache(suppress_st_warning=True)
    def data_reader(dataPath:str) -> pd.DataFrame :
        data = pd.read_csv(dataPath) #pd.read_csv(dataPath, decimal=',')
        # prepro = Preprocessing()
        # data = prepro.preprocess(df)
        # data = data[['BarCode', 'Face', 'Cell', 'Point', 'Group' , 'Output Joules' , 'Charge (v)', 'Residue (v)', 'Force L N','Force L N_1', 'Y/M/D hh:mm:ss']]
        data.rename(columns={'BarCode':'Barcode', 'Output Joules': 'Joules', 'Charge (v)':'Charge', 'Residue (v)':'Residue','Force L N':'Force_N', 'Force L N_1':'Force_N_1', 'Y/M/D hh:mm:ss': 'Datetime'}, inplace=True)
        data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']] = data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].apply(np.float32)

        return data


    # @st.cache(allow_output_mutation=True)
    def cls_loader(clsPatah):
        return load(clsPatah)


    # @st.cache(suppress_st_warning= True)
    def kalman_loader(modelPath, dim:int=5, latend_dim:int=20) ->DiscreteKalmanFilter:
        model = DiscreteKalmanFilter(dim=5, latent_dim=latend_dim)
        checkpoint = torch.load(modelPath)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    # def kalman_loader(dim:int=5, latend_dim:int=20) ->DiscreteKalmanFilter:
    #     model = DiscreteKalmanFilter(dim=dim, latent_dim=latend_dim)
    #     checkpoint = torch.load(KALMAN_PATH_FROM_ROOT)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     return model

    def kalman_train(steps:int=500, verbose:bool=True) -> DiscreteKalmanFilter:
        model = DiscreteKalmanFilter(dim=5, latent_dim=20)
        optim = torch.optim.Adam(model.parameters())
        for i in range(1, steps):
            optim.zero_grad()
            loss = model(torch.Tensor(y))
            if verbose:
                if i % 100 == 0:
                    print(f'Iter {i}, loss: {loss:.4f}')
            loss.backward()
            optim.step()
        return model

    def kalman_forecast(model , y, forecast_steps:int=448) -> tuple:
        pred_mu_1, pred_sigma_1, x, P = model.iterate_disc_sequence(torch.Tensor(y))#iterate_disc_sequence(torch.Tensor(y))
        pred_mu_2, pred_sigma_2 = model.forecasting(forecast_steps, x, P)
        # pred_mu = torch.cat([pred_mu_1, pred_mu_2]).detach().cpu().numpy()
        # pred_sigma = torch.cat([pred_sigma_1, pred_sigma_2]).detach().cpu().numpy()
        return pred_mu_2.detach().cpu().numpy(), pred_sigma_2.detach().cpu().numpy(), model


    scaler = load(scalerPath)
    kl_model = kalman_loader(modelPath=modelPath)


    # df = pd.read_csv(dataPath)
    # st.write(df.columns)
    # data = data_reader(dataPath=dataPath)
#######################################################################


    with st.sidebar.container():
        with st.form(key='my_form'):
            with st.expander('Chart and Pack control'):
                st.subheader('Chart and Pack control')

                feature = st.selectbox("Feature to be display", ['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1'], index=0)
                max_x = st.slider ( "Select the max length for scatter plot", min_value=112, max_value=10 * 448, step=112,
                                    key='max_lenh' )

                # st.subheader("Table control input")
            with st.expander("Model control"):
                st.subheader('Model Control')

                model_choice = st.selectbox (
                    "Choose the model",
                    ("Isolation Forest","Random Forest", "Thresholding") )

                drift_model = st.selectbox (
                    "Choose the Data shift detector",
                    ("Wasserschein distance","Petite", "Mean averaging") )

                forecast_lenght = st.slider ( "Forecasting steps", min_value=1, max_value=448, step=1,
                                            key='forecast_length' )


            # submitted = st.form_submit_button('Apply')

            stop_bt, rerun_bt, show_bt = st.columns ( (1, 1, 1) )
            SADS_submit = show_bt.form_submit_button ( "Start" )
            stop_submit = stop_bt.form_submit_button ( "Stop" )
            rerun_submit = rerun_bt.form_submit_button ( "Rerun" )

            with st.expander('See instructions'):
                st.subheader("Plot setting")
                st.markdown ( """
                - Forecasting steps: Maximum length of the plot
                - Stop: Stop the simulation
                -
                - Choose the model :
                    - ***Repeat***: Model based on the repeated labeling method
                        - Labeling strategy provided by the WAM technik
                    - ***Iforest***: Model base on Isolation forest labeling method
                        - Unsupervised labeling mechanisme for anomalies. A clustering based method

                        """ )

    ######################### SETING UP THE TABS ########################################

    counter = count ( 0 )
    pack_view, table_view, chart_view = st.tabs(["Pack", "🗃Table", "📈 Charts"])

    # display
    joul_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> Display the output joules </p> </h2></center>'
    chart_view.markdown ( joul_title, unsafe_allow_html=True )
    show_joules = chart_view.empty()
    show_all, py_chart = chart_view.columns ( (2, 1) )
    show_all, py_chart = show_all.empty(), py_chart.empty()
    # show_dist = chart_view.empty()
    kalman_view = chart_view.empty()
    kalman_view , show_dist = chart_view.columns ( (2, 1) )

    kalman_view, show_dist = kalman_view.empty(), show_dist.empty()


    table_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> Display the table </p> </h2></center>'
    table_view.markdown ( table_title, unsafe_allow_html=True )
    day_left, time_right = table_view.columns ( 2 )
    good_weld = day_left.empty ()
    bad_weld = time_right.empty ()



    pack_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> Show the Data battery pack </p> </h2></center>'
    pack_view.markdown ( pack_title, unsafe_allow_html=True )

    pack_1, pack_2 = pack_view.columns(2)
    pack_vew1 = pack_1.empty()
    pack_vew2 = pack_2.empty()


    fig = make_subplots ( rows=3, cols=1 )
    # fig  = go.Figure()
    kalman_col = ['OJ_mu', 'Charge_mu', 'Residue_mu', 'ForceL_mu','ForceR_mu','OJ_std', 'Charge_std', 'Residue_std', 'ForceL_std','ForceR_std','Measuremnt']
    df_fif = pd.DataFrame(columns= kalman_col)


    if SADS_submit:
        face_1 =  np.ones(shape=(14,16))*0.
        face_2 =  np.ones(shape=(14,16))*0.

        face_1_maske = np.ones ( shape=(14,16) )
        face_2_maske = np.ones ( shape=(14,16) )

        face_1_repeat = np.zeros(shape=(14,16))
        face_2_repeat = np.zeros(shape=(14,16))

        face_1_repeat_mask =  np.ones(shape=(14,16))
        face_2_repeat_mask = np.ones(shape = (14,16))


        time_plot_1_1 = 0
        time_plot_1_2 = 0
        time_plot_2_1 = 0
        time_plot_2_2 = 0
        plot_count_1 = 0
        plot_count_2 = 0
        plot_count_repeat_1 = 0
        plot_count_repeat_2 = 0


        model = cls_loader(clsPath)


        result_old = deque(maxlen=WINDOW_SIZE)
        result_old_bar = deque(maxlen=WINDOW_SIZE)
        result_new = deque(maxlen=WINDOW_SIZE)
        result_new_bar = deque(maxlen=WINDOW_SIZE)
        TRAIN_MODEL = False
        # while True:  # stop_forecast == 'continue':
        while fetch_entity_series ()   and not stop_submit:
            fig = make_subplots ( rows=4, cols=1 )
            time_count = next ( counter )

            rr = fetch_entity_series ()
            to_predict = pd.DataFrame ( rr.dict() )
            # st.write(to_predict)
            to_predict[['face', 'cell', 'point']] = to_predict[['face', 'cell', 'point']].apply(lambda x: x.str.extract('(\d+)', expand=False).astype(int)) # extract the number from the string
            rr = to_predict[ ['joules', 'charge', 'residue', 'force_n', 'force_n_1',]].astype(float).values # Process_update  if get_data_from_entity( NGSY entity )
            answer = model.predict ( scaler.transform(rr) )
            to_predict['anomaly'] = answer

            ############################# SETTING THE SHIFT DETECTION #########################################

            #strip the barcode
            to_predict['barcode'] = to_predict['barcode'].str.strip()

            bar = to_predict['barcode'].to_list()
            # set pack display color to dark blue
            if time_count == 0:
                pack_info.markdown (f'<center> <h2> <p style="font-family:fantasy; font-size: 14px;"> Current pack: {bar} </p> </h2></center>', unsafe_allow_html=True )

                old_bar = bar
                record_new=False

                df_chart = to_predict.copy()

                df_chart['anomaly'] = answer
            else:

                if old_bar != bar:
                    pack_info.empty()
                    pack_info.markdown (f'<center> <h2> <p style="font-family:fantasy; font-size: 14px;"> Current pack: {bar} </p> </h2></center>', unsafe_allow_html=True )
                    record_new = True
                    old_bar = bar
                if not record_new:
                    result_old.append(to_predict['joules'].to_list()[0])
                    result_old_bar.append(bar)
                else :
                    result_new.append(to_predict['joules'].to_list()[0])
                    result_new_bar.append(bar)

                #### Testing the shift when the window is full ####

                if len(result_new) == WINDOW_SIZE and record_new:
                    record_new = False
                    st.write("Testing the shift")
                    st.write("Old result", result_old)
                    st.write("New result", result_new)


                    if drift_model == "Wasserschein distance":
                        res = utils.wasser_KL_test(result_old, result_new, threshold=0.1)
                        if res.drift:
                            TRAIN_MODEL = True
                            info.warning("Shift detected, Need to retrain the model: Wasserstein distance + KL divergence")
                            # info.empty()
                    elif drift_model == 'Petite':
                        to_test = deque(maxlen=WINDOW_SIZE*2)
                        to_test.extend(result_old)
                        to_test.extend(result_new)
                        res = utils.pettitt_test(to_test, alpha=0.8)
                        if res.cp > WINDOW_SIZE -1 and res.cp < WINDOW_SIZE +1:
                            TRAIN_MODEL = True
                            info.warning("Shift detected, Need to retrain the model: method PETTITE")


                    result_old.extend(result_new)
                    result_new = deque(maxlen=WINDOW_SIZE)
                    result_new_bar = deque(maxlen=WINDOW_SIZE)

                ####################################### END #######################################
                df_new = to_predict.copy()
                df_new['anomaly'] = answer
                df_chart = pd.concat ( [df_chart, df_new], ignore_index=True )

                ################# TRAINING THE MODEL ############################
                if TRAIN_MODEL:
                    # info.empty()
                    info_train.info("Retraining the model")
                    if model_choice == "Isolation Forest":
                        model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1,
                                                            max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42,
                                                            verbose=1)
                    elif model_choice == "One Class SVM":
                        model = OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001,
                                                            nu=0.5, shrinking=True, cache_size=200, verbose=False,

                                                            max_iter=-1, random_state=42)
                    elif model_choice == "Local Outlier Factor":
                        model = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30,
                                                                metric='minkowski', p=2, metric_params=None,
                                                                contamination=0.1, novelty=False, n_jobs=-1)
                    elif model_choice == "Elliptic Envelope":
                        model = EllipticEnvelope(contamination=0.1, random_state=42)

                    with st.echo():
                        info_train.info("Fitting model...")
                        model.fit(scaler.transform(df_chart[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1',]].values))
                        info_train.info("Model fitting completed.")
                        cls_saver(model, clsPath)
                        TRAIN_MODEL = False
                    # train the kalman mode l
                    kalman_train(df_chart, kalmanPath)


            test = df_chart[-max_x:]
            good_weld.dataframe ( df_chart[df_chart['anomaly'] == False] )
            bad_weld.dataframe ( df_chart[df_chart['anomaly'] == True] )
            # set_trace()
            haha = test.copy ()
            haha['anomaly'] = haha['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )



            ################################# SHOWING THE CHART VIEW
            fig_px = px.scatter ( haha, y='joules', color='anomaly',
                                  color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
            # fig_px.update_layout ( width=1500, height=250, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
            fig_px.update_layout ( plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
            show_joules.plotly_chart ( fig_px, use_container_width=True )


            fig.append_trace ( go.Scatter ( y=test['force_n'], x=test.index, mode='lines', name='Force_N',
                                            line_color='blue' if True else 'red' ), row=1, col=1, )
            fig.append_trace ( go.Scatter ( y=test['force_n_1'], x=test.index, mode='lines', name='Force_N_1',
                                            line_color='blue' if True else 'red' ), row=2, col=1, )
            fig.append_trace ( go.Scatter ( y=test['charge'], x=test.index, mode='lines', name='Charge',
                                            line_color='blue' if True else 'red' ), row=3, col=1, )
            fig.append_trace ( go.Scatter ( y=test['residue'], x=test.index, mode='lines', name='Residue',
                                            line_color='blue' if True else 'red' ), row=4, col=1, )
            fig.update_layout ( plot_bgcolor='rgb(206, 237, 240)' )

            show_all.plotly_chart ( fig, use_container_width=True )

            haha_pie = df_chart.copy ()
            haha_pie['anomaly'] = haha_pie['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )

            fig_pi = px.pie ( haha_pie, values='joules', hover_name='anomaly' , names='anomaly', title='the ratio of anomaly vs normal',
                              hole=.3, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) #, color_discrete_sequence=px.colors.sequential.RdBu)
            py_chart.plotly_chart ( fig_pi, use_container_width=True )


            # #
            # if time_count > 2 :
            #     distplot , ax = plt.subplots(constrained_layout=True)
            #     # sns.distplot(test['Joules'], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3},
            #     #   label = "Anomaly")

            #     sns.kdeplot(data=test, x="joules", hue="anomaly", multiple="stack")
            #     show_dist.pyplot(distplot)

            #     del distplot

            del fig, fig_pi, fig_px,

            ######### SHOWING THE PACK VIEW ##################################################


            location_info = to_predict[['face', 'cell', 'point']].values
            row, col  = 14 , 16
            if not time_count:
                old_location = location_info
                cell = location_info[0][1] - 1
                l1, l2 = (cell//col)%row, cell%col
                if location_info[0][0] == 1:  # face1
                    if location_info[0][-1] == 1: # point1
                        face_1[l1*2, l2] = location_info[0][1]
                        face_1_maske[l1*2,l2] = False
                        time_plot_1_1 += 1
                    else: # point2
                        info_train.write(F"CURRENT:POINT 2 plot_count_1: {plot_count_1 + 1 },  time_plot_1_1 {time_plot_1_2}")
                        face_1[l1*2 + 1, l2] = location_info[0][1]
                        face_1_maske[l1*2 + 1, l2] = False
                        if cell%(col)  == 15:
                            pack_test.write(f"location_info[0][1]%(col+1) {location_info[0][1]%(col+1)}")

                            plot_count_1 += 1
                            time_plot_1_1 = 0
                            time_plot_1_2 = 0

                else:   # face2
                    if location_info[0][-1] == 1: # point1

                        face_2[l1*2 , l2] = location_info[0][1]
                        face_2_maske[l1*2 ,l2] = False
                        time_plot_2_1 += 1
                    else: # point2
                        face_2[l1*2+ 1, l2] = location_info[0][1]
                        face_2_maske[l1*2+ 1, l2] = False
                        if cell%(col)  == 15:
                            pack_test.write(f"location_info[0][1]%(col+1) {location_info[0][1]%(col+1)}")
                            plot_count_2 += 1
                            time_plot_2_1 = 0
                            time_plot_2_2 = 0


            elif (old_location != location_info[0]).any() and time_count > 0:

                old_location = location_info
                cell = location_info[0][1] - 1
                l1, l2 = (cell//col)%row, cell%col
                if location_info[0][0] == 1:  # face1
                    if location_info[0][-1] == 1: # point1
                        face_1[l1*2, l2] = location_info[0][1]
                        face_1_maske[l1*2,l2] = False
                        time_plot_1_1 += 1
                    else: # point2
                        info_train.write(F"CURRENT:POINT 2 plot_count_1: {plot_count_1 + 1 },  time_plot_1_1 {time_plot_1_2}")
                        face_1[l1*2 + 1, l2] = location_info[0][1]
                        face_1_maske[l1*2 + 1, l2] = False
                        if cell%(col)  == 15:
                            pack_test.write(f"location_info[0][1]%(col+1) {location_info[0][1]%(col+1)}")
                            plot_count_1 += 1
                            time_plot_1_1 = 0
                            time_plot_1_2 = 0

                else:   # face2
                    if location_info[0][-1] == 1: # point1

                        face_2[l1*2 , l2] = location_info[0][1]
                        face_2_maske[l1*2 ,l2] = False
                        time_plot_2_1 += 1
                    else: # point2
                        face_2[l1*2+ 1, l2] = location_info[0][1]
                        face_2_maske[l1*2+ 1, l2] = False
                        if cell%(col)  == 15:
                            pack_test.write(f"location_info[0][1]%(col+1) {location_info[0][1]%(col+1)}")
                            plot_count_2 += 1
                            time_plot_2_1 = 0
                            time_plot_2_2 = 0
            # else:

            #     old_location = location_info
            #     cell = location_info[0][1]
            #     l1, l2 = (cell//col)%row, cell%col
            #     if location_info[0][0] == 1:  # face1
            #         if location_info[0][-1] == 1: # point1
            #             face_1_repeat[l1+plot_count_repeat_1, l2] = location_info[0][1]
            #             face_1_repeat_mask[l1+plot_count_repeat_1, l2] = False
            #         else: # point2

            #             face_1_repeat[l1 +plot_count_repeat_1+1, l2] = location_info[0][1]
            #             face_1_repeat_mask[l1 +plot_count_repeat_1+1, l2] = False
            #             if location_info[0][1]%col +1 == 16:
            #                 plot_count_repeat_1 += 1

            #     else: # face2
            #         if location_info[0][-1] == 1:
            #             face_2_repeat[l1+plot_count_repeat_2, l2] = location_info[0][1]
            #             face_2_repeat_mask[l1+plot_count_repeat_2, l2] = False
            #         else:
            #             face_2_repeat[l1+plot_count_repeat_2 +1, l2] = location_info[0][1]
            #             face_2_repeat_mask[l1+plot_count_repeat_2 +1, l2] = False
            #             if location_info[0][1]%col +1 == 16:
            #                 plot_count_repeat_2 += 1


            # set_trace()
            if time_count==0:
                last_point = location_info


            face_1_cp_mask = face_1_maske.reshape(14, 16)
            face_2_cp_mask = face_2_maske.reshape(14, 16)

            fig_pack_1, face_ax_1 = plt.subplots (nrows=2, ncols=1, figsize=(5, 5) )
            fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1,  figsize=(5, 5) )
            annot = True
            sns.heatmap(face_1, cmap=ListedColormap(['green', 'red']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_1[0], cbar=False, mask=face_1_maske,\
                        annot=annot, yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],cbar_kws={
                    'pad': .001,
                    'ticks': [0, 1],
                    "shrink": 0.01
                },
                        )

            sns.heatmap(face_2, cmap=ListedColormap(['green', 'red']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_2[0], cbar=False, mask=face_2_maske,\
                        annot=annot, yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],)
            sns.heatmap(face_1_repeat, cmap=ListedColormap(['blue', 'orange']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_1[1], cbar=False, mask=face_1_repeat_mask,\
                        annot=annot, yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],)
            face_ax_1[1].set_title("Reapeted face 1")
            sns.heatmap(face_2_repeat, cmap=ListedColormap(['blue', 'orange']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_2[1], cbar=False, mask=face_2_repeat_mask,\
                        annot=annot, yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],)
            face_ax_2[1].set_title ( "Reapeted face 2" )

            pack_vew1.pyplot(fig_pack_1)#,use_container_width=True)
            pack_vew2.pyplot(fig_pack_2)#,use_container_width=True)
            del fig_pack_1, fig_pack_2






            #########################  TIME SERIE FORECASTING PLACEHOLDER   ###############
            ########## KALMAN FILTERING
            ########## DEEP LEARNING MODEL ( DNN , CNN, RNN )


            # del fig_2, fig_1
            mu, sigma, kl_model = kalman_forecast(model=kl_model, y=rr, forecast_steps=1)
            # mu, sigma = kalman_forecast(kl_model, rr, forecast_steps=max_forecast)
            # set_trace()

            result = np.hstack([mu, sigma[:,:,0],rr[:,0].reshape(-1,1)])
            if time_count==0:
                df_fif = pd.DataFrame(data=result, columns=kalman_col)
            else :
                df_new = pd.DataFrame(data=result, columns=kalman_col)
                df_fif = pd.concat([df_fif, df_new], ignore_index=True)
            df_forecate = df_fif[-max_x:]
            fig = go.Figure([
                    go.Scatter(
                        name='Forecast',
                        # x=df_fif['Time'],
                        y=df_forecate['OJ_mu'],
                        mode='lines',
                        marker=dict(color='red', size=5),
                        showlegend=True
                    ),
                    go.Scatter(
                        name='Measurement',
                        # x=df_forecate['Time'],
                        y=df_forecate['Measuremnt'],
                        mode='markers',
                        marker=dict(color='green', size=5),
                        showlegend=True
                    ),
                    go.Scatter(
                        name='Upper Bound',
                        # x=df_forecate['Time'],
                        y=df_forecate['OJ_mu']+df_forecate['OJ_std'],
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=1),
                        showlegend=False
                    ),
                    go.Scatter(
                        name='Lower Bound',
                        # x=df_forecate['Time'],
                        y=df_forecate['OJ_mu']-df_forecate['OJ_std'],
                        marker=dict(color="#444"),
                        line=dict(width=1),
                        mode='lines',
                        fillcolor='rgba(68, 68, 68, 0.3)',
                        fill='tonexty',
                        showlegend=False
                    )
                ])

            kalman_view.plotly_chart(fig, use_container_width=True)
            # time.sleep ( 0.1)

    # if SADS_submit:
    #     face_1 =  np.ones(shape=(14,16))*0.
    #     face_2 =  np.ones(shape=(14,16))*0.

    #     face_1_maske = np.ones ( shape=(14,16) )
    #     face_2_maske = np.ones ( shape=(14,16) )

    #     face_1_repeat = np.zeros(shape=(14,16))
    #     face_2_repeat = np.zeros(shape=(14,16))

    #     face_1_repeat_mask =  np.ones(shape=(14,16))
    #     face_2_repeat_mask = np.ones(shape = (14,16))


    #     time_plot_1_1 = 0
    #     time_plot_1_2 = 0
    #     time_plot_2_1 = 0
    #     time_plot_2_2 = 0
    #     plot_count_1 = 0
    #     plot_count_1 = 0
    #     plot_count_2 = 0

    #     model = cls_loader(clsPath)


    #     result_old = deque(maxlen=WINDOW_SIZE)
    #     result_old_bar = deque(maxlen=WINDOW_SIZE)
    #     result_new = deque(maxlen=WINDOW_SIZE)
    #     result_new_bar = deque(maxlen=WINDOW_SIZE)

    #     while True:  # stop_forecast == 'continue':
    #         fig = make_subplots ( rows=4, cols=1 )
    #         time_count = next ( counter )

    #         rr = fetch_entity_series ()
    #         to_predict = pd.DataFrame ( rr.dict() )
    #         result = to_predict[['face', 'cell', 'point']].apply(lambda x: x.str.extract('(\d+)', expand=False).astype(int)).values
    #         rr = to_predict[ ['joules', 'charge', 'residue', 'force_n', 'force_n_1',]].values
    #         answer = model.predict ( scaler.transform(rr) )

    #         ###

    #         ############################# SETTING THE SHIFT DETECTION #########################################


    #         bar = to_predict['barcode'].to_list()

    #         if time_count == 0:
    #             old_bar = bar
    #             record_new=False

    #             df_chart = to_predict.copy()

    #             df_chart['anomaly'] = answer
    #         else:
    #             if old_bar != bar:
    #                 record_new = True
    #                 old_bar = bar
    #             if not record_new:
    #                 result_old.append(to_predict['joules'].to_list()[0])
    #                 result_old_bar.append(bar)
    #             else :
    #                 result_new.append(to_predict['joules'].to_list()[0])
    #                 result_new_bar.append(bar)

    #             #### now testing the shift hapanned ##################

    #             if len(result_new) == WINDOW_SIZE and record_new:
    #                 record_new = False

    #                 if drift_model == "Wasserschein distance":
    #                     st.write("Wasserschein distance")
    #                     st.write(result_old)
    #                     st.write(result_new)
    #                     res = utils.wasser_KL_test(result_old, result_new, threshold=0.1)
    #                     if res.drift:
    #                         info.info("Shift detected, Need to retrain the model: Wasserstein distance + KL divergence")
    #                         info.empty()
    #                 elif drift_model == 'Petite':
    #                     to_test = deque(maxlen=WINDOW_SIZE*2)
    #                     to_test.extend(result_old)
    #                     to_test.extend(result_new)
    #                     res = utils.pettitt_test(to_test, alpha=0.8)
    #                     if res.cp > WINDOW_SIZE -1 and res.cp < WINDOW_SIZE +1:
    #                         info.info("Shift detected, Need to retrain the model: method PETTITE")


    #                 result_old.extend(result_new)
    #                 result_new = deque(maxlen=WINDOW_SIZE)
    #                 result_new_bar = deque(maxlen=WINDOW_SIZE)

    #         ####################################### END #######################################
    #             df_new = to_predict.copy()
    #             df_new['anomaly'] = answer
    #             df_chart = pd.concat ( [df_chart, df_new], ignore_index=True )

    #         test = df_chart[-max_x:]
    #         good_weld.dataframe ( df_chart[df_chart['anomaly'] == False] )
    #         bad_weld.dataframe ( df_chart[df_chart['anomaly'] == True] )
    #         # set_trace()
    #         haha = test.copy ()
    #         haha['anomaly'] = haha['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )



    #         ################################# SHOWING THE CHART VIEW
    #         fig_px = px.scatter ( haha, y='joules', color='anomaly',
    #                               color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
    #         # fig_px.update_layout ( width=1500, height=250, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
    #         fig_px.update_layout ( plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
    #         show_joules.plotly_chart ( fig_px, use_container_width=True )


    #         fig.append_trace ( go.Scatter ( y=test['force_n'], x=test.index, mode='lines', name='Force_N',
    #                                         line_color='blue' if True else 'red' ), row=1, col=1, )
    #         fig.append_trace ( go.Scatter ( y=test['force_n_1'], x=test.index, mode='lines', name='Force_N_1',
    #                                         line_color='blue' if True else 'red' ), row=2, col=1, )
    #         fig.append_trace ( go.Scatter ( y=test['charge'], x=test.index, mode='lines', name='Charge',
    #                                         line_color='blue' if True else 'red' ), row=3, col=1, )
    #         fig.append_trace ( go.Scatter ( y=test['residue'], x=test.index, mode='lines', name='Residue',
    #                                         line_color='blue' if True else 'red' ), row=4, col=1, )
    #         fig.update_layout ( plot_bgcolor='rgb(206, 237, 240)' )

    #         show_all.plotly_chart ( fig, use_container_width=True )

    #         haha_pie = df_chart.copy ()
    #         haha_pie['anomaly'] = haha_pie['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )

    #         fig_pi = px.pie ( haha_pie, values='joules', hover_name='anomaly' , names='anomaly', title='the ratio of anomaly vs normal',
    #                           hole=.3, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) #, color_discrete_sequence=px.colors.sequential.RdBu)
    #         py_chart.plotly_chart ( fig_pi, use_container_width=True )


    #         #
    #         if time_count > 2 :
    #             distplot , ax = plt.subplots(constrained_layout=True)
    #             # sns.distplot(test['Joules'], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3},
    #             #   label = "Anomaly")

    #             sns.kdeplot(data=test, x="joules", hue="anomaly", multiple="stack")
    #             show_dist.pyplot(distplot)

    #             del distplot

    #         del fig, fig_pi, fig_px,

    #         ######### SHOWING THE PACK VIEW ##################################################

    #         location_info = to_predict[['face', 'cell', 'point']].apply(lambda x: x.str.extract('(\d+)', expand=False).astype(int))
    #         location_info = location_info.values
    #         if not time_count:
    #             old_location = location_info
    #             if location_info[0][0] == 1:  # face1
    #                 if location_info[0][-1] == 1:
    #                     face_1[plot_count_1, time_plot_1_1] = answer
    #                     face_1_maske[plot_count_1, time_plot_1_1] = False
    #                     time_plot_1_1 += 1
    #                 else:

    #                     face_1[plot_count_1 + 1, time_plot_1_2] = answer
    #                     face_1_maske[plot_count_1 + 1, time_plot_1_2] = False
    #                     time_plot_1_2 += 1
    #                 if time_plot_1_2 == 16:
    #                     plot_count_1 += 2
    #                     time_plot_1_1 = 0
    #                     time_plot_1_2 = 0

    #             else:
    #                 if location_info[0][-1] == 1:
    #                     face_2[plot_count_2, time_plot_2_1] = answer
    #                     face_2_maske[plot_count_2, time_plot_2_1] = False
    #                     time_plot_2_1 += 1
    #                 else:
    #                     face_2[plot_count_2 + 1, time_plot_2_2] = answer
    #                     face_2_maske[plot_count_2 + 1, time_plot_2_2] = False
    #                     time_plot_2_2 += 1
    #                 if time_plot_2_2 == 16:
    #                     plot_count_2 += 2
    #                     time_plot_2_2 = 0
    #                     time_plot_2_1 = 0
    #             if plot_count_1 == 14:
    #                 plot_count_1 = 0
    #                 face_1 = np.ones ( shape=(14, 16) ) * 0.
    #                 face_1_maske = np.ones ( shape=(14, 16) )
    #                 face_1_repeat = np.zeros ( shape=(14, 16) )
    #                 face_1_repeat_mask = np.ones ( shape=(14, 16) )
    #             if plot_count_2 == 14:
    #                 plot_count_2 = 0

    #                 face_2 = np.ones ( shape=(14, 16) ) * 0.
    #                 face_2_maske = np.ones ( shape=(14, 16) )
    #                 face_2_repeat = np.zeros ( shape=(14, 16) )
    #                 face_2_repeat_mask = np.ones ( shape=(14, 16) )


    #         elif (old_location != location_info[0]).any() and time_count > 0:

    #             old_location = location_info
    #             if location_info[0][0] == 1:  # face1
    #                 if location_info[0][-1] == 1:
    #                     face_1[plot_count_1, time_plot_1_1] = answer
    #                     face_1_maske[plot_count_1, time_plot_1_1] = False
    #                     time_plot_1_1 += 1
    #                 else:

    #                     face_1[plot_count_1 + 1, time_plot_1_2] = answer
    #                     face_1_maske[plot_count_1 + 1, time_plot_1_2] = False
    #                     time_plot_1_2 += 1
    #                 if time_plot_1_2 == 16:
    #                     plot_count_1 += 2
    #                     time_plot_1_1 = 0
    #                     time_plot_1_2 = 0

    #             else:
    #                 if location_info[0][-1] == 1:
    #                     face_2[plot_count_2, time_plot_2_1] = answer
    #                     face_2_maske[plot_count_2, time_plot_2_1] = False
    #                     time_plot_2_1 += 1
    #                 else:
    #                     face_2[plot_count_2 + 1, time_plot_2_2] = answer
    #                     face_2_maske[plot_count_2 + 1, time_plot_2_2] = False
    #                     time_plot_2_2 += 1
    #                 if time_plot_2_2 == 16:
    #                     plot_count_2 += 2
    #                     time_plot_2_2 = 0
    #                     time_plot_2_1 = 0
    #             if plot_count_1 == 14:
    #                 plot_count_1 = 0
    #                 face_1 = np.ones ( shape=(14, 16) ) * 0.
    #                 face_1_maske = np.ones ( shape=(14, 16) )
    #                 face_1_repeat = np.zeros ( shape=(14, 16) )
    #                 face_1_repeat_mask = np.ones ( shape=(14, 16) )
    #             if plot_count_2 == 14:
    #                 plot_count_2 = 0

    #                 face_2 = np.ones ( shape=(14, 16) ) * 0.
    #                 face_2_maske = np.ones ( shape=(14, 16) )
    #                 face_2_repeat = np.zeros ( shape=(14, 16) )
    #                 face_2_repeat_mask = np.ones ( shape=(14, 16) )



    #         else:

    #             old_location = location_info
    #             if location_info[0][0] == 1:  # face1
    #                 if location_info[0][-1] == 1:
    #                     face_1_repeat[plot_count_1, time_plot_1_1] = answer
    #                     face_1_repeat_mask[plot_count_1, time_plot_1_1] = False
    #                 else:

    #                     face_1_repeat[plot_count_1 + 1, time_plot_1_2] = answer
    #                     face_1_repeat_mask[plot_count_1 + 1, time_plot_1_2] = False
    #             else:
    #                 if location_info[0][-1] == 1:
    #                     face_2_repeat[plot_count_2, time_plot_2_1] = answer
    #                     face_2_repeat_mask[plot_count_2, time_plot_2_1] = False
    #                 else:
    #                     face_2_repeat[plot_count_2 + 1, time_plot_2_2] = answer
    #                     face_2_repeat_mask[plot_count_2 + 1, time_plot_2_2] = False

    #         # set_trace()
    #         if time_count==0:
    #             last_point = location_info


    #         face_1_cp_mask = face_1_maske.reshape(14, 16)
    #         face_2_cp_mask = face_2_maske.reshape(14, 16)

    #         fig_pack_1, face_ax_1 = plt.subplots (nrows=2, ncols=1, figsize=(5, 5) )
    #         fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1,  figsize=(5, 5) )

    #         sns.heatmap(face_1, cmap=ListedColormap(['green', 'red']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_1[0], cbar=False, mask=face_1_maske,\
    #                     yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],cbar_kws={
    #                 'pad': .001,
    #                 'ticks': [0, 1],
    #                 "shrink": 0.01
    #             },
    # )
    #         sns.heatmap(face_2, cmap=ListedColormap(['green', 'red']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_2[0], cbar=False, mask=face_2_maske,\
    #                     yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],)
    #         sns.heatmap(face_1_repeat, cmap=ListedColormap(['green', 'red']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_1[1], cbar=False, mask=face_1_repeat_mask,\
    #                     yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],)
    #         face_ax_1[1].set_title("Reapeted face 1")
    #         sns.heatmap(face_2_repeat, cmap=ListedColormap(['green', 'red']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_2[1], cbar=False, mask=face_2_repeat_mask,\
    #                     yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],)
    #         face_ax_2[1].set_title ( "Reapeted face 2" )

    #         pack_vew1.pyplot(fig_pack_1)#,use_container_width=True)
    #         pack_vew2.pyplot(fig_pack_2)#,use_container_width=True)
    #         del fig_pack_1, fig_pack_2






    #         #########################  TIME SERIE FORECASTING PLACEHOLDER   ###############
    #         ########## KALMAN FILTERING
    #         ########## DEEP LEARNING MODEL ( DNN , CNN, RNN )


    #         # del fig_2, fig_1
    #         mu, sigma, kl_model = kalman_forecast(model=kl_model, y=rr, forecast_steps=1)
    #         # mu, sigma = kalman_forecast(kl_model, rr, forecast_steps=max_forecast)
    #         # set_trace()

    #         result = np.hstack([mu, sigma[:,:,0],rr[:,0].reshape(-1,1)])
    #         if time_count==0:
    #             df_fif = pd.DataFrame(data=result, columns=kalman_col)
    #         else :
    #             df_new = pd.DataFrame(data=result, columns=kalman_col)
    #             df_fif = pd.concat([df_fif, df_new], ignore_index=True)
    #         df_forecate = df_fif[-max_x:]
    #         fig = go.Figure([
    #                 go.Scatter(
    #                     name='Forecast',
    #                     # x=df_fif['Time'],
    #                     y=df_forecate['OJ_mu'],
    #                     mode='lines',
    #                     marker=dict(color='red', size=5),
    #                     showlegend=True
    #                 ),
    #                 go.Scatter(
    #                     name='Measurement',
    #                     # x=df_forecate['Time'],
    #                     y=df_forecate['Measuremnt'],
    #                     mode='markers',
    #                     marker=dict(color='green', size=5),
    #                     showlegend=True
    #                 ),
    #                 go.Scatter(
    #                     name='Upper Bound',
    #                     # x=df_forecate['Time'],
    #                     y=df_forecate['OJ_mu']+df_forecate['OJ_std'],
    #                     mode='lines',
    #                     marker=dict(color="#444"),
    #                     line=dict(width=1),
    #                     showlegend=False
    #                 ),
    #                 go.Scatter(
    #                     name='Lower Bound',
    #                     # x=df_forecate['Time'],
    #                     y=df_forecate['OJ_mu']-df_forecate['OJ_std'],
    #                     marker=dict(color="#444"),
    #                     line=dict(width=1),
    #                     mode='lines',
    #                     fillcolor='rgba(68, 68, 68, 0.3)',
    #                     fill='tonexty',
    #                     showlegend=False
    #                 )
    #             ])

    #         kalman_view.plotly_chart(fig, use_container_width=True)
    #         time.sleep ( 0.1)