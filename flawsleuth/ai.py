import pickle
import numpy as np
import joblib
from xgboost import XGBClassifier
from flawsleuth.ngsy import FloatAttr
from flawsleuth.ngsy import  WeldingMachineEntity, AnomalyDetectionEntity, ForecastngEntity
from flawsleuth.kalman import DiscreteKalmanFilter
import torch



ANOMALY_MODEL_PATH_FROM_ROOT_REPEAT = 'data/cls_new.joblib'
ANOMALY_MODEL_PATH_FROM_ROOT_IFOREST = 'data/ifor_cls_new.joblib'
ANOMALY_MODEL_PATH_FROM_ROOT_XGB = 'data/xgb_fine_tuned.pkl'
KALMAN_PATH_FROM_ROOT = 'data/kalman_update.pf'
SCALER_PATH = 'data/scaler_new.joblib'

# def predict(joules: float) -> bool:
#     with open(ANOMALY_MODEL_PATH_FROM_ROOT, 'rb') as open_file:
#         model = pickle.load(open_file)
#         label = model.predict(np.array(joules).reshape(-1,1))[0]
#         return label

def kalman_loader(dim:int=5, latend_dim:int=20) ->DiscreteKalmanFilter:
    model = DiscreteKalmanFilter(dim=dim, latent_dim=latend_dim)
    checkpoint = torch.load(KALMAN_PATH_FROM_ROOT)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

model_repeat = joblib.load ( ANOMALY_MODEL_PATH_FROM_ROOT_REPEAT )
model_iforest = joblib.load ( ANOMALY_MODEL_PATH_FROM_ROOT_IFOREST )
scaler = joblib.load ( SCALER_PATH )
kalman_model = kalman_loader()

with open(ANOMALY_MODEL_PATH_FROM_ROOT_XGB , mode='rb') as f:
    model_xgb = pickle.load(f)

def predict (machine: WeldingMachineEntity, model_type: bool = 0) -> AnomalyDetectionEntity:
    label = predict_anomaly ( machine, model_type=model_type )
    return AnomalyDetectionEntity ( id='1',
                                    Label=FloatAttr.new ( label ) )

def kalman_forecast (machine: WeldingMachineEntity,forecast_steps:int= 448) -> ForecastngEntity:
    mu, std =  kalman_forecast( machine, forecast_steps=forecast_steps )
    return ForecastngEntity( id='1', Label=FloatAttr.new ( label ) )


def predict_anomaly (machine: WeldingMachineEntity, model_type: bool = 0) -> bool:
    global model_repeat
    global model_iforest
    global model_xgb
    global scaler
    x = np.array ( [machine.joules, machine.charge , machine.residue , machine.force_n ,
                    machine.force_n_1 ] )
    scaled = scaler.transform ( x.reshape ( 1, -1 ))
    if model_type==1:
        return model_repeat.predict ( scaled )[0]
    elif model_type==2:
        return model_xgb.predict(scaled)[0]
    return model_iforest.predict ( scaled )[0]

def kalman_forecast (machine: WeldingMachineEntity, forecast_steps:int) -> tuple:
    global kalman_model
    x = np.array ( [machine.joules, machine.charge , machine.residue , machine.force_n ,  machine.force_n_1 ] )
    scaled = scaler.transform ( x.reshape ( 1, -1 ))
    pred_mu_1, pred_sigma_1, x, P = kalman_model.iterate_disc_sequence ( torch.Tensor ( x ) )
    pred_mu_2, pred_sigma_2 = kalman_model.forecasting ( forecast_steps, x, P )
    pred_mu = torch.cat([pred_mu_1, pred_mu_2]).detach().cpu().numpy()
    pred_sigma = torch.cat([pred_sigma_1, pred_sigma_2]).detach().cpu().numpy()
    return pred_mu, pred_sigma


