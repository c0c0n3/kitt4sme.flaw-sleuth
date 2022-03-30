import numpy as np
import pickle


ANOMALY_MODEL_PATH_FROM_ROOT = 'data/anomaly_detection.pkl'


def predict(joules: float) -> bool:
    with open(ANOMALY_MODEL_PATH_FROM_ROOT, 'rb') as open_file:
        model = pickle.load(open_file)
        label = model.predict(np.array(joules).reshape(-1,1))[0]
        return label
