from fipy.ngsi.entity import EntitySeries
from fipy.ngsi.headers import FiwareContext
from fipy.ngsi.quantumleap import QuantumLeapClient
import pandas as pd
from uri import URI

from flawsleuth.ngsy import WeldingMachineEntity


# TODO these values should come from config and HTTP headers!
# Hard-coded to match the values in the tests.sim app.
QUANTUMLEAP_INTERNAL_BASE_URL = 'http://quantumleap:8668'
TENANT = 'wamtechnik'
SERVICE_PATH = '/'
ENTITY_ID = 'urn:ngsi-ld:WeldingMachine:1'


def quantumleap_client() -> QuantumLeapClient:
    base_url = URI(QUANTUMLEAP_INTERNAL_BASE_URL)
    ctx = FiwareContext(service=TENANT, service_path=SERVICE_PATH)
    return QuantumLeapClient(base_url, ctx)


def fetch_entity_series() -> EntitySeries:
    entity_type = WeldingMachineEntity(id='').type
    quantumleap = quantumleap_client()
    return quantumleap.entity_series(
        entity_id=ENTITY_ID, entity_type=entity_type,
        entries_from_latest=10
    )


def fetch_data_frame() -> pd.DataFrame:
    r = fetch_entity_series()
    time_indexed_df = pd.DataFrame(r.dict()).set_index('index')
    return time_indexed_df
