from datetime import datetime
import json
import random
import pandas as pd
from typing import List, Optional
from uri import URI
from itertools import count

from fipy.ngsi.entity import FloatAttr, TextAttr
from fipy.ngsi.headers import FiwareContext
from fipy.ngsi.orion import OrionClient
from fipy.ngsi.quantumleap import QuantumLeapClient
from fipy.sim.sampler import DevicePoolSampler
from fipy.wait import wait_for_orion, wait_for_quantumleap

from flawsleuth.ngsy import WeldingMachineEntity
from tests.util.data_process import Preprocessing

import streamlit as st
import time

# FILEPATH = 'tests/util/Welding_data_new.csv'
FILEPATH = 'tests/util/Shiftedt_data.csv'
# FILEPATH = 'tests/util/final.csv'

TENANT = 'wamtechnik'
ORION_EXTERNAL_BASE_URL = 'http://localhost:1026'
QUANTUMLEAP_INTERNAL_BASE_URL = 'http://quantumleap:8668'
QUANTUMLEAP_EXTERNAL_BASE_URL = 'http://localhost:8668'

QUANTUMLEAP_SUB = {
    "description": "Notify QuantumLeap of changes to any entity.",
    "subject": {
        "entities": [
            {
                "idPattern": ".*"
            }
        ]
    },
    "notification": {
        "http": {
            "url": f"{QUANTUMLEAP_INTERNAL_BASE_URL}/v2/notify"
        }
    }
}
COUNTER = count(0)

def orion_client(service_path: Optional[str] = None,
                 correlator: Optional[str] = None) -> OrionClient:
    base_url = URI(ORION_EXTERNAL_BASE_URL)
    ctx = FiwareContext(service=TENANT, service_path=service_path,
                        correlator=correlator)
    return OrionClient(base_url, ctx)


def wait_on_orion():
    wait_for_orion(orion_client())


class SubMan:

    def __init__(self):
        self._orion = orion_client()

    def create_quantumleap_sub(self):
        self._orion.subscribe(QUANTUMLEAP_SUB)

    def create_subscriptions(self) -> List[dict]:
        self.create_quantumleap_sub()
        return self._orion.list_subscriptions()

# NOTE. Subscriptions and FIWARE service path.
# The way it behaves for subscriptions is a bit counter intuitive.
# You'd expect that with a header of 'fiware-servicepath: /' Orion would
# notify you of changes to *any* entities in the tree, similar to queries.
# But in actual fact, to do that you'd have to omit the service path header,
# which is what we do here. Basically the way it works is that if you
# specify a service path, then Orion only considers entities right under
# the last node in the service path, but not any other entities that might
# sit further down below. E.g. if your service tree looks like (e stands
# for entity)
#
#                        /
#                     p     q
#                  e1   r     e4
#                     e2 e3
#
# then a subscription with a service path of '/' won't catch any entities
# at all whereas one with a service path of '/p' will consider changes to
# e1 but not e2 nor e3.


def create_subscriptions():
    print(
        f"Creating catch-all {TENANT} entities subscription for QuantumLeap.")

    man = SubMan()
    orion_subs = man.create_subscriptions()
    formatted = json.dumps(orion_subs, indent=4)

    print("Current subscriptions in Orion:")
    print(formatted)


def quantumleap_client() -> QuantumLeapClient:
    base_url = URI(QUANTUMLEAP_EXTERNAL_BASE_URL)
    ctx = FiwareContext(service=TENANT, service_path='/')  # (*)
    return QuantumLeapClient(base_url, ctx)
# NOTE. Orion handling of empty service path. We send Orion entities w/ no
# service path in our tests. But when Orion notifies QL, it sends along a
# root service path. So we add it to the context to make queries work.


def wait_on_quantumleap():
    wait_for_quantumleap(quantumleap_client())


class WeldingMachineSampler(DevicePoolSampler):

    def __init__(self, pool_size: int, orion: Optional[OrionClient] = None):
        super().__init__(pool_size, orion if orion else orion_client())
        self.data = pd.read_csv(FILEPATH, decimal=',')

        self.counter = 0


    def new_device_entity_scv(self) -> WeldingMachineEntity:
        seed = random.uniform(0, 1)
        # time_count = next(self.counter)
        # to_predict = self.data[time_count:time_count + 1]


        return WeldingMachineEntity(
            id='34',

            barcode=TextAttr.new('bc-xyz'),
            face=TextAttr.new(random.choice(['f1', 'f2'])),
            cell=TextAttr.new(random.choice(['c1', 'c2'])),
            point=TextAttr.new(random.choice(['p1', 'p2'])),
            group=TextAttr.new(random.choice(['g1', 'g2'])),

            joules=FloatAttr.new(1.0335 + seed),
            charge=FloatAttr.new(2.0335 + seed),
            residue=FloatAttr.new(0.0335 + seed),
            force_n=FloatAttr.new(3.09 + seed),
            force_n_1=FloatAttr.new(4.0335 + seed),

            datetime=TextAttr.new(f"{datetime.now().isoformat()}")
        )

    def new_device_entity(self) -> WeldingMachineEntity:
        seed = random.uniform(0, 1)
        self.counter = next(COUNTER)
        to_predict = self.data.iloc[self.counter:self.counter+1]
        to_predict = to_predict.to_dict()


        to_print = WeldingMachineEntity(
            id='34',

            barcode=TextAttr.new(to_predict['BarCode'][self.counter]),
            face=TextAttr.new(to_predict['Face'][self.counter]),
            cell=TextAttr.new(to_predict['Cell'][self.counter]),
            point=TextAttr.new(to_predict['Point'][self.counter]),
            group=TextAttr.new(to_predict['Group'][self.counter]),
            joules=FloatAttr.new(to_predict['Output Joules'][self.counter]),
            charge=FloatAttr.new(to_predict[ 'Charge (v)'][self.counter]),
            residue=FloatAttr.new(to_predict[ 'Residue (v)'][self.counter]),
            force_n=FloatAttr.new(to_predict[ 'Force L N'][self.counter]),
            force_n_1=FloatAttr.new(to_predict['Force L N_1'][self.counter]),
            datetime=TextAttr.new(f"{datetime.now().isoformat()}")
        )
        # print(to_print)
        # for key, val in to_predict.items():
        #     if key == 'Output Joules':
        #         print(f'key {key}--values type {FloatAttr.new(val[self.counter])} counter { self.counter}')
        #     else :
        #         print ( f'key {key}--values type {  val[self.counter] } counter {self.counter}' )


        return WeldingMachineEntity(
            id='',

            barcode=TextAttr.new(to_predict['BarCode'][self.counter]),
            face=TextAttr.new(to_predict['Face'][self.counter]),
            cell=TextAttr.new(to_predict['Cell'][self.counter]),
            point=TextAttr.new(to_predict['Point'][self.counter]),
            group=TextAttr.new(to_predict['Group'][self.counter]),
            joules=FloatAttr.new(to_predict['Output Joules'][self.counter]),
            charge=FloatAttr.new(to_predict[ 'Charge (v)'][self.counter]),
            residue=FloatAttr.new(to_predict[ 'Residue (v)'][self.counter]),
            force_n=FloatAttr.new(to_predict[ 'Force L N'][self.counter]),
            force_n_1=FloatAttr.new(to_predict['Force L N_1'][self.counter]),
            datetime=TextAttr.new(f"{datetime.now().isoformat()}")
        )
        # return WeldingMachineEntity(
        #     id='',

        #     barcode=TextAttr.new(to_predict['BarCode'][self.counter]),
        #     face=TextAttr.new(to_predict['Face'][self.counter]),
        #     cell=TextAttr.new(to_predict['Cell'][self.counter]),
        #     point=TextAttr.new(to_predict['Point'][self.counter]),
        #     group=TextAttr.new(to_predict['Group'][self.counter]),
        #     joules=FloatAttr.new(to_predict['Output Joules'][self.counter]),
        #     charge=FloatAttr.new(to_predict[ 'Charge (v)'][self.counter]),
        #     residue=FloatAttr.new(to_predict[ 'Residue (v)'][self.counter]),
        #     force_n=FloatAttr.new(to_predict[ 'Force L N'][self.counter]),
        #     force_n_1=FloatAttr.new(to_predict['Force L N_1'][self.counter]),
        #     datetime=TextAttr.new(f"{datetime.now().isoformat()}")
        # )
        # return WeldingMachineEntity(
        #     id='34',
        #
        #     barcode=TextAttr.new('bc-xyz'),
        #     face=TextAttr.new(random.choice(['f1', 'f2'])),
        #     cell=TextAttr.new(random.choice(['c1', 'c2'])),
        #     point=TextAttr.new(random.choice(['p1', 'p2'])),
        #     group=TextAttr.new(random.choice(['g1', 'g2'])),
        #     joules=FloatAttr.new(1.0335 + seed),
        #     charge=FloatAttr.new(2.0335 + seed),
        #     residue=FloatAttr.new(0.0335 + seed),
        #     force_n=FloatAttr.new(3.09 + seed),
        #     force_n_1=FloatAttr.new(4.0335 + seed),
        #     datetime=TextAttr.new(f"{datetime.now().isoformat()}")
        # )