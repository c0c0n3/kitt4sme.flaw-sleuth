import pandas as pd

from fipy.ngsi.entity import EntitySeries


# TODO hard-coded stuff for app scaffolding to be replaced w/ actual QL call.

entity_query_result = {
    "entityId": "urn:ngsi-ld:Bot:2",
    "entityType": "Bot",
    "index": ["2022-03-28T18:03:18.923+00:00", "2022-03-28T18:03:20.458+00:00",
                "2022-03-28T18:03:22.011+00:00"],
    "attributes": [
        {
            "attrName": "direction",
            "values": ["S", "N", "N"]
        },
        {
            "attrName": "speed",
            "values": [1.308673138, 1.935175709, 1.451720504]
        }
    ]
}


def fetch_data_frame() -> pd.DataFrame:
    r = EntitySeries.from_quantumleap_format(entity_query_result)
    time_indexed_df = pd.DataFrame(r.dict()).set_index('index')
    return time_indexed_df
