from typing import Optional

from fipy.ngsi.entity import BaseEntity, FloatAttr, TextAttr


class WeldingMachineEntity(BaseEntity):
    type = 'WeldingMachine'

    barcode: Optional[TextAttr]
    face: Optional[TextAttr]
    cell: Optional[TextAttr]
    point: Optional[TextAttr]
    group: Optional[TextAttr]

    joules: Optional[FloatAttr]
    charge: Optional[FloatAttr]
    residue: Optional[FloatAttr]
    force_n: Optional[FloatAttr]
    force_n_1: Optional[FloatAttr]

    datetime: Optional[TextAttr]
