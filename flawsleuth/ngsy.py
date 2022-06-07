from typing import Optional
from pydantic import BaseModel
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
 
class RawReading ( BaseModel ):
    Barcode: Optional[str]
    Face: Optional[str]
    Cell: Optional[str]
    Point: Optional[str]
    Group: Optional[str]
    Joules: Optional[float]
    Charge: Optional[float]
    Residue: Optional[float]
    Force_N: Optional[float]
    Force_N_1: Optional[float]
    Datetime: Optional[str]

    def to_machine_entity (self, entity_id) -> WeldingMachineEntity:
        e = WeldingMachineEntity ( id=entity_id )

        e.Barcode = TextAttr.new ( self.Barcode )
        e.Face = TextAttr.new ( self.Face )
        e.Cell = TextAttr.new ( self.Cell )
        e.Point = TextAttr.new ( self.Point )
        e.Group = TextAttr.new ( self.Group )
        e.Joules = FloatAttr.new ( self.Joules )
        e.Charge = FloatAttr.new ( self.Charge )
        e.Residue = FloatAttr.new ( self.Residue )
        e.Force_N = FloatAttr.new ( self.Force_N )
        e.Force_N_1 = FloatAttr.new ( self.Force_N_1 )
        e.Datetime = TextAttr.new ( self.Datetime )
        return e

class AnomalyDetectionEntity ( BaseEntity ):
    type = 'AnomalyDetection'
    # sensor = dict
    Label: FloatAttr

class ForecastngEntity ( BaseEntity ) :
    type = 'AnomalyDetection'
    # sensor = dict
    mu: FloatAttr
    std:FloatAttr