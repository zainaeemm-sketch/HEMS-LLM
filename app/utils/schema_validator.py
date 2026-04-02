# app/utils/schema_validator.py
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Optional, Literal
from datetime import date

class ApplianceSetting(BaseModel):
    name: str
    start_time: Optional[str] = None  # "HH:MM"
    end_time: Optional[str] = None    # "HH:MM"
    can_shift: Optional[bool] = False

    @validator("start_time", "end_time")
    def validate_time_format(cls, v):
        if v in (None, ""):
            return None
        s = str(v)
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError("Time must be HH:MM")
        hh, mm = parts
        if not (hh.isdigit() and mm.isdigit()):
            raise ValueError("Time must be HH:MM numeric")
        h = int(hh)
        m = int(mm)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError("Time must be valid 24h HH:MM")
        return f"{h:02d}:{m:02d}"

class HEMSParameters(BaseModel):
    city: str
    start_date: date
    end_date: date
    user_type: Literal["residential", "industrial"]
    appliances: List[ApplianceSetting] = Field(default_factory=list)
    Tmin: float
    Tmax: float
    max_power: Optional[float] = None
    do_not_disturb: Optional[List[str]] = Field(default_factory=list)
    solar_pv_capacity: Optional[float] = 0.0
    pv_forecast: Optional[List[float]] = Field(default_factory=list)

    # extra fields we keep in the stored payload (not required by the form)
    forecast_source: Optional[str] = None
    forecast_metrics: Optional[dict] = Field(default_factory=dict)
    weather_hourly: Optional[dict] = None
    optimization_results: Optional[dict] = None

    @validator("end_date")
    def validate_date_range(cls, v, values):
        if "start_date" in values and v < values["start_date"]:
            raise ValueError("End date must be after start date")
        return v

    @validator("Tmax")
    def validate_temperature(cls, v, values):
        if "Tmin" in values and v <= values["Tmin"]:
            raise ValueError("Tmax must be greater than Tmin")
        return v

    @validator("appliances")
    def validate_appliances(cls, v):
        for a in v or []:
            if not a.name or not isinstance(a.name, str):
                raise ValueError("Each appliance must have a valid name.")
        return v
