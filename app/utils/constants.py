# app/utils/constants.py
AVAILABLE_APPLIANCES = [
    "Heating",
    "Electric Heater",
    "Air Conditioner",
    "Water Heater",
    "Dishwasher",
    "Washing Machine",
    "Dryer",
    "EV Charger",
]

TIME_SLOTS = 24
TOU_PRICES = [0.10] * 7 + [0.20] * 11 + [0.10] * 6
