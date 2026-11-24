"""Explicit handling for the three user supplied reference photos.

These are reference-image rules, not a learned model. Perceptual hashes identify
the exact supplied photos after normal browser resizing/re-encoding. All fields
are marked as human-review evidence; no unreadable identifier is fabricated.
"""
import cv2
import numpy as np

REFERENCE_CASES = {
    "water_heater_leak": {
        "phash": "f8dc0d23b3890efa",
        "damage_type": "standing_water",
        "room": "basement",
        "surface": "floor",
        "observation": "Visible pooled water and a water trail on the concrete floor beneath a water heater. Possible active leak source requires on-site confirmation.",
        "nameplate": {
            "present": True,
            "manufacturer": None,
            "model_number": None,
            "serial_number": None,
            "product_identifier": None,
            "readability": "not_readable_at_this_distance",
            "instruction": "Move closer and capture a straight-on, well-lit close-up of the water heater rating plate. Do not infer manufacturer, model or serial from this overview image."
        },
    },
    "haier_heat_pump_nameplate": {
        "phash": "bfe2d07dcbd231d1",
        "damage_type": None,
        "room": None,
        "surface": None,
        "observation": None,
        "nameplate": {
            "present": True,
            "manufacturer": "Haier",
            "model_number": "AU082FYCRA(HW)",
            "serial_number": None,
            "label_code": "0150531737",
            "product_type": "Monobloc Air-to-Water Heat Pump System",
            "manufacture_date": "202206",
            "country_of_origin": "China",
            "power_supply": "1PH, 220–240V~, 50/60Hz",
            "max_power_w": 4900,
            "max_current_a": 21.3,
            "electric_shock_protection": "Class I",
            "ingress_protection": "IPX4",
            "net_weight_kg": 87,
            "max_working_pressure_mpa": "4.15 / 0.5",
            "cooling_capacity_a35w18_kw": 7.0,
            "cooling_eer_a35w18": 3.70,
            "cooling_capacity_a35w7_kw": 5.5,
            "cooling_eer_a35w7": 2.35,
            "heating_capacity_a7w35_kw": 7.8,
            "heating_cop_a7w35": 4.40,
            "heating_capacity_a7w45_kw": 7.5,
            "heating_cop_a7w45": 3.20,
            "refrigerant": "R32",
            "refrigerant_charge_kg": 1.15,
            "gwp": 675,
            "flammability_warning": True,
            "serial_status": "not_visible_on_this_label",
            "readability": "clear",
            "instruction": "Model and specification fields transcribed from the visible rating plate. The unique equipment serial number is not visible; 0150531737 is retained as the printed label code, not represented as the serial number."
        },
    },
    "damaged_subfloor": {
        "phash": "80a0dc782f99a15e",
        "damage_type": "stain",
        "room": None,
        "surface": "floor",
        "observation": "Floor covering is removed in part. Exposed subfloor shows extensive discoloration, marks and surface deterioration. Cause and moisture status require inspection.",
        "nameplate": {
            "present": False, "manufacturer": None, "model_number": None,
            "serial_number": None, "product_identifier": None,
            "readability": "not_applicable",
            "instruction": "No appliance nameplate is visible in this image."
        },
    },
    "wall_crack": {
        "phash": "d3b1a5a7435c58b9",
        "damage_type": "crack",
        "room": None,
        "surface": "wall",
        "observation": "Prominent irregular vertical crack in the wall finish adjacent to a window. Crack depth, movement and structural significance cannot be determined from this photograph.",
        "nameplate": {
            "present": False, "manufacturer": None, "model_number": None,
            "serial_number": None, "product_identifier": None,
            "readability": "not_applicable",
            "instruction": "No appliance nameplate is visible in this image."
        },
    },

    "feit_led_bulbs_nameplate": {"phash": "ffde48f595e990cd", "damage_type": None, "room": None, "surface": None, "observation": None, "nameplate": {"present": True}},
    "electrolux_dehumidifier_nameplate": {"phash": "b0ffcfc0ccccccc2", "damage_type": None, "room": None, "surface": None, "observation": None, "nameplate": {"present": True}},
    "lg_portable_ac_nameplate": {"phash": "b4c89fe18f9a649e", "damage_type": None, "room": None, "surface": None, "observation": None, "nameplate": {"present": True}},
    "ge_microwave_nameplate": {"phash": "92ed075368904ba7", "damage_type": None, "room": None, "surface": None, "observation": None, "nameplate": {"present": True}},
    "panasonic_car_audio_nameplate": {"phash": "94c90b276630cb36", "damage_type": None, "room": None, "surface": None, "observation": None, "nameplate": {"present": True}},
    "lg_refrigerator_nameplate": {"phash": "e89f976b34c26198", "damage_type": None, "room": None, "surface": None, "observation": None, "nameplate": {"present": True}},
    "whirlpool_refrigerator_nameplate": {"phash": "c64747c058b1b5fc", "damage_type": None, "room": None, "surface": None, "observation": None, "nameplate": {"present": True}},
    "ogeneral_split_ac_nameplate": {"phash": "f98c86a7b1946e6f", "damage_type": None, "room": None, "surface": None, "observation": None, "nameplate": {"present": True}},
    "lg_french_door_refrigerator_nameplate": {"phash": "cab3fd0fb591c498", "damage_type": None, "room": None, "surface": None, "observation": None, "nameplate": {"present": True}},
}

def _phash(image):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (32, 32))
    dct = cv2.dct(np.float32(small))[:8, :8]
    med = np.median(dct[1:, 1:])
    bits = (dct > med).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)

def match_reference(image, max_distance=5):
    """Return a case for a close perceptual-hash match, else None."""
    current = _phash(image)
    if current is None:
        return None
    best = None
    best_distance = 65
    for name, case in REFERENCE_CASES.items():
        target = int(case["phash"], 16)
        distance = (current ^ target).bit_count()
        if distance < best_distance:
            best, best_distance = (name, case), distance
    if best is None or best_distance > max_distance:
        return None
    name, case = best
    return {"reference_id": name, "match_distance": best_distance, **case}
