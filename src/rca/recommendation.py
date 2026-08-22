SENSOR_METADATA = {
    "sensor_2": {
        "name": "LPC outlet total temperature",
        "component": "Low-Pressure Compressor",
        "hpc_related": False
    },
    "sensor_3": {
        "name": "HPC outlet total temperature",
        "component": "High-Pressure Compressor",
        "hpc_related": True
    },
    "sensor_4": {
        "name": "LPT outlet total temperature",
        "component": "Low-Pressure Turbine",
        "hpc_related": False
    },
    "sensor_6": {
        "name": "Total temperature at nozzle throat",
        "component": "Nozzle",
        "hpc_related": False
    },
    "sensor_7": {
        "name": "HPC outlet static pressure",
        "component": "High-Pressure Compressor",
        "hpc_related": True
    },
    "sensor_8": {
        "name": "Fan inlet static pressure",
        "component": "Fan",
        "hpc_related": False
    },
    "sensor_9": {
        "name": "Bypass ratio",
        "component": "Nacelle",
        "hpc_related": False
    },
    "sensor_11": {
        "name": "HPC outlet static pressure",
        "component": "High-Pressure Compressor",
        "hpc_related": True
    },
    "sensor_12": {
        "name": "Fuel flow to Ps30 ratio",
        "component": "Combustor",
        "hpc_related": False
    },
    "sensor_13": {
        "name": "Corrected fan speed",
        "component": "Fan",
        "hpc_related": False
    },
    "sensor_14": {
        "name": "Corrected core speed",
        "component": "Core",
        "hpc_related": False
    },
    "sensor_17": {
        "name": "Bleed enthalpy",
        "component": "Bleed Air System",
        "hpc_related": False
    },
    "sensor_20": {
        "name": "Demanded fan speed",
        "component": "Fan Control",
        "hpc_related": False
    },
    "sensor_21": {
        "name": "LPT coolant bleed",
        "component": "Low-Pressure Turbine",
        "hpc_related": False
    }
}


def get_sensor_id(feature):
    feature = feature.lower()

    for sensor_id in SENSOR_METADATA:
        if sensor_id in feature:
            return sensor_id

    return None


def explain_feature(feature):
    sensor_id = get_sensor_id(feature)

    if sensor_id is None:
        return {
            "sensor": feature,
            "meaning": "Unknown model feature",
            "component": "Unknown"
        }

    metadata = SENSOR_METADATA[sensor_id]

    return {
        "sensor": sensor_id,
        "meaning": metadata["name"],
        "component": metadata["component"]
    }


def analyze_root_cause(top_factors):
    hpc_factors = []

    for factor in top_factors:
        sensor_id = get_sensor_id(factor["feature"])

        if sensor_id is None:
            continue

        if SENSOR_METADATA[sensor_id]["hpc_related"]:
            hpc_factors.append(factor)

    if len(hpc_factors) >= 2:
        confidence = "high"
    elif len(hpc_factors) == 1:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "component": "High-Pressure Compressor",
        "fault": "HPC degradation",
        "confidence": confidence,
        "evidence": hpc_factors
    }


def get_recommendation(rul, root_cause):
    if rul < 20:
        priority = "High"
        action = (
            "Schedule inspection of the High-Pressure Compressor "
            "and investigate the sensor factors contributing to the "
            "low RUL prediction."
        )

    elif rul < 50:
        priority = "Medium"
        action = (
            "Increase monitoring of the High-Pressure Compressor "
            "and schedule preventive inspection."
        )

    else:
        priority = "Low"
        action = (
            "Continue monitoring the engine condition and track "
            "the identified sensor factors."
        )

    return {
        "priority": priority,
        "action": action
    }