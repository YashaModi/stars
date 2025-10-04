"""util file for the 'stars' project"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import skyfield 
import pickle


from datetime import datetime, timedelta
from skyfield.api import load, utc
from sklearn.preprocessing import StandardScaler

# Define constants for feature engineering (Orbital Periods in Days)
PERIOD_YEAR = 365.25          # Earth's orbital period

# Dictionary for Sidereal Periods (Used to make feature generation dynamic)
PLANET_PERIODS = {
    'mercury': 87.97,
    'venus': 224.70,
    'earth': PERIOD_YEAR,
    'mars': 686.98,
    'jupiter': 4332.6,
    'saturn': 10759.2,
    'uranus': 30685.4,
    'neptune': 60189.0
}


def save_scaler(scaler: StandardScaler, filepath: str):
    """
    Saves the fitted StandardScaler object to a file using pickle.
    This is essential to ensure the same scaling is used for prediction as was used for training.
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved successfully to: {filepath}")
    except Exception as e:
        print(f"ERROR: Failed to save scaler to {filepath}: {e}")

def load_scaler(filepath: str) -> StandardScaler or None:
    """
    Loads a fitted StandardScaler object from a file using pickle.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: Scaler file not found at {filepath}.")
        return None
    try:
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded successfully from: {filepath}")
        return scaler
    except Exception as e:
        print(f"ERROR: Failed to load scaler from {filepath}: {e}")
        return None

def generate_planetary_ephemeris_df(
    target_planet: str, 
    start_date: datetime, 
    end_date: datetime, 
    time_step: timedelta = timedelta(days=1),
    ephemeris_file: str = 'de421.bsp'
) -> pd.DataFrame:
    """
    Generates a DataFrame containing the geocentric position (RA, Dec) 
    for a given celestial body over a specified time range.

    Args:
        target_planet (str): The name of the planet to track (e.g., 'mars', 'jupiter').
        start_date (datetime): The start date for the data range.
        end_date (datetime): The end date for the data range.
        time_step (timedelta): The interval between data points (default: 1 day).
        ephemeris_file (str): The name of the JPL ephemeris file (default: 'de421.bsp').

    Returns:
        pd.DataFrame: A DataFrame with time, Julian date, Right Ascension (RA), Declination (Dec),
        geocentric XYZ coordinates (in AU).
    """
    try:
        # Load the JPL ephemeris data file and timescale
        planets = load(ephemeris_file)
        ts = load.timescale()

    except Exception as e:
        print(f"ERROR: Could not load ephemeris file '{ephemeris_file}'. Check file existence and Skyfield installation.")
        print(e)
        return pd.DataFrame()

    # Determine the correct name for Skyfield lookup
    target_name = target_planet.lower()
    
    # List of bodies whose center is explicitly listed in de421.bsp (inner planets/Moon)
    inner_bodies = ['mercury', 'venus', 'earth', 'mars', 'moon']
    
    # Use barycenter for outer planets since de421.bsp only provides barycenter data for them
    if target_name not in inner_bodies:
        skyfield_target_name = target_name + ' barycenter'
        print(f"Note: Using '{skyfield_target_name}' for target lookup.")
    else:
        skyfield_target_name = target_name

    # Get the target body and the observer (Earth)
    try:
        target = planets[skyfield_target_name]
    except KeyError:
        # Check against the derived name, which should exist if the input was valid
        print(f"ERROR: Skyfield target '{skyfield_target_name}' (derived from '{target_planet}') not found in ephemeris data.")
        return pd.DataFrame()
        
    earth = planets['earth']

    # 1. Generate a list of dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += time_step

    if not dates:
        print("ERROR: Time range resulted in zero data points.")
        return pd.DataFrame()

    # 2. Convert Python datetimes to Skyfield time objects
    # Skyfield requires time zone aware datetime objects (using UTC here)
    timezone_aware_dates = [d.replace(tzinfo=utc) for d in dates]
    t = ts.utc(timezone_aware_dates)

    # 3. Calculate Geocentric Position (Position as seen from Earth)
    # astrometric = position of target relative to the observer (Earth)
    astrometric = earth.at(t).observe(target)

    # Get Right Ascension, Declination, and distance
    ra, dec, distance = astrometric.radec()

    # Get the astrometric XYZ coordinates in AU
    pos_vector_au = astrometric.xyz.au # 3xN NumPy array

    # 4. Calculate Geocentric Position (Sun Position as seen from Earth for features)
    sun = planets['sun']
    astrometric_sun = earth.at(t).observe(sun)
    pos_sun_au = astrometric_sun.xyz.au

    # 5. Create the DataFrame (Your Dataset)
    data = {
            'Time_UTC': dates,
            'Julian_Date': t.tdb,
            'RA_deg': ra.degrees,          
            'Dec_deg': dec.degrees,        
            'Distance_AU': distance.au,    
            
            # Geocentric XYZ coordinates in AU (Core Targets for Training)
            'X_au': pos_vector_au[0],
            'Y_au': pos_vector_au[1],
            'Z_au': pos_vector_au[2],

             # Sun Geocentric XYZ coordinates (Stabilization Features)
            'Sun_X_au': pos_sun_au[0],
            'Sun_Y_au': pos_sun_au[1],
            'Sun_Z_au': pos_sun_au[2]


        }

    df = pd.DataFrame(data)
    print(f"Dataset for {target_planet.capitalize()} created successfully with {len(df)} data points.")
    
    return df

def add_astronomy_features(df: pd.DataFrame, target_planet: str) -> pd.DataFrame:
    """
    Adds polynomial and sinusoidal features related to orbital mechanics dynamically 
    based on the target planet's period and the resulting synodic period with Earth.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'Julian_Date'.
        target_planet (str): The name of the target planet (e.g., 'mars', 'jupiter').

    Returns:
        pd.DataFrame: The DataFrame with new feature columns added.
    """
    
    # --- Dynamic Period Lookup ---
    target_planet_key = target_planet.lower()
    if target_planet_key not in PLANET_PERIODS:
        raise ValueError(f"Period for target planet '{target_planet}' not found in PLANET_PERIODS dictionary.")

    P_target = PLANET_PERIODS[target_planet_key]

    # Calculate Synodic Period (P_syn)
    # Formula: 1/P_syn = |1/P_Earth - 1/P_target|
    P_synodic = 1 / abs((1 / PERIOD_YEAR) - (1 / P_target))
    
    print(f"Calculated {target_planet}'s Synodic Period with Earth: {P_synodic:.2f} days.")

    # 1. Create Base Time Index (normalized Julian Date)
    JD_min = df.Julian_Date.min()
    df["Time_Index"] = df.Julian_Date - JD_min
    time_index = df["Time_Index"]

    # 2. Polynomial Features
    df["Time_Index_2"] = time_index ** 2
    df["Time_Index_3"] = time_index ** 3
    df["Time_Index_4"] = time_index ** 4

    # 3. Sinusoidal/Cyclic Features
    
    # Earth's Annual Cycle Features
    df["Sin_Year"] = np.sin(2 * np.pi * time_index / PERIOD_YEAR)
    df["Cos_Year"] = np.cos(2 * np.pi * time_index / PERIOD_YEAR)

    # Target Planet's Orbital Cycle Features (Dynamic)
    df[f'Sin_{target_planet_key.capitalize()}'] = np.sin(2 * np.pi * time_index / P_target)
    df[f'Cos_{target_planet_key.capitalize()}'] = np.cos(2 * np.pi * time_index / P_target)
    
    # Earth-Target Synodic Cycle Features (Dynamic)
    df["Sin_Synodic"] = np.sin(2 * np.pi * time_index / P_synodic)
    df["Cos_Synodic"] = np.cos(2 * np.pi * time_index / P_synodic)

    # 4. Interaction Features
    df['Sin_Year_Sin_Synodic'] = df['Sin_Year'] * df['Sin_Synodic']
    df['Sin_Year_Cos_Synodic'] = df['Sin_Year'] * df['Cos_Synodic']
    df['Cos_Year_Sin_Synodic'] = df['Cos_Year'] * df['Sin_Synodic']
    df['Cos_Year_Cos_Synodic'] = df['Cos_Year'] * df['Cos_Synodic']
    
    print(f"Added dynamic features (Time Index, Polynomial, Earth Cycle, Target Cycle, Synodic Cycle, Interaction) to the DataFrame.")
    
    return df

def xyz_to_radec(pred_au):
    """
    Converts predicted equatorial rectangular coordinates (X, Y, Z in AU) 
    to Right Ascension (RA) and Declination (Dec) in degrees.
    1. Extract Coordinates
    2. Calculate Predicted Distance (r) where r = sqrt(X^2 + Y^2 + Z^2)
    3. Calculate Declination (Dec) - The angle North/South where Dec = arcsin(Z / r) 
    4. Calculate Right Ascension (RA) - The angle East/West where RA = arctan2(Y, X)
    5. Convert Dec and RA to degrees from radians
    6. Normalize RA to the 0 to 360 degree range (adding +360 for negative value)
    """

    X_pred = pred_au[:, 0]
    Y_pred = pred_au[:, 1]
    Z_pred = pred_au[:, 2]
    
    r_pred = np.sqrt(X_pred**2 + Y_pred**2 + Z_pred**2)
    
    dec_rad = np.arcsin(Z_pred / r_pred)
    dec_deg = np.degrees(dec_rad)
    
    ra_rad = np.arctan2(Y_pred, X_pred)
    ra_deg = np.degrees(ra_rad)
    
    # arctan2 returns results in the range [-180, 180]. Add 360 to negative values.
    ra_deg = np.where(ra_deg < 0, ra_deg + 360, ra_deg)
    
    # Return results as a DataFrame for easy handling
    results_df = pd.DataFrame({
        'Predicted_RA_deg': ra_deg,
        'Predicted_Dec_deg': dec_deg
    })
    
    return results_df

def models_loader(directory: str, models_lis: list, target_planet: str) -> list:
    """
    Loads Keras models from a specified directory, checking both .keras and .h5 extensions.
    The model filenames are expected to be prefixed by the target planet name.
    
    Args:
        directory (str): The folder containing the Keras model files.
        models_lis (list): A list of model suffixes (e.g., ['mm0', 'mm1', 'mm2']).
        target_planet (str): The name of the planet the model was trained for (e.g., 'mars').

    Returns:
        list: A list of loaded Keras Model objects.
    """
    models = []
    
    if not os.path.exists(directory):
        print(f"ERROR: Model directory not found: {directory}")
        return models
    
    print(f"Attempting to load {len(models_lis)} models for {target_planet} from '{directory}'...")
    
    for suffix in models_lis:
        # Base filename format: '{target_planet}_position_predictor_{suffix}'
        base_filename = f'{target_planet}_position_predictor_{suffix}'
        path_keras = os.path.join(directory, f'{base_filename}.keras')
        path_h5 = os.path.join(directory, f'{base_filename}.h5')
        
        loaded = False
        model = None

        # --- 2. Try to load .keras (Preferred) ---
        if os.path.exists(path_keras):
            print(f"Attempting to load: {path_keras}")
            model_path_to_use = path_keras
            
        # --- 3. Fallback to .h5 ---
        elif os.path.exists(path_h5):
            print(f"File not found at {path_keras}. Falling back to: {path_h5}")
            model_path_to_use = path_h5
            
        else:
            print(f"ERROR: Model file not found for {target_planet} with suffix '{suffix}'. Checked both .keras and .h5 extensions.")
            continue # Skip to the next model in the list
            
        # --- 4. Load the found model ---
        try:
            # tf.keras.models.load_model handles both .keras and .h5 formats
            model = tf.keras.models.load_model(model_path_to_use)
            models.append(model)
            loaded = True
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load Keras model from {model_path_to_use}: {e}")
            # If loading fails (e.g., file corruption), we stop trying to load this specific model
            
    if models:
        print(f"Successfully loaded {len(models)} models.")
    else:
        print(f"No models were successfully loaded for {target_planet}.")

    return models