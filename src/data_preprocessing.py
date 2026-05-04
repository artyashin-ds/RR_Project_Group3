import pandas as pd
import numpy as np
import regex as re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df_cars = pd.read_csv(f'../data/used_cars.csv')

# cleaning the integer column by removing uncessary signs as mi, ., $ and spaces
df_cars['milage'] = df_cars['milage'].astype(str).str.replace("mi.","")
df_cars['milage'] = df_cars['milage'].str.replace(",","").astype(int)
df_cars['price'] = df_cars['price'].str.replace("$","")
df_cars['price'] = df_cars['price'].str.replace(",","").astype(int)

# cleaning the fuel type column by replacing the nan values with electric and not supported with hydrogen and plug in hybrid with hybrid and E85 flex fuel with flex fuel
df_cars['fuel_type'] = df_cars['fuel_type'].astype(str).replace('nan' ,'Electric')
df_cars['fuel_type'] = df_cars['fuel_type'].astype(str).replace('not supported', 'Hydrogen')
df_cars['fuel_type'] = df_cars['fuel_type'].astype(str).replace('Plug-In Hybrid', 'Hybrid')
df_cars['fuel_type'] = df_cars['fuel_type'].astype(str).replace('E85 Flex Fuel', 'Flex_Fuel')

# checking the value counts of the fuel type column to see if the cleaning process was successful
df_cars['fuel_type'].value_counts(dropna=False).reset_index()

# extraction of engine size
def extract_engine_size(engine_str):
    if pd.isna(engine_str):
        return None    
    engine_str = str(engine_str).lower()
    
    # Skip electric motors - they dont have engine size in liters
    if 'electric' in engine_str:
        return 'Electric'      
    # Look for patterns like:
    # "3.7L", "2.0L" (with decimal and L)
    # "3L", "2L" (without decimal and L)  
    # "3.5 Liter", "2.4 Liter" (with Liter)
    pattern = r'(\d+\.\d+)\s*(?:l|liter)|(\d+)\s*(?:l|liter)'
    matches = re.findall(pattern, engine_str)
    
    if matches:
        for match in matches:
            if match[0]:  
                return float(match[0])
            elif match[1]:  
                return float(match[1])
    
    return None


def extract_horsepower(engine_str):
    if pd.isna(engine_str):
        return None
    # Look for patterns like "300.0HP", "292.0HP", "120HP"
    pattern = r'(\d+\.\d+)HP|(\d+)HP'
    matches = re.findall(pattern, str(engine_str))
    if matches:
        for match in matches:
            for group in match:
                if group:
                    return float(group)
    return None

# Apply the extraction functions to create new columns for engine size and horsepower
df_cars['engine_liters'] = df_cars['engine'].apply(extract_engine_size)
df_cars['horsepower'] = df_cars['engine'].apply(extract_horsepower)


def get_transmission_type(trans_str):
    if pd.isna(trans_str) or trans_str in ['–', '2', 'SCHEDULED FOR OR IN PRODUCTION']:
        return 'Unknown'
    
    trans_str = str(trans_str).upper()
    
    if any(x in trans_str for x in ['CVT', 'VARIABLE']):
        return 'CVT'
    elif any(x in trans_str for x in ['MANUAL', 'M/T', 'MT']):
        return 'Manual'
    elif any(x in trans_str for x in ['DUAL-CLUTCH', 'DCT', 'PDK']):
        return 'Dual-Clutch'
    elif any(x in trans_str for x in ['AUTOMATIC', 'A/T', 'AT']):
        return 'Automatic'
    elif 'F' in trans_str:  # Common abbreviation for Automatic
        return 'Automatic'
    else:
        return 'Other'

def extract_gears(trans_str):
    if pd.isna(trans_str):
        return None
    
    trans_str = str(trans_str)
    
    # Look for patterns like "6-Speed", "8-Speed", "10-Speed", "6-Spd", "8-Spd"
    pattern = r'(\d+)[\s-]*(?:Speed|Spd)'
    matches = re.findall(pattern, trans_str, re.IGNORECASE)
    if matches:
        return int(matches[0])
    
    # Handle special cases
    if '1-Speed' in trans_str or 'Single-Speed' in trans_str:
        return 1
    elif 'CVT' in trans_str:  # CVT doesn't have fixed gears
        return 1
    elif trans_str in ['Automatic', 'Manual', 'M/T', 'A/T']:
        return np.nan  # Unknown number of gears
    
    return None

# Create new columns for transmission type and number of gears
df_cars['transmission_type'] = df_cars['transmission'].apply(get_transmission_type)
df_cars['num_gears'] = df_cars['transmission'].apply(extract_gears)

# Dropping columns that should be transformed and later these columns will encoded
df_ver2 = df_cars[['brand', 'model', 'model_year', 'milage', 'fuel_type', 'ext_col', 'int_col', 'accident', 'clean_title',
       'price', 'engine_liters', 'horsepower', 'transmission_type',
       'num_gears']]

# filling the missing values in the accident column with None reported
df_ver2['accident'] = df_ver2['accident'].fillna('None reported')

# creating dummy variables for the fuel type and transmission type columns and creating a binary variable for the accident column
fuel_type_df_dummy = pd.get_dummies(df_ver2['fuel_type'].replace('–','not_provided'), prefix='fuel_type')
transmission_type = pd.get_dummies(df_ver2['transmission_type'] , prefix='transmission_type')
accident_binary = pd.DataFrame(np.where(df_ver2['accident']=='None reported', 1, 0)).rename(columns={0:'Accident'})

# creating a binary variable for the clean title column
df_ver2['clean_title']  = pd.DataFrame(np.where(df_ver2['clean_title']=='Yes', 1, 0)).rename(columns={0:'Clean_title'})
df_ver2['engine_liters'] = df_ver2['engine_liters'].replace('Electric',0)
df_ver2['model_clean'] = df_ver2['brand'] +'_' + df_ver2['model'].apply(lambda x: '_'.join(str(x).split(' ')[:2]))

# Concatenating the original dataframe with the new dummy variables and binary variables
df_ver_3 = df_ver2.copy()

# Concatenating the original dataframe with the new dummy variables and binary variables
df_ver_3 = pd.concat([df_ver_3, fuel_type_df_dummy, transmission_type,  accident_binary], axis=1)

# Fill missing engine liters using the median of the respective Brand.
# Logic: BMWs generally have similar engine sizes, distinct from Toyotas.
df_ver_3['engine_liters'] = df_ver_3['engine_liters'].fillna(
                                                        df_ver_3.groupby('brand')['engine_liters'].transform('median')
                                                            )

# Pass 1 (High Precision): Fill HP based on Brand AND Engine Size.
# Logic: A 2.0L BMW engine usually has a specific HP range (e.g., ~250hp), 
# which is different from a 2.0L Toyota engine (~170hp).

df_ver_3['horsepower'] = df_ver_3['horsepower'].fillna(
                    df_ver_3.groupby(['brand', 'engine_liters'])['horsepower'].transform('median')
                                                        )
# Pass 2 (Fallback): If Pass 1 failed (e.g., unknown engine size), 
# fill using the median HP of the Brand only.
df_ver_3['horsepower'] = df_ver_3['horsepower'].fillna(
                    df_ver_3.groupby(['brand'])['horsepower'].transform('median')
                                                        )

# Pass 1 (Most Specific): Look for the exact same Car Model and Transmission.
# Logic: A "BMW 3-Series Automatic" almost always has the same gear count.
df_ver_3['num_gears']= df_ver_3['num_gears'].fillna(
                    df_ver_3.groupby(['brand', 'model_clean', 'transmission_type'])['num_gears'].transform('median')
                            )
# Pass 2 (Generalization): If the Model is rare/unknown, look at the Brand + Transmission.
# Logic: "All BMW Automatics" likely share similar gear counts (e.g., 8 gears).
df_ver_3['num_gears']= df_ver_3['num_gears'].fillna(
                    df_ver_3.groupby(['brand', 'transmission_type'])['num_gears'].transform('median')
                            )
# Pass 3 (Final Safety Net): Fill remaining using Transmission Type only.
# Logic: If Brand is unknown, a generic "Automatic" usually implies ~6 gears.
df_ver_3['num_gears']= df_ver_3['num_gears'].fillna(
                    df_ver_3.groupby(['brand'])['num_gears'].transform('median')
                            )

# After all passes, if there are still missing values (e.g., very rare brands), we can fill with the overall median or a placeholder value.
df_ver_3 = df_ver_3.drop(['accident', 'fuel_type', 'clean_title', 'model', 'transmission_type'] , axis=1)

# Drop any remaining rows with missing values (if any)
df_ver_3  = df_ver_3.dropna()

# Calculate car age
current_year = 2024
df_ver_3['car_age'] = current_year - df_ver_3['model_year']

# Calculate annual mileage
df_ver_3['annual_mileage'] = round(df_ver_3['milage'] / df_ver_3['car_age'].replace(0, 1), 2)

# Age-miles ratio (how hard the car was used)
df_ver_3['miles_per_year']  = df_ver_3['milage'] / (df_ver_3['car_age'] + 1)

# scale numerical features (like Mileage)
# 1. Price (Target): Log Transform
# We do this to predict percentages, not raw dollars
df_ver_3['price_log'] = np.log1p(df_ver_3['price']) 

# 2. Mileage (Feature): Log Transform AND Scale
# Log first to fix the distribution (diminishing impact)
df_ver_3['milage_log'] = np.log1p(df_ver_3['milage']) 
# Then Scale to fix the "high numbers"
scaler = MinMaxScaler()
df_ver_3['milage_scaled'] = scaler.fit_transform(df_ver_3[['milage_log']])
df_ver_3['miles_per_year_scaled'] = scaler.fit_transform(df_ver_3[['miles_per_year']])
df_ver_3['annual_mileage_scaled'] = scaler.fit_transform(df_ver_3[['annual_mileage']])

# 3. Year (Feature): Scale
df_ver_3['car_age_scaled'] = scaler.fit_transform(df_ver_3[['car_age']])

# Drop original columns that have been transformed
df_ver_3 = df_ver_3.drop(columns=['price', 'milage', 'milage_log', 'miles_per_year', 'annual_mileage', 'car_age', 'model_year'])


# Prepare the final dataset for modeling by separating features and target variable, and then splitting into training and testing sets.
X = df_ver_3.drop('price_log', axis=1)
y = df_ver_3['price_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=908)
# X_train.shape, X_test.shape

def target_encode_train_test(X_train, y_train, X_test, col, m=10):

    # 1. Calculate Global Mean (or Median) on Train
    global_val = y_train.median()
    
    # 2. Aggregate data on Train
    temp_df = X_train.copy()
    temp_df['target'] = y_train
    agg = temp_df.groupby(col)['target'].agg(['count', 'median'])
    
    # 3. Calculate Smoothed Values
    counts = agg['count']
    means = agg['median']
    smooth_scores = (counts * means + m * global_val) / (counts + m)
    
    # 4. Create the Dictionary (Map)
    map_dict = smooth_scores.to_dict()
    
    # --- TRANSFORMATION PHASE ---
    # 5. Map to Train
    X_train_encoded = X_train[col].map(map_dict)
    
    # 6. Map to Test
    # CRITICAL: If a category in Test wasn't in Train, fill with Global Value
    X_test_encoded = X_test[col].map(map_dict).fillna(global_val)
    
    return X_train_encoded, X_test_encoded

# Target encode the specified columns in both training and testing sets
cols_to_target_encode = ['brand', 'model_clean', 'ext_col', 'int_col']

for col in cols_to_target_encode:
    X_train[col], X_test[col] = target_encode_train_test(
        X_train, y_train, X_test, col, m=10
    )

print(X_train.shape, X_test.shape)

# Export X_train, X_test, y_train, y_test to CSV files for later use in modeling (replace if exists)
X_train.to_csv(f'../data/clean/X_train.csv', index=False)
X_test.to_csv(f'../data/clean/X_test.csv', index=False)
y_train.to_csv(f'../data/clean/y_train.csv', index=False)
y_test.to_csv(f'../data/clean/y_test.csv', index=False)