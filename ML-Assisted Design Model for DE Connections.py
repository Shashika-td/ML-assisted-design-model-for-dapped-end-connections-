import numpy as np
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load("trained_GA_ANN_model.pkl")
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file not found")
    exit()

# Load the scaler used in training (To scale input parameters before making predictions)
try:
    scaler = joblib.load("scaler.pkl") 
    print("Scaler loaded successfully!")
except FileNotFoundError:
    print("Error: Scaler file not found")
    exit()

# Define the input parameters
Expected_value = 300 # Target capacity of the dappe-end connection
Threshold = 5  # Error margin (eg: 300 ± 5)
h_value = 400  # Depth od the dappe-end connections (Range: 200 - 500 mm)
ad_value = 0.47  # a/d ratio of the connections (Range: 0.23 - 1.51)


number_of_gen_data = 10000  # Number of devisions samples

# Define min-max values for each variable feature
min_max_vals = {
    "Num of Flex_layers": {"min": 1, "max": 4},
    "Flex_Spacing": {"min": 0, "max": 210},
    "Flex_RF_Strength": {"min": 67.62, "max": 1282.2},
    "Number of Han_layers": {"min": 1, "max": 6},
    "Hanger_spacing": {"min": 0, "max": 180},
    "Hanger_Strength": {"min": 58.73, "max": 1456.3},
    "Com. Strength": {"min": 25, "max": 71.6}
}
# Generate random input data within min-max ranges
gen_data = np.zeros((number_of_gen_data, len(variable_features)))  # Pre-allocate array

for i, feature in enumerate(variable_features):
    min_val = min_max_vals[feature]["min"]
    max_val = min_max_vals[feature]["max"]

    if feature in ["Num of Flex_layers", "Number of Han_layers"]:
        # Generate integers directly for Nh and Nv
        gen_data[:, i] = np.random.randint(min_val, max_val + 1, size=number_of_gen_data)
    else:
        # Generate continuous values within min-max
        gen_data[:, i] = np.random.uniform(min_val, max_val, size=number_of_gen_data)

# Convert generated data to DataFrame
df_gen = pd.DataFrame(gen_data, columns=variable_features)

# Add use defined values for Depth (h) and a/d
df_gen.insert(0, "Depth (h)", h_value)
df_gen.insert(1, "a/d", ad_value)

# Apply constraints: Set Sh = 0 when Nh = 1, or Sv = 0 when Nv = 1
df_gen.loc[df_gen["Num of Flex_layers"] == 1, "Flex_Spacing"] = 0
df_gen.loc[df_gen["Number of Han_layers"] == 1, "Hanger_spacing"] = 0

# Round Hanger_Spacing values to nearest 5
df_gen.loc[df_gen["Number of Han_layers"] > 1, "Hanger_spacing"] = (df_gen["Hanger_spacing"] / 5).round() * 5

# Round Flex_Spacing values to nearest 5
df_gen.loc[df_gen["Num of Flex_layers"] > 1, "Flex_Spacing"] = (df_gen["Flex_Spacing"] / 5).round() * 5

# Ensure that all flexural layers are placed within the beam
condition = (df_gen["Num of Flex_layers"] - 1) * df_gen["Flex_Spacing"] - 100 <= df_gen["Depth (h)"] 
df_gen = df_gen[condition]


# Ensure that input parameters are scaled, as the GA-ANN was trained on scaled data.
df_gen_scaled = scaler.transform(df_gen)

# Predict DE_Capacity using the trained GA-ANN model
df_gen["prediction"] = model.predict(df_gen_scaled)

# Compute absolute error
df_gen["error"] = df_gen["prediction"] - Expected_value
df_gen["abs_error"] = abs(df_gen["error"])
df_gen["abs_error_%"] = (df_gen["abs_error"] / abs(Expected_value)) * 100  # Convert to percentage

# Sort and filter results within the threshold range
df_filtered = df_gen[df_gen["abs_error"] < Threshold].reset_index(drop=True)

# Check if filtered data is empty
if df_filtered.empty:
    print("No generated design combinations satisfy the threshold condition")
else:
    print(f"{len(df_filtered)} design combinations found within the threshold range")
    
    # Save the final filtered inputs to CSV
    output_file = "Predicted_parameters.csv"
    df_filtered.to_csv(output_file, index=False)
    print(f"Filtered results saved to {output_file}")
    
    # Show some filtered results
    print("\nPredicted parameters (First 5 rows):")
    print(df_filtered.head())
