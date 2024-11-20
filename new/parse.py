import re

# Path to your file
file_path = r"C:\Users\cnort\Desktop\zeros4.txt"

# Function to parse gamma values from the file
def parse_zeros(file_path):
    gamma_values = []
    with open(file_path, 'r') as file:
        for line in file:
            # Match numerical lines (excluding headers and descriptions)
            match = re.match(r"^\s*([\d.]+)\s*$", line)
            if match:
                gamma_values.append(float(match.group(1)))
    return gamma_values

# Load the gamma values
gamma_values = parse_zeros(file_path)

# Display the first few parsed gamma values
print("Loaded gamma values:")
print(gamma_values[:10])  # Display first 10 values
