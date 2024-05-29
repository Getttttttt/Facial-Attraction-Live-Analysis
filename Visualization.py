import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = './attractiveness_scores.csv'
data = pd.read_csv(file_path)

# Extract the frame numbers from the filenames
data['frame_number'] = data['filename'].str.extract('(\d+)').astype(int)

# Sort the data by frame number
data = data.sort_values('frame_number')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data['frame_number'], data['attractiveness_score'], linestyle='-')
plt.xlabel('Frame Number')
plt.ylabel('Attractiveness Score')
plt.title('Attractiveness Score over Time')
plt.grid(True)
plt.show()
