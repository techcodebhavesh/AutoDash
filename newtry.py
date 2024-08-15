import pandas as pd
import matplotlib.pyplot as plt
import json

# Load data from CSV
df = pd.read_csv("F:/Mayur/vit/innov8ors/ollama/AutoDash/app/csv/test.csv")

# Prepare the data for plotting
melted_df = pd.DataFrame({'genre': df['genre'], 'popularity': df['popularity'], 'id': df['id']})
grouped_df = melted_df.groupby('genre')['popularity'].mean().reset_index()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(grouped_df['genre'], grouped_df['popularity'])
plt.xlabel('Genre')
plt.ylabel('Mean Popularity')
plt.title('Trend of Popularity by Genre')
plt.savefig('plot.png')  # Save the plot to a file
result = {'type': 'plot', 'value': 'plot.png'}
print(result)

# Additional code to extract data
# Get current axes
ax = plt.gca()

# Get data from the first line in the axes
line = ax.get_lines()[0]
x_data = line.get_xdata().tolist()
y_data = line.get_ydata().tolist()

# Prepare the data to send to the frontend
plot_data = {'x': x_data, 'y': y_data}
print(plot_data)

# Convert the plot_data to JSON
json_plot_data = json.dumps(plot_data)
print(json_plot_data)
