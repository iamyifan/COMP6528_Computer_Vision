import numpy as np
import matplotlib.pyplot as plt

path = 'Task2/images/mountain2.jpg'
save_path = 'Task2/mountain2_point.npy'
# Load an image
mountain = plt.imread(path)
# Display the image
plt.imshow(mountain)
# Use plt.ginput() to select points
selected_points = plt.ginput(n=8, timeout=0)
# Close the plot
plt.close()
# Print the selected points
print("Selected points:", selected_points)
np.save(save_path, selected_points)