import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data from your runs folder
csv_path = r'runs\detect\train\results.csv'
df = pd.read_csv(csv_path)

# Strip whitespace from column names just in case
df.columns = df.columns.str.strip()

# 2. Create the plot
plt.figure(figsize=(12, 5))

# Plotting Training vs Validation Loss
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.title('Train vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting mAP (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='green')
plt.title('Model Accuracy (mAP@50)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy Score')
plt.legend()

plt.tight_layout()
plt.savefig('signzy_performance_graphs.png') # Saves the graph for your report
plt.show()