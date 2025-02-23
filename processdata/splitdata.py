import numpy as np
pre = "...."
from sklearn.model_selection import train_test_split
# Initialize lists to store the split data

y = []
# List of files and their corresponding labels
files = ["anxiety", "bipolar", "depression", "mentalhealth"]
labels = [0, 1, 2, 3]
X_concatenated_list = []
# Load data and create labels
for file, label in zip(files, labels):
    X_temp1 = np.load(pre + file + "_r.npy")
    X_temp2 = np.load(pre + file + "_s.npy")
    X_concatenated = np.hstack((X_temp1, X_temp2))
    X_concatenated_list.append(X_concatenated)
   
    y.extend([label] * (len(X_temp1) // 2))  # Assuming every 2 rows correspond to one user

# Stack the data arrays vertically (row-wise)
X= np.vstack(X_concatenated_list)
print(X.shape)
y = np.array(y)

# Reshape X so that each user's data is grouped together
X_grouped = X.reshape(-1, 2, X.shape[1])  # Group by user (2 rows per user)

# Flatten the labels to match the grouped data
y_grouped = y.reshape(-1)

# Split the grouped data into training (60%) and temp (40%)
X_temp, X_test, y_temp, y_test = train_test_split(X_grouped, y_grouped, test_size=0.2, stratify=y_grouped, random_state=42)

# Further split the temp set into validation (20%) and test (20%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)


n_train = len(y_train)
n_val = len(y_val)
n_test = len(y_test)


# Reshape the data back to its original shape
X_train = X_train[:,0,:]
X_val = X_val[:,0,:]
X_test = X_test[:,0,:]



np.save(pre + "X_train_new.npy", X_train)
np.save(pre + "X_val_new.npy", X_val)
np.save(pre + "X_test_new.npy", X_test)
np.save(pre + "y_train_new.npy", y_train)
np.save(pre + "y_val_new.npy", y_val)
np.save(pre + "y_test_new.npy", y_test)