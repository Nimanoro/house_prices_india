import torch 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim


data = pd.read_csv('Hyderabad.csv')
image=plt.scatter(data['Area'], data['Price'])

numeric_data = data.select_dtypes(include=['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8'])

# Perform one-hot encoding on categorical columns
categorical_data = data.select_dtypes(include=['object'])
encoded_data = pd.get_dummies(categorical_data)

# Concatenate numeric and encoded categorical data
processed_data = pd.concat([numeric_data, encoded_data], axis=1)

# Convert processed data to PyTorch tensor
data_tensor = torch.tensor(processed_data.values, dtype=torch.float32)

# Split the data into input features and target variable
inputs = data_tensor[1:1800, 1:]
targets = data_tensor[1:1800,0]

# Define a linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)
    
# Create an instance of the linear regression model
input_size = inputs.size(1)
output_size = 1
model = LinearRegression(input_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss after each epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    # Calculate mean absolute percentage error (MAPE)
    absolute_percentage_error = torch.abs((targets - outputs) / targets)
    mape = torch.mean(absolute_percentage_error) * 100
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    

# Test the model
weights = model.linear.weight.data.numpy().flatten()

test_inputs = data_tensor[1800:2000, 1:]
test_output = data_tensor[1800:2000,0]

predicted_prices = model(test_inputs).squeeze()
print(f'Predicted Price: {predicted_prices}')

# Plot the weights
plt.bar(range(len(weights)), weights)
plt.xlabel('Weight Index')
plt.ylabel('Weight Value')
plt.title('Weights of Linear Regression Model')
plt.show()




