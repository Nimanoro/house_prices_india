import torch 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import torch.optim as optim
from data import data_tensor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Define the model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(37, 1)
       
    def forward(self, x):
        predict_y = self.linear(x)
        return predict_y

# Create an instance of the model
loaded_model = LinearRegression()

# Load the saved model
loaded_model.load_state_dict(torch.load("linear_regression_model.pth"))
loaded_model.eval()

# Assuming data_tensor is defined as a numpy array
test_inputs = torch.tensor(data_tensor[1500:2100, 1:38], dtype=torch.float32)
test_outputs = torch.tensor(data_tensor[1500:2100, 0], dtype=torch.float32)

#a plot out of the weights of our model
import matplotlib.pyplot as plt

weights = []
for index, param in enumerate(loaded_model.parameters()):
    weights.extend(param.data.flatten().tolist())

# Generate indices based on the length of weights
indices = list(range(len(weights)))

# Plotting the weights
plt.figure(figsize=(10, 6))
plt.plot(indices, weights, 'bo')
plt.xlabel('Parameter Index')
plt.ylabel('Weight Values')
plt.title('Weights vs. Parameter Index')
plt.show()
# Calculate the percentage error for each test input
percentage_errors = []
predicted_outputs = []
with torch.no_grad():
    for i in range(test_inputs.size(0)):
        input_i = test_inputs[i]
        output_i = test_outputs[i]
        predicted_output = loaded_model(input_i)
        absolute_difference = torch.abs(predicted_output - output_i)
        percentage_error = (absolute_difference / output_i) * 100
        percentage_errors.append(percentage_error.item())
        predicted_outputs.append(predicted_output.item())
        

# Calculate the average percentage error
average_percentage_error = np.mean(percentage_errors)
print("Average Percentage Error:", average_percentage_error)
print(predicted_output)

# Plot the predicted outputs
plt.plot(range(test_outputs.size(0)), test_outputs.numpy(), label='Actual')
plt.plot(range(test_outputs.size(0)), predicted_outputs, label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Output')
plt.legend()
plt.show()
