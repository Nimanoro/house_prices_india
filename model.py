import torch 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import torch.optim as optim
#loading the data with panda library
data = pd.read_csv('Hyderabad.csv')
data=data.drop('Location', axis=1)
data_array = data.to_numpy()
data_tensor = torch.tensor(data_array, dtype=torch.float32)

#making the model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(37, 1)
       
    def forward(self, x):
        predict_y = self.linear(x)
        return predict_y

#create model
linear_model = LinearRegression()

# criterion and optimizer
criterion = torch.nn.MSELoss()
learning_rate=0.01
optimizer = torch.optim.RMSprop(linear_model.parameters(), lr=learning_rate)

inputs = data_tensor[1:1800, 1:38]
outputs= data_tensor[1:1800, 0]

num_epochs = 1100
for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(inputs.size(0)):
        # Clear gradients
        optimizer.zero_grad()

        # Get a single input and output
        input_i = inputs[i]
        output_i = outputs[i]

        # Reshape the input to match the expected size (1, 37)
        input_i = input_i.unsqueeze(0)

        # Forward pass
        predicted_output_i = linear_model(input_i)

        # Compute loss
        loss = criterion(predicted_output_i, output_i)

        # Accumulate total loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        absolute_difference = torch.abs(predicted_output_i - output_i)

# Calculate the percentage error
    percentage_error = (absolute_difference / output_i) * 100

# Compute the average percentage error
    average_percentage_error = torch.mean(100-percentage_error)

    print(f"Average Accuracy Percentage: {average_percentage_error.item()}%")
    # Print average loss for the epoch
    average_loss = total_loss / inputs.size(0)
    print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}')


path = "linear_regression_model.pth"

# Save the model
torch.save(linear_model.state_dict(), path)