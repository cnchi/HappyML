import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset, random_split


class Sequential(nn.Module):
    def __init__(self):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList()
    
    def add(self, layer, kernel_initializer=None, activation=None):
        # Define Python dictionary for weight initializers-------------
        initializer_dict = {
            "random_uniform": init.uniform_,
            "random_normal": init.normal_,
            "glorot_uniform": init.xavier_uniform_,
            "glorot_normal": init.xavier_normal_,
            "he_uniform": init.kaiming_uniform_,
            "he_normal": init.kaiming_normal_
        }

        # Check the parameter kernel_initializer
        if isinstance(kernel_initializer, str):
            kernel_initializer = initializer_dict.get(kernel_initializer, None)
            if kernel_initializer is None:
                raise ValueError(f"Initializer Error: Not support string {kernel_initializer}, 'None' applied.")

        if kernel_initializer is not None and hasattr(layer, 'weight'):
            kernel_initializer(layer.weight)

        self.layers.append(layer)

        # Define Python dictionary for activation function-------------
        activation_dict = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=-1),  # Note: Softmax required assigning dim
            "tanh": nn.Tanh(),
            "linear": None  # No activation function required for linear
        }

        # Check the parameter activation
        if isinstance(activation, str):
            activation = activation_dict.get(activation, None)
            if activation is None:
                raise ValueError(f"Activation Error: Not support string {activation}, 'None' applied.")

        if activation is not None:
            self.layers.append(activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def compile(self, optimizer="adam", loss="categorical_crossentropy"):
        # Define Python dictionary for loss functions-------------
        loss_dict = {
            "mse": nn.MSELoss(),
            "binary_crossentropy": nn.BCELoss(),
            "categorical_crossentropy": nn.CrossEntropyLoss()
        }

        # Check the parameter loss
        if isinstance(loss, str):
            self.criterion = loss_dict.get(loss, None)
            if loss is None:
                raise ValueError(f"Loss Error: Not support string {loss}, 'None' applied.")
        else:
            self.criterion = loss

        # Define Python dictionary for optimizers-------------
        lr = 0.0001
        optimizer_dict = {
            "sgd": optim.SGD(self.parameters(), lr=lr),
            "adagrad": optim.Adagrad(self.parameters(), lr=lr),
            "adadelta": optim.Adadelta(self.parameters(), lr=lr),
            "rmsprop": optim.RMSprop(self.parameters(), lr=lr),
            "adam": optim.Adam(self.parameters(), lr=lr),
            "adamax": optim.Adamax(self.parameters(), lr=lr),
            "nadam": optim.NAdam(self.parameters(), lr=lr)
        }

        # Check the parameter optimizer
        if isinstance(optimizer, str):
            self.optimizer = optimizer_dict.get(optimizer, None)
            if optimizer is None:
                raise ValueError(f"Optimizer Error: Not support string {optimizer}, 'None' applied.")
        else:
            self.optimizer = optimizer

    def summary(self):
        print(self)

    def fit(self, x=None, y=None, validation_split=0.0, batch_size=None, epochs=1, shuffle=True):
      # Split the dataset into training and validation sets
      dataset_size = len(x)
      val_size = int(dataset_size * validation_split)
      train_size = dataset_size - val_size
      train_dataset, val_dataset = random_split(TensorDataset(x, y), [train_size, val_size])

      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

      # Start the training loop
      for epoch in range(epochs):
        # Change the model to training mode
        self.train()
        # Save the metrics for the training set
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, Y_batch in train_loader:
          # Clear the gradients for this batch
          self.optimizer.zero_grad()

          # Compute the Y_pred for this batch
          Y_pred = self(X_batch)
          # Calculate the loss for this batch
          loss = self.criterion(Y_pred, Y_batch.squeeze())
          # Find the gradients for the loss
          loss.backward()
          # Update the weights using the optimizer
          self.optimizer.step()

          # Calculate the loss for this batch
          train_loss += loss.item() * X_batch.size(0)
          # Find the maximum probability to get the predicted class
          _, predicted = torch.max(Y_pred.data, 1)
          # Add the total number of this batch
          train_total += Y_batch.size(0)
          # Add the number of correct predictions
          # Note: .item() is used to get the value of the tensor
          train_correct += (predicted == Y_batch).sum().item()

        # Change the model to evaluation mode
        self.eval()
        # Save the metrics for the validation set
        val_loss, val_correct, val_total = 0.0, 0, 0
        # Close the autograd for validation
        with torch.no_grad():
          for X_batch, Y_batch in val_loader:
            # Compute the Y_pred for this batch
            Y_pred = self(X_batch)
            # Calculate the loss for this batch
            loss = self.criterion(Y_pred, Y_batch)

            # Calculate the metrics for this batch
            val_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(Y_pred.data, 1)
            val_total += Y_batch.size(0)
            val_correct += (predicted == Y_batch).sum().item()

        # Calculate the average loss and accuracy for this epoch
        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total

        # Print the metrics for this epoch
        print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

    def predict(self, x, batch_size=None):
        # Set the model to evaluation mode
        self.eval()
        predictions = []

        # Create a DataLoader if a batch_size is provided
        if batch_size is not None:
            dataloader = DataLoader(x, batch_size=batch_size)
        else:
            dataloader = DataLoader(x, batch_size=len(x))

        # No need to track gradients for prediction
        with torch.no_grad():
            for inputs in dataloader:
                # Get the predicted classes
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Append predictions to list
                predictions.append(predicted)

        # Concatenate all tensors in the list into a single tensor
        return torch.cat(predictions)

    def evaluate(self, x=None, y=None, batch_size=None):
        # Set the model to evaluation mode
        self.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0

        # Create a DataLoader if a batch_size is provided
        if batch_size is not None:
            dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
        else:
            dataloader = DataLoader(TensorDataset(x, y), batch_size=len(x), shuffle=False)

        # No need to track gradients for evaluation
        with torch.no_grad():
            for X_batch, Y_batch in dataloader:
                # Compute the Y_pred for this batch
                Y_pred = self(X_batch)
                # Calculate the loss for this batch
                loss = self.criterion(Y_pred, Y_batch.squeeze())

                # Calculate the metrics for this batch
                test_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(Y_pred.data, 1)
                test_total += Y_batch.size(0)
                test_correct += (predicted == Y_batch).sum().item()

        # Calculate the average loss and accuracy for the testing dataset
        test_loss /= test_total
        test_acc = test_correct / test_total

        return test_loss, test_acc
