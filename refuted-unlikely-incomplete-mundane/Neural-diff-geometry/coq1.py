import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    import geoopt
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False

def get_user_choice():
    print("Select Optimization Method:")
    print("1. Standard Optimization (SGD)")
    if GEOOPT_AVAILABLE:
        print("2. Geodesic Optimization (Riemannian SGD)")
    else:
        print("2. Geodesic Optimization (Requires 'geoopt' library, not installed)")

    choice = input("Enter the number of your choice (1 or 2): ")
    if choice == '1':
        return 'standard'
    elif choice == '2' and GEOOPT_AVAILABLE:
        return 'geodesic'
    else:
        print("Invalid choice or 'geoopt' library not available. Defaulting to Standard Optimization.")
        return 'standard'

# Define the neural network model
class ManifoldNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, optimization_method):
        super(ManifoldNetwork, self).__init__()
        self.optimization_method = optimization_method
        self.activation = nn.ReLU()

        if self.optimization_method == 'geodesic':
            # Define manifold parameters for geodesic optimization
            manifold = geoopt.EuclideanStiefelExact()
            # Corrected shapes to satisfy manifold requirements
            self.layer1_weight = geoopt.ManifoldParameter(
                torch.randn(hidden_dim, input_dim), manifold=manifold
            )
            self.layer2_weight = geoopt.ManifoldParameter(
                torch.randn(hidden_dim, output_dim), manifold=manifold
            )
        else:
            # Standard parameters for standard optimization
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if self.optimization_method == 'geodesic':
            # Adjusted forward pass without transposing layer2_weight
            x = self.activation(x @ self.layer1_weight.t())
            x = x @ self.layer2_weight
        else:
            x = self.activation(self.layer1(x))
            x = self.layer2(x)
        return x

# Define the loss function
def lagrangian(output, target):
    return nn.MSELoss()(output, target)

# Training function
def train_model(model, data, target, epochs, lr):
    if model.optimization_method == 'geodesic':
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_history = []
    print("\nTraining started...\n")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = lagrangian(output, target)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # Display progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    print("\nTraining completed.\n")
    return loss_history

# Generate synthetic data
def generate_data(num_samples, input_dim):
    data = torch.randn(num_samples, input_dim)
    noise = torch.randn(num_samples) * 0.5
    target = (data[:, 0] * 3 - data[:, 1] * 2 + noise).unsqueeze(1)
    return data, target

def main():
    # Parameters
    input_dim = 2
    hidden_dim = 5
    output_dim = 1
    num_samples = 1000
    epochs = 100
    lr = 0.01

    # Get user choice for optimization method
    optimization_method = get_user_choice()

    # Generate data
    data, target = generate_data(num_samples, input_dim)

    # Initialize model
    model = ManifoldNetwork(input_dim, hidden_dim, output_dim, optimization_method)

    # Train model
    loss_history = train_model(model, data, target, epochs=epochs, lr=lr)

    # Plot the loss curve
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if GEOOPT_AVAILABLE or get_user_choice() != 'geodesic':
        main()
    else:
        # If geoopt is not available and user wants geodesic optimization
        print("The 'geoopt' library is required for geodesic optimization but is not installed.")
        install_geoopt = input("Would you like to install 'geoopt' now? (y/n): ").lower()
        if install_geoopt == 'y':
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "geoopt"])
            print("'geoopt' installed. Please rerun the script.")
        else:
            print("Proceeding with Standard Optimization.")
            main()
