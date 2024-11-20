import torch
import mpmath
import numpy as np
from tqdm import tqdm

# Convert Riemann zeta zeros to torch tensors
def load_riemann_zeros_torch(num_zeros):
    """
    Load nontrivial zeros of the Riemann zeta function as torch tensors.
    Parameters:
        num_zeros (int): Number of zeros to load.
    Returns:
        torch.Tensor: Tensor of zeros.
    """
    zeros = [mpmath.im(mpmath.zetazero(n)) for n in range(1, num_zeros + 1)]
    return torch.tensor(zeros, dtype=torch.float32, device="cuda")

# Define the Riemann-Siegel Z-function on CPU (mpmath integration)
def z_function_cpu(t):
    """
    Compute the Riemann-Siegel Z-function using mpmath.
    Parameters:
        t (list): Points at which to evaluate Z(t).
    Returns:
        list: Real part of Z(t).
    """
    return [float(mpmath.zeta(0.5 + 1j * float(val)).real) for val in t]

# Compute derivatives using finite differences
def compute_derivatives_gpu(zeros, max_order, h=1e-6):
    """
    Compute normalized derivatives of the Riemann-Siegel Z-function on GPU.
    Parameters:
        zeros (torch.Tensor): Nontrivial zeros of the Riemann zeta function.
        max_order (int): Maximum order of derivatives to compute.
        h (float): Step size for finite differences.
    Returns:
        torch.Tensor: Normalized derivatives K_n(rho).
    """
    num_zeros = len(zeros)
    derivatives = torch.zeros((num_zeros, max_order), device="cuda")

    for n in tqdm(range(1, max_order + 1), desc="Computing Derivatives"):
        for i, zero in enumerate(zeros.cpu().numpy()):  # Transfer zero to CPU for mpmath
            zero = float(zero)
            f_plus = z_function_cpu([zero + h])[0]
            f_minus = z_function_cpu([zero - h])[0]
            derivative = (f_plus - f_minus) / (2 * h)

            # Normalize derivative
            derivatives[i, n - 1] = torch.abs(torch.tensor(derivative, device="cuda")) / (zero ** (n - 0.5))

    return derivatives

# Main Function
def main():
    # Parameters
    num_zeros = 20  # Number of zeros to compute
    max_order = 5    # Maximum order of derivatives
    h = 1e-6         # Step size for finite differences

    # Load zeros onto GPU
    print("Loading Riemann zeta zeros...")
    zeros = load_riemann_zeros_torch(num_zeros)

    # Compute derivatives on GPU
    print("Computing derivatives...")
    derivatives = compute_derivatives_gpu(zeros, max_order, h)

    # Transfer results back to CPU for further processing
    derivatives_cpu = derivatives.cpu().numpy()
    print("Derivatives Computed:")
    print(derivatives_cpu)

if __name__ == "__main__":
    main()
