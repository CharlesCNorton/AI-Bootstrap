import numpy as np
from mpmath import mp
mp.dps = 50  # Set precision

def riemann_siegel_theta_derivatives(t, n_max=4):
    """
    Compute derivatives of θ(t) up to order n_max
    """
    results = []
    for n in range(n_max + 1):
        if n == 0:
            # Base theta function
            theta = t/2 * mp.log(t/(2*mp.pi)) - t/2 - mp.pi/8
            results.append(theta)
        elif n == 1:
            # First derivative
            deriv = mp.log(t/(2*mp.pi))/2 + 1/2
            results.append(deriv)
        else:
            # Higher derivatives follow a pattern
            deriv = mp.power(-1, n-1) * mp.factorial(n-1)/(2 * mp.power(t, n-1))
            results.append(deriv)
    return results

def verify_derivative_relationship(zero, order=4):
    """
    Verify the relationship between successive derivatives at a zero
    """
    theta_derivs = riemann_siegel_theta_derivatives(zero, order)

    # Compute the main sum derivatives
    N = int(mp.sqrt(zero/(2*mp.pi)))

    # For each derivative order
    sum_derivs = []
    for n in range(1, order + 1):
        sum_n = 0
        for k in range(1, N + 1):
            # Complex phase term
            phase = theta_derivs[0] - zero*mp.log(k)

            # Apply Leibniz rule for derivatives
            terms = []
            for j in range(n + 1):
                coef = mp.binomial(n, j)
                if j == 0:
                    term1 = mp.power(-1, j) * theta_derivs[j]
                else:
                    term1 = mp.power(-1, j) * theta_derivs[j]

                term2 = -mp.log(k) if j == 1 else 0

                terms.append(coef * (term1 + term2))

            sum_n += mp.power(k, -1/2) * sum(terms) * mp.exp(1j * phase)

        sum_derivs.append(sum_n)

    return sum_derivs

# Test for first few zeros
zeros = [
    14.134725141734693,
    21.022039638771554,
    25.010857580145688
]

print("Verifying derivative relationships:")
for zero in zeros:
    print(f"\nAnalyzing zero ρ ≈ {zero}")
    derivs = verify_derivative_relationship(zero)

    # Output ratios between successive derivatives
    for i in range(len(derivs) - 1):
        ratio = abs(derivs[i+1]/derivs[i])
        print(f"Ratio between derivatives {i+1} and {i+2}: {ratio}")

        # Verify our conjectured relationship
        predicted = mp.exp(mp.pi * (i+1))
        error = abs(ratio - predicted)/predicted
        print(f"Relative error from prediction: {error}")
