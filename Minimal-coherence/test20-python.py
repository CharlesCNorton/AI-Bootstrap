
import numpy as np
import matplotlib.pyplot as plt

def C(n):
    if n in [2, 3]:
        return n - 1
    elif n in [4, 5]:
        return 2 * n - 3
    elif n >= 6:
        return 2 * n - 1
    else:
        raise ValueError("n must be an integer greater than or equal to 2")

# Known values from the paper
known_values = {
    2: 1,
    3: 2,
    4: 5,
    5: 7,
    6: 11,
    7: 13
}

def test_known_values():
    print("Testing known values of C(n):\n")
    all_tests_pass = True
    for n, expected_C in known_values.items():
        computed_C = C(n)
        test_pass = computed_C == expected_C
        all_tests_pass &= test_pass
        print(f"n = {n}: Computed C(n) = {computed_C}, Expected C(n) = {expected_C}, Test Pass: {test_pass}")
    if all_tests_pass:
        print("\nAll known value tests passed!")
    else:
        print("\nSome tests failed. Please check the implementation.")

def analyze_phase_transitions(max_n=15):
    n_values = np.arange(2, max_n + 1)
    C_values = np.array([C(n) for n in n_values])

    # Define the phases based on the given function
    phases = []
    for n in n_values:
        if n in [2, 3]:
            phases.append('Foundational')
        elif n in [4, 5]:
            phases.append('Transitional')
        else:
            phases.append('Linear')

    # Detect phase transitions
    transition_points = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            transition_points.append(n_values[i])

    print("\nAnalyzing phase transitions:\n")
    for n in transition_points:
        print(f"Phase transition detected at n = {n}")

    # Plot C(n) with phase transitions
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, C_values, marker='o', linestyle='-')
    for n in transition_points:
        plt.axvline(x=n, color='red', linestyle='--', label=f'Transition at n={n}')
    plt.xlabel('Dimension n')
    plt.ylabel('C(n)')
    plt.title('Minimal Coherence Conditions C(n) vs. Dimension n')
    plt.legend()
    plt.grid(True)
    plt.show()

def extended_analysis(max_n=20):
    n_values = np.arange(2, max_n + 1)
    C_values = np.array([C(n) for n in n_values])

    # Define phases
    phases = []
    for n in n_values:
        if n in [2, 3]:
            phases.append('Foundational')
        elif n in [4, 5]:
            phases.append('Transitional')
        else:
            phases.append('Linear')

    # Calculate differences and growth rates
    diffs = np.diff(C_values)
    growth_rates = diffs / np.diff(n_values)

    print("\nGrowth rates and phases:")
    for n, rate, phase in zip(n_values[1:], growth_rates, phases[1:]):
        print(f"n = {n}, Growth rate = {rate}, Phase = {phase}")

    # Analyze parity
    parity = ['Even' if c % 2 == 0 else 'Odd' for c in C_values]
    print("\nParity of C(n):")
    for n, c, p in zip(n_values, C_values, parity):
        print(f"n = {n}, C(n) = {c}, Parity = {p}")

    # Modular patterns
    for mod in range(2, 6):
        modulo_values = C_values % mod
        print(f"\nC(n) mod {mod}:")
        for n, val in zip(n_values, modulo_values):
            print(f"n = {n}, C(n) mod {mod} = {val}")

if __name__ == "__main__":
    test_known_values()
    analyze_phase_transitions()
    extended_analysis()
