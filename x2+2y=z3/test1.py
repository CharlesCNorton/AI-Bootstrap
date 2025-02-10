"""
first_script.py

Search for integer solutions to the equation x^2 + 2^y = z^3
over a small, user-defined range of x and y.

Usage:
  1. Set XMAX and YMAX to define the search space.
  2. Run in Python on Windows:
       python first_script.py
  3. Observe any solutions found in that range.
"""

import math

def is_perfect_cube(n: int) -> bool:
    """
    Check if integer n is a perfect cube by rounding the
    cubic root and verifying.
    """
    if n < 0:
        c_approx = int(round((-n)**(1/3)))
        # Check around c_approx to handle rounding issues
        return any((-(c_approx + delta))**3 == n for delta in [-1, 0, 1])
    else:
        c_approx = int(round(n**(1/3)))
        return any((c_approx + delta)**3 == n for delta in [-1, 0, 1])

def search_solutions(xmax: int, ymax: int):
    """
    Returns a list of (x, y, z) solutions to x^2 + 2^y = z^3
    within the specified bounds: |x| <= xmax, 0 <= y <= ymax.
    """
    sol_list = []
    for x in range(-xmax, xmax + 1):
        for y in range(ymax + 1):
            lhs = x*x + (2 ** y)
            # Check if lhs is a perfect cube
            if is_perfect_cube(lhs):
                # Identify the actual z by integer cubic root
                z_approx = int(round(lhs ** (1/3)))
                # Check integers around z_approx to confirm
                for delta in [-2, -1, 0, 1, 2]:
                    z_candidate = z_approx + delta
                    if z_candidate**3 == lhs:
                        sol_list.append((x, y, z_candidate))
    return sol_list

if __name__ == "__main__":
    # Adjust these bounds as needed
    XMAX = 50
    YMAX = 20

    solutions = search_solutions(XMAX, YMAX)
    # Remove duplicates and sort by y, then x, then z
    unique_solutions = sorted(set(solutions), key=lambda s: (s[1], s[0], s[2]))

    print(f"Searching for solutions to x^2 + 2^y = z^3 with |x|<={XMAX}, 0<=y<={YMAX}...\n")
    if not unique_solutions:
        print("No solutions found in this range.")
    else:
        print("Solutions found (x, y, z):")
        for (x, y, z) in unique_solutions:
            print(f"  x={x}, y={y}, z={z}  =>  {x**2} + 2^{y} = {z**3}")
    print(f"\nTotal solutions: {len(unique_solutions)}")
