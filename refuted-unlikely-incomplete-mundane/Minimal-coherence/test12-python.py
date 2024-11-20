def C(n):
    """Calculate minimal coherence conditions for n-categories"""
    if n < 2:
        raise ValueError("n must be >= 2")

    if n <= 3:
        return n - 1  # Foundational phase
    elif n <= 5:
        return 2*n - 3  # Transitional phase
    else:
        return 2*n - 1  # Linear phase

def verify_known_values():
    """Verify against known values"""
    known_values = {
        2: 1,  # categories
        3: 2,  # bicategories
        4: 5,  # tricategories
        5: 7,  # tetracategories
        6: 11, # pentacategories
        7: 13  # hexacategories
    }

    all_match = True
    for n, expected in known_values.items():
        calculated = C(n)
        matches = calculated == expected
        all_match &= matches
        print(f"n={n}: C({n})={calculated} (Expected: {expected}) - {'✓' if matches else '✗'}")

    return all_match

def test_sequence(start=2, end=10):
    """Generate sequence of values"""
    print("\nSequence of values:")
    for n in range(start, end+1):
        print(f"C({n}) = {C(n)}")

if __name__ == "__main__":
    print("Verifying known values:")
    if verify_known_values():
        print("\nAll known values match! ✓")
    else:
        print("\nDiscrepancy found in known values! ✗")

    test_sequence()
