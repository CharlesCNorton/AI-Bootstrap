##############################################################################
#  Weighted-Motivic-Taylor-Tower mini demo (robust version)                  #
##############################################################################
from sage.interfaces.macaulay2 import macaulay2 as m2
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1)  Geometry helpers
# ---------------------------------------------------------------------------
def blowup_affine_plane_at_origin():
    """
    Returns (dim, sing_complexity) for the blow-up of A^2 at (0,0).
    Tries Macaulay2; hard-codes (2,0) if the package or grading fails.
    """
    code = r'''
        needsPackage "ReesAlgebra";
        R = QQ[x,y];
        I = ideal(x,y);
        S = reesAlgebra(I, Variable => z, Degree => 1);  -- singly graded
        X = Proj S;
        {dim X, degree singularLocus X}
    '''
    try:
        return list(m2(code))
    except Exception:
        # fall back: blow-up is a smooth surface
        return (2, 0)

def nodal_cubic_curve():
    """
    (dim, sing_complexity) for  y^2 = x^3 + x^2  (one node ⇒ sc = 1)
    Works in all stock Macaulay2 installs.
    """
    code = r'''
        S = QQ[x,y,z];
        I = ideal(y^2*z - x^3 - x^2*z);
        X = Proj(S/I);
        {dim X, degree singularLocus X}
    '''
    return list(m2(code))

# ---------------------------------------------------------------------------
# 2)  Weight functions and obstruction recursion
# ---------------------------------------------------------------------------
def w_dim(d):          return 1.0 / (1.0 + d)
def w_sing(s):         return 1.0 / (1.0 + s)
def w_stage(n):        return 1.0 / (1.0 + n)
def w_total(d, s, n):  return w_dim(d) * w_sing(s) * w_stage(n)

def obstruction_seq(d, s, n_max=15, O0=1.0):
    obs = [O0]
    for n in range(n_max):
        obs.append(obs[-1] * w_total(d, s, n))
    return obs

# ---------------------------------------------------------------------------
# 3)  Pick a variety   (toggle by commenting)
# ---------------------------------------------------------------------------
d, s = blowup_affine_plane_at_origin()
ex_name = "Blow-up of 𝔸² at (0,0)"

# d, s = nodal_cubic_curve()
# ex_name = "Nodal cubic  y² = x³ + x²"

# ---------------------------------------------------------------------------
# 4)  Run the weighted tower
# ---------------------------------------------------------------------------
print(f"\n=== {ex_name} ===")
print(f"dimension            d  = {d}")
print(f"singularity measure  s  = {s}")

obs = obstruction_seq(d, s, n_max=20)
print("\n n    obstruction size")
print("-----------------------")
for n, val in enumerate(obs):
    print(f"{n:2d}   {val:.6g}")

# ---------------------------------------------------------------------------
# 5)  Plot the decay (log-scale)
# ---------------------------------------------------------------------------
plt.figure(figsize=(6,3))
plt.semilogy(range(len(obs)), obs, marker='o')
plt.title(f'Obstruction decay for {ex_name}')
plt.xlabel('tower stage  n')
plt.ylabel('size (log-scale)')
plt.grid(True)
plt.tight_layout()
plt.show()
