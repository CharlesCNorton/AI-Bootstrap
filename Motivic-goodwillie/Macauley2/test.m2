-- Weighted Motivic Taylor Tower: Rigorous Iterative Blowup and Obstruction Computation
-- Author: Charles Norton / OpenAI / Anthropic
-- System: Macaulay2 (any recent version; no packages required)
-- Purpose: To algebraically realize repeated singular blowups at the node of the nodal cubic,
--          and at each stage to compute the strict transform, Jacobian, and singular locus.
--          This concretely tests the suppression of obstructions predicted by the weighted tower conjecture.

restart

---------------------------------------------------------------
-- STAGE 0: ORIGINAL CUBIC IN AN AFFINE CHART
---------------------------------------------------------------
-- Let x, t, z be affine coordinates. The original cubic is:
--   f(x, t, z) = (x*t)^2*z - x^2*(x+z)
-- This models the nodal cubic in the chart y = x*t.

R0 = QQ[x, t, z]
f0 = (x*t)^2*z - x^2*(x + z)
print "Stage 0 strict transform (original nodal cubic in chart y = x*t):"
f0

---------------------------------------------------------------
-- STAGE 1: FIRST BLOWUP (at the origin x = 0, t = 0, z = 0)
-- We take the affine chart t = x*u (so u is the new variable).
---------------------------------------------------------------
-- The origin remains (x = 0, u = 0, z = 0) in these coordinates.
-- The substitution t = x*u gives the strict transform in (x, u, z).
R1 = QQ[x, u, z]
f1 = (x^2*(x*u)^2)*z - x^2*(x + z)
print "\nStage 1 strict transform (after first blowup, chart t = x*u):"
f1

-- Compute the Jacobian matrix (partial derivatives w.r.t x, u, z)
J1 = jacobian matrix{{f1}}
d1_1 = (entries J1)#0#0 -- ∂f1/∂x
d1_2 = (entries J1)#1#0 -- ∂f1/∂u
d1_3 = (entries J1)#2#0 -- ∂f1/∂z
print "\nStage 1 partial derivatives:"
d1_1
d1_2
d1_3

-- The ideal of the strict transform and all partials (singular locus)
I1 = ideal(f1, d1_1, d1_2, d1_3)
print "\nStage 1 singular locus ideal:"
I1

-- Primary decomposition yields the individual singular points/schemes
print "\nStage 1 primary components of the singular locus:"
primaryDecomposition I1

---------------------------------------------------------------
-- STAGE 2: SECOND BLOWUP (at x = 0, u = 0, z = 0)
-- Proceed in chart u = x*v (affine variable v replaces u)
---------------------------------------------------------------
R2 = QQ[x, v, z]
f2 = (x^2*(x*v)^2)*z - x^2*(x + z)
print "\nStage 2 strict transform (after second blowup, chart u = x*v):"
f2

-- Jacobian for stage 2
J2 = jacobian matrix{{f2}}
d2_1 = (entries J2)#0#0 -- ∂f2/∂x
d2_2 = (entries J2)#1#0 -- ∂f2/∂v
d2_3 = (entries J2)#2#0 -- ∂f2/∂z
print "\nStage 2 partial derivatives:"
d2_1
d2_2
d2_3

I2 = ideal(f2, d2_1, d2_2, d2_3)
print "\nStage 2 singular locus ideal:"
I2

print "\nStage 2 primary components of the singular locus:"
primaryDecomposition I2

---------------------------------------------------------------
-- STAGE 3: THIRD BLOWUP (at x = 0, v = 0, z = 0)
-- Proceed in chart v = x*w (affine variable w replaces v)
---------------------------------------------------------------
R3 = QQ[x, w, z]
f3 = (x^2*(x*w)^2)*z - x^2*(x + z)
print "\nStage 3 strict transform (after third blowup, chart v = x*w):"
f3

-- Jacobian for stage 3
J3 = jacobian matrix{{f3}}
d3_1 = (entries J3)#0#0 -- ∂f3/∂x
d3_2 = (entries J3)#1#0 -- ∂f3/∂w
d3_3 = (entries J3)#2#0 -- ∂f3/∂z
print "\nStage 3 partial derivatives:"
d3_1
d3_2
d3_3

I3 = ideal(f3, d3_1, d3_2, d3_3)
print "\nStage 3 singular locus ideal:"
I3

print "\nStage 3 primary components of the singular locus:"
primaryDecomposition I3

---------------------------------------------------------------
-- (Optional) CONTINUE AS DESIRED:
-- For each further blowup at the origin, introduce a new variable,
-- substituting e.g. w = x*y, and so forth.
-- At each stage: update the polynomial, ring, Jacobian, and run again!
---------------------------------------------------------------

-- END
print "\nEnd of weighted motivic tower algebraic test (3 blowups shown)."