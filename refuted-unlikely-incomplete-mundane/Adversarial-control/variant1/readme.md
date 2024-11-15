Given a dynamic network G = (V, E), where a subset of nodes V_A ⊂ V exhibit adversarial behavior to prevent expansion, determine the optimal propagation strategy S: V → V such that an autonomous agent maximizes the controlled subgraph G_C ⊂ G while minimizing the total resource usage R, under adversarial and uncertain conditions.

Mathematical Objective:
S* = argmax_S |V_C|  subject to  R(S) ≤ R₀

where:
  R(S) is the resource cost associated with strategy S.
  V_C ⊆ V is the set of nodes controlled by the agent following strategy S.
  V_A ⊂ V adaptively restricts and modifies connections to limit expansion.
  
Conditions:
  - The agent must operate under uncertain resource availability at each node.
  - The strategy must account for adversarial dynamics and optimize both control expansion and resilience.
