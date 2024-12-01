Require Import UniMath.Foundations.All.
Require Import UniMath.CategoryTheory.Core.Categories.

Local Open Scope cat.

Record Suspension (C : category) : UU := {
  susp_map : ob C -> ob C;
  susp_mor : ∏ (a b : ob C), (a --> b) -> (susp_map a --> susp_map b);
  susp_id : ∏ (a : ob C), susp_mor a a (identity a) = identity (susp_map a);
  susp_comp : ∏ (a b c : ob C) (f : a --> b) (g : b --> c),
              susp_mor a c (f · g) = susp_mor a b f · susp_mor b c g;
  base_point : ob C;
  zero_section : ∏ (a : ob C), base_point --> a;
  point_susp : base_point --> susp_map base_point
}.

Record LoopSpace' (C : category) : UU := {
  loop_map : ob C -> ob C;
  loop_mor : ∏ (a b : ob C), (a --> b) -> (loop_map a --> loop_map b);
  loop_id : ∏ (a : ob C), loop_mor a a (identity a) = identity (loop_map a);
  loop_comp : ∏ (a b c : ob C) (f : a --> b) (g : b --> c),
              loop_mor a c (f · g) = loop_mor a b f · loop_mor b c g
}.

Definition get_susp_map {C : category} (Σ : Suspension C) := @susp_map C Σ.
Definition get_susp_mor {C : category} (Σ : Suspension C) := @susp_mor C Σ.
Definition get_loop_map {C : category} (Ω : LoopSpace' C) := @loop_map C Ω.
Definition get_loop_mor {C : category} (Ω : LoopSpace' C) := @loop_mor C Ω.

Record SuspensionLoopAdjunction (C : category) : UU := {
  susp : Suspension C;
  loops : LoopSpace' C;
  
  unit : ∏ (a : ob C), a --> get_loop_map loops (get_susp_map susp a);
  
  counit : ∏ (a : ob C), get_susp_map susp (get_loop_map loops a) --> a;
  
  triangle_id1 : ∏ (a : ob C),
    unit (get_loop_map loops a) · get_loop_mor loops _ _ (counit a) = identity (get_loop_map loops a);
    
  triangle_id2 : ∏ (a : ob C),
    get_susp_mor susp _ _ (unit a) · counit (get_susp_map susp a) = identity (get_susp_map susp a)
}.

Definition suspension_natural_transformation {C : category} 
  (Σ1 Σ2 : Suspension C) 
  (η : ∏ (a : ob C), get_susp_map Σ1 a --> get_susp_map Σ2 a) : UU :=
  ∏ (a b : ob C) (f : a --> b),
    η a · get_susp_mor Σ2 a b f = get_susp_mor Σ1 a b f · η b.

Definition loop_natural_transformation {C : category}
  (Ω1 Ω2 : LoopSpace' C)
  (η : ∏ (a : ob C), get_loop_map Ω1 a --> get_loop_map Ω2 a) : UU :=
  ∏ (a b : ob C) (f : a --> b),
    η a · get_loop_mor Ω2 a b f = get_loop_mor Ω1 a b f · η b.