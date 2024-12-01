Require Import UniMath.Foundations.All.
Require Import UniMath.CategoryTheory.Core.Categories.

Local Open Scope cat.

Definition functor_preserves_id {C : category} 
  (F_ob : ob C -> ob C)
  (F_mor : ∏ (a b : ob C), (a --> b) -> (F_ob a --> F_ob b)) : UU :=
  ∏ (a : ob C), F_mor a a (identity a) = identity (F_ob a).

Definition functor_preserves_comp {C : category}
  (F_ob : ob C -> ob C)
  (F_mor : ∏ (a b : ob C), (a --> b) -> (F_ob a --> F_ob b)) : UU :=
  ∏ (a b c : ob C) (f : a --> b) (g : b --> c),
    F_mor a c (f · g) = F_mor a b  f · F_mor b c g.

Lemma id_composition_neutral {C : category} {a b : ob C} (f : a --> b) :
  f · identity b = f.
Proof.
  apply id_right.
Qed.

Lemma composition_id_neutral {C : category} {a b : ob C} (f : a --> b) :
  identity a · f = f.
Proof.
  apply id_left.
Qed.

Lemma composition_assoc {C : category} {a b c d : ob C} 
  (f : a --> b) (g : b --> c) (h : c --> d) :
  (f · g) · h = f · (g · h).
Proof.
  apply assoc'.
Qed.

Record P1Suspension (C : category) : UU := {
  p1_susp_map : ob C -> ob C;
  p1_susp_mor : ∏ (a b : ob C), (a --> b) -> (p1_susp_map a --> p1_susp_map b);
  p1_susp_id_preservation : functor_preserves_id p1_susp_map p1_susp_mor;
  p1_susp_comp_preservation : functor_preserves_comp p1_susp_map p1_susp_mor;
  p1_base : ob C;
  p1_inclusion : ∏ (a : ob C), a --> p1_susp_map a;
  p1_projection : ∏ (a : ob C), p1_susp_map a --> a
}.

Definition get_p1_susp_map {C : category} (P : P1Suspension C) := @p1_susp_map C P.
Definition get_p1_susp_mor {C : category} (P : P1Suspension C) := @p1_susp_mor C P.

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

Definition get_susp_map {C : category} (Σ : Suspension C) := @susp_map C Σ.

Record P1Stable (C : category) : UU := {
  stable_susp : Suspension C;
  stable_p1 : P1Suspension C;
  stability_iso : ∏ (X : ob C), 
    get_susp_map stable_susp X --> get_p1_susp_map stable_p1 X
}.

Definition IsIsomorphism {C : category} {a b : ob C} (f : a --> b) : UU :=
  ∑ (g : b --> a), (f · g = identity a) × (g · f = identity b).