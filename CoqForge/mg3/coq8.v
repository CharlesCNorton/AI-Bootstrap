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
Definition IsIsomorphism {C : category} {a b : ob C} (f : a --> b) : UU :=
  ∑ (g : b --> a), (f · g = identity a) × (g · f = identity b).

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
Definition get_susp_mor {C : category} (Σ : Suspension C) := @susp_mor C Σ.
Definition get_base_point {C : category} (Σ : Suspension C) := @base_point C Σ.
Definition get_point_susp {C : category} (Σ : Suspension C) := @point_susp C Σ.
Definition get_p1_inclusion {C : category} (P : P1Suspension C) := @p1_inclusion C P.

Record P1Stable (C : category) : UU := {
  stable_susp : Suspension C;
  stable_p1 : P1Suspension C;
  stability_iso : ∏ (X : ob C), 
    get_susp_map stable_susp X --> get_p1_susp_map stable_p1 X;
  stability_is_iso : ∏ (X : ob C),
    IsIsomorphism (stability_iso X);
  stability_natural : ∏ (X Y : ob C) (f : X --> Y),
    stability_iso X · get_p1_susp_mor stable_p1 X Y f = 
    get_susp_mor stable_susp X Y f · stability_iso Y
}.

Definition get_stable_susp {C : category} (PS : P1Stable C) : Suspension C :=
  match PS with
  | Build_P1Stable _ susp p1 iso is_iso nat => susp
  end.

Definition get_stable_p1 {C : category} (PS : P1Stable C) : P1Suspension C :=
  match PS with
  | Build_P1Stable _ susp p1 iso is_iso nat => p1
  end.

Definition get_stability_iso {C : category} (PS : P1Stable C) : 
  ∏ (X : ob C), get_susp_map (get_stable_susp PS) X --> get_p1_susp_map (get_stable_p1 PS) X :=
  match PS with
  | Build_P1Stable _ susp p1 iso is_iso nat => iso
  end.

Definition get_stability_is_iso {C : category} (PS : P1Stable C) :
  ∏ (X : ob C), IsIsomorphism (get_stability_iso PS X) :=
  match PS with
  | Build_P1Stable _ susp p1 iso is_iso nat => is_iso
  end.

Definition get_stability_inverse {C : category} (PS : P1Stable C) (X : ob C) : 
  get_p1_susp_map (get_stable_p1 PS) X --> get_susp_map (get_stable_susp PS) X :=
  pr1 (get_stability_is_iso PS X).

Definition get_stability_natural {C : category} (PS : P1Stable C) :
  ∏ (X Y : ob C) (f : X --> Y),
    get_stability_iso PS X · get_p1_susp_mor (get_stable_p1 PS) X Y f = 
    get_susp_mor (get_stable_susp PS) X Y f · get_stability_iso PS Y :=
  match PS with
  | Build_P1Stable _ susp p1 iso is_iso nat => nat
  end.

Definition cancelR_iso {C : category} {a b c : ob C} 
  (h : b --> c) (H : IsIsomorphism h) {f g : a --> b} :
  f · h = g · h -> f = g.
Proof.
  intro p.
  set (hinv := pr1 H).
  set (hsec := pr1 (pr2 H)).
  rewrite <- (id_right f).
  rewrite <- (id_right g).
  rewrite <- hsec.
  rewrite <- assoc'.
  rewrite p.
  rewrite assoc'.
  rewrite hsec.
  rewrite id_right.
  reflexivity.
Qed.

Definition post_comp_with_iso_is_inj {C : category} {a b c : ob C} 
  (f g : a --> b) (h : b --> c) (H : IsIsomorphism h) :
  f · h = g · h -> f = g.
Proof.
  apply (cancelR_iso h H).
Qed.
