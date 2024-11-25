Require Import Coq.Reals.Reals.
Require Import Coq.Sets.Ensembles.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Arith.Compare_dec.
Open Scope R_scope.

(* Previous structures *)
Record MotivicSpace := {
  underlying_type : Type;
  A1_homotopy : Type;
  base_scheme : Type;
  transfer : underlying_type -> underlying_type
}.

Record VectorBundle := {
  rank : nat;
  chern_class : R
}.

Record Automorphism := {
  dimension : nat;
  determinant : R;
  trace : R
}.

Inductive K0 : Type :=
  | K0_zero : K0
  | K0_vb : VectorBundle -> K0
  | K0_sum : K0 -> K0 -> K0.

Inductive K1 : Type :=
  | K1_zero : K1
  | K1_auto : Automorphism -> K1
  | K1_sum : K1 -> K1 -> K1
  | K1_inv : K1 -> K1.

(* K₀ equality *)
Inductive K0_eq : K0 -> K0 -> Prop :=
  | K0_eq_zero : K0_eq K0_zero K0_zero
  | K0_eq_vb : forall v1 v2,
      rank v1 = rank v2 ->
      chern_class v1 = chern_class v2 ->
      K0_eq (K0_vb v1) (K0_vb v2)
  | K0_eq_sum : forall a1 a2 b1 b2,
      K0_eq a1 b1 ->
      K0_eq a2 b2 ->
      K0_eq (K0_sum a1 a2) (K0_sum b1 b2).

(* K₁ equality *)
Inductive K1_eq : K1 -> K1 -> Prop :=
  | K1_eq_zero : K1_eq K1_zero K1_zero
  | K1_eq_auto : forall a1 a2,
      dimension a1 = dimension a2 ->
      determinant a1 = determinant a2 ->
      trace a1 = trace a2 ->
      K1_eq (K1_auto a1) (K1_auto a2)
  | K1_eq_sum : forall a1 a2 b1 b2,
      K1_eq a1 b1 ->
      K1_eq a2 b2 ->
      K1_eq (K1_sum a1 a2) (K1_sum b1 b2)
  | K1_eq_inv : forall a b,
      K1_eq a b ->
      K1_eq (K1_inv a) (K1_inv b).

(* Prove K₁ equality is reflexive *)
Theorem K1_eq_refl : forall x, K1_eq x x.
Proof.
  induction x; simpl.
  - apply K1_eq_zero.
  - apply K1_eq_auto; reflexivity.
  - apply K1_eq_sum; assumption.
  - apply K1_eq_inv; assumption.
Qed.

(* Prove K₁ equality is symmetric - fixed proof *)
Theorem K1_eq_sym : forall x y, K1_eq x y -> K1_eq y x.
Proof.
  intros x y H.
  induction H; simpl.
  - apply K1_eq_zero.
  - apply K1_eq_auto; auto.
  - apply K1_eq_sum; auto.
  - apply K1_eq_inv; auto.
Qed.

(* Extended MotivicKTheory *)
Record MotivicKTheory := {
  space : MotivicSpace;
  k0_lift : K0 -> underlying_type space;
  k1_lift : K1 -> underlying_type space;
  k0_lift_vb_eq : forall v1 v2,
    rank v1 = rank v2 ->
    chern_class v1 = chern_class v2 ->
    k0_lift (K0_vb v1) = k0_lift (K0_vb v2);
  k0_lift_sum : forall a b,
    k0_lift (K0_sum a b) = k0_lift a;
  k1_lift_auto_eq : forall a1 a2,
    dimension a1 = dimension a2 ->
    determinant a1 = determinant a2 ->
    trace a1 = trace a2 ->
    k1_lift (K1_auto a1) = k1_lift (K1_auto a2);
  k1_lift_sum : forall a b,
    k1_lift (K1_sum a b) = k1_lift a;
  k1_lift_inv : forall a,
    k1_lift (K1_inv a) = k1_lift a
}.

(* Prove K₁ lift respects equality *)
Theorem k1_lift_respects_eq : forall (mk : MotivicKTheory) (a b : K1),
  K1_eq a b -> k1_lift mk a = k1_lift mk b.
Proof.
  intros mk a b H.
  induction H; simpl.
  - reflexivity.
  - apply (k1_lift_auto_eq mk); assumption.
  - rewrite (k1_lift_sum mk a1 a2).
    rewrite (k1_lift_sum mk b1 b2).
    assumption.
  - rewrite (k1_lift_inv mk a).
    rewrite (k1_lift_inv mk b).
    assumption.
Qed.