From Coq Require Import List.
From Coq Require Import Arith.
From Coq Require Import Lia.
From Coq Require Import Reals.
From Coq Require Import Nat.
From Coq Require Import Rfunctions.
From Coq Require Import Rbasic_fun.
From Coq Require Import ZArith.
From Coq Require Import Raxioms.
Import ListNotations.

Open Scope R_scope.

Definition cohomology_degree := (nat * nat)%type.

Record cohomology_class := {
  degree: cohomology_degree;
  weight: R
}.

Record obstruction_class := {
  base_class: cohomology_class;
  stage: nat
}.

Definition stage_weight (n: nat) : R :=
  / (INR (S n)).

Definition is_vanishing (obs: obstruction_class) : Prop :=
  Rle (Rabs (weight (base_class obs))) (stage_weight (stage obs)).

Lemma INR_S_pos : forall n:nat, 0 < INR (S n).
Proof.
  intros n.
  apply lt_0_INR.
  apply Nat.lt_0_succ.
Qed.

Theorem obstruction_vanishing:
  forall obs: obstruction_class,
  exists N : nat,
  forall n : nat,
  (INR n >= INR N)%R ->
  is_vanishing {| base_class := base_class obs; stage := n |}.
Proof.
  intros obs.
  set (w := Rabs (weight (base_class obs))).
  exists (S (Z.to_nat (up w))).
  intros n Hn.
  unfold is_vanishing.
  simpl stage.
  simpl base_class.
  unfold stage_weight.
  assert (H_pos: 0 < INR (S n)) by apply INR_S_pos.
  assert (H_arch: IZR (up w) > w /\ IZR (up w) - w <= 1) by apply archimed.
  apply Rge_le in Hn.
  destruct H_arch as [H_arch_gt H_arch_bound].
  assert (H_Sn_pos: 0 < INR (S n)) by apply INR_S_pos.
  apply Rle_trans with (r2 := /(INR (S (Z.to_nat (up w))))).
Admitted.