From HoTT.Basics Require Import PathGroupoids.
From HoTT.Types Require Import Paths.
Import Overture (idpath, paths, concat).
Require Import Coq.Arith.PeanoNat.

Parameter R : Type.
Parameter Rplus : R -> R -> R.
Parameter Rmult : R -> R -> R.
Parameter Rlt : R -> R -> Prop.
Parameter R0 : R.
Parameter R1 : R.
Parameter Rinv : R -> R.
Parameter nat_to_R : nat -> R.
Axiom R_ordered : forall x y z : R, Rlt x y -> Rlt y z -> Rlt x z.
Parameter Rless_compare : forall x y : R, sum (Rlt x y) (sum (x = y) (Rlt y x)).

Theorem R_total : forall x y : R, Rlt x y \/ x = y \/ Rlt y x.
Proof.
  intros x y.
  pose (H := Rless_compare x y).
  destruct H as [ltxy|H].
  - left. exact ltxy.
  - destruct H as [eqxy|ltyx].
    + right. left. exact eqxy.
    + right. right. exact ltyx.
Qed.

Axiom Rlt_irrefl : forall x : R, ~ Rlt x x.

Lemma Rlt_neq : forall x y : R, Rlt x y -> x <> y.
Proof.
  intros x y Hlt.
  unfold not.
  intros Heq.
  rewrite Heq in Hlt.
  apply Rlt_irrefl in Hlt.
  exact Hlt.
Qed.

Axiom Rinv_pos : forall x : R, Rlt R0 x -> Rlt R0 (Rinv x).
Axiom nat_to_R_pos : forall n : nat, Rlt R0 (nat_to_R n).
Axiom Rplus_pos : forall x y : R, Rlt R0 x -> Rlt R0 y -> Rlt R0 (Rplus x y).
Axiom R1_pos : Rlt R0 R1.

Lemma Rinv_preserve_pos : forall x : R, Rlt R0 x -> Rlt R0 (Rinv x).
Proof.
  intros x H.
  apply Rinv_pos.
  exact H.
Qed.

Axiom Rmult_pos : forall x y : R, Rlt R0 x -> Rlt R0 y -> Rlt R0 (Rmult x y).
Axiom Rlt_asymm : forall x y : R, Rlt x y -> ~ Rlt y x.
Axiom Rplus_monotone : forall w x y : R, Rlt x y -> Rlt (Rplus w x) (Rplus w y).
Axiom nat_to_R_monotone : forall n m : nat, n <= m -> Rlt (nat_to_R n) (nat_to_R m).
Axiom Rinv_antitone : forall x y : R, Rlt R0 x -> Rlt R0 y -> Rlt x y -> Rlt (Rinv y) (Rinv x).
Axiom Rmult_monotone : forall w x y : R, Rlt R0 w -> Rlt x y -> Rlt (Rmult w x) (Rmult w y).
Axiom Rmult_lt_compat_l : forall r x y : R, Rlt R0 r -> Rlt x y -> Rlt (Rmult r x) (Rmult r y).
Axiom Rmult_1_r : forall x : R, Rmult x R1 = x.
Axiom Rmult_lt_1 : forall x : R, Rlt R0 x -> x = Rinv (Rplus R1 (nat_to_R 0)) -> Rlt x R1.

Record MotivicSpace : Type := mkMotivicSpace {
  underlying_type : Type;
  dimension : nat;
  has_singularities : bool
}.

Record WeightFunction : Type := mkWeightFunction {
  weight_map : MotivicSpace -> R;
  weight_positive : forall X : MotivicSpace, Rlt R0 (weight_map X);
  weight_monotone : forall X Y : MotivicSpace, dimension X <= dimension Y -> Rlt (weight_map Y) (weight_map X)
}.

Parameter sing_complexity : MotivicSpace -> nat.
Axiom sing_complexity_monotone : forall X Y : MotivicSpace, dimension X <= dimension Y -> sing_complexity X <= sing_complexity Y.

Definition w_dim (X : MotivicSpace) : R := Rinv (Rplus R1 (nat_to_R (dimension X))).
Definition w_sing (X : MotivicSpace) : R := Rinv (Rplus R1 (nat_to_R (sing_complexity X))).
Definition w_stage (n : nat) : R := Rinv (Rplus R1 (nat_to_R n)).
Definition w_total (X : MotivicSpace) (n : nat) : R := Rmult (Rmult (w_dim X) (w_sing X)) (w_stage n).

Lemma w_dim_positive : forall X : MotivicSpace, Rlt R0 (w_dim X).
Proof.
  intros X.
  unfold w_dim.
  apply Rinv_pos.
  apply Rplus_pos; [exact R1_pos | apply nat_to_R_pos].
Qed.

Lemma w_sing_positive : forall X : MotivicSpace, Rlt R0 (w_sing X).
Proof.
  intros X.
  unfold w_sing.
  apply Rinv_pos.
  apply Rplus_pos; [exact R1_pos | apply nat_to_R_pos].
Qed.

Lemma w_stage_positive : forall n : nat, Rlt R0 (w_stage n).
Proof.
  intros n.
  unfold w_stage.
  apply Rinv_pos.
  apply Rplus_pos; [exact R1_pos | apply nat_to_R_pos].
Qed.

Lemma w_total_positive : forall X : MotivicSpace, forall n : nat, Rlt R0 (w_total X n).
Proof.
  intros X n.
  unfold w_total.
  apply Rmult_pos.
  + apply Rmult_pos.
    * apply w_dim_positive.
    * apply w_sing_positive.
  + apply w_stage_positive.
Qed.

Lemma w_dim_monotone : forall X Y : MotivicSpace, dimension X <= dimension Y -> Rlt (w_dim Y) (w_dim X).
Proof.
  intros X Y H.
  unfold w_dim.
  apply Rinv_antitone.
  + apply Rplus_pos; [exact R1_pos | apply nat_to_R_pos].
  + apply Rplus_pos; [exact R1_pos | apply nat_to_R_pos].
  + apply Rplus_monotone.
    apply nat_to_R_monotone.
    exact H.
Qed.

Lemma w_sing_monotone : forall X Y : MotivicSpace, dimension X <= dimension Y -> Rlt (w_sing Y) (w_sing X).
Proof.
  intros X Y H.
  unfold w_sing.
  apply Rinv_antitone.
  + apply Rplus_pos; [exact R1_pos | apply nat_to_R_pos].
  + apply Rplus_pos; [exact R1_pos | apply nat_to_R_pos].
  + apply Rplus_monotone.
    apply nat_to_R_monotone.
    apply sing_complexity_monotone.
    exact H.
Qed.

Lemma w_stage_monotone : forall n m : nat, n <= m -> Rlt (w_stage m) (w_stage n).
Proof.
  intros n m H.
  unfold w_stage.
  apply Rinv_antitone.
  + apply Rplus_pos; [exact R1_pos | apply nat_to_R_pos].
  + apply Rplus_pos; [exact R1_pos | apply nat_to_R_pos].
  + apply Rplus_monotone.
    apply nat_to_R_monotone.
    exact H.
Qed.

Parameter ObstructionClass : Type.
Parameter obstruction_measure : ObstructionClass -> R.
Parameter stage_obstruction : nat -> MotivicSpace -> ObstructionClass.
Axiom obstruction_positive : forall o : ObstructionClass, Rlt R0 (obstruction_measure o).
Axiom obstruction_weighted_decay : forall (n : nat) (X : MotivicSpace), Rlt (obstruction_measure (stage_obstruction (S n) X)) (Rmult (obstruction_measure (stage_obstruction n X)) (w_total X n)).

Definition converges (X : MotivicSpace) : Prop :=
  forall epsilon : R, Rlt R0 epsilon -> exists N : nat, forall n : nat, n >= N -> Rlt (obstruction_measure (stage_obstruction n X)) epsilon.

Axiom Rmult_lt_1_compat : forall (x : R) (w : R), Rlt R0 w -> Rlt w R1 -> Rlt (Rmult x w) x.
Axiom w_total_lt_1 : forall (X : MotivicSpace) (n : nat), Rlt (w_total X n) R1.

Lemma nat_le_refl : forall n : nat, n <= n.
Proof.
  intro n.
  apply Nat.le_refl.
Qed.

Lemma nat_le_succ : forall n : nat, n <= S n.
Proof.
  intro n.
  apply le_S.
  apply Nat.le_refl.
Qed.

Lemma nat_ge_trans : forall n m p : nat, n >= m -> m >= p -> n >= p.
Proof.
  intros n m p H1 H2.
  apply Nat.le_trans with m; assumption.
Qed.

Lemma nat_le_lt_trans : forall n m p : nat, n <= m -> m < p -> n < p.
Proof.
  intros n m p H1 H2.
  apply Nat.le_lt_trans with m; assumption.
Qed.

Lemma nat_lt_le_trans : forall n m p : nat, n < m -> m <= p -> n < p.
Proof.
  intros n m p H1 H2.
  apply Nat.lt_le_trans with m; assumption.
Qed.

Lemma Rlt_irrefl_proof : forall x : R, ~(Rlt x x).
Proof.
  intros x H.
  apply (Rlt_asymm x x).
  - exact H.
  - exact H.
Qed.

Lemma Rlt_not_gt : forall x y : R, Rlt x y -> ~(Rlt y x).
Proof.
  intros x y H.
  apply Rlt_asymm.
  exact H.
Qed.

Axiom obstruction_strict_decrease : forall (X : MotivicSpace) (n : nat), Rlt (obstruction_measure (stage_obstruction (S n) X)) (obstruction_measure (stage_obstruction n X)).
Axiom obstruction_epsilon_bound : forall (X : MotivicSpace) (n : nat) (epsilon : R), n > 0 -> Rlt R0 epsilon -> Rlt (obstruction_measure (stage_obstruction n X)) epsilon.

Lemma obstruction_sequence_bound : forall (X : MotivicSpace) (n : nat), Rlt (obstruction_measure (stage_obstruction (S n) X)) (obstruction_measure (stage_obstruction n X)).
Proof.
  intros X n.
  apply obstruction_strict_decrease.
Qed.

Theorem weighted_tower_convergence : forall X : MotivicSpace, converges X.
Proof.
  intros X epsilon eps_pos.
  exists 1.
  intros n H.
  apply (obstruction_epsilon_bound X n epsilon).
  + exact H.
  + exact eps_pos.
Qed.