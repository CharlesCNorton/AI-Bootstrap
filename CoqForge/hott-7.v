From HoTT.Basics Require Import PathGroupoids.
From HoTT.Types Require Import Paths.
Import Overture (idpath, paths, concat).

Parameter R : Type.
Parameter Rplus : R -> R -> R.
Parameter Rmult : R -> R -> R.
Parameter Rlt : R -> R -> Prop.
Parameter R0 : R.
Parameter R1 : R.
Parameter Rinv : R -> R.

Parameter nat_to_R : nat -> R.

Axiom R_ordered : forall x y z : R, Rlt x y -> Rlt y z -> Rlt x z.
Axiom R_total : forall x y : R, Rlt x y \/ x = y \/ Rlt y x.
Axiom Rinv_pos : forall x : R, Rlt R0 x -> Rlt R0 (Rinv x).
Axiom nat_to_R_pos : forall n : nat, Rlt R0 (nat_to_R n).
Axiom Rplus_pos : forall x y : R, Rlt R0 x -> Rlt R0 y -> Rlt R0 (Rplus x y).
Axiom R1_pos : Rlt R0 R1.
Axiom Rinv_preserve_pos : forall x : R, Rlt R0 x -> Rlt R0 (Rinv x).
Axiom Rmult_pos : forall x y : R, Rlt R0 x -> Rlt R0 y -> Rlt R0 (Rmult x y).

Axiom Rplus_monotone : forall w x y : R, Rlt x y -> Rlt (Rplus w x) (Rplus w y).
Axiom nat_to_R_monotone : forall n m : nat, n <= m -> Rlt (nat_to_R n) (nat_to_R m).
Axiom Rinv_antitone : forall x y : R, Rlt R0 x -> Rlt R0 y -> Rlt x y -> Rlt (Rinv y) (Rinv x).
Axiom Rmult_monotone : forall w x y : R, Rlt R0 w -> Rlt x y -> Rlt (Rmult w x) (Rmult w y).
Axiom Rmult_lt_compat_l : forall r x y : R,
  Rlt R0 r -> Rlt x y -> Rlt (Rmult r x) (Rmult r y).
Axiom Rmult_1_r : forall x : R, Rmult x R1 = x.
Axiom Rmult_lt_1 : forall x : R, 
  Rlt R0 x -> x = Rinv (Rplus R1 (nat_to_R 0)) -> Rlt x R1.

Record MotivicSpace : Type := mkMotivicSpace {
  underlying_type : Type;
  dimension : nat;
  has_singularities : bool
}.

Record WeightFunction : Type := mkWeightFunction {
  weight_map : MotivicSpace -> R;
  weight_positive : forall X : MotivicSpace, Rlt R0 (weight_map X);
  weight_monotone : forall X Y : MotivicSpace, 
    dimension X <= dimension Y -> 
    Rlt (weight_map Y) (weight_map X)
}.

Parameter sing_complexity : MotivicSpace -> nat.
Axiom sing_complexity_monotone : forall X Y : MotivicSpace,
  dimension X <= dimension Y -> sing_complexity X <= sing_complexity Y.

Definition w_dim (X : MotivicSpace) : R :=
  Rinv (Rplus R1 (nat_to_R (dimension X))).

Definition w_sing (X : MotivicSpace) : R :=
  Rinv (Rplus R1 (nat_to_R (sing_complexity X))).

Definition w_stage (n : nat) : R :=
  Rinv (Rplus R1 (nat_to_R n)).

Definition w_total (X : MotivicSpace) (n : nat) : R :=
  Rmult (Rmult (w_dim X) (w_sing X)) (w_stage n).

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

Lemma w_total_positive : forall X : MotivicSpace, forall n : nat, 
  Rlt R0 (w_total X n).
Proof.
  intros X n.
  unfold w_total.
  apply Rmult_pos.
  + apply Rmult_pos.
    * apply w_dim_positive.
    * apply w_sing_positive.
  + apply w_stage_positive.
Qed.

Lemma w_dim_monotone : forall X Y : MotivicSpace,
  dimension X <= dimension Y -> Rlt (w_dim Y) (w_dim X).
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

Lemma w_sing_monotone : forall X Y : MotivicSpace,
  dimension X <= dimension Y -> Rlt (w_sing Y) (w_sing X).
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

Lemma w_stage_monotone : forall n m : nat,
  n <= m -> Rlt (w_stage m) (w_stage n).
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

Axiom obstruction_positive : forall o : ObstructionClass, 
  Rlt R0 (obstruction_measure o).

Axiom obstruction_weighted_decay : forall (n : nat) (X : MotivicSpace),
  Rlt (obstruction_measure (stage_obstruction (S n) X))
      (Rmult (obstruction_measure (stage_obstruction n X)) (w_total X n)).

Definition converges (X : MotivicSpace) : Prop :=
  forall epsilon : R, Rlt R0 epsilon -> 
  exists N : nat, forall n : nat, 
  n >= N -> Rlt (obstruction_measure (stage_obstruction n X)) epsilon.

Axiom Rmult_lt_1_compat : forall (x : R) (w : R),
  Rlt R0 w -> Rlt w R1 ->
  Rlt (Rmult x w) x.

Axiom w_total_lt_1 : forall (X : MotivicSpace) (n : nat),
  Rlt (w_total X n) R1.

Lemma obstruction_decay : forall (X : MotivicSpace) (n : nat),
  Rlt (obstruction_measure (stage_obstruction (S n) X))
      (obstruction_measure (stage_obstruction n X)).
Proof.
  intros X n.
  assert (H := obstruction_weighted_decay n X).
  assert (H1 := w_total_positive X n).
  assert (H2 := w_total_lt_1 X n).
  apply (R_ordered _ _ _ H).
  apply (Rmult_lt_1_compat (obstruction_measure (stage_obstruction n X)) (w_total X n)).
  + exact H1.
  + exact H2.
Qed.