(* Import the standard Reals library *)
Require Import Reals.
Open Scope R_scope.

(* Replace nat_to_R with INR from the standard library *)
Definition nat_to_R := INR.

(* Define MotivicSpace with dimension and singularity information *)
Record MotivicSpace : Type := mkMotivicSpace {
  underlying_type : Type;
  dimension : nat;
  has_singularities : bool
}.

(* Define sing_complexity as a concrete function *)
Definition sing_complexity (X : MotivicSpace) : nat :=
  dimension X.

(* Prove sing_complexity_monotone *)
Lemma sing_complexity_monotone : forall X Y : MotivicSpace,
  (dimension X <= dimension Y)%nat -> (sing_complexity X <= sing_complexity Y)%nat.
Proof.
  intros X Y Hdim.
  unfold sing_complexity.
  exact Hdim.
Qed.

(* Define weight functions *)
Definition w_dim (X : MotivicSpace) : R :=
  / (1 + nat_to_R (dimension X)).

Definition w_sing (X : MotivicSpace) : R :=
  / (1 + nat_to_R (sing_complexity X)).

Definition w_stage (n : nat) : R :=
  / (1 + nat_to_R n).

Definition w_total (X : MotivicSpace) (n : nat) : R :=
  w_dim X * w_sing X * w_stage n.

(* Prove positivity of weight functions *)
Lemma w_dim_positive : forall X : MotivicSpace, 0 < w_dim X.
Proof.
  intros X.
  unfold w_dim.
  
  (* Prove 1/(1+nat_to_R(dimension X)) > 0 *)
  
  (* Step 1: Show 1 + nat_to_R(dimension X) > 0 *)
  assert (0 < 1 + nat_to_R (dimension X)).
  {
    (* Step 1a: Show 1 > 0 *)
    assert (0 < 1) by apply Rlt_0_1.
    
    (* Step 1b: Show nat_to_R(dimension X) ≥ 0 *)
    assert (0 <= nat_to_R (dimension X)).
    {
      unfold nat_to_R.
      (* Use a very basic property: for any nat n, INR n ≥ 0 *)
      apply pos_INR.
    }
    
    (* Step 1c: If a > 0 and b ≥ 0, then a + b > 0 *)
    apply Rplus_lt_le_0_compat; assumption.
  }
  
  (* Step 2: If r > 0 then 1/r > 0 *)
  apply Rinv_0_lt_compat.
  exact H.
Qed.

Lemma w_sing_positive : forall X : MotivicSpace, 0 < w_sing X.
Proof.
  intros X.
  unfold w_sing.
  
  (* Prove 1/(1+nat_to_R(sing_complexity X)) > 0 *)
  
  (* Step 1: Show 1 + nat_to_R(sing_complexity X) > 0 *)
  assert (0 < 1 + nat_to_R (sing_complexity X)).
  {
    (* Step 1a: Show 1 > 0 *)
    assert (0 < 1) by apply Rlt_0_1.
    
    (* Step 1b: Show nat_to_R(sing_complexity X) ≥ 0 *)
    assert (0 <= nat_to_R (sing_complexity X)).
    {
      unfold nat_to_R.
      (* For any nat n, INR n ≥ 0 *)
      apply pos_INR.
    }
    
    (* Step 1c: If a > 0 and b ≥ 0, then a + b > 0 *)
    apply Rplus_lt_le_0_compat; assumption.
  }
  
  (* Step 2: If r > 0 then 1/r > 0 *)
  apply Rinv_0_lt_compat.
  exact H.
Qed.

Lemma w_stage_positive : forall n : nat, 0 < w_stage n.
Proof.
  intros n.
  unfold w_stage.
  
  (* Prove 1/(1+nat_to_R n) > 0 *)
  
  (* Step 1: Show 1 + nat_to_R n > 0 *)
  assert (0 < 1 + nat_to_R n).
  {
    (* Step 1a: Show 1 > 0 *)
    assert (0 < 1) by apply Rlt_0_1.
    
    (* Step 1b: Show nat_to_R n ≥ 0 *)
    assert (0 <= nat_to_R n).
    {
      unfold nat_to_R.
      (* For any nat n, INR n ≥ 0 *)
      apply pos_INR.
    }
    
    (* Step 1c: If a > 0 and b ≥ 0, then a + b > 0 *)
    apply Rplus_lt_le_0_compat; assumption.
  }
  
  (* Step 2: If r > 0 then 1/r > 0 *)
  apply Rinv_0_lt_compat.
  exact H.
Qed.

Lemma w_total_positive : forall X : MotivicSpace, forall n : nat, 
  0 < w_total X n.
Proof.
  intros X n.
  unfold w_total.
  
  (* Prove w_dim X * w_sing X * w_stage n > 0 *)
  
  (* If a > 0 and b > 0, then a * b > 0 *)
  apply Rmult_lt_0_compat.
  
  (* First product: w_dim X * w_sing X > 0 *)
  - apply Rmult_lt_0_compat.
    + apply w_dim_positive.
    + apply w_sing_positive.
  
  (* Second factor: w_stage n > 0 *)
  - apply w_stage_positive.
Qed.

Require Import Coq.Arith.PeanoNat.

Lemma le_lt_or_eq : forall n m : nat, (n <= m)%nat -> (n < m \/ n = m)%nat.
Proof.
  intros n m H.
  induction H.
  
  (* Base case: n <= n (reflexivity) *)
  - right. reflexivity.
  
  (* Inductive case: n <= m -> n <= S m *)
  - (* If n < m or n = m, then n < S m *)
    left. apply le_n_S. assumption.
Qed.
