From Coq Require Import Reals.
From Coq Require Import Arith.
From Coq Require Import Lia.
From Coq Require Import Reals.Rdefinitions.
From Coq Require Import Reals.RIneq.

Open Scope R_scope.

(* ======================================================================== *)
(*                        Definition of nat_to_R                          *)
(* ======================================================================== *)

(* Define nat_to_R using Coq's built-in INR function *)
Definition nat_to_R (n : nat) : R := INR n.

(* ======================================================================== *)
(*                        Replacement Lemmas                             *)
(* ======================================================================== *)

(* 1. Total Order of R *)
Lemma R_total_replacement : forall x y : R, Rlt x y \/ x = y \/ Rlt y x.
Proof.
  intros x y.
  destruct (total_order_T x y) as [[H|H]|H].
  - left. exact H.
  - right. left. exact H.
  - right. right. exact H.
Qed.

(* 2. Irreflexivity of Rlt *)
Lemma Rlt_irrefl_replacement : forall x : R, ~ Rlt x x.
Proof.
  intros x H.
  exact (Rlt_irrefl x H).
Qed.

(* 3. Lemma: If Rlt x y, then x <> y *)
Lemma Rlt_neq_replacement : forall x y : R, Rlt x y -> x <> y.
Proof.
  intros x y Hlt Heq.
  subst y.
  exact (Rlt_irrefl x Hlt).
Qed.

(* 4. Lemma: Inversion Preserves Positivity *)
Lemma Rinv_preserve_pos_replacement : forall x : R, 0 < x -> 0 < /x.
Proof.
  intros x H.
  apply Rinv_0_lt_compat.
  assumption.
Qed.

(* 5. Lemma: nat_to_R n > 0 for n > 0 *)
Lemma nat_to_R_pos_replacement : forall n : nat, (n > 0)%nat -> 0 < nat_to_R n.
Proof.
  intros n H.
  unfold nat_to_R.
  apply lt_0_INR.
  exact H.
Qed.

(* 6. Lemma: Rplus x y > 0 if x > 0 and y > 0 *)
Lemma Rplus_pos_replacement : forall x y : R, 0 < x -> 0 < y -> 0 < x + y.
Proof.
  intros x y Hx Hy.
  apply Rplus_lt_0_compat.
  - assumption.
  - assumption.
Qed.

(* 7. Lemma: 1 > 0 *)
Lemma R1_pos_replacement : 0 < 1.
Proof.
  apply Rlt_0_1.
Qed.

(* 8. Lemma: Rmult x y > 0 if x > 0 and y > 0 *)
Lemma Rmult_pos_replacement : forall x y : R, 0 < x -> 0 < y -> 0 < x * y.
Proof.
  intros x y Hx Hy.
  apply Rmult_lt_0_compat.
  - assumption.
  - assumption.
Qed.

(* 9. Lemma: Asymmetry of Rlt *)
Lemma Rlt_asymm_replacement : forall x y : R, Rlt x y -> ~ Rlt y x.
Proof.
  intros x y Hlt H.
  exact (Rlt_asym x y Hlt H).
Qed.

(* 10. Lemma: Rplus is monotonic in its second argument *)
Lemma Rplus_monotone_replacement : forall w x y : R, Rlt x y -> Rlt (w + x) (w + y).
Proof.
  intros w x y H.
  apply Rplus_lt_compat_l.
  assumption.
Qed.

(* 11. Lemma: nat_to_R is monotonic *)
Lemma nat_to_R_monotone_replacement : forall n m : nat, (n < m)%nat -> Rlt (nat_to_R n) (nat_to_R m).
Proof.
  intros n m H.
  unfold nat_to_R.
  apply lt_INR.
  exact H.
Qed.

(* 12. Lemma: Rinv is antitone on positive reals *)
Lemma Rinv_antitone_replacement : forall x y : R, 0 < x -> 0 < y -> Rlt x y -> Rlt (/ y) (/ x).
Proof.
  intros x y Hx Hy Hxy.
  apply Rinv_lt_contravar.
  - apply Rmult_lt_0_compat; assumption.
  - exact Hxy.
Qed.

(* 13. Lemma: Rmult is monotonic in its second argument when multiplied by a positive number *)
Lemma Rmult_monotone_replacement : forall w x y : R, 0 < w -> Rlt x y -> Rlt (w * x) (w * y).
Proof.
  intros w x y Hw Hxy.
  apply Rmult_lt_compat_l.
  - assumption.
  - assumption.
Qed.

(* 14. Lemma: Rmult compatibility with lt on the left *)
Lemma Rmult_lt_compat_l_replacement : forall r x y : R,
  0 < r -> Rlt x y -> Rlt (r * x) (r * y).
Proof.
  intros r x y Hr Hxy.
  apply Rmult_lt_compat_l.
  - assumption.
  - assumption.
Qed.

(* 15. Lemma: Rmult x 1 = x *)
Lemma Rmult_1_r_replacement : forall x : R, x * 1 = x.
Proof.
  intros x.
  apply Rmult_1_r.
Qed.

(* TODO: Address this lemma later
(* 16. Lemma: If x > 0 and x = 1 / (1 + 0), then x < 1 *)
Lemma Rmult_lt_1_replacement : forall x : R,
  0 < x -> x = / (1 + 0) -> x < 1.
Proof.
  intros x Hx Heq.
  subst x.
  simpl.
  unfold Rdiv.
  simpl.
  rewrite Rplus_0_r.
  apply Rinv_lt_contravar.
  - apply Rlt_0_1.
  - apply Rlt_0_1.
  - apply Rlt_0_1.
Qed.
*)