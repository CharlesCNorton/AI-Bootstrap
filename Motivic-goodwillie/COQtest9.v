From Coq Require Import List.
From Coq Require Import Arith.
From Coq Require Import Lia.
Import ListNotations.

(* Import previous structures *)
Record bidegree := mk_bidegree {
  p_index : nat;
  q_index : nat
}.

Record term := mk_term {
  degree : bidegree;
  level : nat
}.

Record differential_value := mk_diff_value {
  value : nat;
  is_zero : bool
}.

(* New: Recursive differential structure *)
Fixpoint recursive_diff (t: term) (depth: nat) : differential_value :=
  match depth with
  | 0 => mk_diff_value 0 true
  | S n => 
      match level t with
      | 0 => mk_diff_value 0 true
      | S m => mk_diff_value (if le_gt_dec (value (recursive_diff t n)) 2 
                             then value (recursive_diff t n)
                             else 2) false
      end
  end.

(* Recursive differential is bounded *)
Theorem recursive_diff_bounded :
  forall (t: term) (depth: nat),
  value (recursive_diff t depth) <= 2.
Proof.
  intros t depth.
  induction depth.
  - simpl. auto.
  - simpl.
    destruct (level t).
    + auto.
    + destruct (le_gt_dec (value (recursive_diff t depth)) 2).
      * exact l.
      * auto.
Qed.

(* Zero level implies zero differential at any depth *)
Theorem recursive_diff_zero_level :
  forall (t: term) (depth: nat),
  level t = 0 ->
  is_zero (recursive_diff t depth) = true.
Proof.
  intros t depth H.
  induction depth.
  - simpl. reflexivity.
  - simpl. rewrite H. reflexivity.
Qed.