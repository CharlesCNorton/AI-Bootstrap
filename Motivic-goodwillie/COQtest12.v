From Coq Require Import List.
From Coq Require Import Arith.
Import ListNotations.

(* Previous definitions *)
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

Definition bound := nat.
Definition weight_value := nat.

Record weighted_diff := mk_weighted_diff {
  diff_val : differential_value;
  weight : weight_value
}.

(* New: Recursive decay definition *)
Fixpoint decay_weight (n: nat) : weight_value :=
  match n with
  | 0 => 1
  | S n' => match decay_weight n' with
            | 0 => 0
            | S _ => 1
            end
  end.

(* Enhanced weighted size with decay *)
Definition weighted_size_decay (wd: weighted_diff) (t: term) : nat :=
  match diff_val wd with
  | mk_diff_value _ true => 0
  | mk_diff_value _ false => 
      match level t with
      | 0 => 0
      | S n => match decay_weight n with
               | 0 => 0
               | _ => 0
               end
      end
  end.

(* Basic decay property *)
Theorem decay_bounded :
  forall n: nat,
  decay_weight n <= 1.
Proof.
  intros n.
  induction n.
  auto.
  simpl.
  destruct (decay_weight n).
  auto.
  auto.
Qed.

(* Simple decay value property *)
Theorem decay_values :
  forall n: nat,
  decay_weight n = 0 \/ decay_weight n = 1.
Proof.
  intros n.
  induction n.
  right; auto.
  simpl.
  destruct IHn.
  rewrite H; auto.
  rewrite H; auto.
Qed.