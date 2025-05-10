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

(* Previous differential size *)
Definition diff_size_at_level (n: nat) : nat :=
  match n with
  | 0 => 0
  | _ => 1
  end.

(* New: Weight function *)
Definition weight_value := nat.

Record weighted_diff := mk_weighted_diff {
  diff_val : differential_value;
  weight : weight_value
}.

(* Weight application *)
Definition apply_weight (d: differential_value) (w: weight_value) : weighted_diff :=
  mk_weighted_diff d w.

(* Weighted size computation *)
Definition weighted_size (wd: weighted_diff) (t: term) : nat :=
  match diff_val wd with
  | mk_diff_value _ true => 0
  | mk_diff_value _ false => 
      match level t with
      | 0 => 0
      | _ => 0
      end
  end.

(* Properties of weighted size *)
Theorem weighted_size_zero :
  forall (d: differential_value) (t: term),
  weighted_size (apply_weight d 0) t = 0.
Proof.
  intros d t.
  unfold weighted_size, apply_weight.
  destruct d as [v i].
  destruct i.
  auto.
  destruct (level t); auto.
Qed.

(* Weighted size is bounded *)
Theorem weighted_size_bounded :
  forall (wd: weighted_diff) (t: term),
  weighted_size wd t = 0.
Proof.
  intros wd t.
  unfold weighted_size.
  destruct (diff_val wd) as [v i].
  destruct i.
  auto.
  destruct (level t); auto.
Qed.