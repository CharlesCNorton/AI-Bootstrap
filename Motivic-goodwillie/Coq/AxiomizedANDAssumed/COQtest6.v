From Coq Require Import List.
From Coq Require Import Arith.
From Coq Require Import Lia.
Import ListNotations.

(* Basic index type for spectral sequences *)
Record bidegree := mk_bidegree {
  p_index : nat;
  q_index : nat
}.

(* Basic structure for a term in spectral sequence *)
Record term := mk_term {
  degree : bidegree;
  level : nat  (* 'r' in dᵣᵖ,ᑫ *)
}.

(* Differential source and target *)
Definition diff_source (t: term) : bidegree := degree t.

Definition diff_target (t: term) : bidegree :=
  mk_bidegree 
    (p_index (degree t) + level t)
    (q_index (degree t) - level t).

(* Basic differential property - source and target connect properly *)
Theorem diff_connection :
  forall t: term,
  p_index (diff_target t) = p_index (diff_source t) + level t.
Proof.
  intros t.
  unfold diff_target, diff_source.
  simpl.
  reflexivity.
Qed.