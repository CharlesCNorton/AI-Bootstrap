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

(* New: Growth control structures *)
Definition bound_constant := 2.

(* Bounded differential value *)
Definition is_bounded (d: differential_value) : Prop :=
  value d <= bound_constant.

(* Growth control function *)
Definition control_growth (t: term) : differential_value :=
  match level t with
  | 0 => mk_diff_value 0 true
  | S n => mk_diff_value (if (le_lt_dec (S n) bound_constant) 
                         then S n 
                         else bound_constant) false
  end.

(* Key theorem: Growth is bounded *)
Theorem control_growth_bounded :
  forall t: term,
  is_bounded (control_growth t).
Proof.
  intros t.
  unfold is_bounded, control_growth.
  destruct (level t).
  - unfold bound_constant. auto with arith.
  - unfold bound_constant.
    destruct (le_lt_dec (S n) 2); auto with arith.
Qed.

(* Growth control respects zero differentials *)
Theorem control_growth_zero :
  forall t: term,
  level t = 0 ->
  is_zero (control_growth t) = true.
Proof.
  intros t H.
  unfold control_growth.
  rewrite H.
  reflexivity.
Qed.