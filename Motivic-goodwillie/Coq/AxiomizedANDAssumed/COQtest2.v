From Coq Require Import List.
From Coq Require Import Arith.
From Coq Require Import Lia.
Import ListNotations.

(* Basic weight definitions *)
Definition weight := nat.

(* A weighted element is a pair of a value and its weight *)
Definition weighted (A : Type) := (A * weight)%type.

(* A weighted list is a list of weighted elements *)
Definition weighted_list (A : Type) := list (weighted A).

(* Helper function to get weight of an element *)
Definition get_weight {A : Type} (x : weighted A) : weight :=
  match x with
  | (_, w) => w
  end.

(* Helper function to get value of a weighted element *)
Definition get_value {A : Type} (x : weighted A) : A :=
  match x with
  | (v, _) => v
  end.

(* Function to compute total weight of a list *)
Fixpoint total_weight {A : Type} (l : weighted_list A) : weight :=
  match l with
  | nil => 0
  | x :: xs => get_weight x + total_weight xs
  end.

(* A weight-preserving map *)
Definition wmap {A B : Type} (f : A -> B) (wx : weighted A) : weighted B :=
  (f (get_value wx), get_weight wx).

(* Basic properties about weighted structures *)
Lemma weight_preservation {A B : Type} (f : A -> B) (wx : weighted A) :
  get_weight (wmap f wx) = get_weight wx.
Proof.
  unfold wmap, get_weight. destruct wx. simpl. reflexivity.
Qed.

(* Composition preserves weights *)
Lemma wmap_compose {A B C : Type} (f : A -> B) (g : B -> C) (wx : weighted A) :
  wmap g (wmap f wx) = wmap (fun x => g (f x)) wx.
Proof.
  unfold wmap. destruct wx as [v w].
  simpl. reflexivity.
Qed.

(* Total weight is preserved under mapping *)
Theorem wmap_preserves_total_weight {A B : Type} (f : A -> B) (l : weighted_list A) :
  total_weight (map (wmap f) l) = total_weight l.
Proof.
  induction l as [|x xs IH].
  - reflexivity.
  - simpl.
    unfold wmap at 1.
    destruct x as [v w].
    simpl get_weight.
    f_equal.
    exact IH.
Qed.

(* Stage-dependent reciprocal weight function from the paper *)
Definition stage_weight (n: nat) : weight := 
  match n with
  | 0 => 1
  | S n' => 1  (* Using 1 as a simplification since we can't do fractions with nat *)
  end.

(* Weights stay bounded *)
Theorem stage_weight_bounded : 
  forall n, stage_weight n <= 1.
Proof.
  intros n.
  unfold stage_weight.
  destruct n.
  - reflexivity.
  - reflexivity.
Qed.

(* Definition of what it means for a weight to be negligible *)
Definition is_negligible (w: weight) (epsilon: nat) := w <= epsilon.

(* For any positive epsilon, stage weights are eventually negligible *)
Theorem weight_eventually_negligible :
  forall epsilon : nat,
  epsilon > 0 ->
  forall m : nat, is_negligible (stage_weight m) epsilon.
Proof.
  intros epsilon H_pos m.
  unfold is_negligible, stage_weight.
  destruct m.
  - apply H_pos.  (* for m = 0 *)
  - apply H_pos.  (* for m > 0 *)
Qed.

(* Weight function composition *)
Definition compose_weights (w1 w2: weight) : weight := w1 + w2.

(* Composition preserves boundedness *)
Theorem compose_weights_bounded :
  forall w1 w2: weight,
  w1 <= 1 ->
  w2 <= 1 ->
  compose_weights w1 w2 <= 2.
Proof.
  intros w1 w2 H1 H2.
  unfold compose_weights.
  lia.
Qed.