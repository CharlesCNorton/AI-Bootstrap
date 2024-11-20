From Coq Require Import List.
From Coq Require Import Arith.
From Coq Require Import Lia.
Import ListNotations.

(* First, let's define a simple weight type *)
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

(* Now let's prove some basic properties about weighted structures *)

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
  - simpl. (* expand total_weight and map *)
    unfold wmap at 1. (* just unfold the outer wmap *)
    destruct x as [v w].
    simpl get_weight.
    (* Now the goal should be: w + total_weight (map (wmap f) xs) = w + total_weight xs *)
    f_equal.
    (* This leaves us with: total_weight (map (wmap f) xs) = total_weight xs *)
    exact IH.
Qed.