Require Import UniMath.Foundations.All.
Require Import UniMath.Foundations.NaturalNumbers.
Require Import UniMath.Foundations.Sets.
Require Import UniMath.Foundations.Propositions.


Definition natHSet : hSet := make_hSet nat isasetnat.

Definition isInjective {X Y : hSet} (f : X → Y) :=
  ∏ x1 x2 : X, f x1 = f x2 -> x1 = x2.

Definition DedekindInfinite (X : hSet) :=
  ∑ (f : natHSet → X), isInjective f.

Definition DedekindFinite (X : hSet) := ¬ (DedekindInfinite X).

Lemma DedekindInfinite_nonempty (X : hSet) :
  DedekindInfinite X -> ∥ X ∥.
Proof.
  intros [f _]. (* f : natHSet → X and we have an injective function, but we don't need injectivity here *)
  apply hinhpr.
  exact (f 0%nat).
Qed.

Lemma natHSet_is_DedekindInfinite : DedekindInfinite natHSet.
Proof.
  (* We can use the identity function on natHSet as an injection *)
  exists (fun x => x).
  intros x1 x2 eq.
  exact eq.
Qed.

Definition boolHSet : hSet := make_hSet bool isasetbool.

Inductive empty : Type := . (* no constructors *)

Lemma natneq0and1 : 0%nat = 1%nat -> hfalse.
Proof.
  intro p.
  (* Define P: a family indexed by nat *)
  set (P := λ n:nat, match n with 
                     | 0%nat => unit
                     | S _   => hfalse
                     end).

  (* transportf moves tt : P(0)=unit along p:0=1 to P(1)=hfalse *)
  apply (transportf P p tt).
Qed.

Lemma natneq0and2 : 0%nat = 2%nat -> hfalse.
Proof.
  intro p.
  (* Define a family P where P(0)=unit and P(2)=hfalse, so transporting tt along p yields hfalse *)
  set (P := λ n:nat, match n with
                     | 0%nat => unit
                     | 1%nat => unit
                     | 2%nat => hfalse
                     | _ => unit
                     end).
  exact (transportf P p tt).
Qed.

Lemma natneq2and0 : 2%nat = 0%nat -> hfalse.
Proof.
  intro p.
  apply natneq0and2.
  apply pathsinv0.
  exact p.
Qed.














