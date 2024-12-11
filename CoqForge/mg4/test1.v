Require Import UniMath.Foundations.All.
Require Import UniMath.Foundations.NaturalNumbers.
Require Import UniMath.Foundations.Sets.

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
