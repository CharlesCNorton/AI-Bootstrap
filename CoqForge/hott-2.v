From HoTT.Basics Require Import PathGroupoids.
From HoTT.Types Require Import Paths.
Import Overture (idpath, paths, concat).

Lemma iter3_double {A : Type} {a : A} (p : paths a a) :
  paths (iter3 2 p) (concat p p).
Proof.
  simpl.
  rewrite (concat_p1 p).
  reflexivity.
Qed.