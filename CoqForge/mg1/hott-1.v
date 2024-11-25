(* Import specific modules from HoTT *)
From HoTT.Basics Require Import PathGroupoids.
From HoTT.Types Require Import Paths.
Import Overture (idpath, paths).

(* Define a simple theorem *)
Definition my_test_theorem {A : Type} (x : A) : paths x x := idpath.

(* Prove the theorem interactively *)
Theorem my_example_proof : forall (A : Type) (x : A), paths x x.
Proof.
  intros A x. (* Introduce variables into the proof context *)
  apply idpath. (* Use the reflexivity of paths *)
Qed.
