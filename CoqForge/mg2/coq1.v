s(* First, load the required libraries *)
Require Import Coq.Reals.Reals.
Require Import Coq.micromega.Lra.
Open Scope R_scope.

(* Define p(x) and its derivatives *)
Definition p (x : R) : R := x^2 + x + 1.
Definition dp (x : R) : R := 2 * x + 1.
Definition ddp (x : R) : R := 2.

(* Define q(x) and its derivatives *)
Definition q (x : R) : R := x^3 + 2 * x^2 + x.
Definition dq (x : R) : R := 3 * x^2 + 4 * x + 1.
Definition ddq (x : R) : R := 6 * x + 4.
Definition dddq (x : R) : R := 6.

(* Taylor series and residuals for p(x) *)
Definition T2_p (x : R) : R := p 0 + dp 0 * x + (ddp 0 * x^2) / 2.
Definition R2_p (x : R) : R := p x - T2_p x.

(* Taylor series and residuals for q(x) *)
Definition T2_q (x : R) : R := q 0 + dq 0 * x + (ddq 0 * x^2) / 2.
Definition R2_q (x : R) : R := q x - T2_q x.

(* Define composite function p(q(x)) *)
Definition composite (x : R) : R := p (q x).

(* Taylor series and residual for composite function *)
Definition T2_composite (x : R) : R := T2_p (T2_q x).
Definition R2_composite (x : R) : R := composite x - T2_composite x.

(* First set of proofs: T2 explicit forms *)
Lemma T2_p_is_taylor : forall x : R, 
  T2_p x = 1 + x + x^2.
Proof.
  intros.
  unfold T2_p, p, dp, ddp.
  field.
Qed.

Lemma T2_q_is_taylor : forall x : R, 
  T2_q x = x + 2 * x^2.
Proof.
  intros.
  unfold T2_q, q, dq, ddq.
  field.
Qed.

Lemma T2_composite_is_taylor : forall x : R, 
  T2_composite x = 1 + (x + 2*x^2) + (x + 2*x^2)^2.
Proof.
  intros.
  unfold T2_composite, T2_p.
  rewrite T2_q_is_taylor.
  unfold p, dp, ddp.
  simpl.
  field.
Qed.

(* Second set of proofs: R2 residual analysis *)
Lemma R2_p_explicit : forall x : R,
  R2_p x = 0.
Proof.
  intros.
  unfold R2_p.
  rewrite T2_p_is_taylor.
  unfold p.
  ring.
Qed.

Lemma R2_q_explicit : forall x : R,
  R2_q x = x^3.
Proof.
  intros.
  unfold R2_q.
  rewrite T2_q_is_taylor.
  unfold q.
  ring.
Qed.

Lemma R2_composite_explicit : forall x : R,
  R2_composite x = (q x)^2 + q x + 1 - (1 + (x + 2*x^2) + (x + 2*x^2)^2).
Proof.
  intros.
  unfold R2_composite, composite.
  unfold p.
  rewrite T2_composite_is_taylor.
  unfold q.
  field.
Qed.

(* Third-order Taylor series definitions *)
Definition T3_p (x : R) : R := p 0 + dp 0 * x + (ddp 0 * x^2) / 2 + 0.
Definition T3_q (x : R) : R := q 0 + dq 0 * x + (ddq 0 * x^2) / 2 + (dddq 0 * x^3) / 6.
Definition T3_composite (x : R) : R := T3_p (T3_q x).

(* Residuals for third-order *)
Definition R3_p (x : R) : R := p x - T3_p x.
Definition R3_q (x : R) : R := q x - T3_q x.
Definition R3_composite (x : R) : R := composite x - T3_composite x.

(* Third set of proofs: T3 explicit forms *)
Lemma T3_p_is_taylor : forall x : R, 
  T3_p x = 1 + x + x^2.
Proof.
  intros.
  unfold T3_p, p, dp, ddp.
  field.
Qed.

Lemma T3_q_is_taylor : forall x : R, 
  T3_q x = x + 2*x^2 + x^3.
Proof.
  intros.
  unfold T3_q, q, dq, ddq, dddq.
  field.
Qed.

Lemma T3_composite_is_taylor : forall x : R,
  T3_composite x = 1 + (x + 2*x^2 + x^3) + (x + 2*x^2 + x^3)^2.
Proof.
  intros.
  unfold T3_composite, T3_p.
  rewrite T3_q_is_taylor.
  unfold p, dp, ddp.
  field.
Qed.

(* Fourth set of proofs: R3 residual analysis *)
Lemma R3_p_explicit : forall x : R,
  R3_p x = 0.
Proof.
  intros.
  unfold R3_p.
  rewrite T3_p_is_taylor.
  unfold p.
  ring.
Qed.

Lemma R3_q_explicit : forall x : R,
  R3_q x = 0.
Proof.
  intros.
  unfold R3_q.
  rewrite T3_q_is_taylor.
  unfold q.
  ring.
Qed.

Lemma R3_composite_explicit : forall x : R,
  R3_composite x = (q x)^2 + q x + 1 - (1 + (x + 2*x^2 + x^3) + (x + 2*x^2 + x^3)^2).
Proof.
  intros.
  unfold R3_composite, composite.
  unfold p.
  rewrite T3_composite_is_taylor.
  unfold q.
  field.
Qed.

(* Final convergence theorems *)
Theorem p_exact_at_2 : forall x : R,
  p x = T2_p x.
Proof.
  intros.
  unfold p.
  rewrite T2_p_is_taylor.
  ring.
Qed.

Theorem q_exact_at_3 : forall x : R,
  q x = T3_q x.
Proof.
  intros.
  unfold q.
  rewrite T3_q_is_taylor.
  ring.
Qed.