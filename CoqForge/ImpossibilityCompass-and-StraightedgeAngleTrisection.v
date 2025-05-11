From Coq Require Import QArith.QArith.
From Coq Require Import QArith.Qreals.

From Coq Require Import Reals.Reals.
Open Scope R_scope.      (* so +, *, etc. default to ℝ *)

Inductive constructible : R -> Prop :=
| C_Q    : forall q : Q,
             constructible (Q2R q)
| C_add  : forall x y : R,
             constructible x -> constructible y ->
             constructible (x + y)
| C_sub  : forall x y : R,
             constructible x -> constructible y ->
             constructible (x - y)
| C_mul  : forall x y : R,
             constructible x -> constructible y ->
             constructible (x * y)
| C_inv  : forall x : R,
             constructible x -> x <> 0 ->
             constructible (/ x)
| C_sqrt : forall x : R,
             constructible x -> 0 <= x ->
             constructible (sqrt x).

From Coq Require Import Field.          (* gives the “field” tactic *)

Lemma constructible_0 : constructible 0.
Proof.
  (* First rewrite 0 as Q2R 0%Q, then apply the constructor C_Q. *)
  replace 0 with (Q2R 0%Q).
  - constructor.                       (* C_Q, with q := 0%Q *)
  - unfold Q2R; simpl; field.          (* proves 0 = Q2R 0%Q *)
Qed.

Lemma constructible_1 : constructible 1.
Proof.
  (* Rewrite 1 as the rational 1%Q embedded in R, then apply C_Q. *)
  replace 1 with (Q2R 1%Q).
  - constructor.                      (* uses C_Q with q := 1%Q *)
  - unfold Q2R; simpl; field.         (* proves 1 = Q2R 1%Q *)
Qed.

Lemma constructible_opp :
  forall x : R, constructible x -> constructible (- x).
Proof.
  intros x Hx.
  replace (- x) with (0 - x) by field.
  apply C_sub; [apply constructible_0 | exact Hx].
Qed.

Lemma constructible_add :
  forall x y : R, constructible x -> constructible y ->
                  constructible (x + y).
Proof.
  intros x y Hx Hy.
  apply C_add; assumption.
Qed.

Lemma constructible_sub :
  forall x y : R, constructible x -> constructible y ->
                  constructible (x - y).
Proof.
  intros x y Hx Hy.
  apply C_sub; assumption.
Qed.

Lemma constructible_mul :
  forall x y : R, constructible x -> constructible y ->
                  constructible (x * y).
Proof.
  intros x y Hx Hy.
  apply C_mul; assumption.
Qed.

Lemma constructible_inv :
  forall x : R, constructible x -> x <> 0 -> constructible (/ x).
Proof. intros x Hx Hnz; apply C_inv; assumption. Qed.

Lemma constructible_sqrt_nonneg :
  forall x : R, constructible x -> 0 <= x -> constructible (sqrt x).
Proof. intros; eapply C_sqrt; eauto. Qed.

Lemma constructible_Q :
  forall q : Q, constructible (Q2R q).
Proof.
  intro q.
  constructor.                (* directly uses rule C_Q *)
Qed.

Lemma constructible_div :
  forall x y : R,
    constructible x -> constructible y -> y <> 0 ->
    constructible (x / y).
Proof.
  intros x y Hx Hy Hnz.
  unfold Rdiv.                      (* x / y  ≡  x * / y *)
  apply constructible_mul.
  - exact Hx.
  - apply constructible_inv; assumption.
Qed.

From Coq Require Import Lra.      (* for the “lra” tactic *)

Lemma constructible_sqrt2 : constructible (sqrt 2).
Proof.
  (* Step 1: 2 is constructible because it’s rational. *)
  apply constructible_sqrt_nonneg.
  - replace 2 with (Q2R 2%Q) by (unfold Q2R; simpl; field).
    constructor.                    (* C_Q with q := 2%Q *)
  - lra.                            (* 0 ≤ 2 *)
Qed.

Lemma constructible_minus1 : constructible (-1).
Proof.
  apply constructible_opp.
  apply constructible_1.
Qed.

Lemma constructible_pow :
  forall x : R, forall n : nat,
    constructible x -> constructible (pow x n).
Proof.
  intros x n Hx.
  induction n.
  - simpl.                       (* pow x 0 = 1 *)
    apply constructible_1.
  - simpl.                       (* pow x (S n) = x * pow x n *)
    apply constructible_mul; assumption.
Qed.

Lemma constructible_half : constructible (1 / 2).
Proof.
  apply constructible_div.
  - apply constructible_1.                               (* numerator 1 *)
  - replace 2 with (Q2R 2%Q) by (unfold Q2R; simpl; field). (* denominator 2 *)
    constructor.                                          (* C_Q for 2%Q *)
  - lra.                                                  (* 2 ≠ 0 *)
Qed.

Lemma constructible_cubic_expr :
  forall x : R,
    constructible x ->
    constructible (4 * pow x 3 - 3 * x - (1 / 2)).
Proof.
  intros x Hx.

  (* pow x 3 is constructible *)
  assert (Hpow : constructible (pow x 3)).
  { apply constructible_pow; assumption. }

  (* constants 4 and 3 are constructible (they’re rationals) *)
  assert (H4 : constructible 4).
  { replace 4 with (Q2R 4%Q) by (unfold Q2R; simpl; field); constructor. }
  assert (H3 : constructible 3).
  { replace 3 with (Q2R 3%Q) by (unfold Q2R; simpl; field); constructor. }

  (* term₁ = 4·x³  and  term₂ = 3·x *)
  assert (Hterm1 : constructible (4 * pow x 3))
    by (apply constructible_mul; [exact H4 | exact Hpow]).
  assert (Hterm2 : constructible (3 * x))
    by (apply constructible_mul; [exact H3 | exact Hx]).

  (* term₁ − term₂ is constructible *)
  assert (Hsub1 : constructible (4 * pow x 3 - 3 * x))
    by (apply constructible_sub; [exact Hterm1 | exact Hterm2]).

  (* subtract ½ to finish the desired expression *)
  apply constructible_sub; [exact Hsub1 | apply constructible_half].
Qed.

Require Import List.
Import ListNotations.

(* Horner form:  poly_eval [a0; a1; …; an] x  =  a0 + x*(a1 + x*( … + x*an)). *)
Fixpoint poly_eval (coeffs : list Q) (x : R) : R :=
  match coeffs with
  | []        => 0
  | c :: rest => Q2R c + x * poly_eval rest x
  end.

Lemma poly_eval_constructible :
  forall (coeffs : list Q) (x : R),
    constructible x ->
    constructible (poly_eval coeffs x).
Proof.
  intros coeffs x Hx.
  induction coeffs as [| c rest IH].
  - simpl.                         (* []  →  0 *)
    apply constructible_0.
  - simpl.                         (* c :: rest *)
    (* first show c is constructible *)
    assert (Hc : constructible (Q2R c)) by (constructor).
    (* poly_eval rest x is constructible by IH *)
    assert (Hrest : constructible (poly_eval rest x)) by exact IH.
    (* x * poly_eval rest x is constructible *)
    assert (Hmul : constructible (x * poly_eval rest x))
      by (apply constructible_mul; assumption).
    (* sum with the leading coefficient *)
    apply constructible_add; assumption.
Qed.

Inductive pow2 : nat -> Prop :=
| pow2_1 : pow2 1
| pow2_double : forall n, pow2 n -> pow2 (2 * n).

Lemma pow2_2 : pow2 2.
Proof.
  (* 1. Prove the more general statement pow2 (2*1). *)
  assert (H : pow2 (2 * 1)).
  { apply pow2_double. apply pow2_1. }

  (* 2. Simplify 2*1 to 2 inside H, then finish. *)
  simpl in H.            (* 2 * 1  ⟹  2 *)
  exact H.
Qed.

Lemma pow2_4 : pow2 4.
Proof.
  (* First show pow2 (2*2) using pow2_double and pow2_2. *)
  assert (H : pow2 (2 * 2)).
  { apply pow2_double. apply pow2_2. }

  (* Simplify 2*2 to 4, yielding the desired result. *)
  simpl in H.            (* 2 * 2  ⟹  4 *)
  exact H.
Qed.

Definition in_Q (x : R) : Prop :=
  exists q : Q, x = Q2R q.

Lemma in_Q_constructible :
  forall x : R, in_Q x -> constructible x.
Proof.
  intros x [q Hx].              (* unfold “x is in ℚ” as x = Q2R q *)
  subst x.                      (* replace x by Q2R q in the goal *)
  constructor.                  (* C_Q *)
Qed.

From Coq Require Import Nat.

(* For a non-empty coefficient list [a0; …; an] we set
     poly_degree coeffs  =  n   (since length = n + 1).       
   For the empty list we let the degree be 0. *)
Definition poly_degree (coeffs : list Q) : nat :=
  Nat.pred (List.length coeffs).

Lemma poly_degree_const :
  forall a : Q, poly_degree [a] = 0%nat.
Proof.
  intros a.            (* length [a] = 1, so degree = pred 1 = 0 *)
  simpl.               (* goal becomes 0%nat = 0%nat *)
  reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(*  Algebraic-over-ℚ predicate with explicit degree                    *)
(* ------------------------------------------------------------------ *)
Require Import Coq.Lists.List.
Import ListNotations.

(**  [algebraic_deg x n] means:
     there is a non-empty list of rational coefficients [cs] such that

       •  the polynomial encoded by [cs] has degree exactly [n], and
       •  [x] is a root of that polynomial.                            *)

Definition algebraic_deg (x : R) (n : nat) : Prop :=
  exists cs : list Q,
       cs <> []                             (* non-empty list          *)
    /\ poly_degree cs = n                   (* degree = n              *)
    /\ poly_eval cs x = 0.                  (* x is a root             *)

Lemma Q2R_opp_plus_cancel :
  forall q : Q, Q2R (- q)%Q + Q2R q = 0.
Proof.
  intro q.
  rewrite Q2R_opp.   (* − (Q2R q) + Q2R q *)
  lra.               (* closes  a + (−a) = 0 *)
Qed.

Lemma poly_degree_two :
  forall a b : Q, poly_degree [a ; b] = 1%nat.
Proof.
  intros a b.
  unfold poly_degree; simpl.   (* length [a;b] = 2,  pred 2 = 1 *)
  reflexivity.
Qed.

Lemma Q2R_one : Q2R 1%Q = 1.
Proof.
  unfold Q2R; simpl; field.
Qed.

(* The polynomial  (-q) + 1·X  vanishes at  X = q. *)
Lemma poly_eval_linear_root :
  forall q : Q,
    poly_eval [(- q)%Q ; 1%Q] (Q2R q) = 0.
Proof.
  intro q.
  simpl.                                  (* unfold poly_eval           *)
  rewrite Rmult_0_r.                      (* ... + _ * 0 → ... + _ + 0  *)
  rewrite Rplus_0_r.                      (* drop the + 0               *)
  rewrite Q2R_one.                        (* replace Q2R 1%Q by 1       *)
  rewrite Rmult_1_r.                      (* _ * 1 → _                  *)
  apply Q2R_opp_plus_cancel.              (* −q + q = 0                 *)
Qed.

(* Every rational number is algebraic of degree 1. *)
Lemma algebraic_deg_rational :
  forall q : Q, algebraic_deg (Q2R q) 1.
Proof.
  intro q.
  set (cs := [(- q)%Q ; 1%Q]).            (* coefficients: constant, then X *)
  unfold algebraic_deg.
  exists cs; split.
  - discriminate.                         (* cs ≠ [] *)
  - split.
    + unfold cs, poly_degree; simpl; reflexivity.   (* degree = 1 *)
    + unfold cs. apply poly_eval_linear_root.       (* root equality *)
Qed.

Lemma pow2_mul :
  forall m n : nat, pow2 m -> pow2 n -> pow2 (m * n)%nat.
Proof.
  intros m n Hm Hn.
  induction Hm as [| k Hk IH].
  (* ---- base: m = 1 ---- *)
  - rewrite Nat.mul_1_l.
    exact Hn.

  (* ---- step: m = 2 * k ---- *)
  - (* rewrite  ((2*k)*n)  as  2*(k*n)  so we can use [pow2_double] *)
    replace (((2%nat * k)%nat) * n)%nat
      with (2%nat * (k * n))%nat
      by (rewrite Nat.mul_assoc; reflexivity).
    apply pow2_double.
    exact IH.
Qed.

(* --------------------------------------------------------------- *)
(*   “Algebraic over ℚ”  without fixing the degree up-front        *)
(* --------------------------------------------------------------- *)
Definition algebraic (x : R) : Prop :=
  exists n : nat, algebraic_deg x n.

(* Every rational number is algebraic (degree-1 case from earlier). *)
Lemma algebraic_rational :
  forall q : Q, algebraic (Q2R q).
Proof.
  intro q.
  exists 1%nat.
  apply algebraic_deg_rational.
Qed.

(* --------------------------------------------------------------- *)
(*  Flip the sign of every odd-indexed coefficient in a Q-list     *)
(* --------------------------------------------------------------- *)
Fixpoint alt_sign (coeffs : list Q) (odd : bool) : list Q :=
  match coeffs with
  | []        => []
  | c :: rest =>
      let c' := if odd then (- c)%Q else c in
      c' :: alt_sign rest (negb odd)
  end.

Lemma pow2_8 : pow2 8.
Proof.
  (* First build pow2 (2*4) with the doubling rule… *)
  assert (H : pow2 (2 * 4)).
  { apply pow2_double. apply pow2_4. }

  (* …then simplify 2*4 to 8 and finish. *)
  simpl in H.          (* 2 * 4 ⟹ 8 *)
  exact H.
Qed.

Lemma pow2_16 : pow2 16.
Proof.
  (* Build pow2 (2*8) via the doubling rule… *)
  assert (H : pow2 (2 * 8)).
  { apply pow2_double. apply pow2_8. }

  (* …then simplify 2*8 to 16. *)
  simpl in H.          (* 2 * 8  ⟹  16 *)
  exact H.
Qed.

From Coq Require Import Lia. 

Lemma pow2_not_3 :
  forall n : nat, pow2 n -> n <> 3%nat.
Proof.
  intros n Hpow.
  induction Hpow.
  - discriminate.              (* base: 1 ≠ 3 *)
  - intro Heq.                 (* goal: 2*k ≠ 3 *)
    lia.                       (* even ≠ odd ⇒ contradiction *)
Qed.

Lemma pow2_even_or_one :
  forall n : nat,
    pow2 n ->
    (n = 1%nat \/ exists k : nat, n = (2 * k)%nat).
Proof.
  intros n Hp.
  induction Hp as [| k Hk IH].
  - left; reflexivity.                       (* base: n = 1 *)
  - right.                                   (* step: n = 2 * k is even *)
    exists k; reflexivity.
Qed.

Lemma alt_sign_length :
  forall (cs : list Q) (b : bool),
    length (alt_sign cs b) = length cs.
Proof.
  intros cs b.
  revert b.                         (* keep b generic during induction *)
  induction cs as [| c rest IH]; intros b; simpl.
  - reflexivity.                    (* base: both are 0 *)
  - rewrite IH. reflexivity.        (* step: lengths agree after recurse *)
Qed.

Lemma poly_eval_alt_sign :
  forall (cs : list Q) (x : R),
    poly_eval (alt_sign cs false) (- x) =  poly_eval cs x /\
    poly_eval (alt_sign cs true)  (- x) = - poly_eval cs x.
Proof.
  induction cs as [| c rest IH]; intros x.
  - simpl; split; lra.                        (* base: empty list *)
  - simpl.                                    (* unfold poly_eval/alt_sign *)
    destruct (IH x) as [IHfalse IHtrue].
    split.
    + (* b = false branch *)
      simpl.                                  (* head coeff stays c *)
      rewrite IHtrue.                         (* tail equality *)
      lra.
    + (* b = true branch *)
      simpl.                                  (* head coeff becomes -c *)
      rewrite Q2R_opp.                        (* rewrite -c as Q2R (-c) *)
      rewrite IHfalse.                        (* tail equality *)
      lra.
Qed.

Lemma algebraic_opp :
  forall x : R, algebraic x -> algebraic (- x).
Proof.
  intros x [n [cs [Hnz [Hdeg Hroot]]]].

  (* Because [cs] is non-empty we can peel its head. *)
  destruct cs as [| c rest].
  { exfalso. apply Hnz. reflexivity. }

  (* Define the sign-flipped list (it is obviously non-empty). *)
  set (cs_neg := alt_sign (c :: rest) true).

  (* 1.  (-x) satisfies the new polynomial. *)
  assert (Hroot_neg : poly_eval cs_neg (- x) = 0).
  { unfold cs_neg.
    pose proof (poly_eval_alt_sign (c :: rest) x) as [_ Htail].
    rewrite Hroot in Htail. lra. }

  (* 2.  Degree is preserved (alt_sign keeps the length). *)
  assert (Hdeg_neg : poly_degree cs_neg = n).
  { unfold cs_neg, poly_degree.
    rewrite alt_sign_length. exact Hdeg. }

  (* 3.  cs_neg is non-empty. *)
  assert (Hnz_neg : cs_neg <> []).
  { unfold cs_neg. intro Hcontra. discriminate Hcontra. }

  (* Package the witnesses. *)
  exists n, cs_neg; repeat split; assumption.
Qed.


(* no global  Open Scope Q_scope. *)
Fixpoint poly_add (p q : list Q) : list Q :=
  match p, q with
  | [], _              => q
  | _,  []             => p
  | a :: p', b :: q'   => (a + b)%Q :: poly_add p' q'   (* note %Q *)
  end.

Lemma poly_eval_poly_add :
  forall (p q : list Q) (x : R),
    poly_eval (poly_add p q) x = poly_eval p x + poly_eval q x.
Proof.
  induction p as [| a p' IH]; intros q x; simpl.
  - (* p = [] *)
    destruct q; simpl; lra.
  - (* p = a :: p' *)
    destruct q as [| b q']; simpl.
    + lra.                                   (* q = [] *)
    + rewrite Q2R_plus.                      (* Q2R (a+b)%Q → Q2R a + Q2R b *)
      rewrite IH.                            (* induction hypothesis *)
      lra.
Qed.

Fixpoint poly_mul (p q : list Q) : list Q :=
  match p with
  | []      => []
  | a :: p' =>
      (* multiply each coefficient of q by a, then shift and recurse *)
      List.map (fun b => (a * b)%Q) q
      ++ (0%Q :: poly_mul p' q)
  end.

Fixpoint Rpow (x : R) (n : nat) : R :=
  match n with
  | O    => 1
  | S n' => x * Rpow x n'
  end.
