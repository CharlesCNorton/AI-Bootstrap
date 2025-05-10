(* ================================================================= *)
(* FILE: WeightedMotivicTaylorTowerFormalization_Original.v          *)
(* DESCRIPTION: Formalization of components for the Weighted         *)
(* Motivic Taylor Tower Conjecture.                     *)
(* (Strictly reformatted for comments and layout only)  *)
(* ================================================================= *)

(* ================================================================= *)
(* SECTION 1: PREAMBLE - IMPORTS AND BASIC SETTINGS                  *)
(* ================================================================= *)

Require Import Reals.
Require Import Coq.Arith.PeanoNat. (* For S, O, nat equality etc. *)
Require Import Coq.Reals.RIneq.    (* For R inequality lemmas *)
Require Import Coq.micromega.Lra.  (* For lra tactic *)
Require Import Coq.ZArith.ZArith.  (* For Z, Coq's integers *)
Require Import Coq.micromega.Lia.  (* For lia tactic (linear integer/natural arithmetic) *)
(* Note: Some imports like Lia might be implicitly pulled by others like Lra in some setups,
   but explicit imports are generally good practice. *)

Open Scope R_scope.

(* ================================================================= *)
(* SECTION 2: CORE DEFINITIONS AND BASIC PROPERTIES                  *)
(* ================================================================= *)

(* ---------- Nat to Real Conversion ---------- *)

(* Definition for converting natural numbers to reals. *)
Definition nat_to_R := INR.

(* ---------- MotivicSpace Record ---------- *)

(* Record representing a MotivicSpace with its dimension and singularity status. *)
Record MotivicSpace : Type := mkMotivicSpace {
  underlying_type : Type;
  dimension : nat;
  has_singularities : bool
}.

(* ---------- Singularity Complexity ---------- *)

(* Definition of singularity complexity for a MotivicSpace.
   Currently simplified to be equal to its dimension. *)
Definition sing_complexity (X : MotivicSpace) : nat :=
  dimension X.

(* Lemma showing monotonicity of sing_complexity based on its definition. *)
Lemma sing_complexity_monotone : forall X Y : MotivicSpace,
  (dimension X <= dimension Y)%nat -> (sing_complexity X <= sing_complexity Y)%nat.
Proof.
  intros X Y Hdim.
  unfold sing_complexity.
  exact Hdim.
Qed.

(* ---------- Weight Function Definitions ---------- *)

(* Definition of the dimension-based weight function. *)
Definition w_dim (X : MotivicSpace) : R :=
  / (1 + nat_to_R (dimension X)).

(* Definition of the singularity-based weight function. *)
Definition w_sing (X : MotivicSpace) : R :=
  / (1 + nat_to_R (sing_complexity X)).

(* Definition of the stage-based weight function. *)
Definition w_stage (n : nat) : R :=
  / (1 + nat_to_R n).

(* Definition of the total weight function. *)
Definition w_total (X : MotivicSpace) (n : nat) : R :=
  w_dim X * w_sing X * w_stage n.

(* ================================================================= *)
(* SECTION 3: PROPERTIES OF WEIGHT FUNCTIONS                         *)
(* ================================================================= *)

(* ---------- Positivity of Weight Functions ---------- *)

(* Lemma proving that w_dim is always positive. *)
Lemma w_dim_positive : forall X : MotivicSpace, 0 < w_dim X.
Proof.
  intros X.
  unfold w_dim.
  (* Prove 1/(1+nat_to_R(dimension X)) > 0 *)
  (* Step 1: Show 1 + nat_to_R(dimension X) > 0 *)
  assert (0 < 1 + nat_to_R (dimension X)).
  {
    (* Step 1a: Show 1 > 0 *)
    assert (0 < 1) by apply Rlt_0_1.
    (* Step 1b: Show nat_to_R(dimension X) >= 0 *)
    assert (0 <= nat_to_R (dimension X)).
    {
      unfold nat_to_R.
      (* Use a very basic property: for any nat n, INR n >= 0 *)
      apply pos_INR.
    }
    (* Step 1c: If a > 0 and b >= 0, then a + b > 0 *)
    apply Rplus_lt_le_0_compat; assumption.
  }
  (* Step 2: If r > 0 then 1/r > 0 *)
  apply Rinv_0_lt_compat.
  exact H.
Qed.

(* Lemma proving that w_sing is always positive. *)
Lemma w_sing_positive : forall X : MotivicSpace, 0 < w_sing X.
Proof.
  intros X.
  unfold w_sing.
  (* Prove 1/(1+nat_to_R(sing_complexity X)) > 0 *)
  (* Step 1: Show 1 + nat_to_R(sing_complexity X) > 0 *)
  assert (0 < 1 + nat_to_R (sing_complexity X)).
  {
    (* Step 1a: Show 1 > 0 *)
    assert (0 < 1) by apply Rlt_0_1.
    (* Step 1b: Show nat_to_R(sing_complexity X) >= 0 *)
    assert (0 <= nat_to_R (sing_complexity X)).
    {
      unfold nat_to_R.
      (* For any nat n, INR n >= 0 *)
      apply pos_INR.
    }
    (* Step 1c: If a > 0 and b >= 0, then a + b > 0 *)
    apply Rplus_lt_le_0_compat; assumption.
  }
  (* Step 2: If r > 0 then 1/r > 0 *)
  apply Rinv_0_lt_compat.
  exact H.
Qed.

(* Lemma proving that w_stage is always positive. *)
Lemma w_stage_positive : forall n : nat, 0 < w_stage n.
Proof.
  intros n.
  unfold w_stage.
  (* Prove 1/(1+nat_to_R n) > 0 *)
  (* Step 1: Show 1 + nat_to_R n > 0 *)
  assert (0 < 1 + nat_to_R n).
  {
    (* Step 1a: Show 1 > 0 *)
    assert (0 < 1) by apply Rlt_0_1.
    (* Step 1b: Show nat_to_R n >= 0 *)
    assert (0 <= nat_to_R n).
    {
      unfold nat_to_R.
      (* For any nat n, INR n >= 0 *)
      apply pos_INR.
    }
    (* Step 1c: If a > 0 and b >= 0, then a + b > 0 *)
    apply Rplus_lt_le_0_compat; assumption.
  }
  (* Step 2: If r > 0 then 1/r > 0 *)
  apply Rinv_0_lt_compat.
  exact H.
Qed.

(* Lemma proving that w_total is always positive. *)
Lemma w_total_positive : forall X : MotivicSpace, forall n : nat,
  0 < w_total X n.
Proof.
  intros X n.
  unfold w_total.
  (* Prove w_dim X * w_sing X * w_stage n > 0 *)
  (* If a > 0 and b > 0, then a * b > 0 *)
  apply Rmult_lt_0_compat.
  (* First product: w_dim X * w_sing X > 0 *)
  - apply Rmult_lt_0_compat.
    + apply w_dim_positive.
    + apply w_sing_positive.
  (* Second factor: w_stage n > 0 *)
  - apply w_stage_positive.
Qed.

(* ---------- Helper Lemmas for Nat and Real Properties ---------- *)

(* Lemma: If n <= m, then n < m or n = m. *)
Lemma le_lt_or_eq : forall n m : nat, (n <= m)%nat -> (n < m \/ n = m)%nat.
Proof.
  intros n m H.
  induction H.
  (* Base case: n <= n (reflexivity) *)
  - right. reflexivity.
  (* Inductive case: n <= m -> n <= S m *)
  - (* If n < m or n = m, then n < S m *)
    left. apply le_n_S. assumption.
Qed.

(* Lemma: For any nat n, INR n >= 0. (This is also pos_INR) *)
Lemma Rle_0_INR : forall n : nat, 0 <= INR n.
Proof.
  induction n.
  - (* n = 0 *) simpl. lra.
  - (* n = S n' *) rewrite S_INR. lra.
Qed.

(* ---------- Antitonicity of Weight Functions ---------- *)

(* Lemma proving that w_dim is antitone with respect to dimension. *)
Lemma w_dim_antitone : forall X Y : MotivicSpace,
  (dimension X <= dimension Y)%nat -> w_dim Y <= w_dim X.
Proof.
  intros X Y Hdim.
  unfold w_dim, nat_to_R.
  apply Rinv_le_contravar.
  - (* Positivity of denominators *)
    assert (0 <= INR (dimension X)) by apply pos_INR.
    lra.
  - (* Monotonicity of denominators *)
    apply Rplus_le_compat_l.
    apply le_INR; assumption.
Qed.

(* Lemma proving that w_sing is antitone. *)
Lemma w_sing_antitone : forall X Y : MotivicSpace,
  (dimension X <= dimension Y)%nat -> w_sing Y <= w_sing X.
Proof.
  intros X Y Hdim.
  unfold w_sing, sing_complexity, nat_to_R.
  apply Rinv_le_contravar.
  - (* Positivity of denominators *)
    assert (0 <= INR (dimension X)) by apply pos_INR.
    lra.
  - (* Monotonicity of denominators *)
    apply Rplus_le_compat_l.
    apply le_INR; assumption.
Qed.

(* Lemma proving that w_stage is antitone. *)
Lemma w_stage_antitone : forall n m : nat,
  (n <= m)%nat -> w_stage m <= w_stage n.
Proof.
  intros n m Hnm.
  unfold w_stage, nat_to_R.
  apply Rinv_le_contravar.
  - (* Positivity of denominators *)
    assert (0 <= INR n) by apply pos_INR.
    lra.
  - (* Monotonicity of denominators *)
    apply Rplus_le_compat_l.
    apply le_INR; assumption.
Qed.

(* Lemma for combined antitonicity of w_dim * w_sing. *)
Lemma w_dim_sing_antitone : forall X Y : MotivicSpace,
  (dimension X <= dimension Y)%nat ->
  w_dim Y * w_sing Y <= w_dim X * w_sing X.
Proof.
  intros X Y Hdim.
  assert (HdimX_pos : 0 <= w_dim X) by (left; apply w_dim_positive).
  assert (HsingY_pos : 0 <= w_sing Y) by (left; apply w_sing_positive).
  assert (HdimY_pos : 0 <= w_dim Y) by (left; apply w_dim_positive).

  (* Step 1: Show w_dim Y <= w_dim X *)
  assert (Hdim_antitone : w_dim Y <= w_dim X) by (apply w_dim_antitone; assumption).

  (* Step 2: Show w_sing Y <= w_sing X *)
  assert (Hsing_antitone : w_sing Y <= w_sing X) by (apply w_sing_antitone; assumption).

  (* Now clearly handle multiplication *)
  apply Rle_trans with (w_dim X * w_sing Y).
  - (* Use w_dim monotonicity, multiply by positive w_sing Y *)
    apply Rmult_le_compat_r; assumption.
  - (* Now use w_sing monotonicity, multiply by positive w_dim X *)
    apply Rmult_le_compat_l; assumption.
Qed.

(* Lemma for antitonicity of w_total. *)
Lemma w_total_antitone : forall X Y : MotivicSpace, forall n m : nat,
  (dimension X <= dimension Y)%nat ->
  (n <= m)%nat ->
  w_total Y m <= w_total X n.
Proof.
  intros X Y n m Hdim Hnm.
  unfold w_total.

  (* positivity assertions clearly named *)
  assert (HdimX_pos : 0 <= w_dim X) by (left; apply w_dim_positive).
  assert (HsingX_pos : 0 <= w_sing X) by (left; apply w_sing_positive).
  assert (HstageN_pos : 0 <= w_stage n) by (left; apply w_stage_positive).
  assert (HstageM_pos : 0 <= w_stage m) by (left; apply w_stage_positive).

  (* Use the helper lemma we just proved *)
  assert (Hdim_sing_antitone : w_dim Y * w_sing Y <= w_dim X * w_sing X).
  { apply w_dim_sing_antitone; assumption. }

  (* Now multiply clearly by stage weights *)
  apply Rle_trans with ((w_dim X * w_sing X) * w_stage m).
  - (* Multiply right side by w_stage m *)
    apply Rmult_le_compat_r; assumption.
  - (* Multiply left side by w_stage n *)
    apply Rmult_le_compat_l.
    + apply Rmult_le_pos; assumption.
    + apply w_stage_antitone; assumption.
Qed.

(* ================================================================= *)
(* SECTION 4: ABSTRACT OBSTRUCTION AND CONVERGENCE FRAMEWORK         *)
(* ================================================================= *)

(* ---------- Obstruction Types ---------- *)

(* Record for an ObstructionClass, containing its value and stage. *)
Record ObstructionClass : Type := mkObstruction {
  obstruction_value : R;  (* numeric measure of the obstruction *)
  obstruction_stage : nat (* stage of the obstruction *)
}.

(* Type alias for Obstruction as a real number. *)
Definition Obstruction := R.

(* Definition of an ObstructionSeq as a sequence of real numbers indexed by nat. *)
Definition ObstructionSeq := nat -> Obstruction.

(* ---------- Obstruction Bounding Lemma (with Premise) ---------- *)

(* Lemma stating a one-step bound for an obstruction sequence,
   given a general premise H_bound about how obstructions are bounded. *)
Lemma obstruction_bounding :
  forall (X : MotivicSpace) (O : ObstructionSeq) (n : nat),
    0 <= O n ->
    (forall k, 0 <= O k) ->
    (forall k, O (S k) <= O k * w_total X k) -> (* The core premise *)
    O (S n) <= O n * w_total X n.
Proof.
  intros X O n HOn_pos H_all_pos H_bound.
  apply H_bound.
Qed.

(* ---------- Convergence Definition ---------- *)

(* Definition of convergence for an ObstructionSeq. *)
Definition obstruction_converges (O : ObstructionSeq) : Prop :=
  forall ε : R, ε > 0 -> exists N : nat, forall n : nat,
    (n >= N)%nat -> O n < ε.

(* ---------- Helper Lemma for Real Inverses ---------- *)

(* Lemma regarding inequality of real inverses. *)
Lemma Rinv_le : forall a b : R,
  0 < a -> 0 < b -> a <= b -> / b <= / a.
Proof.
  intros a b Ha Hb Hab.
  apply Rle_Rinv; assumption.
Qed.

(* ================================================================= *)
(* SECTION 5: POLYNOMIAL APPROXIMATION STRUCTURES                    *)
(* ================================================================= *)

(* ---------- PolyApprox Record and Property ---------- *)

(* Record to structure polynomial approximation data. *)
Record PolyApprox := {
  approx_space : MotivicSpace; (* the motivic space being approximated *)
  approx_stage : nat;          (* stage of the polynomial approximation *)
  approx_dim_bound : nat;      (* dimension bound at this approximation *)
  approx_sing_bound : nat      (* singularity bound at this approximation *)
}.

(* Proposition defining when a PolyApprox respects its bounds. *)
Definition PolyApproximation (F : MotivicSpace -> Type) (P : PolyApprox) : Prop :=
  (dimension (approx_space P) <= approx_dim_bound P)%nat /\
  (sing_complexity (approx_space P) <= approx_sing_bound P)%nat.

(* ---------- Model for Obstructions from PolyApprox ---------- *)

(* Definition of what constitutes an obstruction based on differences in PolyApprox bounds. *)
Definition is_obstruction
  (F : MotivicSpace -> Type)
  (P Q : PolyApprox) (* Q is the next approximation after P *)
  (obs : Obstruction) : Prop :=
    obs = (* explicitly define how obs measures the "difference" between approximations P and Q *)
    R_dist (INR (approx_dim_bound P)) (INR (approx_dim_bound Q)) +
    R_dist (INR (approx_sing_bound P)) (INR (approx_sing_bound Q)).

(* Definition of an obstruction sequence based on a sequence of PolyApprox. *)
Definition obstruction_sequence
  (F : MotivicSpace -> Type)
  (approxs : nat -> PolyApprox) : ObstructionSeq :=
    fun n => R_dist (INR (approx_dim_bound (approxs n))) (INR (approx_dim_bound (approxs (S n)))) +
              R_dist (INR (approx_sing_bound (approxs n))) (INR (approx_sing_bound (approxs (S n)))).

(* Lemma: The modeled obstruction sequence is non-negative. *)
Lemma obstruction_sequence_nonnegative :
  forall (F : MotivicSpace -> Type) (approxs : nat -> PolyApprox) (n : nat),
    0 <= obstruction_sequence F approxs n.
Proof.
  intros F approxs n.
  unfold obstruction_sequence.
  unfold R_dist.
  apply Rplus_le_le_0_compat; apply Rabs_pos.
Qed.

(* Helper lemma for Rabs (a-b). *)
Lemma Rabs_diff_le_sum :
  forall a b : R, 0 <= a -> 0 <= b -> Rabs (a - b) <= a + b.
Proof.
  intros a b Ha Hb.
  unfold Rabs.
  destruct (Rcase_abs (a - b)).
  - (* case a - b < 0 *)
    apply Ropp_le_cancel.
    rewrite Ropp_minus_distr.
    lra.
  - (* case a - b >= 0 *)
    lra.
Qed.

(* Lemma providing an upper bound for the modeled obstruction sequence. *)
Lemma obstruction_sequence_bound :
  forall (F : MotivicSpace -> Type) (approxs : nat -> PolyApprox) (n : nat),
    obstruction_sequence F approxs n <=
      (INR (approx_dim_bound (approxs n)) + INR (approx_dim_bound (approxs (S n)))) +
      (INR (approx_sing_bound (approxs n)) + INR (approx_sing_bound (approxs (S n)))).
Proof.
  intros F approxs n.
  unfold obstruction_sequence.
  apply Rplus_le_compat.
  - apply Rabs_diff_le_sum; apply pos_INR.
  - apply Rabs_diff_le_sum; apply pos_INR.
Qed.

(* ---------- Archimedean Properties and INR/Z Conversions ---------- *)

(* Lemma relating INR to IZR for non-negative Z. *)
Lemma INR_Ztonat :
  forall z : Z, (0 <= z)%Z -> INR (Z.to_nat z) = IZR z.
Proof.
  intros z Hz.
  rewrite INR_IZR_INZ.
  rewrite Z2Nat.id; auto.
Qed.

(* Lemma for Archimedean property: case where up r is non-negative. *)
Lemma archimedean_nat_nonneg_case :
  forall r : R, (0 <= up r)%Z -> exists N : nat, r < INR N.
Proof.
  intros r Hz_nonneg.
  destruct (archimed r) as [H1 H2]. (* H2 is r < IZR (up r) *)
  exists (Z.to_nat (up r)).
  rewrite INR_Ztonat; auto.
Qed.

(* General Archimedean property for nat. *)
Lemma archimedean_nat :
  forall r : R, exists N : nat, r < INR N.
Proof.
  intros r.
  destruct (archimed r) as [H_up_gt_r _]. (* This is IZR (up r) - 1 <= r < IZR (up r) *)
                                        (* Let's use the second part directly, H2. *)
  destruct (archimed r) as [_ H_r_lt_IZR_up_r].
  destruct (Z_le_gt_dec 0 (up r)).
  - (* Case up r >= 0 *)
    apply archimedean_nat_nonneg_case; auto.
  - (* Case up r < 0. Then up r <= -1. IZR (up r) <= -1. So r < -1.
       We need N such that r < INR N. N=0 gives INR N = 0. Since r < -1 < 0, N=0 works. *)
    exists 0%nat.
    simpl. (* INR 0 is 0 *)
    apply Rlt_trans with (r2 := IZR (up r)); auto.
    apply IZR_lt. lia.
Qed.

(* ---------- More Properties of w_stage ---------- *)

(* Lemma for an upper bound on w_stage (it's actually equality). *)
Lemma w_stage_upper_bound : forall n : nat,
  w_stage n <= / INR (n + 1).
Proof.
  intros n.
  unfold w_stage, nat_to_R.
  rewrite plus_INR.
  simpl INR at 1. (* Simplifies INR 1 to 1 for R if not already done by plus_INR *)
  rewrite Rplus_comm.
  right; reflexivity. (* Rle_refl *)
Qed.

(* Helper for Rinv_lt_contravar, useful for strict inequalities. *)
Lemma Rinv_lt_contravar_standard : forall x y : R,
  0 < x -> 0 < y -> y < x -> / x < / y.
Proof.
  intros x y Hx Hy H.
  apply Rinv_lt_contravar.
  - apply Rmult_lt_0_compat; assumption. (* x*y > 0 *)
  - assumption. (* y < x *)
Qed.

(* ---------- Improving Approximation Sequences ---------- *)

(* Definition for a sequence of PolyApprox being "improving". *)
Definition improving_approxs (approxs : nat -> PolyApprox) : Prop :=
  forall n,
    (approx_dim_bound (approxs (S n)) < approx_dim_bound (approxs n))%nat \/
    ((approx_dim_bound (approxs (S n)) = approx_dim_bound (approxs n))%nat /\
     (approx_sing_bound (approxs (S n)) < approx_sing_bound (approxs n))%nat).

(* Lemma: Dimension bounds are non-increasing in an improving sequence. *)
Lemma improving_dim_bound_nonincreasing :
  forall approxs,
    improving_approxs approxs ->
    forall n, (approx_dim_bound (approxs (S n)) <= approx_dim_bound (approxs n))%nat.
Proof.
  intros approxs H n.
  specialize (H n).
  destruct H as [Hlt | [Heq Hsing]].
  - apply Nat.lt_le_incl; assumption.
  - rewrite Heq. lia.
Qed.

(* Lemma relating singularity and dimension bounds in an improving sequence. *)
Lemma improving_sing_bound_nonincreasing :
  forall approxs,
    improving_approxs approxs ->
    forall n, (approx_sing_bound (approxs (S n)) <= approx_sing_bound (approxs n))%nat \/
              (approx_dim_bound (approxs (S n)) < approx_dim_bound (approxs n))%nat.
Proof.
  intros approxs H n.
  specialize (H n).
  destruct H as [Hdim_lt | [Hdim_eq Hsing_lt]].
  - right; assumption.
  - left; apply Nat.lt_le_incl; assumption.
Qed.

(* ================================================================= *)
(* SECTION 6: WEIGHTED TOWER STRUCTURES AND PROPERTIES               *)
(* ================================================================= *)

(* ---------- WeightedApprox Record and WeightedTower Definition ---------- *)

(* Record for a weighted approximation stage. *)
Record WeightedApprox := mkWeightedApprox {
  w_approx_poly : PolyApprox;      (* The polynomial approximation at this stage *)
  w_approx_threshold : R;          (* The weight threshold ω(n) *)
  w_approx_threshold_pos : 0 < w_approx_threshold (* Ensure threshold is positive *)
}.

(* Definition of a WeightedTower as a sequence of WeightedApprox. *)
Definition WeightedTower := nat -> WeightedApprox.

(* ---------- Properties of Weighted Towers ---------- *)

(* Definition for a WeightedTower having properly decreasing weight thresholds. *)
Definition proper_weighted_tower (tower : WeightedTower) : Prop :=
  forall n, w_approx_threshold (tower (S n)) < w_approx_threshold (tower n).

(* Lemma: Weight thresholds in a proper tower are positive. *)
Lemma proper_tower_positive_weights :
  forall (tower : WeightedTower) (n : nat),
    proper_weighted_tower tower ->
    0 < w_approx_threshold (tower n).
Proof.
  intros tower n Hproper.
  apply (w_approx_threshold_pos (tower n)).
Qed.

(* Definition for a WeightedTower being "improving" based on its PolyApprox sequence. *)
Definition improving_weighted_tower (tower : WeightedTower) : Prop :=
  improving_approxs (fun n => w_approx_poly (tower n)).

(* Lemma: Dimension bounds are non-increasing in an improving weighted tower. *)
Lemma improving_tower_dim_nonincreasing :
  forall (tower : WeightedTower),
    improving_weighted_tower tower ->
    forall n, (approx_dim_bound (w_approx_poly (tower (S n))) <=
               approx_dim_bound (w_approx_poly (tower n)))%nat.
Proof.
  intros tower Himproving n.
  unfold improving_weighted_tower in Himproving.
  set (approxs := fun m => w_approx_poly (tower m)).
  assert (H: (approx_dim_bound (approxs (S n)) <= approx_dim_bound (approxs n))%nat).
  { apply improving_dim_bound_nonincreasing. exact Himproving. }
  unfold approxs in H.
  exact H.
Qed.

(* Lemma: If dimension is stable in an improving tower, singularity bound decreases. *)
Lemma improving_tower_sing_decreasing_when_dim_stable :
  forall (tower : WeightedTower),
    improving_weighted_tower tower ->
    forall n,
      (approx_dim_bound (w_approx_poly (tower (S n))) =
       approx_dim_bound (w_approx_poly (tower n)))%nat ->
      (approx_sing_bound (w_approx_poly (tower (S n))) <
       approx_sing_bound (w_approx_poly (tower n)))%nat.
Proof.
  intros tower Himproving n Hdim_stable.
  unfold improving_weighted_tower in Himproving.
  set (approxs := fun m => w_approx_poly (tower m)).
  specialize (Himproving n).
  unfold approxs in *.
  destruct Himproving as [Hdim_lt | [Hdim_eq Hsing_lt]].
  - exfalso. rewrite Hdim_stable in Hdim_lt.
    apply (Nat.lt_irrefl _ Hdim_lt).
  - exact Hsing_lt.
Qed.

(* Lemma: Weight thresholds are strictly decreasing across multiple steps in a proper tower. *)
Lemma tower_decreasing_weights :
  forall (tower : WeightedTower) (n m : nat),
    proper_weighted_tower tower ->
    (n < m)%nat ->
    w_approx_threshold (tower m) < w_approx_threshold (tower n).
Proof.
  intros tower n m Hproper Hnm.
  induction m.
  - inversion Hnm.
  - destruct (Nat.eq_dec n m) as [Heq | Hneq].
    + rewrite Heq.
      apply (Hproper m).
    + assert (Hlt: (n < m)%nat) by lia.
      assert (H: w_approx_threshold (tower m) < w_approx_threshold (tower n)) by (apply IHm; assumption).
      apply Rlt_trans with (w_approx_threshold (tower m)).
      * apply (Hproper m).
      * assumption.
Qed.

(* ---------- Model for Tower Fibers ---------- *)

(* Definition of the "fiber" for a tower, based on differences in PolyApprox bounds. *)
Definition tower_fiber (tower : WeightedTower) (n : nat) : R :=
  match n with
  | O => 0
  | S m =>
      let curr := tower n in
      let prev := tower m in
      R_dist (INR (approx_dim_bound (w_approx_poly curr)))
             (INR (approx_dim_bound (w_approx_poly prev))) +
      R_dist (INR (approx_sing_bound (w_approx_poly curr)))
             (INR (approx_sing_bound (w_approx_poly prev)))
  end.

(* Lemma: The modeled tower fiber is non-negative. *)
Lemma tower_fiber_nonnegative :
  forall (tower : WeightedTower) (n : nat),
    0 <= tower_fiber tower n.
Proof.
  intros tower n.
  unfold tower_fiber.
  destruct n.
  - (* Case n = 0 *)
    right; reflexivity.
  - (* Case n = S m *)
    apply Rplus_le_le_0_compat.
    + apply Rabs_pos. (* R_dist is based on Rabs which is non-negative *)
    + apply Rabs_pos.
Qed.

(* ================================================================= *)
(* SECTION 7: OMEGA FUNCTION (STAGE WEIGHT)                          *)
(* ================================================================= *)

(* ---------- Definition and Basic Properties of omega ---------- *)

(* Definition of omega(n) as 1 / INR(S n). *)
Definition omega (n : nat) : R := / INR (S n).

(* Lemma: omega(n) is positive. *)
Lemma omega_pos : forall n, 0 < omega n.
Proof.
  intros n. unfold omega.
  apply Rinv_0_lt_compat.
  apply lt_0_INR. lia.
Qed.

(* Lemma: omega(n) is strictly decreasing. *)
Lemma omega_decreasing : forall n : nat, omega (S n) < omega n.
Proof.
  intro n. unfold omega.
  apply Rinv_lt_contravar.
  - apply Rmult_lt_0_compat; apply lt_0_INR; lia.
  - rewrite !S_INR. lra.
Qed.

(* ---------- Relation between w_stage and omega ---------- *)

(* Lemma: w_stage(n) is equal to omega(n). *)
Lemma w_stage_eq_omega : forall n, w_stage n = omega n.
Proof.
  intro n. unfold w_stage, omega, nat_to_R.
  rewrite S_INR.
  rewrite Rplus_comm.
  reflexivity.
Qed.

(* Lemma: w_stage(n) is strictly decreasing. *)
Lemma w_stage_decreasing : forall n : nat, w_stage (S n) < w_stage n.
Proof.
  intro n.
  rewrite !w_stage_eq_omega.
  apply omega_decreasing.
Qed.

(* Lemma: w_total is non-increasing in stage index for a fixed X. *)
Lemma w_total_stage_nonincreasing :
  forall (X : MotivicSpace) n,
    w_total X (S n) <= w_total X n.
Proof.
  intros X n.
  unfold w_total.
  set (c := w_dim X * w_sing X).
  assert (Hc_pos : 0 <= c).
  { unfold c; apply Rmult_le_pos; left; [apply w_dim_positive|apply w_sing_positive]. }
  rewrite w_stage_eq_omega with (n:=S n).
  rewrite w_stage_eq_omega with (n:=n).
  apply Rmult_le_compat_l; [exact Hc_pos | apply Rlt_le, omega_decreasing].
Qed.

(* ================================================================= *)
(* SECTION 8: CUMULATIVE WEIGHT PRODUCT AND OBSTRUCTION BOUNDING     *)
(* ================================================================= *)

(* ---------- Cumulative Product of w_total ---------- *)

(* Fixpoint definition for the cumulative product of w_total terms. *)
Fixpoint w_total_prod (X : MotivicSpace) (n : nat) : R :=
  match n with
  | 0   => 1
  | S k => w_total_prod X k * w_total X k
  end.

(* Lemma: The cumulative product w_total_prod is positive. *)
Lemma w_total_prod_pos :
  forall (X : MotivicSpace) n, 0 < w_total_prod X n.
Proof.
  intros X n.
  induction n.
  - simpl. lra.
  - simpl.
    apply Rmult_lt_0_compat; [exact IHn|].
    unfold w_total.
    apply Rmult_lt_0_compat.
    + apply Rmult_lt_0_compat; [apply w_dim_positive|apply w_sing_positive].
    + apply w_stage_positive.
Qed.

(* Lemma for unfolding w_total_prod at S n. *)
Lemma w_total_prod_S :
  forall (X : MotivicSpace) n,
    w_total_prod X (S n) = w_total_prod X n * w_total X n.
Proof.
  intros X n; simpl; reflexivity.
Qed.

(* ---------- Recursive Bounding Lemmas ---------- *)

(* Lemma for one step of the obstruction bound induction. *)
Lemma obstruction_bound_step
      (X : MotivicSpace) (O : ObstructionSeq) (k : nat) :
  (forall j, O (S j) <= O j * w_total X j) ->      (* Stepwise bound premise *)
  O k <= O 0%nat * w_total_prod X k ->             (* Inductive hypothesis *)
  O (S k) <= O 0%nat * w_total_prod X (S k).       (* Goal for S k *)
Proof.
  intros Hstep IH.
  rewrite w_total_prod_S.
  eapply Rle_trans.
  - apply Hstep.
  - assert (Hw : 0 <= w_total X k) by
      (left; apply w_total_positive).
    apply Rmult_le_compat_r with (r := w_total X k) in IH; [| exact Hw ].
    replace (O 0%nat * w_total_prod X k * w_total X k)
      with (O 0%nat * (w_total_prod X k * w_total X k))
      in IH by (rewrite Rmult_assoc; reflexivity).
    exact IH.
Qed.

(* Lemma for the cumulative recursive bound on obstructions. *)
Lemma obstruction_recursive_bound
      (X : MotivicSpace) (O : ObstructionSeq) :
  (forall j, O (S j) <= O j * w_total X j) ->  (* Stepwise bound premise *)
  forall n, O n <= O 0%nat * w_total_prod X n.
Proof.
  intros Hstep.
  fix IH 1.
  intro n.
  destruct n.
  - simpl. rewrite Rmult_1_r. right; reflexivity.
  - apply obstruction_bound_step with (k:=n) in IH; auto.
Qed.

(* ================================================================= *)
(* SECTION 9: MISCELLANEOUS UTILITIES (Factorial, w_stage_INR)       *)
(* ================================================================= *)

(* Lemma clarifying w_stage in terms of INR (S n). *)
Lemma w_stage_INR : forall n : nat,
  w_stage n = / INR (S n).
Proof.
  intro n.
  unfold w_stage, nat_to_R.
  rewrite S_INR.
  rewrite Rplus_comm.
  reflexivity.
Qed.

(* ---------- Factorial Function and its Positivity ---------- *)

(* Fixpoint definition of factorial for natural numbers. *)
Fixpoint fact (n : nat) : nat :=
  match n with
  | 0   => 1
  | S k => (S k) * fact k
  end.

(* Lemma: fact n is positive (as a natural number). *)
Lemma fact_nat_pos : forall n, (0 < fact n)%nat.
Proof.
  induction n; simpl; lia.
Qed.

(* Lemma: INR (fact n) is positive (as a real number). *)
Lemma fact_pos : forall n, 0 < INR (fact n).
Proof.
  intro n.
  apply lt_0_INR, fact_nat_pos.
Qed.

(* ================================================================= *)
(* END OF SCRIPT                                                     *)
(* ================================================================= *)
