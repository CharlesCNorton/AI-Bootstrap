(* Import the standard Reals library *)
Require Import Reals.
Open Scope R_scope.

(* Replace nat_to_R with INR from the standard library *)
Definition nat_to_R := INR.

(* Define MotivicSpace with dimension and singularity information *)
Record MotivicSpace : Type := mkMotivicSpace {
  underlying_type : Type;
  dimension : nat;
  has_singularities : bool
}.

(* Define sing_complexity as a concrete function *)
Definition sing_complexity (X : MotivicSpace) : nat :=
  dimension X.

(* Prove sing_complexity_monotone *)
Lemma sing_complexity_monotone : forall X Y : MotivicSpace,
  (dimension X <= dimension Y)%nat -> (sing_complexity X <= sing_complexity Y)%nat.
Proof.
  intros X Y Hdim.
  unfold sing_complexity.
  exact Hdim.
Qed.

(* Define weight functions *)
Definition w_dim (X : MotivicSpace) : R :=
  / (1 + nat_to_R (dimension X)).

Definition w_sing (X : MotivicSpace) : R :=
  / (1 + nat_to_R (sing_complexity X)).

Definition w_stage (n : nat) : R :=
  / (1 + nat_to_R n).

Definition w_total (X : MotivicSpace) (n : nat) : R :=
  w_dim X * w_sing X * w_stage n.

(* Prove positivity of weight functions *)
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
    
    (* Step 1b: Show nat_to_R(dimension X) ≥ 0 *)
    assert (0 <= nat_to_R (dimension X)).
    {
      unfold nat_to_R.
      (* Use a very basic property: for any nat n, INR n ≥ 0 *)
      apply pos_INR.
    }
    
    (* Step 1c: If a > 0 and b ≥ 0, then a + b > 0 *)
    apply Rplus_lt_le_0_compat; assumption.
  }
  
  (* Step 2: If r > 0 then 1/r > 0 *)
  apply Rinv_0_lt_compat.
  exact H.
Qed.

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
    
    (* Step 1b: Show nat_to_R(sing_complexity X) ≥ 0 *)
    assert (0 <= nat_to_R (sing_complexity X)).
    {
      unfold nat_to_R.
      (* For any nat n, INR n ≥ 0 *)
      apply pos_INR.
    }
    
    (* Step 1c: If a > 0 and b ≥ 0, then a + b > 0 *)
    apply Rplus_lt_le_0_compat; assumption.
  }
  
  (* Step 2: If r > 0 then 1/r > 0 *)
  apply Rinv_0_lt_compat.
  exact H.
Qed.

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
    
    (* Step 1b: Show nat_to_R n ≥ 0 *)
    assert (0 <= nat_to_R n).
    {
      unfold nat_to_R.
      (* For any nat n, INR n ≥ 0 *)
      apply pos_INR.
    }
    
    (* Step 1c: If a > 0 and b ≥ 0, then a + b > 0 *)
    apply Rplus_lt_le_0_compat; assumption.
  }
  
  (* Step 2: If r > 0 then 1/r > 0 *)
  apply Rinv_0_lt_compat.
  exact H.
Qed.

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

Require Import Coq.Arith.PeanoNat.

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

Require Import Coq.Reals.RIneq.
Require Import Coq.micromega.Lra.

(* If you don't already have a lemma that INR n >= 0 for all n, define it: *)
Lemma Rle_0_INR : forall n : nat, 0 <= INR n.
Proof.
  induction n.
  - (* n = 0 *) simpl. lra.
  - (* n = S n' *) rewrite S_INR. lra.
Qed.

Record ObstructionClass : Type := mkObstruction {
  obstruction_value : R;  (* numeric measure of the obstruction *)
  obstruction_stage : nat (* stage of the obstruction *)
}.


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

(* For simplicity, define obstruction as a real number measuring "size" *)
Definition Obstruction := R.

(* Sequence of obstructions indexed by stage *)
Definition ObstructionSeq := nat -> Obstruction.

Lemma obstruction_bounding :
  forall (X : MotivicSpace) (O : ObstructionSeq) (n : nat),
    0 <= O n ->
    (forall k, 0 <= O k) ->
    (forall k, O (S k) <= O k * w_total X k) ->
    O (S n) <= O n * w_total X n.
Proof.
  intros X O n HOn_pos H_all_pos H_bound.
  apply H_bound.
Qed.

Definition obstruction_converges (O : ObstructionSeq) : Prop :=
  forall ε : R, ε > 0 -> exists N : nat, forall n : nat,
    (n >= N)%nat -> O n < ε.

Require Import Coq.Reals.Reals.

Lemma Rinv_le : forall a b : R,
  0 < a -> 0 < b -> a <= b -> / b <= / a.
Proof.
  intros a b Ha Hb Hab.
  apply Rle_Rinv; assumption.
Qed.

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

Record PolyApprox := {
  approx_space : MotivicSpace; (* the motivic space being approximated *)
  approx_stage : nat;          (* stage of the polynomial approximation *)
  approx_dim_bound : nat;      (* dimension bound at this approximation *)
  approx_sing_bound : nat      (* singularity bound at this approximation *)
}.

(* Given a functor F from MotivicSpace to Type, a polynomial approximation P 
   ensures the motivic space at stage approx_stage has dimensions and singularities
   bounded by approx_dim_bound and approx_sing_bound respectively. *)
Definition PolyApproximation (F : MotivicSpace -> Type) (P : PolyApprox) : Prop :=
  (dimension (approx_space P) <= approx_dim_bound P)%nat /\
  (sing_complexity (approx_space P) <= approx_sing_bound P)%nat.

(* An obstruction measures how far F(X) is from being approximated by the polynomial approximation at the given stage *)
Definition is_obstruction 
  (F : MotivicSpace -> Type) 
  (P Q : PolyApprox) (* Q is the next approximation after P *)
  (obs : Obstruction) : Prop :=
    obs = (* explicitly define how obs measures the "difference" between approximations P and Q *)
    R_dist (INR (approx_dim_bound P)) (INR (approx_dim_bound Q)) +
    R_dist (INR (approx_sing_bound P)) (INR (approx_sing_bound Q)).

(* Explicit obstruction sequence based on successive polynomial approximations *)
Definition obstruction_sequence 
  (F : MotivicSpace -> Type) 
  (approxs : nat -> PolyApprox) : ObstructionSeq :=
    fun n => R_dist (INR (approx_dim_bound (approxs n))) (INR (approx_dim_bound (approxs (S n)))) +
             R_dist (INR (approx_sing_bound (approxs n))) (INR (approx_sing_bound (approxs (S n)))).

Lemma obstruction_sequence_nonnegative :
  forall (F : MotivicSpace -> Type) (approxs : nat -> PolyApprox) (n : nat),
    0 <= obstruction_sequence F approxs n.
Proof.
  intros F approxs n.
  unfold obstruction_sequence.
  unfold R_dist.
  apply Rplus_le_le_0_compat; apply Rabs_pos.
Qed.

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

Require Import Coq.Reals.Reals.
Require Import Coq.ZArith.ZArith.
Require Import Coq.micromega.Lia.

Lemma INR_Ztonat :
  forall z : Z, (0 <= z)%Z -> INR (Z.to_nat z) = IZR z.
Proof.
  intros z Hz.
  rewrite INR_IZR_INZ.
  rewrite Z2Nat.id; auto.
Qed.

Lemma archimedean_nat_nonneg_case :
  forall r : R, (0 <= up r)%Z -> exists N : nat, r < INR N.
Proof.
  intros r Hz_nonneg.
  destruct (archimed r) as [H1 H2].
  exists (Z.to_nat (up r)).
  rewrite INR_Ztonat; auto.
Qed.

Lemma archimedean_nat :
  forall r : R, exists N : nat, r < INR N.
Proof.
  intros r.
  destruct (archimed r) as [H_up_gt_r _].
  destruct (Z_le_gt_dec 0 (up r)).
  - apply archimedean_nat_nonneg_case; auto.
  - exists 0%nat.
    simpl.
    apply Rlt_trans with (r2 := IZR (up r)); auto.
    apply IZR_lt. lia.
Qed.

Lemma w_stage_upper_bound : forall n : nat,
  w_stage n <= / INR (n + 1).
Proof.
  intros n.
  unfold w_stage, nat_to_R.
  rewrite plus_INR.
  simpl INR at 1.
  rewrite Rplus_comm.
  right; reflexivity.
Qed.

Lemma Rinv_lt_contravar_standard : forall x y : R,
  0 < x -> 0 < y -> y < x -> / x < / y.
Proof.
  intros x y Hx Hy H.
  apply Rinv_lt_contravar.
  - apply Rmult_lt_0_compat; assumption.
  - assumption.
Qed.

(* Explicit improvement condition for approximations *)
Definition improving_approxs (approxs : nat -> PolyApprox) : Prop :=
  forall n, 
    (approx_dim_bound (approxs (S n)) < approx_dim_bound (approxs n))%nat \/ 
    ((approx_dim_bound (approxs (S n)) = approx_dim_bound (approxs n))%nat /\
     (approx_sing_bound (approxs (S n)) < approx_sing_bound (approxs n))%nat).

(* Immediate consequence: dimension bounds are non-increasing *)
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

(* Define a single stage of the weighted tower *)
Record WeightedApprox := mkWeightedApprox {
  w_approx_poly : PolyApprox;     (* The polynomial approximation at this stage *)
  w_approx_threshold : R;         (* The weight threshold ω(n) *)
  w_approx_threshold_pos : 0 < w_approx_threshold  (* Ensure threshold is positive *)
}.

(* Define a weighted tower as a sequence of weighted approximations *)
Definition WeightedTower := nat -> WeightedApprox.

(* Define when a tower has properly decreasing weight thresholds *)
Definition proper_weighted_tower (tower : WeightedTower) : Prop :=
  forall n, w_approx_threshold (tower (S n)) < w_approx_threshold (tower n).

(* Define the fiber between consecutive stages of the weighted tower *)
Definition tower_fiber (tower : WeightedTower) (n : nat) : R :=
  match n with
  | O => 0  (* No fiber before the first stage *)
  | S m => 
      let curr := tower n in
      let prev := tower m in
      (* Measure the "difference" between stages using R_dist *)
      R_dist (INR (approx_dim_bound (w_approx_poly curr))) 
             (INR (approx_dim_bound (w_approx_poly prev))) +
      R_dist (INR (approx_sing_bound (w_approx_poly curr)))
             (INR (approx_sing_bound (w_approx_poly prev)))
  end.

(* Lemma: tower_fiber is always non-negative *)
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

(* Lemma: In a proper weighted tower, weight thresholds are positive *)
Lemma proper_tower_positive_weights :
  forall (tower : WeightedTower) (n : nat),
    proper_weighted_tower tower ->
    0 < w_approx_threshold (tower n).
Proof.
  intros tower n Hproper.
  apply (w_approx_threshold_pos (tower n)).
Qed.

(* Definition: A weighted tower is improving if its polynomial approximations are improving *)
Definition improving_weighted_tower (tower : WeightedTower) : Prop :=
  improving_approxs (fun n => w_approx_poly (tower n)).

(* Lemma: In an improving weighted tower, dimension bounds are non-increasing *)
Lemma improving_tower_dim_nonincreasing :
  forall (tower : WeightedTower),
    improving_weighted_tower tower ->
    forall n, (approx_dim_bound (w_approx_poly (tower (S n))) <= 
              approx_dim_bound (w_approx_poly (tower n)))%nat.
Proof.
  intros tower Himproving n.
  unfold improving_weighted_tower in Himproving.
  
  (* Define the function we're applying explicitly *)
  set (approxs := fun m => w_approx_poly (tower m)).
  
  (* Now apply the lemma with this explicit function *)
  assert (H: (approx_dim_bound (approxs (S n)) <= approx_dim_bound (approxs n))%nat).
  { apply improving_dim_bound_nonincreasing. exact Himproving. }
  
  (* Unfold the definition to complete the proof *)
  unfold approxs in H.
  exact H.
Qed.

(* Lemma: In an improving weighted tower with stable dimensions, 
   singularity bounds are decreasing *)
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
  
  (* Define the function we're applying explicitly *)
  set (approxs := fun m => w_approx_poly (tower m)).
  
  (* Apply improving_approxs to this function *)
  specialize (Himproving n).
  unfold approxs in *.
  
  (* By definition of improving_approxs, we have two cases *)
  destruct Himproving as [Hdim_lt | [Hdim_eq Hsing_lt]].
  
  (* Case 1: dimension strictly decreases *)
  - (* But this contradicts our assumption that dimensions are equal *)
    rewrite Hdim_stable in Hdim_lt.
    exfalso. apply (Nat.lt_irrefl _ Hdim_lt).
    
  (* Case 2: dimension stays the same and singularity decreases *)
  - exact Hsing_lt.
Qed.

(* Lemma: In a weighted tower, weight thresholds form a strictly decreasing sequence *)
Lemma tower_decreasing_weights :
  forall (tower : WeightedTower) (n m : nat),
    proper_weighted_tower tower ->
    (n < m)%nat ->
    w_approx_threshold (tower m) < w_approx_threshold (tower n).
Proof.
  intros tower n m Hproper Hnm.
  induction m.
  
  (* Base case: m = 0, which contradicts n < m *)
  - inversion Hnm.
  
  (* Inductive case: m = S m' *)
  - destruct (Nat.eq_dec n m) as [Heq | Hneq].
    
    (* Case: n = m *)
    + rewrite Heq.
      apply (Hproper m).
      
    (* Case: n < m *)
    + assert (Hlt: (n < m)%nat) by lia.
      assert (H: w_approx_threshold (tower m) < w_approx_threshold (tower n)) by (apply IHm; assumption).
      apply Rlt_trans with (w_approx_threshold (tower m)).
      * apply (Hproper m).
      * assumption.
Qed.
