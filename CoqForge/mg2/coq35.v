Require Import Coq.Reals.Reals.
Require Import Coq.Sets.Ensembles.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Arith.Compare_dec.
Require Import Coq.Logic.FunctionalExtensionality.
Require Import Coq.Program.Equality.
Require Import Coq.Lists.List.
Import ListNotations.
Open Scope R_scope.

Record BaseSpace := {
  underlying_type : Type;
  base_scheme : Type;
  transfer : underlying_type -> underlying_type
}.

Record MotivicSpace := {
  base : BaseSpace;
  A1_space : BaseSpace
}.

Record VectorBundle := {
  rank : nat;
  chern_class : R;
  chern_nontrivial : chern_class <> 0
}.

Definition example_vector_bundle : VectorBundle :=
  {|
    rank := 1;
    chern_class := 1;
    chern_nontrivial := R1_neq_R0
  |}.

Lemma vector_bundle_exists : exists v : VectorBundle, chern_class v <> 0.
Proof.
  exists example_vector_bundle.
  simpl.
  exact R1_neq_R0.
Qed.

Lemma vector_bundle_general_exists : forall (c : R), c <> 0 -> 
  exists v : VectorBundle, chern_class v = c.
Proof.
  intros c Hc.
  exists {|
    rank := 1;
    chern_class := c;
    chern_nontrivial := Hc
  |}.
  simpl.
  reflexivity.
Qed.

Definition bundle_to_space (v : VectorBundle) : MotivicSpace := {|
  base := {|
    underlying_type := Type;
    base_scheme := Type;
    transfer := fun x => x
  |};
  A1_space := {|
    underlying_type := Type;
    base_scheme := Type;
    transfer := fun x => x
  |}
|}.

Definition default_space := {|
  base := {|
    underlying_type := Type;
    base_scheme := Type;
    transfer := fun x => x
  |};
  A1_space := {|
    underlying_type := Type;
    base_scheme := Type;
    transfer := fun x => x
  |}
|}.


Lemma mult_nonzero_helper : forall r1 r2 : R,
  r1 <> 0 -> r2 <> 0 -> r1 * r2 <> 0.
Proof.
  intros r1 r2 H1 H2.
  intros H.
  apply Rmult_integral in H.
  destruct H; contradiction.
Qed.

Record PolynomialTerm := {
  coefficient : R;
  variables : list MotivicSpace;
  term_degree : nat;
  term_bound : (length variables <= term_degree)%nat;
  zero_coefficient_zero_degree : 
    coefficient = 0%R -> term_degree = 0%nat
}.

Definition evaluate_term (t : PolynomialTerm) (M : MotivicSpace) : R :=
  coefficient t * INR (term_degree t).

Inductive is_polynomial (P : MotivicSpace -> R) (d : nat) : Prop :=
  | poly_constant : forall c : R,
    (forall M, P M = c) ->
    is_polynomial P d
  | poly_linear : forall (c : R) (f : MotivicSpace -> R),
    (forall M, P M = c * f M) ->
    is_polynomial P d
  | poly_sum : forall P1 P2,
    is_polynomial P1 d ->
    is_polynomial P2 d ->
    (forall M, P M = P1 M + P2 M) ->
    is_polynomial P d
  | poly_product : forall P1 P2 d1 d2,
    is_polynomial P1 d1 ->
    is_polynomial P2 d2 ->
    (d1 + d2 <= d)%nat ->
    (forall M, P M = P1 M * P2 M) ->
    is_polynomial P d.

Record ChernClass := {
  degree : nat;
  value : VectorBundle -> R;
  multiplicative : forall (v1 v2 : VectorBundle),
    value {| rank := rank v1 + rank v2;
             chern_class := chern_class v1 * chern_class v2;
             chern_nontrivial := mult_nonzero_helper _ _ (chern_nontrivial v1) (chern_nontrivial v2) |} = value v1 * value v2;
  rank_bound : forall (v : VectorBundle),
    (rank v < degree)%nat -> value v = 0;
  nontrivial : exists v : VectorBundle, value v <> 0;
  naturality : forall (v1 v2 : VectorBundle),
    rank v1 = rank v2 ->
    chern_class v1 = chern_class v2 ->
    value v1 = value v2;
  degree_monotonicity : forall (v : VectorBundle),
    (rank v > degree)%nat ->
    Rabs (value v) <= INR (rank v) * chern_class v;
  value_zero : value {| rank := 0; 
                       chern_class := 1; 
                       chern_nontrivial := R1_neq_R0 |} = 0
}.

Record MotivicExcisiveApprox := {
  m_degree : nat;
  m_weight : nat;
  m_value : MotivicSpace -> R;
  m_error : MotivicSpace -> R;
  m_nontrivial : exists M : MotivicSpace, m_value M <> 0;
  excisive_poly_bound : forall (M : MotivicSpace),
    exists (P : MotivicSpace -> R) (dim : nat),
    is_polynomial P m_degree /\
    (dim > 0)%nat /\
    Rabs (m_value M - P M) <= INR (1/m_weight) * INR(dim) * INR(m_degree);
  excisive_tower_compatibility : forall (M : MotivicSpace) (n : nat),
    (n > m_weight)%nat ->
    exists (P_n : MotivicSpace -> R) (dim : nat),
    is_polynomial P_n m_degree /\
    (dim > 0)%nat /\
    Rabs (m_value M - P_n M) <= 
    INR (1/n) * INR(dim) * INR(m_degree) - INR(1/m_weight) * INR(dim) * INR(m_degree);
  excisive_chern_compatibility : forall (v : VectorBundle) (c : ChernClass),
    (degree c <= m_degree)%nat ->
    exists (P : MotivicSpace -> R),
    is_polynomial P (degree c) /\
    P (bundle_to_space v) = value c v /\
    Rabs (m_value (bundle_to_space v) - P (bundle_to_space v)) <= 
    INR (1/m_weight) * INR(rank v) * INR(degree c);

  excisive_trans : forall X Y Z : MotivicSpace,
    m_value X = m_value Y -> m_value Y = m_value Z -> m_value X = m_value Z
}.

Record TowerLevel := {
  level_num : nat;
  filtration_degree : nat;
  bound_condition : (filtration_degree <= level_num)%nat
}.

Definition scaled_error_bound (n filtration_level: nat) (dim: nat) : R :=
  INR (1/n) * INR(dim) * INR(filtration_level).

Record GeometricFiltration := {
  filtration_level : nat;
  geometric_piece : MotivicSpace -> R;
  filtration_positive : (0 < filtration_level)%nat;
  polynomial_bound : forall (M : MotivicSpace) (n : nat),
    (n > filtration_level)%nat ->
    exists P : MotivicSpace -> R,
    exists dim : nat,
    is_polynomial P filtration_level /\
    (dim > 0)%nat /\
    Rabs (geometric_piece M - P M) <= scaled_error_bound n filtration_level dim;
  polynomial_tower_refinement : forall (M : MotivicSpace) (n m : nat),
    (n > m)%nat -> 
    (m > filtration_level)%nat ->
    exists (P_n P_m : MotivicSpace -> R) (dim : nat),
    is_polynomial P_n filtration_level /\
    is_polynomial P_m filtration_level /\
    (dim > 0)%nat /\
    Rabs (P_n M - P_m M) <= 
    scaled_error_bound n filtration_level dim - scaled_error_bound m filtration_level dim;
  geometric_meaning : forall (v : VectorBundle),
    geometric_piece (bundle_to_space v) = chern_class v * INR filtration_level;
  geometric_meaning_bounded : forall (v : VectorBundle) (n : nat),
    (n <= filtration_level)%nat ->
    Rabs (geometric_piece (bundle_to_space v)) <= 
    Rabs (chern_class v) * INR n;
  chern_polynomial_bound : forall (v : VectorBundle) (c : ChernClass),
    (degree c <= filtration_level)%nat ->
    exists (P : MotivicSpace -> R),
    is_polynomial P (degree c) /\
    P (bundle_to_space v) = value c v /\
    (forall M : MotivicSpace,
      Rabs (geometric_piece M - P M) <= 
      scaled_error_bound (degree c) filtration_level (rank v));
  bound_compatibility : forall (M : MotivicSpace) (n : nat) (v : VectorBundle) (c : ChernClass),
    (n > filtration_level)%nat ->
    (degree c <= filtration_level)%nat ->
    scaled_error_bound n filtration_level (rank v) <= 
    scaled_error_bound (degree c) filtration_level (rank v);
  geometric_A1 : forall (M M' : MotivicSpace),
    geometric_piece M = geometric_piece M'
}.

Record MotivicTower := {
  base_level : TowerLevel;
  tower_height : nat;
  tower_filtration : forall n : nat, (n < tower_height)%nat -> GeometricFiltration;
  filtration_compatible : forall (n m : nat) 
    (Hn: (n < tower_height)%nat) 
    (Hm: (m < tower_height)%nat),
    (n <= m)%nat -> 
    forall M : MotivicSpace,
    Rabs (geometric_piece (tower_filtration n Hn) M) <= 
    Rabs (geometric_piece (tower_filtration m Hm) M);
  level_map : forall (n : nat) (H: (n < tower_height)%nat),
    MotivicExcisiveApprox -> MotivicExcisiveApprox;
  level_map_degree : forall (n : nat) (H: (n < tower_height)%nat) 
    (approx : MotivicExcisiveApprox),
    (m_degree (level_map n H approx) <= m_degree approx)%nat;
  level_map_nontrivial : forall (n : nat) (H: (n < tower_height)%nat)
    (approx : MotivicExcisiveApprox),
    exists M : MotivicSpace, 
    m_value (level_map n H approx) M <> 0;
  level_map_A1 : forall (n : nat) (H: (n < tower_height)%nat)
    (approx : MotivicExcisiveApprox) (M1 M2 : MotivicSpace),
    m_value (level_map n H approx) M1 = m_value (level_map n H approx) M2;
  level_map_filtration : forall (n : nat) (H: (n < tower_height)%nat)
    (approx : MotivicExcisiveApprox) (M : MotivicSpace),
    m_value (level_map n H approx) M = 
    geometric_piece (tower_filtration n H) M;
  tower_excisive_convergence : forall (M : MotivicSpace) (n : nat) (H: (n < tower_height)%nat),
    exists (approx : MotivicExcisiveApprox),
    m_degree approx = filtration_level (tower_filtration n H) /\
    m_weight approx = n /\
    forall (P : MotivicSpace -> R) (dim : nat),
    is_polynomial P (m_degree approx) ->
    (dim > 0)%nat ->
    Rabs (geometric_piece (tower_filtration n H) M - m_value approx M) <=
    INR (1/n) * INR(dim) * INR(filtration_level (tower_filtration n H));
  tower_converges : forall (M : MotivicSpace) (ε : R),
    ε > 0 ->
    exists N : nat,
    (N < tower_height)%nat /\
    forall (n m : nat) (Hn: (n < tower_height)%nat) (Hm: (m < tower_height)%nat),
    (N <= n)%nat -> (N <= m)%nat ->
    Rabs (geometric_piece (tower_filtration n Hn) M - 
          geometric_piece (tower_filtration m Hm) M) < ε
}.

Lemma geometric_meaning_consistent : 
  forall (G : GeometricFiltration) (v : VectorBundle),
  Rabs (geometric_piece G (bundle_to_space v)) <= 
  Rabs (chern_class v) * INR (filtration_level G).
Proof.
  intros G v.
  rewrite (geometric_meaning G v).
  rewrite Rabs_mult.
  assert (HINR: 0 <= INR (filtration_level G)).
  { apply pos_INR. }
  assert (H_abs_INR: Rabs (INR (filtration_level G)) = INR (filtration_level G)).
  { apply Rabs_pos_eq. assumption. }
  rewrite H_abs_INR.
  apply Rmult_le_compat.
  - apply Rabs_pos.
  - apply pos_INR.
  - apply Rle_refl.
  - apply Rle_refl.
Qed.

Lemma geometric_meaning_monotone :
  forall (G : GeometricFiltration) (v : VectorBundle) (n m : nat),
  (n <= m)%nat ->
  (m <= filtration_level G)%nat ->
  Rabs (geometric_piece G (bundle_to_space v)) <= 
  Rabs (chern_class v) * INR m.
Proof.
  intros G v n m H_le H_bound.
  apply Rle_trans with (Rabs (chern_class v) * INR n).
  - apply geometric_meaning_bounded with (n := n).
    eapply Nat.le_trans; eassumption.
  - apply Rmult_le_compat_l.
    + apply Rabs_pos.
    + apply le_INR; assumption.
Qed.


Inductive K0 : Type :=
  | K0_zero : K0
  | K0_vb : VectorBundle -> K0
  | K0_sum : K0 -> K0 -> K0.

Record Automorphism := {
  dimension : nat;
  determinant : R;
  trace : R;
  auto_nontrivial : determinant <> 0
}.

Inductive K1 : Type :=
  | K1_zero : K1
  | K1_auto : Automorphism -> K1
  | K1_sum : K1 -> K1 -> K1
  | K1_inv : K1 -> K1.

Inductive K_compatible : K0 -> K1 -> Prop :=
  | K_comp_zero : K_compatible K0_zero K1_zero
  | K_comp_vb_auto : forall (v : VectorBundle) (a : Automorphism),
      rank v = dimension a ->
      chern_class v = determinant a ->
      K_compatible (K0_vb v) (K1_auto a)
  | K_comp_sum : forall k0_1 k0_2 k1_1 k1_2,
      K_compatible k0_1 k1_1 ->
      K_compatible k0_2 k1_2 ->
      K_compatible (K0_sum k0_1 k0_2) (K1_sum k1_1 k1_2).

Record MotivicKTheory := {
  space : MotivicSpace;
  k0_lift : K0 -> underlying_type (base space);
  k1_lift : K1 -> underlying_type (base space);
  k0_lift_A1 : forall k : K0,
    k0_lift k = k0_lift k;
  k0_lift_vb_eq : forall v1 v2,
    rank v1 = rank v2 ->
    chern_class v1 = chern_class v2 ->
    k0_lift (K0_vb v1) = k0_lift (K0_vb v2);
  k0_lift_sum : forall a b,
    k0_lift (K0_sum a b) = k0_lift a;
  k1_lift_auto_eq : forall a1 a2,
    dimension a1 = dimension a2 ->
    determinant a1 = determinant a2 ->
    trace a1 = trace a2 ->
    k1_lift (K1_auto a1) = k1_lift (K1_auto a2);
  k1_lift_sum : forall a b,
    k1_lift (K1_sum a b) = k1_lift a;
  k1_lift_inv : forall a,
    k1_lift (K1_inv a) = k1_lift a;
  lift_nontrivial : forall v : VectorBundle,
    exists x : underlying_type (base space),
    k0_lift (K0_vb v) = x
}.

Record MotivicKTheory_Extended := {
  base_theory :> MotivicKTheory;
  lift_compatible : forall k0 k1,
    K_compatible k0 k1 ->
    transfer (base (space base_theory)) (k0_lift base_theory k0) = 
    k1_lift base_theory k1
}.

Record MotivicApprox := {
  motivic_degree : nat;
  motivic_value : MotivicSpace -> R;
  A1_invariance : forall (M M' : MotivicSpace),
    motivic_value M = motivic_value M';
  polynomial_error : forall n : nat,
    (motivic_degree < n)%nat -> 
    motivic_value = motivic_value;
  approx_nontrivial : exists M : MotivicSpace, motivic_value M <> 0
}.


Record CrossEffect := {
  arity : nat;
  cross_value : list MotivicSpace -> R;
  degenerate : forall (spaces : list MotivicSpace),
    length spaces = arity ->
    (exists (i j : nat), (i <? j)%nat = true /\ (j <? length spaces)%nat = true /\ 
     List.nth_error spaces i = List.nth_error spaces j) ->
    cross_value spaces = 0
}.

Record MotivicTowerWithCrossEffects := {
  underlying_tower :> MotivicTower;
  cross_effects : nat -> CrossEffect;
  compatibility : forall (n : nat) (H: (n < tower_height underlying_tower)%nat),
    forall (spaces : list MotivicSpace),
    length spaces = n ->
    cross_value (cross_effects n) spaces = 
    geometric_piece (tower_filtration underlying_tower n H) 
      (List.nth 0 spaces default_space);
  cross_effect_arity : forall n : nat,
    arity (cross_effects n) = n
}.


Theorem cross_effect_basic : forall (tower : MotivicTowerWithCrossEffects) 
  (n : nat) (H: (n < tower_height (underlying_tower tower))%nat),
  forall (spaces : list MotivicSpace),
  length spaces = n ->
  cross_value (cross_effects tower n) spaces = 
  geometric_piece (tower_filtration (underlying_tower tower) n H) (List.nth 0 spaces default_space).
Proof.
  intros tower n H spaces Hlen.
  apply (compatibility tower n H spaces Hlen).
Qed.

Theorem tower_gives_motivic_approx : forall (kt : MotivicTower) 
  (n : nat) (H: (n < tower_height kt)%nat),
  exists (M : MotivicApprox),
    motivic_degree M = n /\
    forall (mk : MotivicKTheory_Extended),
    exists (v : VectorBundle),
    motivic_value M (space mk) = 
    geometric_piece (tower_filtration kt n H) (space mk).
Proof.
  intros kt n0 H.
  set (default_bundle := {| rank := 1;
                           chern_class := 1;
                           chern_nontrivial := (fun H : 1 = 0 => R1_neq_R0 H) |}).
  set (filt := tower_filtration kt n0 H).
  assert (Hnz: geometric_piece filt default_space <> 0).
  {
    rewrite (geometric_A1 filt default_space (bundle_to_space default_bundle)).
    rewrite geometric_meaning.
    apply mult_nonzero_helper.
    - exact (chern_nontrivial default_bundle).
    - apply Rgt_not_eq.
      apply lt_0_INR.
      exact (filtration_positive filt).
  }
  exists {| motivic_degree := n0;
            motivic_value := geometric_piece filt;
            A1_invariance := geometric_A1 filt;
            polynomial_error := fun m H' => eq_refl;
            approx_nontrivial := ex_intro _ default_space Hnz |}.
  split.
  - reflexivity.
  - intros mk.
    exists default_bundle.
    reflexivity.
Qed.

Theorem tower_approximations_converge : forall (kt : MotivicTower) 
  (M : MotivicSpace) (ε : R),
  ε > 0 ->
  exists (N : nat) (HN: (N < tower_height kt)%nat),
  forall (n m : nat) (Hn: (n < tower_height kt)%nat) (Hm: (m < tower_height kt)%nat),
  (N <= n)%nat -> (N <= m)%nat ->
  Rabs (geometric_piece (tower_filtration kt n Hn) M - 
        geometric_piece (tower_filtration kt m Hm) M) < ε.
Proof.
  intros kt M ε Hε.
  destruct (tower_converges kt M ε Hε) as [N [HN Hconv]].
  exists N, HN.
  intros n m Hn Hm Hn_N Hm_N.
  apply Hconv; assumption.
Qed.

Lemma nth_error_nth_relationship : 
  forall {A : Type} (l : list A) (n : nat) (d : A),
  (n < length l)%nat ->
  nth_error l n = Some (nth n l d).
Proof.
  intros A l. 
  induction l as [|h t IHl].
  - intros n d H. simpl in H. inversion H.
  - intros n d H. destruct n.
    + simpl. reflexivity.
    + simpl. apply IHl.
      simpl in H.
      (* S n < S (length t) implies n < length t *)
      apply Nat.succ_lt_mono in H.
      exact H.
Qed.

Theorem cross_effect_degenerate_concrete : 
  forall (tower : MotivicTowerWithCrossEffects) (n : nat) 
         (spaces : list MotivicSpace),
  (2 <= n)%nat ->
  length spaces = n ->
  List.nth 0 spaces default_space = List.nth 1 spaces default_space ->
  cross_value (cross_effects tower n) spaces = 0.
Proof.
  intros tower n spaces Hn Hlen Heq.
  rewrite <- (cross_effect_arity tower n) in Hlen.
  apply (degenerate (cross_effects tower n) spaces Hlen).
  exists 0%nat, 1%nat.
  split; [| split].
  - (* 0 < 1 *)
    simpl. reflexivity.
  - (* 1 < length spaces *)
    rewrite Hlen.
    apply Nat.ltb_lt.
    rewrite (cross_effect_arity tower n).
    apply (Nat.lt_le_trans _ 2%nat _ ).
    + apply (Nat.lt_succ_diag_r 1).
    + exact Hn.
  - (* Now we can use nth_error_nth_relationship *)
    assert (H0': (0 < length spaces)%nat).
    { rewrite Hlen. rewrite (cross_effect_arity tower n).
      apply (Nat.lt_le_trans _ 2); [apply Nat.lt_0_2 | exact Hn]. }
    assert (H1': (1 < length spaces)%nat).
    { rewrite Hlen. rewrite (cross_effect_arity tower n).
      apply (Nat.lt_le_trans _ 2); [apply Nat.lt_1_2 | exact Hn]. }
    rewrite (nth_error_nth_relationship spaces 0 default_space H0').
    rewrite (nth_error_nth_relationship spaces 1 default_space H1').
    rewrite Heq.
    reflexivity.
Qed.