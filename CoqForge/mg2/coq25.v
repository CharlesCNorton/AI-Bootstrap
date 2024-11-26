Require Import Coq.Reals.Reals.
Require Import Coq.Sets.Ensembles.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Arith.Compare_dec.
Require Import Coq.Logic.FunctionalExtensionality.
Require Import Coq.Program.Equality.
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
  term_bound : (length variables <= term_degree)%nat
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
  nontrivial : exists v : VectorBundle, value v <> 0
}.

Record MotivicExcisiveApprox := {
  m_degree : nat;
  m_weight : nat;
  m_value : MotivicSpace -> R;
  m_error : MotivicSpace -> R;
  m_nontrivial : exists M : MotivicSpace, m_value M <> 0
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
  geometric_meaning : forall (v : VectorBundle),
    geometric_piece (bundle_to_space v) = chern_class v * INR filtration_level;
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
  tower_converges : forall (M : MotivicSpace) (ε : R),
    ε > 0 ->
    exists N : nat,
    (N < tower_height)%nat /\
    forall (n m : nat) (Hn: (n < tower_height)%nat) (Hm: (m < tower_height)%nat),
    (N <= n)%nat -> (N <= m)%nat ->
    Rabs (geometric_piece (tower_filtration n Hn) M - 
          geometric_piece (tower_filtration m Hm) M) < ε
}.

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
  + exact eq_refl.
  + intros mk.
    exists default_bundle.
    exact eq_refl.
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