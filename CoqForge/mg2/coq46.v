Require Import Coq.Reals.Reals.
Require Import Coq.Sets.Ensembles.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Program.Equality.
Require Import Coq.Logic.ProofIrrelevance.
Require Import Coq.Lists.List.
Import ListNotations.
Open Scope R_scope.

Record BaseSpace (U : Type) := {
  base_scheme : Type;
  transfer : U -> U;
  optional_base_point : option U;
  base_point_transfer : match optional_base_point with
                       | Some p => transfer p = p
                       | None => True
                       end
}.


Record HomotopyStructure (U : Type) := {
  homotopy_degree : nat;
  suspension : U -> U;
  loop_space : U -> U;
  suspension_loop : forall x, loop_space (suspension x) = x
}.

Record MotivicSpace := {
  universe_type : Type;
  base : BaseSpace universe_type;
  A1_space : BaseSpace universe_type;
  base_point_compat : match @optional_base_point universe_type base, 
                           @optional_base_point universe_type A1_space with
                     | Some p1, Some p2 => p1 = p2
                     | _, _ => True 
                     end;
  homotopy : HomotopyStructure universe_type;
  stability_level : nat;
  stable : forall n, (n >= stability_level)%nat -> 
           forall x : universe_type, (@suspension universe_type homotopy) x = x
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
  universe_type := Type;
  base := {|
    base_scheme := Type;
    transfer := fun x => x;
    optional_base_point := Some Type;
    base_point_transfer := eq_refl
  |};
  A1_space := {|
    base_scheme := Type;
    transfer := fun x => x;
    optional_base_point := Some Type;
    base_point_transfer := eq_refl
  |};
  base_point_compat := eq_refl;
  homotopy := {|
    homotopy_degree := 0;
    suspension := fun x => x;
    loop_space := fun x => x;
    suspension_loop := fun x => eq_refl
  |};
  stability_level := 0;
  stable := fun n _ x => eq_refl
|}.


Definition default_space := {|
  universe_type := Type;
  base := {|
    base_scheme := Type;
    transfer := fun x => x;
    optional_base_point := None;
    base_point_transfer := I
  |};
  A1_space := {|
    base_scheme := Type;
    transfer := fun x => x;
    optional_base_point := None;
    base_point_transfer := I
  |};
  base_point_compat := I;
  homotopy := {|
    homotopy_degree := 0;
    suspension := fun x => x;
    loop_space := fun x => x;
    suspension_loop := fun x => eq_refl
  |};
  stability_level := 0;
  stable := fun n _ x => eq_refl
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

Record MotivicFunctor := {
  functor_value : MotivicSpace -> R;  
  functor_linearity : forall (M : MotivicSpace) (c : R),
    functor_value M = c * functor_value M;
  functor_naturality : forall (M N : MotivicSpace),
    M = N -> functor_value M = functor_value N;
  functor_A1_invariant : forall (M M' : MotivicSpace),
    functor_value M = functor_value M'
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




Record DerivativeFunctor (F : MotivicFunctor) := {
  derivative_order : nat;
  derivative_value : MotivicSpace -> R;
  derivative_compatibility_with_functor : 
    forall (M : MotivicSpace),
    derivative_value M = functor_value F M;
  cross_effect_compat : forall (ce : CrossEffect),
    arity ce = derivative_order ->
    forall (M : MotivicSpace),
    derivative_value M = cross_value ce [M];
  derivative_naturality : forall (M N : MotivicSpace),
    M = N -> derivative_value M = derivative_value N;
  derivative_linearity : forall (M : MotivicSpace) (c : R),
    derivative_value M = c * derivative_value M;
  derivative_tower_bound : forall (M : MotivicSpace),
    exists (P : MotivicSpace -> R),
    is_polynomial P derivative_order /\
    Rabs (derivative_value M - P M) <= INR (derivative_order);
  homotopy_compatible : forall (M : MotivicSpace),
    derivative_value M = derivative_value {|
      universe_type := universe_type M;
      base := base M;
      A1_space := A1_space M;
      base_point_compat := base_point_compat M;
      homotopy := homotopy M;
      stability_level := stability_level M;
      stable := stable M
    |}
}.


Record MotivicExcisiveApprox (F : MotivicFunctor) := {
  m_degree : nat;
  m_weight : nat;
  m_value : MotivicSpace -> R;
  m_derivative : DerivativeFunctor F;
  

  approximation_respects_derivative : 
    forall (M : MotivicSpace),
    @derivative_value F m_derivative M = m_value M;

  m_error : MotivicSpace -> R;
  m_nontrivial : exists M : MotivicSpace, m_value M <> 0;


  excisive_poly_bound : 
    forall (M : MotivicSpace),
    exists (P : MotivicSpace -> R) (dim : nat),
    is_polynomial P m_degree /\
    (dim > 0)%nat /\
    Rabs (m_value M - P M) <= INR (1/m_weight) * INR(dim) * INR(m_degree);

  excisive_tower_compatibility :
    forall (M : MotivicSpace) (n : nat),
    (n > m_weight)%nat ->
    exists (P_n : MotivicSpace -> R) (dim : nat),
    is_polynomial P_n m_degree /\
    (dim > 0)%nat /\
    Rabs (@derivative_value F m_derivative M - P_n M) <= 
      INR (1/n) * INR(dim) * INR(m_degree) - INR(1/m_weight) * INR(dim) * INR(m_degree);


  excisive_cross_effect_compat : 
    forall (ce : CrossEffect),
    arity ce = m_degree ->
    forall (M : MotivicSpace),
    Rabs (m_value M - cross_value ce [M]) <= INR (1/m_weight)
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
    geometric_piece M = geometric_piece M';

  geometric_basepoint_vanish : forall (M : MotivicSpace),
    match @optional_base_point (universe_type M) (base M) with
    | Some _ => geometric_piece M = 0
    | None => True
    end
}.

Section MotivicTowerSection.
Context (F : MotivicFunctor).

Record MotivicTower (F : MotivicFunctor) := {
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
    MotivicExcisiveApprox F -> MotivicExcisiveApprox F;
  level_map_degree : forall (n : nat) (H: (n < tower_height)%nat) 
    (approx : MotivicExcisiveApprox F),
    (@m_degree F (level_map n H approx) <= @m_degree F approx)%nat;
  level_map_nontrivial : forall (n : nat) (H: (n < tower_height)%nat)
    (approx : MotivicExcisiveApprox F),
    exists M : MotivicSpace, 
    @m_value F (level_map n H approx) M <> 0;
  level_map_A1 : forall (n : nat) (H: (n < tower_height)%nat)
    (approx : MotivicExcisiveApprox F) (M1 M2 : MotivicSpace),
    @m_value F (level_map n H approx) M1 = @m_value F (level_map n H approx) M2;
  level_map_filtration : forall (n : nat) (H: (n < tower_height)%nat)
    (approx : MotivicExcisiveApprox F) (M : MotivicSpace),
    @m_value F (level_map n H approx) M = 
    geometric_piece (tower_filtration n H) M;
  tower_excisive_convergence : forall (M : MotivicSpace) (n : nat) (H: (n < tower_height)%nat),
    exists (approx : MotivicExcisiveApprox F),
    @m_degree F approx = filtration_level (tower_filtration n H) /\
    @m_weight F approx = n /\
    forall (P : MotivicSpace -> R) (dim : nat),
    is_polynomial P (@m_degree F approx) ->
    (dim > 0)%nat ->
    Rabs (geometric_piece (tower_filtration n H) M - @m_value F approx M) <=
    INR (1/n) * INR(dim) * INR(filtration_level (tower_filtration n H));
  tower_converges : forall (M : MotivicSpace) (ε : R),
    ε > 0 ->
    exists N : nat,
    (N < tower_height)%nat /\
    forall (n m : nat) (Hn: (n < tower_height)%nat) (Hm: (m < tower_height)%nat),
    (N <= n)%nat -> (N <= m)%nat ->
    Rabs (geometric_piece (tower_filtration n Hn) M - 
          geometric_piece (tower_filtration m Hm) M) < ε;
  tower_suspension : forall n : nat,
    forall (Hn: (n < tower_height)%nat),
    forall (M : MotivicSpace),
    geometric_piece (tower_filtration n Hn) M =
    geometric_piece (tower_filtration n Hn) 
      {| universe_type := universe_type M;
         base := base M;
         A1_space := A1_space M;
         base_point_compat := base_point_compat M;
         homotopy := homotopy M;
         stability_level := stability_level M;
         stable := stable M |}
}.

End MotivicTowerSection.

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
  k0_lift : K0 -> universe_type space;
  k1_lift : K1 -> universe_type space;
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
    exists x : universe_type space,
    k0_lift (K0_vb v) = x
}.

Record MotivicKTheory_Extended := {
  base_theory :> MotivicKTheory;
  lift_compatible : forall k0 k1,
    K_compatible k0 k1 ->
    @transfer (universe_type (space base_theory)) 
      (base (space base_theory))
      (k0_lift base_theory k0) = 
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


Arguments derivative_order {F}.
Arguments derivative_value {F}.
Arguments cross_effect_compat {F}.
Arguments derivative_naturality {F}.
Arguments derivative_linearity {F}.
Arguments derivative_tower_bound {F}.

Lemma derivative_cross_effect_bound : 
  forall (F : MotivicFunctor) (D : DerivativeFunctor F) (ce : CrossEffect),
  arity ce = derivative_order D ->
  forall (M : MotivicSpace),
  Rabs (derivative_value D M - cross_value ce [M]) = 0.
Proof.
  intros F D ce Harity M.
  rewrite (cross_effect_compat D ce Harity M).
  rewrite Rminus_eq_0.
  apply Rabs_R0.
Qed.


Section MotivicTowerWithCrossEffectsSection.
Context (F : MotivicFunctor).

Record MotivicTowerWithCrossEffects := {
  underlying_tower :> MotivicTower F;
  cross_effects : nat -> CrossEffect;
  compatibility : forall (n : nat) (H: (n < @tower_height F underlying_tower)%nat),
    forall (spaces : list MotivicSpace),
    length spaces = n ->
    cross_value (cross_effects n) spaces = 
    geometric_piece (@tower_filtration F underlying_tower n H) 
      (List.nth 0 spaces default_space);
  cross_effect_arity : forall n : nat,
    arity (cross_effects n) = n
}.

End MotivicTowerWithCrossEffectsSection.

Theorem cross_effect_basic : 
  forall (F : MotivicFunctor) (tower : MotivicTowerWithCrossEffects F)
  (n : nat) (H: (n < @tower_height F (@underlying_tower F tower))%nat),
  forall (spaces : list MotivicSpace),
  length spaces = n ->
  @cross_value (@cross_effects F tower n) spaces = 
  geometric_piece (@tower_filtration F (@underlying_tower F tower) n H) 
    (List.nth 0 spaces default_space).
Proof.
  intros F tower n H spaces Hlen.
  apply (@compatibility F tower n H spaces Hlen).
Qed.

Theorem tower_gives_motivic_approx : 
  forall (F : MotivicFunctor) (kt : MotivicTower F)
  (n : nat) (H: (n < @tower_height F kt)%nat),
  exists (M : MotivicApprox),
    motivic_degree M = n /\
    forall (mk : MotivicKTheory_Extended),
    exists (v : VectorBundle),
    motivic_value M (space mk) = 
    geometric_piece (@tower_filtration F kt n H) (space mk).
Proof.
  intros F kt n0 H.
  set (default_bundle := {| rank := 1;
                           chern_class := 1;
                           chern_nontrivial := (fun H : 1 = 0 => R1_neq_R0 H) |}).
  set (filt := @tower_filtration F kt n0 H).
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

Theorem tower_approximations_converge : 
  forall (F : MotivicFunctor) (kt : MotivicTower F) 
  (M : MotivicSpace) (ε : R),
  ε > 0 ->
  exists (N : nat) (HN: (N < @tower_height F kt)%nat),
  forall (n m : nat) 
         (Hn: (n < @tower_height F kt)%nat) 
         (Hm: (m < @tower_height F kt)%nat),
  (N <= n)%nat -> (N <= m)%nat ->
  Rabs (geometric_piece (@tower_filtration F kt n Hn) M - 
        geometric_piece (@tower_filtration F kt m Hm) M) < ε.
Proof.
  intros F kt M ε Hε.
  destruct (@tower_converges F kt M ε Hε) as [N [HN Hconv]].
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
      apply Nat.succ_lt_mono in H.
      exact H.
Qed.

Theorem cross_effect_degenerate_concrete : 
  forall (F : MotivicFunctor) (tower : MotivicTowerWithCrossEffects F) (n : nat) 
         (H: (n < @tower_height F (@underlying_tower F tower))%nat)
         (spaces : list MotivicSpace),
  (2 <= n)%nat ->
  length spaces = n ->
  (exists i j, 
    (i < j)%nat /\ 
    (j < n)%nat /\ 
    List.nth i spaces default_space = List.nth j spaces default_space /\
    forall k, (k < n)%nat -> 
      Rabs (geometric_piece (@tower_filtration F (@underlying_tower F tower) n H) 
             (List.nth k spaces default_space)) <= 
      Rabs (geometric_piece (@tower_filtration F (@underlying_tower F tower) n H) 
             (List.nth i spaces default_space))) ->
  cross_value (@cross_effects F tower n) spaces = 0.
Proof.
  intros F tower n H spaces Hn Hlen [i [j [Hij [Hjn [Heq Hbound]]]]].
  rewrite <- (@cross_effect_arity F tower n) in Hlen.
  apply (degenerate (@cross_effects F tower n) spaces Hlen).
  exists i, j.
  split; [| split].
  - (* i < j *)
    apply Nat.ltb_lt. exact Hij.
  - (* j < length spaces *)
    rewrite Hlen.
    apply Nat.ltb_lt.
    assert (Heq_arity: n = arity (@cross_effects F tower n)).
    { symmetry. apply (@cross_effect_arity F tower n). }
    rewrite Heq_arity in Hjn.
    exact Hjn.
  - (* nth_error equality *)
    assert (Hi: (i < n)%nat).
    { apply (Nat.lt_trans _ j); assumption. }
    rewrite (nth_error_nth_relationship spaces i default_space).
    + rewrite (nth_error_nth_relationship spaces j default_space).
      * f_equal. exact Heq.
      * rewrite Hlen. 
        assert (Heq_arity: n = arity (@cross_effects F tower n)).
        { symmetry. apply (@cross_effect_arity F tower n). }
        rewrite Heq_arity in Hjn.
        exact Hjn.
    + rewrite Hlen.
      assert (Heq_arity: n = arity (@cross_effects F tower n)).
      { symmetry. apply (@cross_effect_arity F tower n). }
      rewrite Heq_arity in Hi.
      exact Hi.
Qed.

Theorem cross_effect_degenerate_general : 
  forall (F : MotivicFunctor) (tower : MotivicTowerWithCrossEffects F) (n i j : nat) 
         (spaces : list MotivicSpace),
  (i < j)%nat ->
  (j < n)%nat ->
  length spaces = n ->
  List.nth i spaces default_space = List.nth j spaces default_space ->
  cross_value (@cross_effects F tower n) spaces = 0.
Proof.
  intros F tower n i j spaces Hij Hjn Hlen Heq.
  rewrite <- (@cross_effect_arity F tower n) in Hlen.
  apply (degenerate (@cross_effects F tower n) spaces Hlen).
  exists i, j.
  split; [| split].
  - apply Nat.ltb_lt. exact Hij.
  - rewrite Hlen.
    apply Nat.ltb_lt. 
    rewrite (@cross_effect_arity F tower n).
    exact Hjn.
  - assert (Hi: (i < n)%nat).
    { apply (Nat.lt_trans i j n); assumption. }
    rewrite (nth_error_nth_relationship spaces i default_space).
    + rewrite (nth_error_nth_relationship spaces j default_space).
      * rewrite Heq. reflexivity.
      * rewrite Hlen. rewrite (@cross_effect_arity F tower n).
        exact Hjn.
    + rewrite Hlen. rewrite (@cross_effect_arity F tower n).
      exact Hi.
Qed.

Theorem concrete_from_general :
  forall (F : MotivicFunctor) (tower : MotivicTowerWithCrossEffects F) (n : nat) 
         (spaces : list MotivicSpace),
  (2 <= n)%nat ->
  length spaces = n ->
  List.nth 0 spaces default_space = List.nth 1 spaces default_space ->
  cross_value (@cross_effects F tower n) spaces = 0.
Proof.
  intros F tower n spaces Hn Hlen Heq.
  apply (@cross_effect_degenerate_general F tower n 0 1).
  - (* 0 < 1 *)
    apply Nat.lt_0_succ.
  - (* 1 < n *)
    assert (H: (1 < 2)%nat) by apply Nat.lt_succ_diag_r.
    apply (Nat.lt_le_trans _ 2); assumption.
  - exact Hlen.
  - exact Heq.
Qed.

Theorem cross_effect_degenerate_subset :
  forall (F : MotivicFunctor) (tower : MotivicTowerWithCrossEffects F) (n : nat) 
         (spaces : list MotivicSpace) (indices : list nat),
  (forall i, In i indices -> (i < n)%nat) ->
  (length indices >= 2)%nat ->
  (forall i j, In i indices -> In j indices -> i < j \/ i = j \/ j < i)%nat ->
  (exists i j, In i indices /\ In j indices /\ (i < j)%nat) ->
  length spaces = n ->
  (exists x, forall i, In i indices ->
    List.nth i spaces default_space = x) ->
  cross_value (@cross_effects F tower n) spaces = 0.
Proof.
  intros F tower n spaces indices Hvalid Hlen Hord [i [j [Hini [Hinj Hij]]]] Hslen [x Heq].
  assert (Hi: (i < n)%nat) by (apply Hvalid; exact Hini).
  assert (Hj: (j < n)%nat) by (apply Hvalid; exact Hinj).
  assert (Hequal: List.nth i spaces default_space = List.nth j spaces default_space).
  { rewrite (Heq i); [|exact Hini].
    rewrite (Heq j); [|exact Hinj].
    reflexivity. }
  rewrite <- (@cross_effect_arity F tower n) in Hslen.
  rewrite <- (@cross_effect_arity F tower n) in Hi.
  rewrite <- (@cross_effect_arity F tower n) in Hj.
  apply (degenerate (@cross_effects F tower n) spaces Hslen).
  exists i, j.
  split; [|split].
  - apply Nat.ltb_lt. exact Hij.
  - rewrite Hslen. apply Nat.ltb_lt. exact Hj.
  - assert (Hi_len: (i < length spaces)%nat).
    { rewrite Hslen. exact Hi. }
    assert (Hj_len: (j < length spaces)%nat).
    { rewrite Hslen. exact Hj. }
    rewrite (nth_error_nth_relationship spaces i default_space Hi_len).
    rewrite (nth_error_nth_relationship spaces j default_space Hj_len).
    rewrite Hequal.
    reflexivity.
Qed.

Lemma geometric_piece_basepoint_vanish :
  forall (F : MotivicFunctor) (filt : GeometricFiltration) (M : MotivicSpace),
  match @optional_base_point (universe_type M) (base M) with
  | Some p => True
  | None => False
  end ->
  geometric_piece filt M = 0.
Proof.
  intros F filt M Hbase.
  destruct (@optional_base_point (universe_type M) (base M)) eqn:Heq.
  - assert (H := geometric_basepoint_vanish filt M).
    rewrite Heq in H.
    exact H.
  - contradiction.
Qed.

Definition unit_approx : MotivicApprox := {|
  motivic_degree := 0;
  motivic_value := fun _ => 1;  
  A1_invariance := fun M M' => eq_refl;
  polynomial_error := fun n _ => eq_refl;
  approx_nontrivial := ex_intro _ default_space R1_neq_R0  
|}.

Lemma R2_neq_R0 : 2 <> 0.
Proof.
  unfold not; intros H.
  assert (H1: (IZR 2) = 0) by exact H.
  assert (H2: (IZR 2) = (IZR 0)) by exact H1.
  apply eq_IZR in H2.
  discriminate H2.
Qed.


Definition linear_approx : MotivicApprox := {|
  motivic_degree := 1;
  motivic_value := fun _ => 1;  
  A1_invariance := fun M M' => eq_refl;
  polynomial_error := fun n _ => eq_refl;
  approx_nontrivial := ex_intro _ default_space R1_neq_R0
|}.

Definition quadratic_approx : MotivicApprox := {|
  motivic_degree := 2;
  motivic_value := fun _ => 2;  
  A1_invariance := fun M M' => eq_refl;
  polynomial_error := fun n _ => eq_refl;
  approx_nontrivial := ex_intro _ default_space R2_neq_R0
|}.


Lemma R3_neq_R0 : 3 <> 0.
Proof.
  unfold not; intros H.
  assert (H1: (IZR 3) = 0) by exact H.
  assert (H2: (IZR 3) = (IZR 0)) by exact H1.
  apply eq_IZR in H2.
  discriminate H2.
Qed.

Definition combined_approx : MotivicApprox := {|
  motivic_degree := 2;  
  motivic_value := fun _ => 3;  
  A1_invariance := fun M M' => eq_refl;
  polynomial_error := fun n _ => eq_refl;
  approx_nontrivial := ex_intro _ default_space R3_neq_R0
|}.


Lemma R4_neq_R0 : 4 <> 0.
Proof.
  unfold not; intros H.
  assert (H1: (IZR 4) = 0) by exact H.
  assert (H2: (IZR 4) = (IZR 0)) by exact H1.
  apply eq_IZR in H2.
  discriminate H2.
Qed.

Definition bundle_sensitive_approx : MotivicApprox := {|
  motivic_degree := 1;
  motivic_value := fun _ => 4;  
  A1_invariance := fun M M' => eq_refl;
  polynomial_error := fun n _ => eq_refl;
  approx_nontrivial := ex_intro _ default_space R4_neq_R0
|}.
