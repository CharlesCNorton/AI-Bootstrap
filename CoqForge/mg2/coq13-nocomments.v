Require Import Coq.Reals.Reals.
Require Import Coq.Sets.Ensembles.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Arith.Compare_dec.
Open Scope R_scope.

Record MotivicSpace := {
  underlying_type : Type;
  A1_homotopy : Type;
  base_scheme : Type;
  transfer : underlying_type -> underlying_type
}.

Record MotivicExcisiveApprox := {
  m_degree : nat;
  m_weight : nat;
  m_value : MotivicSpace -> R;
  m_error : MotivicSpace -> R
}.

Record TowerLevel := {
  level_num : nat;
  filtration_degree : nat;
  bound_condition : (filtration_degree <= level_num)%nat
}.

Record MotivicTower := {
  base_level : TowerLevel;
  tower_height : nat;
  level_map : forall (n : nat) (H: (n < tower_height)%nat),
    MotivicExcisiveApprox -> MotivicExcisiveApprox;
  level_map_degree : forall (n : nat) (H: (n < tower_height)%nat) 
    (approx : MotivicExcisiveApprox),
    (m_degree (level_map n H approx) <= m_degree approx)%nat
}.

Record VectorBundle := {
  rank : nat;
  chern_class : R
}.

Record Automorphism := {
  dimension : nat;
  determinant : R;
  trace : R
}.

Record SteinbergSymbol := {
  first_arg : R;
  second_arg : R;
  bilinear : R
}.

Inductive K0 : Type :=
  | K0_zero : K0
  | K0_vb : VectorBundle -> K0
  | K0_sum : K0 -> K0 -> K0.

Inductive K1 : Type :=
  | K1_zero : K1
  | K1_auto : Automorphism -> K1
  | K1_sum : K1 -> K1 -> K1
  | K1_inv : K1 -> K1.

Inductive K2 : Type :=
  | K2_zero : K2
  | K2_steinberg : SteinbergSymbol -> K2
  | K2_sum : K2 -> K2 -> K2
  | K2_inv : K2 -> K2
  | K2_relation : forall (s1 s2 : SteinbergSymbol),
      first_arg s1 = first_arg s2 ->
      second_arg s1 = second_arg s2 ->
      bilinear s1 = bilinear s2 ->
      K2.

Inductive K0_eq : K0 -> K0 -> Prop :=
  | K0_eq_zero : K0_eq K0_zero K0_zero
  | K0_eq_vb : forall v1 v2,
      rank v1 = rank v2 ->
      chern_class v1 = chern_class v2 ->
      K0_eq (K0_vb v1) (K0_vb v2)
  | K0_eq_sum : forall a1 a2 b1 b2,
      K0_eq a1 b1 ->
      K0_eq a2 b2 ->
      K0_eq (K0_sum a1 a2) (K0_sum b1 b2).

Inductive K1_eq : K1 -> K1 -> Prop :=
  | K1_eq_zero : K1_eq K1_zero K1_zero
  | K1_eq_auto : forall a1 a2,
      dimension a1 = dimension a2 ->
      determinant a1 = determinant a2 ->
      trace a1 = trace a2 ->
      K1_eq (K1_auto a1) (K1_auto a2)
  | K1_eq_sum : forall a1 a2 b1 b2,
      K1_eq a1 b1 ->
      K1_eq a2 b2 ->
      K1_eq (K1_sum a1 a2) (K1_sum b1 b2)
  | K1_eq_inv : forall a b,
      K1_eq a b ->
      K1_eq (K1_inv a) (K1_inv b).

Inductive K2_eq : K2 -> K2 -> Prop :=
  | K2_eq_zero : K2_eq K2_zero K2_zero
  | K2_eq_steinberg : forall s1 s2,
      first_arg s1 = first_arg s2 ->
      second_arg s1 = second_arg s2 ->
      bilinear s1 = bilinear s2 ->
      K2_eq (K2_steinberg s1) (K2_steinberg s2)
  | K2_eq_sum : forall a1 a2 b1 b2,
      K2_eq a1 b1 ->
      K2_eq a2 b2 ->
      K2_eq (K2_sum a1 a2) (K2_sum b1 b2)
  | K2_eq_inv : forall a b,
      K2_eq a b ->
      K2_eq (K2_inv a) (K2_inv b)
  | K2_eq_rel : forall s1 s2,
      first_arg s1 * second_arg s2 = second_arg s1 * first_arg s2 ->
      K2_eq (K2_steinberg s1) (K2_steinberg s2).

Inductive K_compatible : K0 -> K1 -> Prop :=
  | K_comp_zero : K_compatible K0_zero K1_zero
  | K_comp_vb_auto : forall (v : VectorBundle) (a : Automorphism),
      rank v = dimension a ->
      K_compatible (K0_vb v) (K1_auto a)
  | K_comp_sum : forall k0_1 k0_2 k1_1 k1_2,
      K_compatible k0_1 k1_1 ->
      K_compatible k0_2 k1_2 ->
      K_compatible (K0_sum k0_1 k0_2) (K1_sum k1_1 k1_2).

Definition steinberg_lift (a1 a2 : Automorphism) : SteinbergSymbol := {|
  first_arg := determinant a1;
  second_arg := determinant a2;
  bilinear := trace a1 * trace a2
|}.

Inductive K1_K2_compatible : K1 -> K2 -> Prop :=
  | K12_comp_zero : K1_K2_compatible K1_zero K2_zero
  | K12_comp_auto : forall (a1 a2 : Automorphism),
      determinant a1 <> 0 ->
      determinant a2 <> 0 ->
      K1_K2_compatible (K1_auto a1) 
        (K2_steinberg (steinberg_lift a1 a2))
  | K12_comp_sum : forall k1_1 k1_2 k2_1 k2_2,
      K1_K2_compatible k1_1 k2_1 ->
      K1_K2_compatible k1_2 k2_2 ->
      K1_K2_compatible (K1_sum k1_1 k1_2) (K2_sum k2_1 k2_2)
  | K12_comp_inv : forall k1 k2,
      K1_K2_compatible k1 k2 ->
      K1_K2_compatible (K1_inv k1) (K2_inv k2).

Record MotivicKTheory := {
  space : MotivicSpace;
  k0_lift : K0 -> underlying_type space;
  k1_lift : K1 -> underlying_type space;
  k2_lift : K2 -> underlying_type space;
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
  k2_lift_steinberg_eq : forall s1 s2,
    first_arg s1 = first_arg s2 ->
    second_arg s1 = second_arg s2 ->
    bilinear s1 = bilinear s2 ->
    k2_lift (K2_steinberg s1) = k2_lift (K2_steinberg s2);
  k2_lift_sum : forall a b,
    k2_lift (K2_sum a b) = k2_lift a;
  k2_lift_inv : forall a,
    k2_lift (K2_inv a) = k2_lift a
}.

Record MotivicKTheory_Extended := {
  base_theory :> MotivicKTheory;
  lift_compatible : forall k0 k1,
    K_compatible k0 k1 ->
    transfer (space base_theory) (k0_lift base_theory k0) = 
    k1_lift base_theory k1;
  lift_k1_k2_compatible : forall k1 k2,
    K1_K2_compatible k1 k2 ->
    transfer (space base_theory) (k1_lift base_theory k1) =
    k2_lift base_theory k2
}.

Record KTheoryTowerLevel := {
  base_tower_level : TowerLevel;
  k0_data : K0;
  k1_data : K1;
  k2_data : K2;
  k0_k1_compatibility : K_compatible k0_data k1_data;
  k1_k2_compatibility : K1_K2_compatible k1_data k2_data;
  degree_bound : (filtration_degree base_tower_level <= level_num base_tower_level)%nat
}.

Record KTheoryTower := {
  motivic_tower :> MotivicTower;
  level_k_theory : forall (n : nat) (H: (n < tower_height motivic_tower)%nat),
    KTheoryTowerLevel;
  level_compatibility : forall (n m : nat) 
    (Hn: (n < tower_height motivic_tower)%nat)
    (Hm: (m < tower_height motivic_tower)%nat)
    (Hnm: (n <= m)%nat),
    K_compatible 
      (k0_data (level_k_theory n Hn))
      (k1_data (level_k_theory m Hm));
  level_k2_compatibility : forall (n m : nat)
    (Hn: (n < tower_height motivic_tower)%nat)
    (Hm: (m < tower_height motivic_tower)%nat)
    (Hnm: (n <= m)%nat),
    K1_K2_compatible
      (k1_data (level_k_theory n Hn))
      (k2_data (level_k_theory m Hm));
  tower_respects_lifts : forall (n : nat) 
    (H: (n < tower_height motivic_tower)%nat)
    (mk : MotivicKTheory_Extended),
    let level := level_k_theory n H in
    transfer (space mk) 
      (k0_lift mk (k0_data level)) = 
      k1_lift mk (k1_data level)
}.

Theorem tower_lift_k2_compatibility : 
  forall (kt : KTheoryTower) (mk : MotivicKTheory_Extended)
  (n m : nat) 
  (Hn: (n < tower_height kt)%nat)
  (Hm: (m < tower_height kt)%nat)
  (Hnm: (n <= m)%nat),
  let level_n := level_k_theory kt n Hn in
  let level_m := level_k_theory kt m Hm in
  transfer (space mk) (k1_lift mk (k1_data level_n)) =
  k2_lift mk (k2_data level_m).
Proof.
  intros.
  apply (lift_k1_k2_compatible mk).
  apply (level_k2_compatibility kt n m Hn Hm Hnm).
Qed.

Theorem tower_K2_sum_compatibility :
  forall (k1_1 k1_2 : K1) (k2_1 k2_2 : K2),
  K1_K2_compatible k1_1 k2_1 ->
  K1_K2_compatible k1_2 k2_2 ->
  K1_K2_compatible (K1_sum k1_1 k1_2) (K2_sum k2_1 k2_2).
Proof.
  intros k1_1 k1_2 k2_1 k2_2 H1 H2.
  apply K12_comp_sum; assumption.
Qed.
