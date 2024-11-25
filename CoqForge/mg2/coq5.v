Require Import Coq.Reals.Reals.
Require Import Coq.Sets.Ensembles.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Arith.Compare_dec.
Require Import Coq.micromega.Lia.
Open Scope R_scope.

(* Previous basic definitions *)
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

Definition cross_effect_calc (n : nat) (approx : MotivicExcisiveApprox) : nat :=
  match n with
  | 0 => 0
  | S k => (m_degree approx) * k
  end.

Definition concrete_tower_map (src dst : TowerLevel)
  (approx : MotivicExcisiveApprox) : MotivicExcisiveApprox := {|
    m_degree := min (m_degree approx) (level_num dst);
    m_weight := min (m_weight approx) (filtration_degree dst);
    m_value := m_value approx;
    m_error := fun s =>
      if le_gt_dec (level_num dst) (level_num src)
      then m_error approx s
      else 0
|}.

Record MotivicTower := {
  base_level : TowerLevel;
  tower_height : nat;
  level_map : forall (n : nat) (H: (n < tower_height)%nat),
    MotivicExcisiveApprox -> MotivicExcisiveApprox;
  (* Add constraint on level_map behavior *)
  level_map_degree : forall (n : nat) (H: (n < tower_height)%nat) 
    (approx : MotivicExcisiveApprox),
    (m_degree (level_map n H approx) <= m_degree approx)%nat
}.

(* Basic properties *)
Lemma tower_level_bound : forall (mt : MotivicTower) (n : nat) 
  (Hn: (n < tower_height mt)%nat) (approx : MotivicExcisiveApprox),
  (m_degree (level_map mt n Hn approx) <= m_degree approx)%nat.
Proof.
  intros.
  apply (level_map_degree mt).
Qed.