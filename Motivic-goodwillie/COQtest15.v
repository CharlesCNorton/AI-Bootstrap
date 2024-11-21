From Coq Require Import List.
From Coq Require Import Arith.
Import ListNotations.

Definition value_type := nat.

Record kernel_relation := mk_kernel_rel {
  src : value_type;
  tgt : value_type;
  relation_holds : src >= tgt
}.

Record kernel_map := mk_kernel {
  kernel_rel : kernel_relation;
  is_valid : bool
}.

Definition source_val (k: kernel_map) : value_type := 
  src (kernel_rel k).

Definition target_val (k: kernel_map) : value_type := 
  tgt (kernel_rel k).

Definition make_kernel_relation (s t: value_type) (pf: s >= t) : kernel_relation :=
  mk_kernel_rel s t pf.

Definition make_kernel (kr: kernel_relation) : kernel_map :=
  mk_kernel kr true.

Record cohom_degree := mk_degree {
  p_deg : nat;
  q_deg : nat
}.

Record real_obstruction := mk_real_obs {
  obs_level : nat;
  obs_degrees : cohom_degree;
  obs_kernel : kernel_map
}.

Definition successive_kernel_relation (k1 k2: kernel_map) : Prop :=
  source_val k2 <= target_val k1.

(* Key property: target is always <= source *)
Theorem target_bound :
  forall k: kernel_map,
  target_val k <= source_val k.
Proof.
  intros k.
  unfold target_val, source_val.
  destruct k, kernel_rel.
  apply relation_holds.
Qed.

(* Now we can prove kernel decrease using basic steps *)
Lemma kernel_decrease :
  forall k1 k2: kernel_map,
  successive_kernel_relation k1 k2 ->
  source_val k2 <= source_val k1.
Proof.
  intros k1 k2 H.
  unfold successive_kernel_relation in H.
  assert (H1: target_val k1 <= source_val k1).
  { apply target_bound. }
  assert (H2: source_val k2 <= target_val k1).
  { exact H. }
  assert (H3: source_val k2 <= source_val k1).
  { unfold source_val, target_val in *.
    destruct k1, k2, kernel_rel0, kernel_rel1.
    apply (Nat.le_trans _ tgt0 src0); assumption. }
  exact H3.
Qed.