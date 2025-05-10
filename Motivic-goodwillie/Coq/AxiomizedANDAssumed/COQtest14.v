From Coq Require Import List.
From Coq Require Import Arith.
Import ListNotations.

(* First: Model the kernel relationship *)
Record kernel_map := mk_kernel {
  source_val : nat;
  target_val : nat;
  (* ker(Pₙʷ F → Pₙ₋₁ʷ F) means source_val >= target_val *)
  is_kernel : bool;
  kernel_proof : source_val >= target_val -> is_kernel = true
}.

(* Model cohomology degrees *)
Record cohom_degree := mk_degree {
  p_deg : nat;  (* p in Hᵖ,ᑫʷ *)
  q_deg : nat   (* q in Hᵖ,ᑫʷ *)
}.

(* Now we can model an actual obstruction *)
Record real_obstruction := mk_real_obs {
  obs_level : nat;
  obs_degrees : cohom_degree;
  obs_kernel : kernel_map;
  (* An obstruction must satisfy the kernel condition *)
  obs_valid : (is_kernel obs_kernel) = true
}.

(* Constructing an obstruction requires proof *)
Definition make_real_obs (n: nat) (deg: cohom_degree) (k: kernel_map) 
  (H: is_kernel k = true) : real_obstruction :=
  mk_real_obs n deg k H.

(* Basic properties of obstruction relationships *)
Theorem kernel_condition :
  forall (k: kernel_map),
  source_val k >= target_val k ->
  is_kernel k = true.
Proof.
  intros k H.
  apply (kernel_proof k H).
Qed.