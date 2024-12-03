Require Import UniMath.Foundations.All.
Require Import UniMath.CategoryTheory.Core.Categories.
Require Vector.

Local Open Scope cat.

Universe i j.

Record SimplicialPresheaf (C : category) : Type := {
  presheaf_ob : ob C → UU;
  presheaf_mor : ∏ {a b : ob C}, (a --> b) → presheaf_ob b → presheaf_ob a;
  presheaf_id : ∏ {a : ob C} (x : presheaf_ob a),
    presheaf_mor (identity a) x = x;
  presheaf_comp : ∏ {a b c : ob C} (f : a --> b) (g : b --> c) (x : presheaf_ob c),
    presheaf_mor f (presheaf_mor g x) = presheaf_mor (f · g) x
}.

Record SimplicialPresheafMorphism {C : category} (F G : SimplicialPresheaf C) : Type := {
  mor_components : ∏ (c : ob C), (@presheaf_ob C F) c → (@presheaf_ob C G) c;
  mor_naturality : ∏ {a b : ob C} (f : a --> b) (x : (@presheaf_ob C F) b),
    mor_components a (@presheaf_mor C F a b f x) = 
    @presheaf_mor C G a b f (mor_components b x)
}.

Definition SimplicialPresheafIdentity {C : category} (F : SimplicialPresheaf C) 
  : SimplicialPresheafMorphism F F := {|
  mor_components := λ c x, x;
  mor_naturality := λ a b f x, idpath _
|}.

Definition SimplicialPresheafComposition {C : category} 
  {F G H : SimplicialPresheaf C}
  (α : SimplicialPresheafMorphism F G) 
  (β : SimplicialPresheafMorphism G H) 
  : SimplicialPresheafMorphism F H := {|
  mor_components := λ c x, 
    (@mor_components _ _ _ β) c (@mor_components _ _ _ α c x);
  mor_naturality := λ a b f x,
    let p1 := @mor_naturality _ _ _ α a b f x in
    let p2 := @mor_naturality _ _ _ β a b f (@mor_components _ _ _ α b x) in
    maponpaths (@mor_components _ _ _ β a) p1 @ p2
|}.



Lemma SimplicialPresheafComp_components {C : category}
  {F G H : SimplicialPresheaf C}
  (α : SimplicialPresheafMorphism F G)
  (β : SimplicialPresheafMorphism G H) :
  ∏ (c : ob C) (x : presheaf_ob _ F c),
  @mor_components _ _ _ (SimplicialPresheafComposition α β) c x =
  @mor_components _ _ _ β c (@mor_components _ _ _ α c x).
Proof.
  intros c x.
  reflexivity.
Qed.


Lemma SimplicialPresheafComp_naturality {C : category}
  {F G H : SimplicialPresheaf C}
  (α : SimplicialPresheafMorphism F G)
  (β : SimplicialPresheafMorphism G H) :
  ∏ {a b : ob C} (f : a --> b) (x : presheaf_ob _ F b),
  @mor_components _ _ _ (SimplicialPresheafComposition α β) a (presheaf_mor _ F f x) =
  presheaf_mor _ H f (@mor_components _ _ _ (SimplicialPresheafComposition α β) b x).
Proof.
  intros a b f x.
  repeat rewrite SimplicialPresheafComp_components.
  destruct α as [α_comp α_nat].
  destruct β as [β_comp β_nat].
  simpl.
  rewrite α_nat, β_nat.
  reflexivity.
Qed.

Definition SimplicialPresheafComp_eq {C : category}
  {F G H K : SimplicialPresheaf C}
  (α_comp : ∏ c, presheaf_ob C F c → presheaf_ob C G c)
  (β_comp : ∏ c, presheaf_ob C G c → presheaf_ob C H c)
  (γ_comp : ∏ c, presheaf_ob C H c → presheaf_ob C K c) :
  ∏ c, presheaf_ob C F c → presheaf_ob C K c :=
  λ c x, γ_comp c (β_comp c (α_comp c x)).

(*WIP HERE*)

Definition functor_preserves_id {C : category} 
  (F_ob : ob C -> ob C)
  (F_mor : ∏ (a b : ob C), (a --> b) -> (F_ob a --> F_ob b)) : UU :=
  ∏ (a : ob C), F_mor a a (identity a) = identity (F_ob a).

Definition functor_preserves_comp {C : category}
  (F_ob : ob C -> ob C)
  (F_mor : ∏ (a b : ob C), (a --> b) -> (F_ob a --> F_ob b)) : UU :=
  ∏ (a b c : ob C) (f : a --> b) (g : b --> c),
    F_mor a c (f · g) = F_mor a b  f · F_mor b c g.

Lemma id_composition_neutral {C : category} {a b : ob C} (f : a --> b) :
  f · identity b = f.
Proof.
  apply id_right.
Qed.

Lemma composition_id_neutral {C : category} {a b : ob C} (f : a --> b) :
  identity a · f = f.
Proof.
  apply id_left.
Qed.

Lemma composition_assoc {C : category} {a b c d : ob C} 
  (f : a --> b) (g : b --> c) (h : c --> d) :
  (f · g) · h = f · (g · h).
Proof.
  apply assoc'.
Qed.

Record P1Suspension (C : category) : UU := {
  p1_susp_map : ob C -> ob C;
  p1_susp_mor : ∏ (a b : ob C), (a --> b) -> (p1_susp_map a --> p1_susp_map b);
  p1_susp_id_preservation : functor_preserves_id p1_susp_map p1_susp_mor;
  p1_susp_comp_preservation : functor_preserves_comp p1_susp_map p1_susp_mor;
  p1_base : ob C;
  p1_inclusion : ∏ (a : ob C), a --> p1_susp_map a;
  p1_projection : ∏ (a : ob C), p1_susp_map a --> a;
  p1_retraction : ∏ (a : ob C), 
    p1_projection a · p1_inclusion a = identity (p1_susp_map a);
  p1_section : ∏ (a : ob C),
    p1_inclusion a · p1_projection a = identity a;
  p1_mor_coherence : ∏ (a b : ob C) (f : a --> b),
    p1_susp_mor a b f · p1_projection b = p1_projection a · f
}.



Definition get_p1_susp_map {C : category} (P : P1Suspension C) := @p1_susp_map C P.
Definition get_p1_susp_mor {C : category} (P : P1Suspension C) := @p1_susp_mor C P.
Definition IsIsomorphism {C : category} {a b : ob C} (f : a --> b) : UU :=
  ∑ (g : b --> a), (f · g = identity a) × (g · f = identity b).

Record Suspension (C : category) : UU := {
  susp_map : ob C -> ob C;
  susp_mor : ∏ (a b : ob C), (a --> b) -> (susp_map a --> susp_map b);
  susp_id : ∏ (a : ob C), susp_mor a a (identity a) = identity (susp_map a);
  susp_comp : ∏ (a b c : ob C) (f : a --> b) (g : b --> c),
              susp_mor a c (f · g) = susp_mor a b f · susp_mor b c g;
  base_point : ob C;
  zero_section : ∏ (a : ob C), base_point --> a;
  point_susp : base_point --> susp_map base_point
}.

Definition get_susp_map {C : category} (Σ : Suspension C) := @susp_map C Σ.
Definition get_susp_mor {C : category} (Σ : Suspension C) := @susp_mor C Σ.
Definition get_base_point {C : category} (Σ : Suspension C) := @base_point C Σ.
Definition get_point_susp {C : category} (Σ : Suspension C) := @point_susp C Σ.
Definition get_p1_inclusion {C : category} (P : P1Suspension C) := @p1_inclusion C P.

Record P1Stable (C : category) : UU := {
  stable_susp : Suspension C;
  stable_p1 : P1Suspension C;
  stability_iso : ∏ (X : ob C), 
    get_susp_map stable_susp X --> get_p1_susp_map stable_p1 X;
  stability_is_iso : ∏ (X : ob C),
    IsIsomorphism (stability_iso X);
  stability_natural : ∏ (X Y : ob C) (f : X --> Y),
    stability_iso X · get_p1_susp_mor stable_p1 X Y f = 
    get_susp_mor stable_susp X Y f · stability_iso Y
}.

Definition get_stable_susp {C : category} (PS : P1Stable C) : Suspension C :=
  match PS with
  | Build_P1Stable _ susp p1 iso is_iso nat => susp
  end.

Definition get_stable_p1 {C : category} (PS : P1Stable C) : P1Suspension C :=
  match PS with
  | Build_P1Stable _ susp p1 iso is_iso nat => p1
  end.

Definition get_stability_iso {C : category} (PS : P1Stable C) : 
  ∏ (X : ob C), get_susp_map (get_stable_susp PS) X --> get_p1_susp_map (get_stable_p1 PS) X :=
  match PS with
  | Build_P1Stable _ susp p1 iso is_iso nat => iso
  end.

Definition get_stability_is_iso {C : category} (PS : P1Stable C) :
  ∏ (X : ob C), IsIsomorphism (get_stability_iso PS X) :=
  match PS with
  | Build_P1Stable _ susp p1 iso is_iso nat => is_iso
  end.

Definition get_stability_inverse {C : category} (PS : P1Stable C) (X : ob C) : 
  get_p1_susp_map (get_stable_p1 PS) X --> get_susp_map (get_stable_susp PS) X :=
  pr1 (get_stability_is_iso PS X).

Definition get_stability_natural {C : category} (PS : P1Stable C) :
  ∏ (X Y : ob C) (f : X --> Y),
    get_stability_iso PS X · get_p1_susp_mor (get_stable_p1 PS) X Y f = 
    get_susp_mor (get_stable_susp PS) X Y f · get_stability_iso PS Y :=
  match PS with
  | Build_P1Stable _ susp p1 iso is_iso nat => nat
  end.

Definition cancelR_iso {C : category} {a b c : ob C} 
  (h : b --> c) (H : IsIsomorphism h) {f g : a --> b} :
  f · h = g · h -> f = g.
Proof.
  intro p.
  set (hinv := pr1 H).
  set (hsec := pr1 (pr2 H)).
  rewrite <- (id_right f).
  rewrite <- (id_right g).
  rewrite <- hsec.
  rewrite <- assoc'.
  rewrite p.
  rewrite assoc'.
  rewrite hsec.
  rewrite id_right.
  reflexivity.
Qed.

Definition post_comp_with_iso_is_inj {C : category} {a b c : ob C} 
  (f g : a --> b) (h : b --> c) (H : IsIsomorphism h) :
  f · h = g · h -> f = g.
Proof.
  apply (cancelR_iso h H).
Qed.

Record BiFunctor (C D : category) : UU := {
  bi_ob : ob C → ob C → ob D;
  bi_mor : ∏ (x1 y1 x2 y2 : ob C),
    (x1 --> y1) → (x2 --> y2) →
    bi_ob x1 x2 --> bi_ob y1 y2;
  bi_id : ∏ (x1 x2 : ob C),
    bi_mor x1 x1 x2 x2 (identity x1) (identity x2) = identity (bi_ob x1 x2);
  bi_comp : ∏ (x1 y1 z1 x2 y2 z2 : ob C)
    (f1 : x1 --> y1) (g1 : y1 --> z1)
    (f2 : x2 --> y2) (g2 : y2 --> z2),
    bi_mor x1 z1 x2 z2 (f1 · g1) (f2 · g2) = 
    bi_mor x1 y1 x2 y2 f1 f2 · bi_mor y1 z1 y2 z2 g1 g2
}.

Record TriFunctor (C D : category) : UU := {
  tri_ob : ob C → ob C → ob C → ob D;
  tri_mor : ∏ (x1 y1 x2 y2 x3 y3 : ob C),
    (x1 --> y1) → (x2 --> y2) → (x3 --> y3) →
    tri_ob x1 x2 x3 --> tri_ob y1 y2 y3;
  tri_id : ∏ (x1 x2 x3 : ob C),
    tri_mor x1 x1 x2 x2 x3 x3 (identity x1) (identity x2) (identity x3) = 
    identity (tri_ob x1 x2 x3);
  tri_comp : ∏ (x1 y1 z1 x2 y2 z2 x3 y3 z3 : ob C)
    (f1 : x1 --> y1) (g1 : y1 --> z1)
    (f2 : x2 --> y2) (g2 : y2 --> z2)
    (f3 : x3 --> y3) (g3 : y3 --> z3),
    tri_mor x1 z1 x2 z2 x3 z3 (f1 · g1) (f2 · g2) (f3 · g3) = 
    tri_mor x1 y1 x2 y2 x3 y3 f1 f2 f3 · tri_mor y1 z1 y2 z2 y3 z3 g1 g2 g3
}.

Record NTuple (C : category) (n : Datatypes.nat) : UU := {
  base_obj : ob C;
  objects : list (ob C);
  obj_length : length objects = n
}.

Definition get_objects {C : category} {n : Datatypes.nat} 
  (t : NTuple C n) : list (ob C) := @objects C n t.

Definition get_length {C : category} {n : Datatypes.nat} 
  (t : NTuple C n) : length (@objects C n t) = n := @obj_length C n t.

Definition ntuple_map {C : category} {n : Datatypes.nat}
  (t1 t2 : NTuple C n) : UU :=
  ∏ (i : Datatypes.nat), 
    List.nth i (@objects C n t1) (@base_obj C n t1) --> 
    List.nth i (@objects C n t2) (@base_obj C n t2).

Definition make_ntuple {C : category} {n : Datatypes.nat} 
  (base : ob C) (obs : list (ob C)) (pf : length obs = n) : NTuple C n := 
  {| base_obj := base; objects := obs; obj_length := pf |}.

Definition ntuple_id {C : category} {n : Datatypes.nat}
  (t : NTuple C n) : ntuple_map t t :=
  λ i, identity (List.nth i (@objects C n t) (@base_obj C n t)).

Definition ntuple_comp {C : category} {n : Datatypes.nat}
  {t1 t2 t3 : NTuple C n}
  (f : ntuple_map t1 t2) (g : ntuple_map t2 t3) : ntuple_map t1 t3 :=
  λ i, f i · g i.

Record MultiFunctor (C D : category) (n : Datatypes.nat) : UU := {
  multi_ob : NTuple C n → ob D;
  multi_mor : ∏ (xs ys : NTuple C n),
    ntuple_map xs ys → multi_ob xs --> multi_ob ys;
  multi_id : ∏ (xs : NTuple C n),
    multi_mor xs xs (ntuple_id xs) = identity (multi_ob xs);
  multi_comp : ∏ (xs ys zs : NTuple C n)
    (f : ntuple_map xs ys) (g : ntuple_map ys zs),
    multi_mor xs zs (ntuple_comp f g) = 
    multi_mor xs ys f · multi_mor ys zs g
}.

Definition get_multi_mor {C D : category} {n : Datatypes.nat}
  (F : MultiFunctor C D n) : 
  ∏ (xs ys : NTuple C n), ntuple_map xs ys → 
  @multi_ob C D n F xs --> @multi_ob C D n F ys := 
  @multi_mor C D n F.

Definition get_multi_id {C D : category} {n : Datatypes.nat}
  (F : MultiFunctor C D n) : 
  ∏ (xs : NTuple C n),
  @multi_mor C D n F xs xs (ntuple_id xs) = identity (@multi_ob C D n F xs) := 
  @multi_id C D n F.

Definition get_multi_comp {C D : category} {n : Datatypes.nat}
  (F : MultiFunctor C D n) :
  ∏ (xs ys zs : NTuple C n)
    (f : ntuple_map xs ys) (g : ntuple_map ys zs),
    @multi_mor C D n F xs zs (ntuple_comp f g) = 
    @multi_mor C D n F xs ys f · @multi_mor C D n F ys zs g :=
  @multi_comp C D n F.

Lemma list_map_preserves_length {A B : UU} (f : A → B) (l : list A) :
  List.length (List.map f l) = List.length l.
Proof.
  induction l.
  - reflexivity.
  - simpl. 
    rewrite IHl.
    reflexivity.
Defined.

Definition lift_to_ntuple {C : category} {n : Datatypes.nat} 
  (x : ob C) : NTuple C n.
Proof.
  use (make_ntuple x (List.repeat x n)).
  induction n.
  - reflexivity.
  - simpl. 
    rewrite IHn.
    reflexivity.
Defined.

Lemma list_repeat_nth {A : UU} (x : A) (n i : Datatypes.nat) :
  List.nth i (List.repeat x n) x = x.
Proof.
  revert i.
  induction n.
  - intro i. simpl. 
    destruct i; reflexivity.
  - intro i. simpl.
    destruct i.
    + reflexivity.
    + simpl. apply IHn.
Defined.

Lemma lift_to_ntuple_nth {C : category} {n : Datatypes.nat}
  (x : ob C) (i : Datatypes.nat) :
  List.nth i (@objects C n (lift_to_ntuple x)) (@base_obj C n (lift_to_ntuple x)) = x.
Proof.
  unfold lift_to_ntuple.
  simpl.
  apply list_repeat_nth.
Defined.

Definition lift_morphism {C : category} {n : Datatypes.nat}
  {x y : ob C} (f : x --> y) : @ntuple_map C n (lift_to_ntuple x) (lift_to_ntuple y).
Proof.
  intro i.
  rewrite !lift_to_ntuple_nth.
  exact f.
Defined.


Lemma lift_to_ntuple_id_preservation {C : category} {n : Datatypes.nat}
  (x : ob C) :
  @lift_morphism C n x x (identity x) = ntuple_id (lift_to_ntuple x).
Proof.
  cbn.
  unfold lift_morphism.
  unfold ntuple_id.
  apply funextsec.
  intro i.
  simpl.
  destruct (lift_to_ntuple_nth x i).
  apply idpath.
Qed.

Lemma lift_morphism_comp {C : category} {n : Datatypes.nat}
  {x y z : ob C} (f : x --> y) (g : y --> z) :
  @lift_morphism C n _ _ (f · g) = 
  ntuple_comp (@lift_morphism C n _ _ f) (@lift_morphism C n _ _ g).
Proof.
  unfold lift_morphism.
  unfold ntuple_comp.
  apply funextsec.
  intro i.
  rewrite (@lift_to_ntuple_nth C n x i).
  rewrite (@lift_to_ntuple_nth C n y i).
  rewrite (@lift_to_ntuple_nth C n z i).
  apply idpath.
Defined.

Definition is_functorial_lift {C : category} {n : Datatypes.nat} : UU :=
  (∏ (a : ob C), @lift_morphism C n a a (identity a) = ntuple_id (lift_to_ntuple a)) ×
  (∏ (a b c : ob C) (f g : a --> b) (h : b --> c),
     f = g -> 
     @lift_morphism C n _ _ (f · h) = 
     ntuple_comp (@lift_morphism C n _ _ g) (@lift_morphism C n _ _ h)).


Lemma p1_suspension_retract {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) (x : ob C) :
  @lift_morphism C n (p1_susp_map C P x) (p1_susp_map C P x)
    (@p1_projection C P x · @p1_inclusion C P x) = 
  @lift_morphism C n (p1_susp_map C P x) (p1_susp_map C P x)
    (identity (p1_susp_map C P x)).
Proof.
  unfold lift_morphism.
  apply funextsec.
  intro i.
  simpl.
  rewrite (@p1_retraction C P x).
  apply idpath.
Qed.

Lemma p1_suspension_section {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) (x : ob C) :
  @lift_morphism C n x x
    (@p1_inclusion C P x · @p1_projection C P x) = 
  @lift_morphism C n x x (identity x).
Proof.
  unfold lift_morphism.
  apply funextsec.
  intro i.
  simpl.
  rewrite (@p1_section C P x).
  apply idpath.
Qed.

Lemma list_nth_repeat_transport {C : category} {n : Datatypes.nat}
  (x : ob C) (i : Datatypes.nat) :
  transportf (λ a : ob C, C⟦a, a⟧) 
    (list_repeat_nth x n i) 
    (identity (List.nth i (List.repeat x n) x)) =
  identity x.
Proof.
  unfold transportf.
  destruct (list_repeat_nth x n i).
  apply idpath.
Qed.

Lemma lift_to_ntuple_nth_id {C : category} {n : Datatypes.nat} (x : ob C) (i : Datatypes.nat) :
  List.nth i (List.repeat x n) x = x.
Proof.
  revert i.
  induction n.
  - intros i; destruct i; simpl; reflexivity.
  - intros i; simpl.
    destruct i.
    + reflexivity.
    + apply IHn.
Qed.

Definition transport_in_C {C : category} (x y : ob C) (p : x = y) : 
  C⟦x, x⟧ → C⟦y, y⟧ := 
  transportf (λ a, C⟦a, a⟧) p.

Lemma transport_in_C_id {C : category} (x y : ob C) (p : x = y) :
  transport_in_C x y p (identity x) = identity y.
Proof.
  destruct p.
  apply idpath.
Qed.

Lemma transport_p1_susp_id {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) (x : ob C) 
  (p : p1_susp_map C P x = p1_susp_map C P x) :
  transport_in_C (p1_susp_map C P x) (p1_susp_map C P x) p (identity (p1_susp_map C P x)) = 
  identity (p1_susp_map C P x).
Proof.
  destruct p.
  apply idpath.
Defined.

Lemma lift_p1_susp_mor_coherence {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x y : ob C} (f : x --> y) (i : Datatypes.nat) :
  List.nth i (@objects C n (lift_to_ntuple (p1_susp_map C P x))) 
    (@base_obj C n (lift_to_ntuple (p1_susp_map C P x))) = 
  p1_susp_map C P (List.nth i (@objects C n (lift_to_ntuple x)) 
    (@base_obj C n (lift_to_ntuple x))).
Proof.
  rewrite !lift_to_ntuple_nth.
  reflexivity.
Qed.

Lemma internal_path_transport {C : category} (x y : ob C) 
  (p : x = y) (f g : C⟦x,y⟧) :
  f = g ->
  internal_paths_rew_r C x y (λ o, C⟦x,o⟧) f p =
  internal_paths_rew_r C x y (λ o, C⟦x,o⟧) g p.
Proof.
  intro h.
  rewrite h.
  apply idpath.
Qed.

Lemma nested_nth_lift_to_ntuple {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) (i : Datatypes.nat) (x : ob C) :
  List.nth i (objects C n (lift_to_ntuple 
    (List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P x)))
             (base_obj C n (lift_to_ntuple (p1_susp_map C P x))))))
    (base_obj C n (lift_to_ntuple 
      (List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P x)))
               (base_obj C n (lift_to_ntuple (p1_susp_map C P x)))))) =
  List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P x)))
    (base_obj C n (lift_to_ntuple (p1_susp_map C P x))).
Proof.
  rewrite !lift_to_ntuple_nth.
  apply idpath.
Defined.

Lemma p1_suspension_lift_nth_eq {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) (x : ob C) (i : Datatypes.nat) :
  List.nth i (List.repeat (p1_susp_map C P x) n) (p1_susp_map C P x) = 
  p1_susp_map C P x.
Proof.
  apply list_repeat_nth.
Qed.

Lemma p1_suspension_lift_mor_eq {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x y : ob C} (f : x --> y) (i : Datatypes.nat) :
  List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P x)))
    (base_obj C n (lift_to_ntuple (p1_susp_map C P x))) =
  List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P x)))
    (base_obj C n (lift_to_ntuple (p1_susp_map C P x))).
Proof.
  apply idpath.
Qed.

Lemma p1_suspension_lift_objects {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x : ob C} (i : Datatypes.nat) :
  List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P x)))
    (base_obj C n (lift_to_ntuple (p1_susp_map C P x))) =
  p1_susp_map C P x.
Proof.
  unfold lift_to_ntuple.
  simpl.
  apply list_repeat_nth.
Qed.

Lemma p1_suspension_lift_structure {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x : ob C} :
  objects C n (lift_to_ntuple (p1_susp_map C P x)) = 
  List.repeat (p1_susp_map C P x) n.
Proof.
  unfold lift_to_ntuple.
  simpl.
  reflexivity.
Qed.

Lemma p1_suspension_lift_structure_mor {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x y : ob C} (i : Datatypes.nat) :
  paths (List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P x)))
         (base_obj C n (lift_to_ntuple (p1_susp_map C P x))))
        (p1_susp_map C P x).
Proof.
  rewrite p1_suspension_lift_structure.
  apply list_repeat_nth.
Qed.

Lemma p1_suspension_lift_morphism_structure {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x y : ob C} (f : x --> y) (i : Datatypes.nat) :
  paths (List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P x)))
                   (base_obj C n (lift_to_ntuple (p1_susp_map C P x))))
        (List.nth i (List.repeat (p1_susp_map C P x) n) (p1_susp_map C P x)).
Proof.
  rewrite p1_suspension_lift_structure.
  apply idpath.
Qed.

Lemma p1_suspension_lift_morphism_objects {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x : ob C} (i : Datatypes.nat) :
  paths (List.nth i (List.repeat (p1_susp_map C P x) n) (p1_susp_map C P x))
        (p1_susp_map C P x).
Proof.
  apply list_repeat_nth.
Qed.


Lemma p1_suspension_lift_morphism_eq {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x : ob C} (i : Datatypes.nat) :
  List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P x)))
             (base_obj C n (lift_to_ntuple (p1_susp_map C P x))) =
  List.nth i (List.repeat (p1_susp_map C P x) n) 
             (p1_susp_map C P x).
Proof.
  rewrite p1_suspension_lift_structure.
  apply idpath.
Qed.

Lemma p1_suspension_lift_morphism_hom {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x y : ob C} (f : x --> y) (i : Datatypes.nat) :
  @lift_morphism C n (p1_susp_map C P x) (p1_susp_map C P y) (p1_susp_mor C P x y f) i =
  @lift_morphism C n (p1_susp_map C P x) (p1_susp_map C P y) (p1_susp_mor C P x y f) i.
Proof.
  apply idpath.
Qed.

Lemma p1_suspension_lift_morphism_domain {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x y : ob C} (i : Datatypes.nat) :
  paths (List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P x)))
                   (base_obj C n (lift_to_ntuple (p1_susp_map C P x))))
        (p1_susp_map C P x) ×
  paths (List.nth i (objects C n (lift_to_ntuple (p1_susp_map C P y)))
                   (base_obj C n (lift_to_ntuple (p1_susp_map C P y))))
        (p1_susp_map C P y).
Proof.
  split.
  - rewrite p1_suspension_lift_structure.
    apply list_repeat_nth.
  - rewrite p1_suspension_lift_structure.
    apply list_repeat_nth.
Qed.

Theorem P1_suspension_preserves_comp {C : category} {n : Datatypes.nat}
  (P : P1Suspension C) {x y z : ob C} (f : x --> y) (g : y --> z) :
  @lift_morphism C n _ _ (p1_susp_mor C P x z (f · g)) =
  ntuple_comp 
    (@lift_morphism C n _ _ (p1_susp_mor C P x y f))
    (@lift_morphism C n _ _ (p1_susp_mor C P y z g)).
Proof.
  unfold lift_morphism, ntuple_comp.
  apply funextsec.
  intro i.
  set (p1 := lift_to_ntuple_nth (p1_susp_map C P x) i).
  set (p2 := lift_to_ntuple_nth (p1_susp_map C P y) i).
  set (p3 := lift_to_ntuple_nth (p1_susp_map C P z) i).
  simpl in *.
  rewrite p1, p2, p3.
  apply p1_susp_comp_preservation.
Qed.

Theorem P1Stable_preserves_id {C : category} (PS : P1Stable C) (X : ob C) :
  get_stability_iso PS X · get_p1_susp_mor (get_stable_p1 PS) X X (identity X) =
  get_susp_mor (get_stable_susp PS) X X (identity X) · get_stability_iso PS X.
Proof.
  apply get_stability_natural.
Qed.

Theorem P1Stable_preserves_comp {C : category} (PS : P1Stable C) 
  {X Y Z : ob C} (f : X --> Y) (g : Y --> Z) :
  get_stability_iso PS X · 
  get_p1_susp_mor (get_stable_p1 PS) X Z (f · g) =
  get_susp_mor (get_stable_susp PS) X Z (f · g) · 
  get_stability_iso PS Z.
Proof.
  apply get_stability_natural.
Qed.

Theorem P1Stable_iso_section {C : category} (PS : P1Stable C) (X : ob C) :
  get_stability_iso PS X · get_stability_inverse PS X = 
  identity (get_susp_map (get_stable_susp PS) X).
Proof.
  apply (pr1 (pr2 (get_stability_is_iso PS X))).
Qed.

Theorem P1Stable_iso_comp_unique {C : category} (PS : P1Stable C) (X : ob C) :
  get_stability_iso PS X · get_stability_inverse PS X = 
  identity (get_susp_map (get_stable_susp PS) X).
Proof.
  apply (pr1 (pr2 (get_stability_is_iso PS X))).
Qed.

Theorem P1Stable_inverse_comp_unique {C : category} (PS : P1Stable C) (X : ob C) :
  get_stability_inverse PS X · get_stability_iso PS X = 
  identity (get_p1_susp_map (get_stable_p1 PS) X).
Proof.
  apply (pr2 (pr2 (get_stability_is_iso PS X))).
Qed.

Theorem P1Stable_iso_cancellation {C : category} (PS : P1Stable C) 
  {X Y : ob C} {f g : X --> get_susp_map (get_stable_susp PS) Y} :
  f · get_stability_iso PS Y = g · get_stability_iso PS Y -> f = g.
Proof.
  intro p.
  apply (cancelR_iso (get_stability_iso PS Y) (get_stability_is_iso PS Y)).
  exact p.
Qed.



