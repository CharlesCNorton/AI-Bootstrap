Require Import UniMath.Foundations.All.
Require Import UniMath.CategoryTheory.Core.Categories.

Local Open Scope cat.

Record Suspension (C : category) : UU := {
  underlying_cat := C;
  susp_map : ob C -> ob C;
  susp_mor : ∏ (a b : ob C), (a --> b) -> (susp_map a --> susp_map b)
}.