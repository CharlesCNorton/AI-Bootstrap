Require Import UniMath.Foundations.All.
Require Import UniMath.CategoryTheory.Core.Categories.

Record Suspension (C : category) : UU := {
  underlying_cat := C 
}.