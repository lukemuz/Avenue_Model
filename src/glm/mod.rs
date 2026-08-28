// GLM fitting module for Avenue_Model
// Fits Generalized Linear Models using IRLS coordinate descent on RatingTables

pub mod fitting;
pub mod inference;
pub mod loss;
pub mod matching;
pub mod penalty;
pub mod redundancy;
pub mod utils;

pub use fitting::{
    fit_glm, fit_glm_with_diagnostics, GLMDiagnostics, GLMOptions, GLMSolver, Normalization,
};
pub use inference::{solve_spd, GLMInference, VariateTerms};
pub use penalty::{PenaltyPlan, TablePenalty};
pub use redundancy::{collective_strength, table_correlations, TablePair, NEAR_ALIAS};
