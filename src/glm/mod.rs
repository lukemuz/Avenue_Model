// GLM fitting module for Avenue_Model
// Fits Generalized Linear Models using IRLS coordinate descent on RatingTables

pub mod fitting;
pub mod loss;
pub mod utils;
pub mod matching;

pub use fitting::{fit_glm, GLMOptions};
