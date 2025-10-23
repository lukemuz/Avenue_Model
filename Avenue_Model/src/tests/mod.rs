mod model_tests;
mod weight_distribution_test;
mod analysis_tests;
mod glm_tests;
mod glm_distribution_tests;
mod glm_benchmarks;
mod glm_realistic_benchmarks;
pub mod testing_utils;

use crate::{
    license_handler::internal_initialize_license,
    rating_model::{process_lgbm_trees, 
        RatingTable, 
        RatingModel,
        FeatureValue,
        LinkFunction,
        expand_and_combine_tables,
        build_analysis_tablemodel,
        build_consolidated_tablemodel,
        combine_all_tables,
        },
    table_estimator::estimate_number_of_tables,
    tests::testing_utils::initialize_test_license
};
use polars::prelude::*;
use polars::frame::DataFrame;
use std::collections::{HashMap, HashSet};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::PyDataFrame;
#[cfg(feature = "python")]
use pyo3::types::PyDict;