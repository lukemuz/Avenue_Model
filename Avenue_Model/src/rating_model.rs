use polars::prelude::*;
use polars::frame::DataFrame;
use polars::series::IntoSeries;
use std::collections::{HashMap, HashSet};
use serde_json::Value;
use std::ops::Add;
use rayon::prelude::*;
use std::sync::Mutex;
use polars::error::PolarsError;
use itertools::Itertools;



use crate::license_handler::validate_current_license;
// Begin RatingTable implementation
#[derive(Debug, Clone)]
enum FeatureType {
    Numeric,
    Categorical,
}
// #[derive(Debug, Clone, Copy)]
// pub enum ObjectiveFunction {
//     Regression,
//     Binary,
//     Poisson,
//     Tweedie,
//     Gamma,
//     Huber
// }

// impl ObjectiveFunction {
//     fn from_objective(objective: &str) -> Self {
//         match objective.to_lowercase().as_str() {
//             "regression" => ObjectiveFunction::Regression,
//             "binary" => ObjectiveFunction::Binary,
//             "poisson" => ObjectiveFunction::Poisson,
//             "tweedie" => ObjectiveFunction::Tweedie,
//             "gamma" => ObjectiveFunction::Gamma,
//             "huber" => ObjectiveFunction::Huber,
//             _ => ObjectiveFunction::Regression,
//         }
//     }
// }

#[derive(Debug, Clone)]
pub enum LinkFunction {
    Identity,    // for 'regression'
    Logit,      // for 'binary'
    Log,        // for 'poisson', 'gamma', 'tweedie'
}

impl LinkFunction {
    pub fn from_objective(objective: &str) -> Self {
        match objective.to_lowercase().as_str() {
            "regression" | "gaussian" => LinkFunction::Identity,
            "binary" | "binomial" | "logistic" => LinkFunction::Logit,
            "poisson" => LinkFunction::Log,
            "tweedie" => LinkFunction::Log,
            "gamma" => LinkFunction::Log,
            _ => LinkFunction::Identity, // default to identity for unknown objectives
        }
    }

    fn inverse(&self, x: f64) -> f64 {
        match self {
            LinkFunction::Identity => x,
            LinkFunction::Logit => 1.0 / (1.0 + (-x).exp()),
            LinkFunction::Log => x.exp(),
        }
    }

    fn to_string(&self) -> String {
        match self {
            LinkFunction::Identity => "identity".to_string(),
            LinkFunction::Logit => "logit".to_string(),
            LinkFunction::Log => "log".to_string(),
        }
    }
}
#[derive(Debug, Clone, Copy)]  // Add Copy here
pub enum FeatureValue {
    Numeric(f64),
    Categorical(i32),  
}

impl From<f64> for FeatureValue {
    fn from(v: f64) -> Self {
        if v.fract() == 0.0 && v >= -999.0 {
            FeatureValue::Categorical(v as i32)  
        } else {
            FeatureValue::Numeric(v)
        }
    }
}

impl From<i32> for FeatureValue {
    fn from(v: i32) -> Self {
        FeatureValue::Categorical(v)
    }
}

#[derive(Debug, Clone)]
pub struct RatingTable {
    pub data: DataFrame,
    // Cache column metadata
    numeric_columns: HashMap<String, usize>,    // column name -> index
    categorical_columns: HashMap<String, usize>, // column name -> index
}

impl RatingTable {
    pub fn new(data: DataFrame, _existing_row_number_col: Option<&str>) -> Self {
        // Remove row_number handling from constructor since we'll generate it on demand
        // Skip license check in test/benchmark mode
        #[cfg(not(test))]
        if !validate_current_license() {
            panic!("License not valid");
        }
        
        let mut numeric_columns = HashMap::new();
        let mut categorical_columns = HashMap::new();
        
        for (idx, col_name) in data.get_column_names().iter().enumerate() {
            if col_name == &"Rating_Factor" {
                continue;
            }
            match data.column(col_name).unwrap().dtype() {
                DataType::Float64 => {
                    numeric_columns.insert(col_name.to_string(), idx);
                },
                DataType::Int32 => {  // Changed from Int64
                    categorical_columns.insert(col_name.to_string(), idx);
                },
                _ => continue,
            }
        }

        Self { 
            data:data.clone(),
            numeric_columns,
            categorical_columns,
        }
    }
    
    pub fn find_row_match(&self, feature_values: &HashMap<String, FeatureValue>) -> Option<usize> {
        // Early return if we don't have all required features
        if !self.has_all_required_features(feature_values) {
            return None;
        }

        // ⭐ OPTIMIZATION: Pre-extract and cache input values + column references
        // CRITICAL: Iterate once to ensure consistent ordering between inputs and columns
        let mut categorical_inputs = Vec::with_capacity(self.categorical_columns.len());
        let categorical_columns_vec: Vec<_> = self.categorical_columns.iter()
            .map(|(col_name, &col_idx)| {
                // Build inputs vector in the same iteration to guarantee order consistency
                if let Some(FeatureValue::Categorical(input_cat)) = feature_values.get(col_name) {
                    categorical_inputs.push(Some(*input_cat));
                } else {
                    categorical_inputs.push(None);
                }
                // Return the cached column reference
                self.data.get_columns()[col_idx].i32().unwrap()
            })
            .collect();
            
        let mut numeric_inputs = Vec::with_capacity(self.numeric_columns.len());
        let numeric_columns_vec: Vec<_> = self.numeric_columns.iter()
            .map(|(col_name, &col_idx)| {
                // Build inputs vector in the same iteration to guarantee order consistency
                if let Some(FeatureValue::Numeric(input_val)) = feature_values.get(col_name) {
                    numeric_inputs.push(Some(*input_val));
                } else {
                    numeric_inputs.push(None);
                }
                // Return the cached column reference
                self.data.get_columns()[col_idx].f64().unwrap()
            })
            .collect();

        let mut best_row: Option<usize> = None;
        let mut used_wildcard = false;

        'row_loop: for i in 0..self.data.height() {
            let mut row_matches = true;
            let mut this_row_used_wildcard = false;

            // ⭐ OPTIMIZATION 3: Direct array access instead of HashMap lookups
            // Check categorical features first
            for (idx, input_cat_opt) in categorical_inputs.iter().enumerate() {
                if let Some(input_cat) = input_cat_opt {
                    let col_val = unsafe { 
                        categorical_columns_vec[idx].get_unchecked(i)
                    };
                    
                    if let Some(table_cat) = col_val {
                        if table_cat != -999 && table_cat != *input_cat {
                            row_matches = false;
                            break;
                        }
                        if table_cat == -999 {
                            this_row_used_wildcard = true;
                        }
                    }
                }
            }

            if !row_matches {
                continue;
            }

            // Check numeric features with cached column references
            for (idx, input_val_opt) in numeric_inputs.iter().enumerate() {
                if let Some(input_val) = input_val_opt {
                    let col_val = unsafe { 
                        numeric_columns_vec[idx].get_unchecked(i)
                    };
                    
                    if let Some(threshold) = col_val {
                        if input_val > &threshold {
                            continue 'row_loop;
                        }
                    }
                }
            }

            // If we get here, we found a match
            // Only update if this is the first match or if we found a more specific match
            if best_row.is_none() || (used_wildcard && !this_row_used_wildcard) {
                best_row = Some(i);
                used_wildcard = this_row_used_wildcard;
            }
        }

        best_row
    }

    // Public accessors for vectorized matching
    pub fn get_categorical_columns(&self) -> &HashMap<String, usize> {
        &self.categorical_columns
    }

    pub fn get_numeric_columns(&self) -> &HashMap<String, usize> {
        &self.numeric_columns
    }

    pub fn get_rating_factor(&self, row: usize) -> f64 {
        let rating_col = self.data.column("Rating_Factor").unwrap();
        unsafe {
            rating_col.f64().unwrap().get_unchecked(row).unwrap_or(f64::NAN)
        }
    }

    pub fn predict(&self, feature_values: &HashMap<String, FeatureValue>) -> f64 {
        // Matches a row to the feature values and returns the rating factor, does not apply the link function
        match self.find_row_match(feature_values) {
            Some(row) => {
                // Get rating factor from the matching row
                self.get_rating_factor(row)
            },
            None => f64::NAN
        }
    }

    pub fn get_rating_factor_batch(&self, row_nums: Series) -> Result<Vec<f64>, PolarsError> {
        let mut df = self.data.clone();
        
        // Create temporary row numbers for joining
        let n_rows = df.height();
        let temp_row_numbers = Series::new("row_number".into(), (0..n_rows).map(|x| x as i32).collect::<Vec<i32>>());
        let df_with_rownums = df.with_column(temp_row_numbers)?;
        
        let mut row_number = row_nums.clone();
        row_number.rename("row_number".into());
        let out_df = DataFrame::new(vec![row_number.into()])?;
        
        // join using temporary row numbers
        let joined = out_df.join(&df_with_rownums, ["row_number"], ["row_number"], JoinArgs::new(JoinType::Left), None)?;
        Ok(joined.column("Rating_Factor")?.f64()?.into_iter().flatten().collect())
    }

    pub fn predict_batch(&self, df: &DataFrame) -> Vec<f64> {
        // Predicts a batch of rows, does not apply the link function
        // Uses parallel processing if the number of rows is greater than the ROW_PARALLEL_THRESHOLD
        // Otherwise, it processes rows sequentially
        const ROW_PARALLEL_THRESHOLD: usize = 10;
        let n_rows = df.height();
        if n_rows > ROW_PARALLEL_THRESHOLD {
            (0..n_rows).into_par_iter()
                .map(|row_idx| {
                    match self.extract_row_features(df, row_idx) {
                        Ok(features) => self.predict(&features),
                        Err(_) => f64::NAN,
                    }
                })
                .collect()
        } else {
            (0..n_rows)
                .map(|row_idx| {
                    match self.extract_row_features(df, row_idx) {
                        Ok(features) => self.predict(&features),
                        Err(_) => f64::NAN,
                    }
                })
                .collect()
        }
    }

    fn round_rating_factor(&self, num_decimals: i32) -> RatingTable {
        // Create a copy of the DataFrame
        let mut new_df = self.data.clone();
        
        // Get the Rating_Factor column, round it, and replace it in the DataFrame
        let rounded = new_df
            .column("Rating_Factor")
            .unwrap()
            .f64()
            .unwrap()
            .apply(|x| match x {
                Some(val) => Some((val * 10f64.powi(num_decimals)).round() / 10f64.powi(num_decimals)),
                None => None
            })
            .into_series();
        
        new_df.with_column(rounded).unwrap();
        
        // Create and return a new RatingTable with the rounded values
        RatingTable::new(new_df, None)
    }

    // Helper methods
    #[inline]
    fn has_all_required_features(&self, feature_values: &HashMap<String, FeatureValue>) -> bool {
        // Check that all features required by the table are present in the input
        self.numeric_columns.keys().all(|k| feature_values.contains_key(k)) &&
        self.categorical_columns.keys().all(|k| feature_values.contains_key(k))
    }

    // Add new method to get feature info
    pub fn get_feature_info(&self) -> HashMap<String, DataType> {
        self.data.get_column_names().iter()
            .filter(|&name| *name != "Rating_Factor")
            .map(|name| {
                (name.to_string(), self.data.column(name).unwrap().dtype().clone())
            })
            .collect()
    }

    // Add extract_row_features method to RatingTable
    fn extract_row_features(&self, df: &DataFrame, row_idx: usize) -> Result<HashMap<String, FeatureValue>, PolarsError> {
        let mut feature_values = HashMap::new();
        
        for col_name in df.get_column_names() {
            let column = df.column(col_name)?;
            match column.dtype() {
                DataType::Float64 => {
                    if let Some(value) = column.f64()?.get(row_idx) {
                        feature_values.insert(col_name.to_string(), FeatureValue::Numeric(value));
                    }
                },
                DataType::Int32 => {
                    if let Some(value) = column.i32()?.get(row_idx) {
                        feature_values.insert(col_name.to_string(), FeatureValue::Categorical(value));
                    }
                },
                _ => continue,
            }
        }
        
        Ok(feature_values)
    }

    pub fn one_way_analysis_table(
        &self, 
        df: &DataFrame,  // ⭐ REFERENCE
        target_column: &str,
        weight_column: Option<&str>
    ) -> Result<DataFrame, PolarsError> {
        crate::analysis::one_way_analysis_table(self, df, target_column, weight_column)  // ⭐ NO CLONE
    }
}

// RatingModel holds multiple RatingTables
#[derive(Clone)]
pub struct RatingModel {
    pub tables: Vec<RatingTable>,
    pub link_function: LinkFunction,
}
impl RatingModel {
    // Private Constructor for internal use
    pub fn new(tables: Vec<RatingTable>, link_function: LinkFunction) -> Self {
        //check license
        if !validate_current_license() {
            panic!("License not valid");
        }
        Self { 
            tables,
            link_function,
        }
    }

    fn get_link_from_model_json(model_json: &str) -> Result<LinkFunction, serde_json::Error> {
        let model_json: Value = serde_json::from_str(model_json)?;
        let objective = model_json.get("objective")
            .and_then(|v| v.as_str())
            .unwrap_or("regression");
        Ok(LinkFunction::from_objective(objective))
    }

    //Constructor method from lgbm json
    pub fn from_lgbm_json(model_json: &str, consolidation_level: &str) -> Result<Self, PolarsError> {
        //check license
        if !validate_current_license() {
            panic!("License not valid");
        }
        let tables = process_lgbm_trees(model_json).map_err(|e| PolarsError::ComputeError(format!("Error processing trees: {}", e).into()))?;
        let link_function = Self::get_link_from_model_json(model_json).map_err(|e| PolarsError::ComputeError(format!("Error getting link function: {}", e).into()))?;
        
        match consolidation_level.to_lowercase().as_str() {
            "max" => Ok(build_consolidated_tablemodel(tables, link_function)),
            "analysis" => Ok(build_analysis_tablemodel(model_json, link_function)?),
            _ => Err(PolarsError::ComputeError(
                format!("Invalid consolidation_level '{}'. Must be 'max' or 'analysis'", 
                    consolidation_level).into()
            ))
        }
    }

    // Constructor from a list of DataFrames
    pub fn from_dataframes(
        tables: Vec<DataFrame>, 
        link_function: &str,
        feature_columns: Option<Vec<String>>,
        existing_row_number_col: Option<&str>
    ) -> Result<Self, PolarsError> {
        //check license
        if !validate_current_license() {
            panic!("License not valid");
        }

        let link_function = LinkFunction::from_objective(link_function);
        
        let rating_tables = tables.into_iter()
            .map(|df| {
                // If feature columns are specified, select only those plus Rating_Factor
                let filtered_df = if let Some(features) = &feature_columns {
                    let mut cols = features.clone();
                    cols.push("Rating_Factor".to_string());
                    // Also include row_number column if it exists
                    if let Some(row_col) = existing_row_number_col {
                        cols.push(row_col.to_string());
                    }
                    df.select(&cols)
                } else {
                    Ok(df)
                }?;
                
                Ok(RatingTable::new(filtered_df, existing_row_number_col))
            })
            .collect::<Result<Vec<RatingTable>, PolarsError>>()?;
        
        Ok(RatingModel::new(rating_tables, link_function))
    }

    pub fn model_tables(&self) -> Vec<DataFrame> {
        //check license
        if !validate_current_license() {
            panic!("License not valid");
        }
        self.tables.iter().map(|t| t.data.clone()).collect()
    }

    /// Consolidates all tables in the model (except the mean table) into a minimal set of combined tables
    pub fn consolidate_tables(&self) -> Self {
        // Keep the mean table (first table) separate
        let mean_table = self.tables[0].clone();
        
        // Combine all other tables
        let combined_tables = combine_all_tables(self.tables[1..].to_vec());
        
        // Create new model with mean table and combined tables
        let mut final_tables = vec![mean_table];
        final_tables.extend(combined_tables);
        
        RatingModel::new(final_tables, self.link_function.clone())
    }

    /// Predicts a single row of features
    pub fn predict_one(&self, feature_values: &HashMap<String, f64>) -> f64 {
        // Validate features first
        if let Err(err) = self.validate_features(feature_values) {
            panic!("Feature validation failed: {}", err);
        }

        let adjustment: f64 = self.tables.iter().enumerate()
            .map(|(i,table)| {
                let converted: HashMap<_, _> = feature_values.iter()
                    .filter_map(|(k, &v)| {
                        if let Ok(col) = table.data.column(k) {
                            let feature_value = match col.dtype() {
                                DataType::Int32 => FeatureValue::Categorical(v as i32),
                                DataType::Float64 => FeatureValue::Numeric(v),
                                _ => return None
                            };
                            Some((k.clone(), feature_value))
                        } else {
                            None
                        }
                    })
                    .collect();
                let pred = table.predict(&converted);
                println!("Table {}: {}", i, pred);
                pred
            })
            .sum();
        
        self.link_function.inverse(adjustment)
    }


    pub fn predict_linear(&self, df: &DataFrame) -> Result<Vec<f64>, PolarsError> {
        // Predicts values without applying the inverse link function

        // Validate DataFrame columns
        let required_features: HashMap<String, DataType> = self.tables[1..].iter()
            .flat_map(|table| table.get_feature_info())
            .collect();

        // Check for missing columns
        let missing_cols: Vec<_> = required_features.keys()
            .filter(|col| !df.get_column_names().iter().any(|c| c.as_str() == col.as_str()))
            .collect();
            
        if !missing_cols.is_empty() {
            return Err(PolarsError::ComputeError(
                format!("Missing required columns: {}", missing_cols.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")).into()
            ));
        }

        // Check column types
        for (col, expected_type) in &required_features {
            let df_col = df.column(col)?;
            if df_col.dtype() != expected_type {
                return Err(PolarsError::ComputeError(
                    format!("Column '{}' has type {:?}, expected {:?}", 
                        col, df_col.dtype(), expected_type).into()
                ));
            }
        }

        //check license
        if !validate_current_license() {
            panic!("License not valid");
        }
        let n_rows = df.height();
        let n_tables = self.tables.len();
        
        // Thresholds could be tuned based on benchmarking
        const ROW_PARALLEL_THRESHOLD: usize = 10;
        const TABLE_PARALLEL_THRESHOLD: usize = 10;
        
        let error_holder = Mutex::new(None);
        
        let predictions: Vec<f64> = match (n_rows > ROW_PARALLEL_THRESHOLD, n_tables > TABLE_PARALLEL_THRESHOLD) {
            (true, false) => { 
                // Many rows, few tables - parallelize rows only
                (0..n_rows).into_par_iter()
                    .map(|row_idx| self.predict_row_sequential(df, row_idx, &error_holder))
                    .collect()
            },
            (false, true) => {
                // Few rows, many tables - parallelize tables only
                (0..n_rows).map(|row_idx| {
                    self.predict_row_parallel(df, row_idx, &error_holder)
                }).collect()
            },
            (true, true) => {
                // Many of both - parallelize both
                (0..n_rows).into_par_iter()
                    .map(|row_idx| self.predict_row_parallel(df, row_idx, &error_holder))
                    .collect()
            },
            (false, false) => {
                // Few of both - no parallelization
                (0..n_rows).map(|row_idx| {
                    self.predict_row_sequential(df, row_idx, &error_holder)
                }).collect()
            }
        };

        Ok(predictions)
    }

    pub fn predict(&self, df: &DataFrame) -> Result<Series, PolarsError> {
        // Predicts values, applying the inverse link function
        let linear_predictions = self.predict_linear(df)?;
        let transformed_predictions = self.apply_link_function(linear_predictions);
        Ok(Series::new("predictions".into(), transformed_predictions))
    }

    pub fn get_link_function(&self) -> String {
        self.link_function.to_string()
    }

    pub fn round_rating_factors(&self, num_decimals: i32) -> RatingModel {
        RatingModel::new(self.tables.iter().map(|t| t.round_rating_factor(num_decimals)).collect(), self.link_function.clone())
    }

    pub fn validate_features(&self, features: &HashMap<String, f64>) -> Result<(), String> {
        // Collect all unique features and their types across all tables
        let mut required_features: HashMap<String, DataType> = HashMap::new();
        
        for table in &self.tables[1..] { // Skip mean table
            for (feat, dtype) in table.get_feature_info() {
                required_features.entry(feat).or_insert(dtype);
            }
        }

        // Check if all required features are present
        let missing_features: Vec<_> = required_features.keys()
            .filter(|feat| !features.contains_key(*feat))
            .collect();
            
        if !missing_features.is_empty() {
            return Err(format!(
                "Missing required features: {}. Required features are: {}", 
                missing_features.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
                required_features.keys().cloned().collect::<Vec<_>>().join(", ")
            ));
        }

        // Validate types (all required features should be convertible to their expected type)
        for (feat, dtype) in &required_features {
            let value = features[feat];
            match dtype {
                DataType::Int32 => {
                    if value.fract() != 0.0 {
                        return Err(format!(
                            "Feature '{}' expects integer values, got {}", 
                            feat, value
                        ));
                    }
                },
                DataType::Float64 => {
                    // All f64 values are valid
                    continue;
                },
                _ => {
                    return Err(format!(
                        "Unsupported data type {:?} for feature '{}'", 
                        dtype, feat
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn apply_link_function(&self, linear_predictions: Vec<f64>) -> Vec<f64> {
        linear_predictions.par_iter().map(|v| self.link_function.inverse(*v)).collect::<Vec<f64>>()
    }

    pub fn one_way_analysis(&self, df: &DataFrame, target_column: &str, weight_column: Option<&str>) -> Result<Vec<DataFrame>, PolarsError> {
        crate::analysis::one_way_analysis(self, df, target_column, weight_column)  // ⭐ NO CLONE
    }
}

impl Add for RatingModel {
    type Output = Result<Self, PolarsError>;

    fn add(self, other: Self) -> Self::Output {
        self.combine(&other)
    }
}

impl RatingModel {
    /// Combines two RatingModels into a single consolidated model
    /// 
    /// # Examples
    /// ```
    /// # use your_crate_name::RatingModel;
    /// # let model1 = RatingModel::from_lgbm_json(/* ... */).unwrap();
    /// # let model2 = RatingModel::from_lgbm_json(/* ... */).unwrap();
    /// 
    /// // Using the combine method
    /// let combined = model1.combine(&model2);
    /// 
    /// // Or using the + operator
    /// let combined = model1 + model2;
    /// ```
    pub fn combine(&self, other: &RatingModel) -> Result<Self, PolarsError> {
        // First check that link functions match
        if self.link_function.to_string() != other.link_function.to_string() {
            return Err(PolarsError::ComputeError(
                format!("Cannot combine models with different link functions: {} and {}", 
                    self.link_function.to_string(), 
                    other.link_function.to_string()
                ).into()
            ));
        }

        // Sum the mean tables if they exist
        let combined_mean = {
            let mean1 = if !self.tables.is_empty() {
                self.tables[0].data
                    .column("Rating_Factor")  // Changed from "overall_mean"
                    .ok()
                    .and_then(|col| col.f64().ok())
                    .and_then(|series| series.get(0))
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            
            let mean2 = if !other.tables.is_empty() {
                other.tables[0].data
                    .column("Rating_Factor")  // Changed from "overall_mean"
                    .ok()
                    .and_then(|col| col.f64().ok())
                    .and_then(|series| series.get(0))
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            
            RatingTable ::new(
                DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![mean1 + mean2]).into()]).unwrap(),
                None
            )
        };
        
        // Start with the combined mean table
        let mut combined_tables = vec![combined_mean];
        
        // Combine all other tables from both models
        let mut tables_to_combine = Vec::new();
        if self.tables.len() > 1 {
            tables_to_combine.extend(self.tables[1..].iter().cloned());
        }
        if other.tables.len() > 1 {
            tables_to_combine.extend(other.tables[1..].iter().cloned());
        }
        
        // Use existing combine_all_tables function
        combined_tables.extend(combine_all_tables(tables_to_combine));
        
        Ok(RatingModel::new(combined_tables, self.link_function.clone()))
    }

    /// Combines multiple RatingModels into a single consolidated model
    /// 
    /// # Examples
    /// ```
    /// # use your_crate_name::RatingModel;
    /// # let model1 = RatingModel::from_lgbm_json(/* ... */).unwrap();
    /// # let model2 = RatingModel::from_lgbm_json(/* ... */).unwrap();
    /// # let model3 = RatingModel::from_lgbm_json(/* ... */).unwrap();
    /// 
    /// let combined = RatingModel::combine_many(vec![model1, model2, model3]);
    /// ```
    pub fn combine_many(models: Vec<RatingModel>) -> Result<Self, PolarsError> {
        if models.is_empty() {
            return Err(PolarsError::ComputeError("No models to combine".into()));
        }
        
        let mut iter = models.into_iter();
        let first = iter.next().unwrap();
        iter.try_fold(first, |acc, model| acc + model)
    }

    fn predict_row_sequential(&self, df: &DataFrame, row_idx: usize, 
                            error_holder: &Mutex<Option<PolarsError>>) -> f64 {
        if error_holder.lock().unwrap().is_some() {
            return 0.0;
        }

        match self.tables[0].extract_row_features(df, row_idx) {
            Ok(feature_map) => {
                let adjustment: f64 = self.tables.iter()
                    .map(|table| table.predict(&feature_map))
                    .sum();
                adjustment
            },
            Err(e) => {
                handle_error(error_holder, row_idx, e);
                0.0
            }
        }
    }

    fn predict_row_parallel(&self, df: &DataFrame, row_idx: usize, 
                          error_holder: &Mutex<Option<PolarsError>>) -> f64 {
        if error_holder.lock().unwrap().is_some() {
            return 0.0;
        }

        match self.tables[0].extract_row_features(df, row_idx) {
            Ok(feature_map) => {
                let adjustment: f64 = self.tables.par_iter()
                    .map(|table| table.predict(&feature_map))
                    .sum();
                adjustment
            },
            Err(e) => {
                handle_error(error_holder, row_idx, e);
                0.0
            }
        }
    }
}
#[derive(Debug, Clone)]
struct SplitNodeInfo {
    feature_name: String,
    threshold: f64,
    decision_type: String,
    is_categorical: bool,
    categories: Vec<i32>,
}
#[derive(Debug, Clone)]
struct PathInfo {
    path: Vec<SplitNodeInfo>,
    is_in_first_tree: bool,
    mean_adjustment: f64,
}
impl PathInfo {
    fn new(path: Vec<SplitNodeInfo>, is_in_first_tree: bool, mean_adjustment: f64) -> Self {
        Self { path, is_in_first_tree, mean_adjustment }
    }

    fn create_df(&self) -> Result<DataFrame, PolarsError> {
        // Initialize maps to collect values for each feature
        let mut numeric_values: HashMap<String, Vec<f64>> = HashMap::new();
        let mut categorical_values: HashMap<String, Vec<i32>> = HashMap::new();

        // Process each node in the path to collect feature values
        for node in &self.path {
            if node.is_categorical {
                let values = categorical_values
                    .entry(node.feature_name.clone())
                    .or_insert_with(Vec::new);
                values.push(-999); // Always include wildcard
                values.extend(&node.categories);
            } else {
                let values = numeric_values
                    .entry(node.feature_name.clone())
                    .or_insert_with(Vec::new);
                values.push(node.threshold);
                // Only add infinity if this is the last threshold for this feature
                if !self.path.iter().any(|n| 
                    n.feature_name == node.feature_name && n.threshold > node.threshold
                ) {
                    values.push(f64::INFINITY);
                }
            }
        }

        // Sort and dedupe values
        for values in numeric_values.values_mut() {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));
            values.dedup_by(|a, b| ((*a) - (*b)).abs() < 1e-10 || ((*a).is_infinite() && (*b).is_infinite()));
        }
        for values in categorical_values.values_mut() {
            values.sort_unstable();
            values.dedup();
        }

        // Convert categorical to feature_values map
        let mut feature_values: HashMap<String, Vec<f64>> = numeric_values;
        for (feature, values) in categorical_values {
            feature_values.insert(
                feature,
                values.into_iter().map(|x| x as f64).collect()
            );
        }

        // Generate all combinations
        let mut combinations = vec![HashMap::new()];
        for (feature, values) in &feature_values {
            let mut new_combinations = Vec::new();
            for combo in combinations {
                for &value in values {
                    let mut new_combo = combo.clone();
                    new_combo.insert(feature.clone(), value);
                    new_combinations.push(new_combo);
                }
            }
            combinations = new_combinations;
        }

        // Convert to DataFrame
        let mut series_vec = Vec::new();
        if !combinations.is_empty() {
            // Create series for each feature
            for (feature, _) in &feature_values {
                let values: Vec<f64> = combinations.iter()
                    .map(|combo| *combo.get(feature).unwrap())
                    .collect();
                // Check if this feature is categorical by looking at path nodes
                let is_categorical = self.path.iter()
                    .any(|node| node.feature_name == *feature && node.is_categorical);

                if is_categorical {
                    // Convert to i32 for categorical columns
                    let cat_values: Vec<i32> = values.iter()
                        .map(|&x| x as i32)
                        .collect();
                    series_vec.push(Series::new(feature.into(), cat_values).into());
                } else {
                    series_vec.push(Series::new(feature.into(), values).into());
                }
            }

            // Add Rating_Factor column (initialized to 0.0, will be updated later)
            series_vec.push(Series::new(
                "Rating_Factor".into(),
                vec![0.0; combinations.len()]
            ).into());
        }

        DataFrame::new(series_vec)
    }
}

struct LeafNodeInfo {
    leaf_value: f64,
    path_info: PathInfo
}
impl LeafNodeInfo {
    fn new(leaf_value: f64, path_info: PathInfo) -> Self {
        Self { leaf_value, path_info }
    }
    fn create_rating_table(&self) -> Result<RatingTable, PolarsError> {
        let mut df = self.path_info.create_df()?;
        let mut mask = Series::new("mask".into(), vec![true; df.height()]);
        
        for node in &self.path_info.path {
            let col = df.column(&node.feature_name)?;
            
            let node_mask = if node.is_categorical {
                let values = col.cast(&DataType::Int32)?;
                match node.decision_type.as_str() {
                    "==" => {
                        // Handle empty categories case
                        if node.categories.is_empty() {
                            // If no categories specified, nothing matches (all false)
                            Series::new("empty_mask".into(), vec![false; df.height()])
                        } else {
                            // For left branch, match when the value is among the categories
                            node.categories.iter().fold(None, |acc, &cat| {
                                let cat_series = Series::new("cat".into(), vec![cat; values.len()]);
                                let eq = values.equal(&cat_series.into()).unwrap();
                                Some(match acc {
                                    Some(a) => a | eq,
                                    None => eq,
                                })
                            }).unwrap_or_else(|| {
                                // Should never reach here if we checked for empty above,
                                // but providing a fallback just in case
                                Series::new("fallback_mask".into(), vec![false; df.height()]).bool().unwrap().clone()
                            }).into_series()
                        }
                    },
                    "!=" => {
                        // Handle empty categories case
                        if node.categories.is_empty() {
                            // If no categories specified, everything matches (all true)
                            Series::new("empty_mask".into(), vec![true; df.height()])
                        } else {
                            // For right branch, match when the value is NOT among the categories
                            let not_in_categories = node.categories.iter().fold(None, |acc, &cat| {
                                let cat_series = Series::new("cat".into(), vec![cat; values.len()]);
                                let eq = values.equal(&cat_series.into()).unwrap();
                                Some(match acc {
                                    Some(a) => a & !eq,
                                    None => !eq,
                                })
                            }).unwrap_or_else(|| {
                                // Fallback - should not be needed but safer
                                Series::new("fallback_mask".into(), vec![true; df.height()]).bool().unwrap().clone()
                            });
                            
                            // Also match -999 for wildcard
                            let wildcard_series = Series::new("wildcard".into(), vec![-999; values.len()]);
                            let wildcard = values.equal(&wildcard_series.into())?;
                            (not_in_categories | wildcard).into_series()
                        }
                    },
                    _ => return Err(PolarsError::ComputeError(
                        format!("Invalid categorical decision type: {}", node.decision_type).into()
                    ))
                }
            } else {
                let values = col.cast(&DataType::Float64)?;
                let threshold_series = Series::new("threshold".into(), vec![node.threshold; values.len()]);
                match node.decision_type.as_str() {
                    "<=" => values.lt_eq(&threshold_series.into())?.into_series(),
                    ">" => values.gt(&threshold_series.into())?.into_series(),
                    _ => return Err(PolarsError::ComputeError(
                        format!("Invalid decision type: {}", node.decision_type).into()
                    ))
                }
            };

            mask = (mask.bool()? & node_mask.bool()?).into_series();
        }

        let rating_factors: Vec<f64> = mask.bool()?.into_iter()
            .map(|v| match v {
                Some(true) => {
                    if self.path_info.is_in_first_tree {
                        self.leaf_value - self.path_info.mean_adjustment
                    } else {
                        self.leaf_value
                    }
                },
                _ => 0.0,
            })
            .collect();

        df.with_column(Series::new("Rating_Factor".into(), rating_factors))?;
        Ok(RatingTable::new(df,None))
    }
    
    
}


// Return RatingTables instead of modifying a vector
fn process_tree(
    node: &Value,
    is_first_tree: bool,
    mean_adjustment: f64,
    model: &Value
) -> Result<Vec<RatingTable>, PolarsError> {
    let mut tables = Vec::new();
    let mut stack = vec![(node, Vec::new(), true)];
    
    while let Some((current_node, path, is_left)) = stack.pop() {
        if current_node.get("leaf_index").is_some() {
            // Process leaf node
            let leaf_value = current_node["leaf_value"].as_f64()
                .ok_or_else(|| PolarsError::ComputeError("Missing leaf value".into()))?;
            
            // Create PathInfo for this leaf
            let path_info = PathInfo::new(
                path.clone(),
                is_first_tree,
                mean_adjustment
            );

            // Create LeafNodeInfo
            let leaf_info = LeafNodeInfo::new(
                leaf_value,
                path_info,

            );

            // Create rating table and collect it
            let rating_table = leaf_info.create_rating_table()?;
            tables.push(rating_table);
            
        } else {
            // Process internal node (split node)
            let feature_idx = current_node["split_feature"].as_i64()
                .ok_or_else(|| PolarsError::ComputeError("Missing split feature".into()))? as usize;
            
            let feature_name = model["feature_names"][feature_idx]
                .as_str()
                .ok_or_else(|| PolarsError::ComputeError("Missing feature name".into()))?
                .to_string();
            
            let decision_type = current_node["decision_type"]
                .as_str()
                .ok_or_else(|| PolarsError::ComputeError("Missing decision type".into()))?;
            
            let is_categorical = decision_type == "==";
            
            // Handle threshold/categories based on split type
            let (threshold, categories) = if is_categorical {
                let cats = match &current_node["threshold"] {
                    Value::String(s) => s.split("||")
                        .filter_map(|v| v.parse::<i32>().ok())
                        .collect::<Vec<i32>>(),
                    Value::Number(n) => vec![n.as_i64().unwrap() as i32],
                    _ => return Err(PolarsError::ComputeError("Invalid categorical threshold".into()))
                };
                (0.0, cats)
            } else {
                let thresh = match &current_node["threshold"] {
                    Value::String(s) => s.parse::<f64>()
                        .map_err(|e| PolarsError::ComputeError(format!("Invalid numeric threshold: {}", e).into()))?,
                    Value::Number(n) => n.as_f64().unwrap(),
                    _ => return Err(PolarsError::ComputeError("Missing threshold".into()))
                };
                (thresh, Vec::new())
            };

            // Create SplitNodeInfo for this node
            let split_info = SplitNodeInfo {
                feature_name: feature_name.clone(),
                threshold,
                decision_type: if is_categorical { "==" } else { "<=" }.to_string(),
                is_categorical,
                categories: categories.clone(),
            };

            // Process children
            if let Some(left_child) = current_node.get("left_child") {
                let mut left_path = path.clone();
                left_path.push(split_info.clone());
                stack.push((left_child, left_path, true));
            }

            if let Some(right_child) = current_node.get("right_child") {
                let mut right_path = path.clone();
                // For right path, adjust the decision type
                let mut right_split_info = split_info.clone();
                right_split_info.decision_type = if is_categorical { "!=" } else { ">" }.to_string();
                right_path.push(right_split_info);
                stack.push((right_child, right_path, false));
            }
        }
    }
    
    Ok(tables)
}

pub fn process_lgbm_trees(model_json: &str) -> Result<Vec<RatingTable>, PolarsError> {
    let model: Value = serde_json::from_str(model_json)
        .map_err(|e| PolarsError::ComputeError(format!("JSON parsing error: {}", e).into()))?;
    
    let mut tables = Vec::new();
    
    // Extract overall mean from first tree's root node
    let mean_adjustment = if let Some(first_tree) = model["tree_info"][0]["tree_structure"].as_object() {
        if let Some(mean) = first_tree["internal_value"].as_f64() {
            let mean_df = DataFrame::new(vec![
                Series::new("Rating_Factor".into(), vec![mean]).into()
            ])?;
            tables.push(RatingTable::new(mean_df,None));
            mean
        } else {
            0.0
        }
    } else {
        0.0
    };
    
    // Process each tree
    let tree_tables: Result<Vec<_>, _> = model["tree_info"].as_array()
        .ok_or_else(|| PolarsError::ComputeError("Missing tree_info array".into()))?
        .par_iter() // Use parallel iterator
        .enumerate()
        .map(|(tree_idx, tree_info)| {
            process_tree(
                &tree_info["tree_structure"],
                tree_idx == 0,
                mean_adjustment,
                &model
            )
        })
        .collect();

    // Combine all tables
    tables.extend(tree_tables?.into_iter().flatten());
    
    Ok(tables)
}


fn handle_error(error_holder: &Mutex<Option<PolarsError>>, row_idx: usize, e: PolarsError) {
    let mut error = error_holder.lock().unwrap();
    if error.is_none() {
        *error = Some(PolarsError::ComputeError(
            format!("Error processing row {}: {}", row_idx, e).into()
        ));
    }
}

pub fn expand_and_combine_tables(table1: &RatingTable, table2: &RatingTable) -> RatingTable {

    // Get unique feature values for each feature from both tables
    let mut numeric_values: HashMap<String, Vec<f64>> = HashMap::new();
    let mut categorical_values: HashMap<String, Vec<i32>> = HashMap::new();

    // Collect values from both tables
    for table in [table1, table2] {
        for col_name in table.data.get_column_names() {
            if col_name == "Rating_Factor" || col_name == "row_number"  { continue; }
            
            let col = table.data.column(col_name).unwrap();
            match col.dtype() {
                DataType::Float64 => {
                    let values = numeric_values.entry(col_name.to_string())
                        .or_insert_with(Vec::new);
                    col.f64()
                        .unwrap()
                        .into_iter()
                        .filter_map(|v| v)
                        .for_each(|v| values.push(v));
                },
                DataType::Int32 => {
                    let values = categorical_values.entry(col_name.to_string())
                        .or_insert_with(Vec::new);
                    col.i32()
                        .unwrap()
                        .into_iter()
                        .filter_map(|v| v)
                        .for_each(|v| values.push(v));
                },
                _ => continue,
            }
        }
    }


    // Dedupe and sort values
    for values in numeric_values.values_mut() {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();
    }

    for values in categorical_values.values_mut() {
        values.sort_unstable();
        values.dedup();
    }

    // Calculate the total size needed for combinations
    let total_combinations = numeric_values.values()
        .map(|v| v.len())
        .chain(categorical_values.values().map(|v| v.len()))
        .product::<usize>();

    // Pre-allocate combinations with calculated size
    let mut combinations = Vec::with_capacity(total_combinations);

    // Initialize combinations with first feature (either numeric or categorical)
    let mut processed_features = HashSet::new();

    if let Some((first_feature, first_values)) = numeric_values.iter().next() {
        for &value in first_values {
            let mut combo = HashMap::new();
            combo.insert(first_feature.clone(), FeatureValue::Numeric(value));
            combinations.push(combo);
        }
        processed_features.insert(first_feature.clone());
    } else if let Some((first_feature, first_values)) = categorical_values.iter().next() {
        for &value in first_values {
            let mut combo = HashMap::new();
            combo.insert(first_feature.clone(), FeatureValue::Categorical(value));
            combinations.push(combo);
        }
        processed_features.insert(first_feature.clone());
    }

    // Add remaining numeric features
    for (feature, values) in numeric_values.iter() {
        if processed_features.contains(feature) {
            continue;
        }
        let mut new_combinations = Vec::new();
        for combo in combinations {
            for &value in values {
                let mut new_combo = combo.clone();
                new_combo.insert(feature.clone(), FeatureValue::Numeric(value));
                new_combinations.push(new_combo);
            }
        }
        combinations = new_combinations;
        processed_features.insert(feature.clone());
    }

    // Add remaining categorical features
    for (feature, values) in categorical_values.iter() {
        if processed_features.contains(feature) {
            continue;
        }
        let mut new_combinations = Vec::new();
        for combo in combinations {
            for &value in values {
                let mut new_combo = combo.clone();
                new_combo.insert(feature.clone(), FeatureValue::Categorical(value));
                new_combinations.push(new_combo);
            }
        }
        combinations = new_combinations;
        processed_features.insert(feature.clone());
    }

    // Create the result table
    let mut table_data: Vec<polars::prelude::Column> = Vec::new();
    
    // Add feature columns
    for (feature, _values) in &numeric_values {
        let column_values: Vec<f64> = combinations.iter()
            .map(|combo| {
                if let Some(FeatureValue::Numeric(v)) = combo.get(feature) {
                    *v
                } else {
                    panic!("Missing numeric value for feature {}", feature);
                }
            })
            .collect();
        table_data.push(Series::new(feature.into(), column_values).into());
    }

    for (feature, _) in &categorical_values {
        let column_values: Vec<i32> = combinations.iter()
            .map(|combo| {
                if let Some(FeatureValue::Categorical(v)) = combo.get(feature) {
                    *v
                } else {
                    panic!("Missing categorical value for feature {}", feature);
                }
            })
            .collect();
        table_data.push(Series::new(feature.into(), column_values).into());
    }

    // Calculate rating factors in parallel
    let rating_factors: Vec<f64> = combinations.par_iter()
        .map(|combo| {
            let rf1 = table1.predict(combo);
            let rf2 = table2.predict(combo);
            rf1 + rf2
        })
        .collect();

    table_data.push(Series::new("Rating_Factor".into(), rating_factors).into());

    // Before creating DataFrame, verify all series have same length
    let series_len = if !table_data.is_empty() {
        table_data[0].len()
    } else {
        0
    };

    if !table_data.iter().all(|s| s.len() == series_len) {
        panic!("Mismatched series lengths");
    }

    let result = RatingTable::new(
        DataFrame::new(table_data).unwrap(),
        None
    );

    result
}

pub fn build_consolidated_tablemodel(tables: Vec<RatingTable>, link_function: LinkFunction) -> RatingModel {
    let mut combined_tables = vec![tables[0].clone()];
    let consolidated = combine_all_tables(tables[1..].to_vec());
    combined_tables.extend(consolidated);
    RatingModel::new(combined_tables, link_function)
}

/// Revised build_analysis_tablemodel that uses internal node and leaf values
/// from the LightGBM JSON to construct lower‐level (analysis) tables.
pub fn build_analysis_tablemodel(model_json: &str, link_function: LinkFunction) -> Result<RatingModel, PolarsError> {
    // Parse the model JSON.
    let model: Value = serde_json::from_str(model_json)
        .map_err(|e| PolarsError::ComputeError(format!("JSON parsing error: {}", e).into()))?;
    
    let mut tables: Vec<RatingTable> = Vec::new();

    // Extract the overall mean from the first tree's root internal value.
    // (This will become the mean table.)
    let mean_adjustment = if let Some(first_tree) = model["tree_info"][0]["tree_structure"].as_object() {
        if let Some(mean) = first_tree["internal_value"].as_f64() {
            let mean_df = DataFrame::new(vec![
                Series::new("Rating_Factor".into(), vec![mean]).into()
            ])?;
            let mean_table = RatingTable::new(mean_df, None);
            tables.push(mean_table);
            mean
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Process each tree using internal node/leaf values for analysis.
    if let Some(tree_info_array) = model["tree_info"].as_array() {
        for (tree_idx, tree_info) in tree_info_array.iter().enumerate() {
            // process_tree_analysis traverses a tree and builds RatingTables
            // from both internal and leaf nodes.
            let node_tables = process_tree_analysis(&tree_info["tree_structure"], tree_idx == 0, mean_adjustment, &model, None)?;
            tables.extend(node_tables);
        }
    } else {
        return Err(PolarsError::ComputeError("Missing tree_info array".into()));
    }

    // Combine tables that have overlapping feature sets if needed.
    let consolidated_tables = combine_all_tables_exact(tables);
    
    Ok(RatingModel::new(consolidated_tables, link_function))
}

/// Combines tables by merging those with overlapping features
pub fn combine_all_tables(mut tables: Vec<RatingTable>) -> Vec<RatingTable> {
    let mut made_changes = true;
    
    while made_changes {
        made_changes = false;
        let mut i = 0;
        
        while i < tables.len() {
            // Create parallel iterator for remaining tables
            let combinations: Vec<_> = ((i + 1)..tables.len())
                .into_par_iter()
                .filter_map(|j| {
                    let columns_i: HashSet<_> = tables[i].data.get_column_names().into_iter().collect();
                    let columns_j: HashSet<_> = tables[j].data.get_column_names().into_iter().collect();
                    
                    if columns_i.is_subset(&columns_j) || columns_j.is_subset(&columns_i) {
                        // Return index and combined table
                        Some((j, if columns_i.len() > columns_j.len() {
                            expand_and_combine_tables(&tables[i], &tables[j])
                        } else {
                            expand_and_combine_tables(&tables[j], &tables[i])
                        }))
                    } else {
                        None
                    }
                })
                .collect();
            
            // Apply combinations if any were found
            if let Some((j, combined)) = combinations.first() {
                // Remove the table at index j and replace table i with combined version
                tables.remove(*j);
                tables[i] = combined.clone();
                made_changes = true;
            }
            
            i += 1;
        }
    }
    
    tables
}

/// Traverses a LightGBM tree (provided as a JSON Value) and creates a collection
/// of RatingTables from internal nodes (using `internal_value`) and leaf nodes (using `leaf_value`).
/// The `is_first_tree` flag and `mean_adjustment` are passed along so that the first tree's
/// nodes can be treated specially (subtracting the overall mean, for example).
fn process_tree_analysis(
    node: &Value,
    is_first_tree: bool,
    mean_adjustment: f64,
    model: &Value,
    parent_value: Option<f64>
) -> Result<Vec<RatingTable>, PolarsError> {
    let mut tables = Vec::new();
    
    // Extract the internal value for the root node
    let root_internal_value = node["internal_value"].as_f64()
        .ok_or_else(|| PolarsError::ComputeError("Missing internal value in root node".into()))?;
    
    // Stack holds (node, path, parent_value, tree_level)
    // tree_level helps track which level of the tree we're in (0 = root)
    let mut stack = vec![(node, Vec::new(), parent_value, 0usize)];
    
    while let Some((current_node, path, parent_val, level)) = stack.pop() {
        // Get the current node's internal value if it exists
        let current_internal_value = current_node.get("internal_value")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        
        // Only create tables for non-root nodes
        if !path.is_empty() && current_node.get("internal_value").is_some() {
            // Calculate the effect value - difference from parent
            let effect_value = match parent_val {
                Some(parent_value) => current_internal_value - parent_value,
                None => current_internal_value
            };
            
            // Create a rating table for this internal node's effect
            let path_info = PathInfo::new(path.clone(), is_first_tree, mean_adjustment);
            let node_info = NodeInfo {
                effect_value,
                path_info,
            };
            tables.push(node_info.create_rating_table()?);
        }
        
        // Process leaf nodes
        if current_node.get("leaf_index").is_some() {
            let leaf_value = current_node["leaf_value"].as_f64()
                .ok_or_else(|| PolarsError::ComputeError("Missing leaf value".into()))?;
            
            // For leaf nodes, the effect is the deviation from its parent internal node
            let effect_value = match parent_val {
                Some(parent_value) => leaf_value - parent_value,
                None => leaf_value
            };
            
            // Create a rating table for the leaf effect
            let path_info = PathInfo::new(path.clone(), is_first_tree, mean_adjustment);
            let node_info = NodeInfo {
                effect_value,
                path_info,
            };
            tables.push(node_info.create_rating_table()?);
        }
        // Process internal nodes with splits
        else if current_node.get("split_feature").is_some() {
            // Extract split feature information
            let feature_idx = current_node["split_feature"].as_i64()
                .ok_or_else(|| PolarsError::ComputeError("Missing split feature".into()))? as usize;
            
            let feature_name = model["feature_names"][feature_idx]
                .as_str()
                .ok_or_else(|| PolarsError::ComputeError("Missing feature name".into()))?
                .to_string();
            
            let decision_type = current_node["decision_type"].as_str()
                .ok_or_else(|| PolarsError::ComputeError("Missing decision type".into()))?;
            
            let is_categorical = decision_type == "==";
            
            // Extract threshold/categories
            let (threshold, categories) = if is_categorical {
                // Handle categorical features
                let cats = match &current_node["threshold"] {
                    Value::String(s) => s.split("||")
                        .filter_map(|v| v.parse::<i32>().ok())
                        .collect(),
                    Value::Number(n) => vec![n.as_i64().unwrap() as i32],
                    _ => return Err(PolarsError::ComputeError("Invalid categorical threshold".into()))
                };
                (0.0, cats)
            } else {
                // Handle numeric features
                let thresh = match &current_node["threshold"] {
                    Value::String(s) => s.parse::<f64>()
                        .map_err(|e| PolarsError::ComputeError(
                            format!("Invalid numeric threshold: {}", e).into()
                        ))?,
                    Value::Number(n) => n.as_f64().unwrap(),
                    _ => return Err(PolarsError::ComputeError("Missing threshold".into()))
                };
                (thresh, Vec::new())
            };
            
            // Create split info for left branch
            let left_split_info = SplitNodeInfo {
                feature_name: feature_name.clone(),
                threshold,
                // Left branch decision type
                decision_type: if is_categorical { "==" } else { "<=" }.to_string(),
                is_categorical,
                categories: categories.clone(),
            };
            
            // Create split info for right branch with proper decision type
            let right_split_info = SplitNodeInfo {
                feature_name, 
                threshold,
                // Right branch decision type - critical for correct path traversal
                decision_type: if is_categorical { "!=" } else { ">" }.to_string(),
                is_categorical,
                categories,
            };
            
            // Process left child
            if let Some(left_child) = current_node.get("left_child") {
                let mut left_path = path.clone();
                // Add left split info to path
                left_path.push(left_split_info);
                // Pass current internal value as parent for the child
                stack.push((left_child, left_path, Some(current_internal_value), level + 1));
            }
            
            // Process right child
            if let Some(right_child) = current_node.get("right_child") {
                let mut right_path = path.clone();
                right_path.push(right_split_info);
                // Pass current internal value as parent for the child
                stack.push((right_child, right_path, Some(current_internal_value), level + 1));
            }
        }
    }
    
    Ok(tables)
}

#[derive(Debug, Clone)]
struct NodeInfo {
    effect_value: f64,
    path_info: PathInfo,
}

impl NodeInfo {
    /// Constructs a RatingTable for this node. This method:
    /// 1. Calls `create_df()` (from PathInfo) to generate a grid of feature values
    ///    corresponding to the node's path.
    /// 2. Constructs a boolean mask by applying each decision in the path.
    /// 3. Computes the Rating_Factor: if the mask is true, then, if the node comes from the first tree,
    ///    subtract the mean_adjustment from the effect_value; otherwise, use effect_value directly.
    /// 4. Builds and returns the new RatingTable.
    fn create_rating_table(&self) -> Result<RatingTable, PolarsError> {
        let mut df = self.path_info.create_df()?;
        let mut mask = Series::new("mask".into(), vec![true; df.height()]);
        
        for node in &self.path_info.path {
            let col = df.column(&node.feature_name)?;
            let node_mask = if node.is_categorical {
                let values = col.cast(&DataType::Int32)?;
                match node.decision_type.as_str() {
                    "==" => {
                        // Handle empty categories case
                        if node.categories.is_empty() {
                            // If no categories specified, nothing matches (all false)
                            Series::new("empty_mask".into(), vec![false; df.height()])
                        } else {
                            // For left branch, match when the value is among the categories
                            node.categories.iter().fold(None, |acc, &cat| {
                                let cat_series = Series::new("cat".into(), vec![cat; values.len()]);
                                let eq = values.equal(&cat_series.into()).unwrap();
                                Some(match acc {
                                    Some(a) => a | eq,
                                    None => eq,
                                })
                            }).unwrap_or_else(|| {
                                // Should never reach here if we checked for empty above,
                                // but providing a fallback just in case
                                Series::new("fallback_mask".into(), vec![false; df.height()]).bool().unwrap().clone()
                            }).into_series()
                        }
                    },
                    "!=" => {
                        // Handle empty categories case
                        if node.categories.is_empty() {
                            // If no categories specified, everything matches (all true)
                            Series::new("empty_mask".into(), vec![true; df.height()])
                        } else {
                            // For right branch, match when the value is NOT among the categories
                            let not_in_categories = node.categories.iter().fold(None, |acc, &cat| {
                                let cat_series = Series::new("cat".into(), vec![cat; values.len()]);
                                let eq = values.equal(&cat_series.into()).unwrap();
                                Some(match acc {
                                    Some(a) => a & !eq,
                                    None => !eq,
                                })
                            }).unwrap_or_else(|| {
                                // Fallback - should not be needed but safer
                                Series::new("fallback_mask".into(), vec![true; df.height()]).bool().unwrap().clone()
                            });
                            
                            // Also match -999 for wildcard
                            let wildcard_series = Series::new("wildcard".into(), vec![-999; values.len()]);
                            let wildcard = values.equal(&wildcard_series.into())?;
                            (not_in_categories | wildcard).into_series()
                        }
                    },
                    _ => return Err(PolarsError::ComputeError(
                        format!("Invalid categorical decision type: {}", node.decision_type).into()
                    ))
                }
            } else {
                // For numeric columns, cast to Float64 and compare.
                let values = col.cast(&DataType::Float64)?;
                let threshold_series = Series::new("threshold".into(), vec![node.threshold; values.len()]);
                match node.decision_type.as_str() {
                    "<=" => values.lt_eq(&threshold_series.into())?.into_series(),
                    ">" => values.gt(&threshold_series.into())?.into_series(),
                    _ => return Err(PolarsError::ComputeError(
                        format!("Invalid decision type: {}", node.decision_type).into()
                    ))
                }
            };
            mask = (mask.bool()? & node_mask.bool()?).into_series();
        }
        
        let rating_factors: Vec<f64> = mask.bool()?.into_iter()
            .map(|v| match v {
                Some(true) => {
                    self.effect_value
                },
                _ => 0.0,
            })
            .collect();
        df.with_column(Series::new("Rating_Factor".into(), rating_factors))?;
        Ok(RatingTable::new(df, None))
    }
}

pub fn combine_all_tables_exact(mut tables: Vec<RatingTable>) -> Vec<RatingTable> {
    let mut made_changes = true;
    while made_changes {
        made_changes = false;
        let mut i = 0;
        while i < tables.len() {
            let columns_i: HashSet<String> = tables[i]
                .data
                .get_column_names()
                .iter()
                .filter(|&n| *n != "Rating_Factor" && *n != "row_number")
                .map(|s| s.to_string())
                .collect();
            let mut found_index = None;
            for j in (i + 1)..tables.len() {
                let columns_j: HashSet<String> = tables[j]
                    .data
                    .get_column_names()
                    .iter()
                    .filter(|&n| *n != "Rating_Factor" && *n != "row_number")
                    .map(|s| s.to_string())
                    .collect();
                if columns_i == columns_j {
                    found_index = Some(j);
                    break;
                }
            }
            if let Some(j) = found_index {
                // Here we combine the two tables (assumed to have an identical feature layout)
                // using your existing 'expand_and_combine_tables' helper function.
                let new_table = expand_and_combine_tables(&tables[i], &tables[j]);
                tables.remove(j);
                tables[i] = new_table;
                made_changes = true;
            } else {
                i += 1;
            }
        }
    }
    tables
}


    