# Rating models

This module contains Avenue's core representation: lookup tables that can be inspected,
combined, predicted, and fitted directly as GLMs.

## At a glance

- `RatingTable` represents one lookup table and its matching rules.
- `RatingModel` combines tables under identity, log, or logit links.
- LightGBM trees can be converted exactly into the same representation.
- Tables can be consolidated without changing predictions.

## Module structure

```
rating_model/
├── mod.rs              - Core domain models and public API
├── lgbm_parser.rs      - LightGBM JSON parsing and tree processing
├── consolidation.rs    - Table combination and consolidation algorithms
└── README.md          - This file
```

## Core components

### Core types ([mod.rs](mod.rs))

#### `RatingTable`

- Represents a single rating table with features and rating factors
- Supports both numeric (Float64) and categorical (Int32) features
- Provides row matching and prediction functionality
- Handles metadata for offset tables and locked rows

#### `RatingModel`

- Contains multiple RatingTables with a link function
- Supports predictions on single records or DataFrames
- Can be constructed from LightGBM JSON models or DataFrames
- Supports model combination and consolidation

#### Link functions

- `Identity` - for regression models
- `Logit` - for binary classification
- `Log` - for Poisson, Gamma, and Tweedie models

#### Metadata

- `TableMetadata` - Table-level configuration (name, offset status, updatability)
- `RowMetadata` - Row-level configuration (offset/locked rows)

### LightGBM Parsing ([lgbm_parser.rs](lgbm_parser.rs))

Handles conversion of LightGBM gradient boosting models to rating tables:

- `process_lgbm_trees()` - Parses LightGBM JSON and extracts all trees
- `build_consolidated_tablemodel()` - Creates maximally consolidated model
- `build_analysis_tablemodel()` - Creates analysis-level model with internal nodes

Internal structures:

- `PathInfo` - Represents a path through a decision tree
- `SplitNodeInfo` - Split node information (threshold, feature, decision type)
- `LeafNodeInfo` - Leaf node with value
- `NodeInfo` - Internal node for analysis models

### Table Consolidation ([consolidation.rs](consolidation.rs))

Algorithms for combining and consolidating rating tables:

- `expand_and_combine_tables()` - Combines two tables with overlapping features
- `combine_all_tables()` - Iteratively combines tables with overlapping features
- `combine_all_tables_exact()` - Combines tables with identical feature sets

## Examples

### From LightGBM

```rust
use avenue_model::rating_model::RatingModel;

// Load from LightGBM JSON
let model = RatingModel::from_lgbm_json(
    model_json_str,
    "max"  // or "analysis" for more detailed tables
)?;

// Make predictions
let predictions = model.predict(&dataframe)?;
```

### From DataFrames

```rust
use avenue_model::rating_model::RatingModel;

let model = RatingModel::from_dataframes(
    vec![table1_df, table2_df],
    "regression",  // objective
    None,         // feature_columns (None = use all)
    None          // existing_row_number_col
)?;
```

### Combine models

```rust
// Combine two models with the same link function
let combined = model1.combine(&model2)?;

// Or use the + operator
let combined = (model1 + model2)?;

// Combine multiple models
let combined = RatingModel::combine_many(vec![model1, model2, model3])?;
```

### Offset tables and locked rows

```rust
// Mark entire table as offset (not updated by GLM)
let offset_table = table.as_offset();

// Add offset table to model
model.add_offset_table(offset_table);

// Lock specific rows
table.set_row_offset(5, true);  // Lock row 5
```

### Consolidate tables

```rust
// Consolidate all tables in a model
let consolidated = model.consolidate_tables();
```

## Feature Matching

### Categorical Features

- Use Int32 type
- Support wildcard matching with `-999` value
- Exact match required unless wildcard present

### Numeric Features

- Use Float64 type
- Threshold-based matching (value ≤ threshold)
- Support infinity thresholds for unbounded ranges

## Performance

The module uses several optimization strategies:

- **Parallel Processing**: Batch predictions use Rayon for parallelization
- **Cached Columns**: Column indices cached to avoid repeated lookups
- **Unsafe Access**: Row matching uses unsafe `get_unchecked` for performance
- **Adaptive Parallelization**: Automatically chooses parallel strategy based on data size

Parallelization thresholds:

- `ROW_PARALLEL_THRESHOLD = 10` - Minimum rows for parallel processing
- `TABLE_PARALLEL_THRESHOLD = 10` - Minimum tables for parallel processing

## Public API

All core types and functions are re-exported from `mod.rs`:

```rust
// Core models
pub use RatingTable;
pub use RatingModel;
pub use LinkFunction;
pub use FeatureValue;
pub use TableMetadata;
pub use RowMetadata;

// LightGBM parsing
pub use process_lgbm_trees;
pub use build_analysis_tablemodel;
pub use build_consolidated_tablemodel;

// Consolidation
pub use expand_and_combine_tables;
pub use combine_all_tables;
```

## Testing

Run the module tests with:

```bash
cargo test --lib
```

Coverage includes:

- Model creation and prediction tests
- LightGBM parsing tests
- Table consolidation tests
- Edge case handling
- Performance benchmarks
