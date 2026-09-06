"""Optional, explicit input adapters; Polars remains the modeling boundary."""
import polars as pl


def from_pandas(data, *, index_column=None):
    """Convert a pandas frame without replacing categories by positional codes.

    String, integer and boolean categories retain their actual labels. Unused
    categorical levels are not learned by Plan fitting. Missing values remain null.
    Mixed-type/float categories and arbitrary Python objects are rejected rather
    than stringified. The pandas index is excluded unless index_column is provided.
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError('Install avenue_model[pandas] to use from_pandas') from error
    if not isinstance(data, pd.DataFrame):
        raise TypeError('from_pandas expects a pandas DataFrame; pass Polars frames directly to Avenue')
    if not data.columns.is_unique or any(not isinstance(name, str) for name in data.columns):
        raise ValueError('Pandas columns must have unique string names')
    frame = data.copy(deep=False)
    if index_column is not None:
        if not isinstance(index_column, str) or not index_column or index_column in frame.columns:
            raise ValueError('index_column must be a new nonempty column name')
        if isinstance(data.index, pd.MultiIndex):
            raise ValueError('Reset a MultiIndex into explicitly named columns before conversion')
        frame = frame.assign(**{index_column: data.index})
    for name in frame.columns:
        column = frame[name]
        if isinstance(column.dtype, pd.CategoricalDtype):
            labels = column.cat.categories
            if pd.api.types.is_bool_dtype(labels.dtype):
                frame = frame.assign(**{name: column.astype('boolean')})
            elif pd.api.types.is_integer_dtype(labels.dtype):
                dtype = 'UInt64' if pd.api.types.is_unsigned_integer_dtype(labels.dtype) else 'Int64'
                frame = frame.assign(**{name: column.astype(dtype)})
            elif all(isinstance(value, str) for value in labels):
                frame = frame.assign(**{name: column.astype('string')})
            else:
                raise TypeError(f'Categorical predictor {name!r} needs homogeneous string, integer or boolean labels; provide an explicit encoding for other category types')
        elif pd.api.types.is_object_dtype(column.dtype):
            if not all(isinstance(value, str) for value in column.dropna()):
                raise TypeError(f'Object column {name!r} must contain strings or nulls; cast numerical, date or mixed objects explicitly')
            frame = frame.assign(**{name: column.astype('string')})
    try:
        return pl.from_pandas(frame, include_index=False, nan_to_null=True)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise TypeError(f'Unsupported pandas input dtype: {error}') from error
