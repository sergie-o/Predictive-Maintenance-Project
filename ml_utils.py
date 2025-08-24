# ml_utils.py


# create_model
# X = country_houses.drop(columns=["price", "id"])
# y = country_houses["price"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# results_linreg = create_model(LinearRegression(), X_train, X_test, y_train, y_test)

# birdseye_view
# birdseye_view(country_houses, n_head=10, show_value_counts=True)

# conduct_univariate
# conduct_univariate(country_houses, "bedrooms", chart="countplot")
# conduct_univariate(country_houses, "price", chart="histogram")

# conduct_bivariate
# conduct_bivariate(country_houses, "sqft_living", "price", chart="scatter")
# onduct_bivariate(country_houses, "grade", "price", chart="barchart_mean")
# conduct_bivariate(country_houses, "condition", "view", chart="countplot_grouped")


# numerical_distribution
# numerical_distribution(country_houses, ncols=4, bins=30, color="#FFCC00")


# numerical_boxplots
# numerical_boxplots(country_houses, ncols=4, color="#A600FF")

# correlation_heatmap
# correlation_heatmap(country_houses, method="pearson", cmap="coolwarm")

# missing_values_plot
# missing_values_plot(country_houses)

# feature_importance_plot
# rf = RandomForestRegressor().fit(X_train, y_train)
# feature_importance_plot(rf, X_train, top_n=15)

# distribution_compare
# distribution_compare(country_houses, feature="price", by="condition", kind="violin")

# compare_models_table
# results = []
# results.append(results_linreg)  # from create_model
# results.append(create_model(RandomForestRegressor(), X_train, X_test, y_train, y_test))

# leaderboard = compare_models_table(results, sort_by="R2 Test")
# print(leaderboard)


import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import math
from IPython.display import display
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Models (regression + a few extras if you need them later)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# ml_utils.py (add below your other functions)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_object_dtype


def _regression_metrics(y_true, y_pred):
    """Compute MSE, RMSE, R2 for a single split."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2


def create_model(
    X,
    y,
    model_type="linear",           # "linear", "ridge", "random_forest", "gboost", "xgboost"
    test_size=0.2,
    scale=False,
    random_state=42,
    ridge_alpha=1.0,               # ridge-only
    xgb_params=None,               # optional dict for XGBRegressor
    rf_params=None,                # optional dict for RandomForestRegressor
    gbr_params=None,               # optional dict for GradientBoostingRegressor
):
    """
    Train a regression model, predict on train & test, print and return metrics.

    Returns
    -------
    model : fitted estimator
    metrics : dict with Train/Test MSE, RMSE, R2
    (X_train, X_test, y_train, y_test) : the split (for convenience)
    """

    # 1) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2) Optional scaling
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 3) Pick model
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=ridge_alpha, random_state=random_state)
    elif model_type == "random_forest":
        params = {"n_estimators": 200, "random_state": random_state}
        if rf_params:
            params.update(rf_params)
        model = RandomForestRegressor(**params)
    elif model_type == "gboost":
        params = {"random_state": random_state}
        if gbr_params:
            params.update(gbr_params)
        model = GradientBoostingRegressor(**params)
    elif model_type == "xgboost":
        params = {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "tree_method": "hist",
        }
        if xgb_params:
            params.update(xgb_params)
        model = XGBRegressor(**params)
    else:
        raise ValueError(
            "Unknown model_type. Use: 'linear', 'ridge', 'random_forest', 'gboost', 'xgboost'.")

    # 4) Fit
    model.fit(X_train, y_train)

    # 5) Predict (train & test)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 6) Metrics (train & test)
    mse_train, rmse_train, r2_train = _regression_metrics(
        y_train, y_pred_train)
    mse_test, rmse_test, r2_test = _regression_metrics(y_test, y_pred_test)

    # 7) Print exactly like your notebook style
    print("MSE Test:", mse_test)
    print("MSE Train:", mse_train)
    print("RMSE Test:", rmse_test)
    print("RMSE Train:", rmse_train)
    print("R² Score Test:", r2_test)
    print("R² Score Train:", r2_train)

    # 8) Bundle metrics for results table
    metrics = {
        "MSE (Train)": mse_train,
        "MSE (Test)": mse_test,
        "RMSE (Train)": rmse_train,
        "RMSE (Test)": rmse_test,
        "R2 (Train)": r2_train,
        "R2 (Test)": r2_test,
    }

    return model, metrics, (X_train, X_test, y_train, y_test), scaler


# ml_utils.py (continued)

def birdseye_view(df, n_head=5, show_value_counts=False, max_unique=10):
    """
    Quick EDA overview of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect
    n_head : int
        How many rows to show for df.head()
    show_value_counts : bool
        If True, show value_counts for categorical columns (up to max_unique unique values)
    max_unique : int
        Maximum unique values to print value_counts for each column
    """

    print("="*80)
    print("🔍 DataFrame Shape:", df.shape)
    print("="*80, "\n")

    print("👀 First Rows (head):")
    display(df.head(n_head))
    print("\n")

    print("="*80)
    print("📊 Data Types & Non-Null Counts (.info()):")
    print("="*80)
    print(df.info())
    print("\n")

    print("="*80)
    print("📈 Descriptive Statistics (.describe()):")
    print("="*80)
    display(df.describe(include="all").transpose())
    print("\n")

    print("="*80)
    print("🧮 Missing Values (NaN) per column:")
    print("="*80)
    print(df.isnull().sum())
    print("\n")

    print("="*80)
    print("📦 Number of Unique Values per column:")
    print("="*80)
    print(df.nunique())
    print("\n")

    print("="*80)
    print("🗂 Duplicate Rows:", df.duplicated().sum())
    print("="*80, "\n")

    if show_value_counts:
        print("📑 Value Counts (for categorical columns with <= max_unique unique values):")
        for col in df.columns:
            if df[col].dtype == "object" or df[col].nunique() <= max_unique:
                print(f"\n{col}:")
                print(df[col].value_counts(dropna=False))


# ml_utils.py (add below your other functions)

def conduct_univariate(
    df: pd.DataFrame,
    col: str,
    feature_type: str = "auto",   # "categorical", "numerical", or "auto"
    chart: str = "all",           # pick one option or "all"
    top_n: int = 15,
    bins: int = 30,
    dropna: bool = True,
    figsize=(8, 5),
    color=None
):
    """
    Univariate visualization helper.

    New chart aliases:
      - Categorical: "countplot", "barchart", "piechart"
      - Numerical: "histogram"

    Existing options:
      - Categorical: "count", "proportion", "barh", "pie"
      - Numerical: "hist_kde", "box", "violin", "ecdf"
    """

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    s = df[col].dropna() if dropna else df[col]

    # Infer feature type
    if feature_type == "auto":
        if is_numeric_dtype(s):
            feature_type = "numerical"
        else:
            feature_type = "categorical" if s.nunique() <= 30 else "numerical"

    # --- categorical ---
    if feature_type == "categorical":
        charts = [chart] if chart != "all" else [
            "countplot", "barchart", "piechart"]

        for ch in charts:
            if ch in ["countplot", "count"]:
                plt.figure(figsize=figsize)
                order = s.value_counts().index[:top_n]
                sns.countplot(y=s, order=order, color=color)
                plt.title(f"{col} — Countplot")
                plt.tight_layout()
                plt.show()

            elif ch in ["barchart", "barh"]:
                plt.figure(figsize=figsize)
                counts = s.value_counts().head(top_n).sort_values(ascending=True)
                counts.plot(kind="barh", color=color)
                plt.title(f"{col} — Barchart (Top {len(counts)})")
                plt.tight_layout()
                plt.show()

            elif ch in ["piechart", "pie"]:
                plt.figure(figsize=(figsize[0], figsize[0]))
                counts = s.value_counts().head(top_n)
                counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)
                plt.title(f"{col} — Piechart")
                plt.ylabel("")
                plt.tight_layout()
                plt.show()

    # --- numerical ---
    elif feature_type == "numerical":
        charts = [chart] if chart != "all" else [
            "histogram", "box", "violin", "ecdf"]

        s = pd.to_numeric(s, errors="coerce").dropna()

        for ch in charts:
            if ch in ["histogram", "hist_kde"]:
                plt.figure(figsize=figsize)
                sns.histplot(s, bins=bins, kde=True, color=color)
                plt.title(f"{col} — Histogram")
                plt.tight_layout()
                plt.show()

            elif ch == "box":
                plt.figure(figsize=figsize)
                sns.boxplot(x=s, color=color)
                plt.title(f"{col} — Boxplot")
                plt.tight_layout()
                plt.show()

            elif ch == "violin":
                plt.figure(figsize=figsize)
                sns.violinplot(x=s, color=color, inner="quartile")
                plt.title(f"{col} — Violin Plot")
                plt.tight_layout()
                plt.show()

            elif ch == "ecdf":
                x = np.sort(s.values)
                y = np.arange(1, len(x) + 1) / len(x)
                plt.figure(figsize=figsize)
                plt.plot(x, y, linewidth=2)
                plt.title(f"{col} — ECDF")
                plt.xlabel(col)
                plt.ylabel("Cumulative Probability")
                plt.grid(alpha=0.25)
                plt.tight_layout()
                plt.show()


def conduct_bivariate(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_type: str = "auto",
    y_type: str = "auto",
    chart: str = "all",
    top_n: int = 12,
    figsize=(8, 5),
    color=None,
    cmap="Blues"
):
    """
    Bivariate visualization helper.

    New chart aliases:
      - Categorical×Categorical: "countplot_grouped", "barchart_stacked"
      - Categorical×Numerical: "barchart_mean"
      - Numerical×Numerical: already had scatter/reg/etc.
    """

    if x not in df.columns or y not in df.columns:
        raise ValueError("x or y column not in DataFrame.")

    # infer types
    def infer(s):
        if is_numeric_dtype(s):
            return "numerical"
        return "categorical"

    if x_type == "auto":
        x_type = infer(df[x])
    if y_type == "auto":
        y_type = infer(df[y])

    # --- cat × num ---
    if x_type == "categorical" and y_type == "numerical":
        charts = [chart] if chart != "all" else [
            "box", "violin", "barchart_mean", "strip"]
        for ch in charts:
            if ch in ["barchart_mean"]:
                plt.figure(figsize=figsize)
                sns.barplot(data=df, x=x, y=y, estimator=np.mean,
                            ci=95, color=color)
                plt.title(f"{y} by {x} — Mean Barchart")
                plt.tight_layout()
                plt.show()

            elif ch == "box":
                sns.boxplot(data=df, x=x, y=y, color=color)
                plt.title(f"{y} by {x} — Boxplot")
                plt.show()

            elif ch == "violin":
                sns.violinplot(data=df, x=x, y=y,
                               inner="quartile", color=color)
                plt.title(f"{y} by {x} — Violin")
                plt.show()

            elif ch == "strip":
                sns.stripplot(data=df, x=x, y=y, alpha=0.5,
                              jitter=0.25, color=color)
                plt.title(f"{y} by {x} — Strip")
                plt.show()

    # --- cat × cat ---
    if x_type == "categorical" and y_type == "categorical":
        charts = [chart] if chart != "all" else [
            "countplot_grouped", "barchart_stacked"]
        for ch in charts:
            if ch == "countplot_grouped":
                plt.figure(figsize=figsize)
                sns.countplot(data=df, x=x, hue=y)
                plt.title(f"{x} by {y} — Grouped Countplot")
                plt.xticks(rotation=30)
                plt.tight_layout()
                plt.show()

            elif ch == "barchart_stacked":
                ct = pd.crosstab(df[x], df[y], normalize="index") * 100
                bottom = np.zeros(len(ct))
                plt.figure(figsize=figsize)
                for col_name in ct.columns:
                    plt.bar(ct.index, ct[col_name],
                            bottom=bottom, label=str(col_name))
                    bottom += ct[col_name].values
                plt.title(f"{x} by {y} — Stacked Barchart (%)")
                plt.legend(title=y, bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.show()


def numerical_distribution(
    df: pd.DataFrame,
    # which numeric cols to plot; None = auto-detect numeric
    columns: list[str] | None = None,
    ncols: int = 4,                      # number of columns in the subplot grid
    bins: int = 30,                      # histogram bins
    color: str = "#FFCC00",              # bar color
    edgecolor: str = "black",            # bar edge color
    figsize: tuple[int, int] = (20, 16),  # overall figure size
    dropna: bool = True,                 # drop NaNs before plotting
    sharex: bool = False,
    sharey: bool = False,
    suptitle: str | None = None          # optional big title for the figure
):
    """
    Plot a grid of histograms for numeric features in a DataFrame.

    Returns
    -------
    fig, axes : matplotlib Figure and flattened Axes array
    """
    # 1) pick numeric columns
    if columns is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_cols = [c for c in columns if c in df.columns]
        # filter to numeric
        num_cols = [
            c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]

    if len(num_cols) == 0:
        raise ValueError(
            "No numeric columns to plot. Provide numeric 'columns' or check your DataFrame dtypes.")

    # 2) grid sizing
    n_plots = len(num_cols)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, sharex=sharex, sharey=sharey)
    # axes could be 2D or 1D depending on nrows/ncols
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])  # single axis case

    # 3) plot each histogram
    for i, col in enumerate(num_cols):
        ax = axes[i]
        s = df[col]
        if dropna:
            s = s.dropna()
        ax.hist(s, bins=bins, color=color, edgecolor=edgecolor)
        ax.set_title(col)
        ax.grid(alpha=0.2)

    # 4) hide any unused axes
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.995)

    plt.tight_layout()
    plt.show()
    return fig, axes


def numerical_boxplots(
    df: pd.DataFrame,
    # numeric cols to plot; None = auto-detect
    columns: list[str] | None = None,
    ncols: int = 4,                       # number of subplot columns
    color: str = "#A600FF",               # box fill color
    figsize: tuple[int, int] = (20, 16),  # overall figure size
    dropna: bool = True,
    suptitle: str | None = None
):
    """
    Plot a grid of horizontal boxplots for numeric features in a DataFrame.

    Each boxplot:
      - facecolor = color (default purple)
      - black border, yellow median, black whiskers & caps
      - red outliers

    Returns
    -------
    fig, axes : matplotlib Figure and flattened Axes array
    """
    # 1) numeric columns
    if columns is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_cols = [c for c in columns if c in df.columns]
        num_cols = [
            c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]

    if len(num_cols) == 0:
        raise ValueError("No numeric columns found to plot.")

    # 2) grid sizing
    n_plots = len(num_cols)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])  # single axis case

    # 3) plot boxplots
    for i, col in enumerate(num_cols):
        ax = axes[i]
        s = df[col].dropna() if dropna else df[col]
        ax.boxplot(
            s,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor=color, color='black'),
            medianprops=dict(color='yellow'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(marker='o', color='red', markersize=5)
        )
        ax.set_title(col, fontsize=10)
        ax.tick_params(axis='x', labelsize=8)

    # 4) hide unused axes
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.995)

    plt.tight_layout()
    plt.show()
    return fig, axes


def correlation_heatmap(df, cols=None, method="pearson", figsize=(10, 8), cmap="coolwarm", mask_upper=True):
    """
    Plot a heatmap of correlations between numeric features.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns
    corr = df[cols].corr(method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, mask=mask, square=True)
    plt.title(f"{method.title()} Correlation Heatmap")
    plt.show()

    def missing_values_plot(df, figsize=(10, 6), color="#FF5733"):
        """
    Bar chart of missing values per column.
    """
    import matplotlib.pyplot as plt

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("✅ No missing values.")
        return

    plt.figure(figsize=figsize)
    missing.plot(kind="bar", color=color)
    plt.title("Missing Values per Column")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    def feature_importance_plot(model, X, top_n=15, figsize=(10, 6), color="#008B8B"):
        """
Plot top-N feature importances for tree-based models.
"""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute.")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    features = X.columns[indices]

    plt.figure(figsize=figsize)
    plt.barh(range(len(indices)), importances[indices], color=color)
    plt.yticks(range(len(indices)), features)
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    def distribution_compare(df, feature, by, kind="box", figsize=(8, 6)):
        """
Compare distribution of a numerical feature across categories.
kind: "box", "violin", "strip"
"""
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    if kind == "box":
        sns.boxplot(data=df, x=by, y=feature)
    elif kind == "violin":
        sns.violinplot(data=df, x=by, y=feature, inner="quartile")
    elif kind == "strip":
        sns.stripplot(data=df, x=by, y=feature, alpha=0.5, jitter=0.25)
    else:
        raise ValueError("kind must be 'box', 'violin', or 'strip'")
    plt.title(f"{feature} by {by}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    def compare_models_table(results, sort_by="R2 Test"):
        """
Create a comparison table from a results list of dicts.
results = [ { "Model": "Linear Regression", "R2 Train": ..., "R2 Test": ..., "MSE (Test)": ... }, ... ]
"""
    import pandas as pd
    df = pd.DataFrame(results)
    df = df.set_index("Model")
    return df.sort_values(by=sort_by, ascending=False)


def create_classification_model(model, X_train, X_test, y_train, y_test, average="binary"):
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_tr = accuracy_score(y_train, y_pred_train)
    acc_te = accuracy_score(y_test,  y_pred_test)
    prec_tr = precision_score(y_train, y_pred_train,
                              average=average, zero_division=0)
    prec_te = precision_score(y_test,  y_pred_test,
                              average=average, zero_division=0)
    rec_tr = recall_score(y_train, y_pred_train,
                          average=average, zero_division=0)
    rec_te = recall_score(y_test,  y_pred_test,
                          average=average, zero_division=0)
    f1_tr = f1_score(y_train, y_pred_train, average=average, zero_division=0)
    f1_te = f1_score(y_test,  y_pred_test, average=average, zero_division=0)

    print(" Classification Results")
    print(f"Accuracy   -> Train: {acc_tr:.3f}, Test: {acc_te:.3f}")
    print(f"Precision  -> Train: {prec_tr:.3f}, Test: {prec_te:.3f}")
    print(f"Recall     -> Train: {rec_tr:.3f}, Test: {rec_te:.3f}")
    print(f"F1 Score   -> Train: {f1_tr:.3f}, Test: {f1_te:.3f}")
    print("\nConfusion Matrix (Test):\n",
          confusion_matrix(y_test, y_pred_test))
    print("\nClassification Report (Test):\n", classification_report(
        y_test, y_pred_test, zero_division=0))

    return {
        "Model": model.__class__.__name__,
        "Accuracy (Train)": acc_tr, "Accuracy (Test)": acc_te,
        "Precision (Train)": prec_tr, "Precision (Test)": prec_te,
        "Recall (Train)": rec_tr, "Recall (Test)": rec_te,
        "F1 (Train)": f1_tr, "F1 (Test)": f1_te
    }


def compare_classification_models_table(results, sort_by="F1 (Test)", ascending=False, round_cols=True):
    """
    Turn a list of classification results into a sorted leaderboard DataFrame.

    Parameters
    ----------
    results : list of dict
        Each dict should look like the return of create_classification_model():
        {
          "Model": "...",
          "Accuracy (Train)": ...,
          "Accuracy (Test)": ...,
          "Precision (Train)": ...,
          "Precision (Test)": ...,
          "Recall (Train)": ...,
          "Recall (Test)": ...,
          "F1 (Train)": ...,
          "F1 (Test)": ...
        }
    sort_by : str
        Column to sort by (default "F1 (Test)").
    ascending : bool
        Sort order.
    round_cols : bool
        If True, round metric columns to 3 decimals for readability.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(results)
    if "Model" in df.columns:
        df = df.set_index("Model")

    # Optional rounding for nice display
    if round_cols:
        for col in df.columns:
            if df[col].dtype.kind in "fc":  # float or complex
                df[col] = df[col].round(3)

    # Sort if column exists
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    return df


def init_results():
    """
    Initialize an empty results list for storing model evaluation dictionaries.
    Returns
    -------
    list
    """
    return []


def log_results(results_list, results_dict, auto_print=True):
    """
    Append model results to the results list and optionally print confirmation.

    Parameters
    ----------
    results_list : list
        The list created by init_results(), where model results are stored.
    results_dict : dict
        The dictionary output of create_model() or create_classification_model().
    auto_print : bool
        If True, print confirmation of append.

    Returns
    -------
    list
    """
    results_list.append(results_dict)
    if auto_print:
        print(f"📌 Added results for {results_dict['Model']} to leaderboard.")
    return results_list


# ---------- Helpers ----------


def to_leaderboard_df(results_list, sort_by="F1 (Test)", round_ndigits=3):
    """Turn a list[dict] (from create_classification_model) into a sorted, rounded DataFrame."""
    df = pd.DataFrame(results_list).copy()
    if "Model" in df.columns:
        df = df.set_index("Model")
    # keep common classification metrics if present
    cols = [c for c in [
        "Accuracy (Train)", "Accuracy (Test)",
        "Precision (Train)", "Precision (Test)",
        "Recall (Train)", "Recall (Test)",
        "F1 (Train)", "F1 (Test)"
    ] if c in df.columns]
    df = df[cols]
    df = df.round(round_ndigits)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    return df


def plotly_leaderboard(df, title="Leaderboard"):
    """Pretty Plotly table from a DataFrame."""
    header_vals = [df.index.name or "Model"] + list(df.columns)
    cell_vals = [df.index.tolist()] + [df[c].tolist() for c in df.columns]

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_vals, fill_color="#E6F2FF", align="left"),
        cells=dict(values=cell_vals, align="left")
    )])
    fig.update_layout(title=title, width=1100, height=400)
    fig.show()


def plot_metric_bars(df, metric="F1 (Test)", title=None):
    """Bar chart for a selected metric."""
    title = title or f"{metric} by Model"
    fig = go.Figure(go.Bar(x=df.index, y=df[metric]))
    fig.update_layout(title=title, xaxis_title="Model",
                      yaxis_title=metric, width=1100, height=400)
    fig.show()


def init_results():
    """Return a fresh list to store model evaluation dicts."""
    return []


def log_results(results_list, results_dict, rename=None, auto_print=True):
    """
    Append one model's results (dict from create_classification_model) to results_list.

    Parameters
    ----------
    results_list : list
        The list created by init_results().
    results_dict : dict
        Output of create_classification_model(...).
    rename : str | None
        If provided, override results_dict["Model"] with this label.
    auto_print : bool
        Print a short confirmation.

    Returns
    -------
    list
    """
    d = dict(results_dict)  # shallow copy
    if rename:
        d["Model"] = rename
    results_list.append(d)
    if auto_print:
        print(f"📌 Added: {d.get('Model', 'Unknown Model')}")
    return results_list


def to_leaderboard_df(results_list, sort_by="F1 (Test)", round_ndigits=3):
    """
    Convert a list of result dicts into a sorted, rounded DataFrame.
    """
    if not results_list:
        return pd.DataFrame()

    df = pd.DataFrame(results_list).copy()
    if "Model" in df.columns:
        df = df.set_index("Model")

    # keep common classification metrics if present
    cols_wanted = [
        "Accuracy (Train)", "Accuracy (Test)",
        "Precision (Train)", "Precision (Test)",
        "Recall (Train)", "Recall (Test)",
        "F1 (Train)", "F1 (Test)"
    ]
    cols = [c for c in cols_wanted if c in df.columns]
    df = df[cols]

    # rounding
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(round_ndigits)

    # sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    return df


def run_classifier_with_optional_smote(
    model, X, y,
    test_size=0.2,
    average="binary",
    random_state=42,
    use_smote=False,
    smote_kwargs=None,
    model_name=None
):
    """
    Split once, optionally apply SMOTE on train, then evaluate via create_classification_model.
    Returns the result dict (ready to be logged).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    suffix = " (baseline)"
    if use_smote:
        sm = SMOTE(random_state=random_state, **
                   (smote_kwargs or {"sampling_strategy": 1.0}))
        X_train, y_train = sm.fit_resample(X_train, y_train)
        suffix = " (+SMOTE)"

    res = create_classification_model(
        model, X_train, X_test, y_train, y_test, average=average)
    # label
    label = model_name or model.__class__.__name__
    res["Model"] = f"{label}{suffix}"
    return res
