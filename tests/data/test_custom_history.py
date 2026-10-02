from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from src.data.custom_history import prepare_history, read_history_table, workbook_sheets
from src.analytics.custom_history import analyze_history, bootstrap_history
from src.optimization.estimators import estimate_portfolio_inputs


def monthly_table():
    return pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=24, freq="ME"),
                         "A": np.tile([0.02, -0.01, 0.03, -0.02], 6),
                         "B": np.tile([-0.01, 0.04, 0.01, 0.02], 6)})


def prepare(table, **kwargs):
    options = dict(date_column="Date", asset_columns=["A", "B"], frequency="Monthly", value_type="Returns")
    options.update(kwargs)
    return prepare_history(table, **options)


def test_monthly_annualization_and_optimizer_estimates():
    history = prepare(monthly_table())
    result = analyze_history(history, np.array([0.6, 0.4]), 0.03)
    weighted = history.returns @ np.array([0.6, 0.4])
    assert result["metrics"]["Annualized return"] == pytest.approx((1 + weighted).prod() ** 0.5 - 1)
    assert result["metrics"]["Annualized volatility"] == pytest.approx(weighted.std() * np.sqrt(12))
    rf_month = 1.03 ** (1 / 12) - 1
    assert result["metrics"]["Sharpe ratio"] == pytest.approx((weighted.mean() - rf_month) / weighted.std() * np.sqrt(12))
    estimates = estimate_portfolio_inputs(history.returns, trading_days=history.periods_per_year)
    np.testing.assert_allclose(estimates.sample_covariance, history.returns.cov() * 12)


def test_prices_percent_and_baseline_agree():
    table = monthly_table()
    expected = prepare(table)
    percent = table.copy()
    percent[["A", "B"]] *= 100
    pd.testing.assert_frame_equal(prepare(percent, return_unit="Percent").returns, expected.returns)
    baseline = pd.DataFrame({"Date": [pd.Timestamp("2019-12-31")], "A": [np.nan], "B": [np.nan]})
    with_baseline = prepare(pd.concat([baseline, table], ignore_index=True))
    pd.testing.assert_frame_equal(with_baseline.returns, expected.returns)
    prices = pd.concat([baseline, table], ignore_index=True)
    prices[["A", "B"]] = (1 + prices[["A", "B"]].fillna(0)).cumprod() * 100
    np.testing.assert_allclose(prepare(prices, value_type="Prices / index levels").returns, expected.returns)


@pytest.mark.parametrize("defect,match", [
    ("duplicate", "Duplicate dates"), ("gap", "consecutive"),
    ("missing", "missing or non-numeric"), ("text", "missing or non-numeric"),
    ("invalid_first", "missing or non-numeric"), ("infinite", "missing or non-numeric"),
    ("loss", "greater than -100%"), ("bad_date", "dates are missing or invalid"),
])
def test_rejects_invalid_histories(defect, match):
    table = monthly_table()
    if defect == "duplicate":
        table.loc[1, "Date"] = table.loc[0, "Date"]
    elif defect == "gap":
        table = table.drop(index=5)
    elif defect == "bad_date":
        table.loc[2, "Date"] = pd.NaT
    elif defect == "invalid_first":
        table[["A", "B"]] = table[["A", "B"]].astype(object)
        table.loc[0, ["A", "B"]] = "not a return"
    else:
        table["A"] = table["A"].astype(object)
        table.loc[3, "A"] = {"missing": np.nan, "text": "bad", "infinite": np.inf, "loss": -1}[defect]
    with pytest.raises(ValueError, match=match):
        prepare(table)


def test_monthly_rejected_as_daily():
    with pytest.raises(ValueError, match="do not look daily"):
        prepare(monthly_table(), frequency="Daily")


def test_daily_annualization_unchanged():
    table = monthly_table()
    table["Date"] = pd.bdate_range("2024-01-01", periods=len(table))
    history = prepare(table, frequency="Daily")
    result = analyze_history(history, np.array([1.0, 0.0]), 0)
    assert result["metrics"]["Annualized volatility"] == pytest.approx(table.A.std() * np.sqrt(252))


def test_csv_locale_and_duplicate_headers():
    table = read_history_table(b"Date;A;B\n31/01/2024;0,01;0,02\n29/02/2024;-0,02;0,03\n", "test.csv", separator=";", decimal=",")
    history = prepare(table, date_format="Day / month / year")
    assert history.returns.iloc[0, 0] == 0.01
    with pytest.raises(ValueError, match="unique"):
        read_history_table(b"Date,A,A\n2024-01-01,1,2", "test.csv")


def test_xlsx_sheet_and_offset_header():
    import openpyxl
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "History"
    sheet.append(["Title"])
    sheet.append(["Date", "A", "B"])
    for row in monthly_table().itertuples(index=False, name=None):
        sheet.append(row)
    content = BytesIO()
    workbook.save(content)
    data = content.getvalue()
    assert workbook_sheets(data, "history.xlsx") == ["History"]
    history = prepare(read_history_table(data, "history.xlsx", sheet="History", header_row=2))
    assert len(history.returns) == 24
    assert history.returns.iloc[0, 0] == 0.02


def test_bootstrap_joint_sampling_preserves_offsetting_assets():
    table = monthly_table()
    table["B"] = -table["A"]
    history = prepare(table)
    result = bootstrap_history(history, np.array([0.5, 0.5]), years=2, simulations=100, block_length=3)
    assert len(result) == 25
    assert result.index[-1] == 2
    np.testing.assert_allclose(result.to_numpy(), 1.0)


def test_bootstrap_reproducibility_and_weight_validation():
    history = prepare(monthly_table())
    args = dict(years=1, simulations=100, block_length=3)
    pd.testing.assert_frame_equal(bootstrap_history(history, np.array([0.5, 0.5]), **args), bootstrap_history(history, np.array([0.5, 0.5]), **args))
    with pytest.raises(ValueError, match="total 100%"):
        analyze_history(history, np.array([0.4, 0.4]), 0.03)


def test_file_limits_and_empty_data():
    with pytest.raises(ValueError, match="20 MB"):
        read_history_table(b"", "empty.csv")
    with pytest.raises(ValueError, match="valid XLSX"):
        workbook_sheets(b"not a zip", "bad.xlsx")
    with pytest.raises(ValueError, match="two return"):
        prepare(monthly_table().iloc[:1])
