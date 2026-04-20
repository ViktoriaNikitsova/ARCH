import io
import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import jarque_bera, kurtosis, norm
from scipy.stats import t as student_t
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.numdiff import approx_hess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

ARTICLE_EURUSD_START = "2010-01-01"
ARTICLE_EURUSD_END = "2016-12-30"
ARTICLE_MEAN_ORDER = (1, 0, 1)
LJUNG_BOX_LAGS = [1, 6, 10, 15, 20]
MODEL_NAMES = ["GARCH", "GARCH-M", "EGARCH", "GJR-GARCH", "APARCH"]
DIST_NAMES = ["normal", "t", "skewt"]
EPS = 1e-12


@dataclass
class FitResult:
    model_name: str
    distribution: str
    success: bool
    message: str
    optimizer_method: str
    params: Dict[str, float]
    stderr: Dict[str, float]
    zvalues: Dict[str, float]
    pvalues: Dict[str, float]
    loglikelihood: float
    aic: float
    bic: float
    aic_per_obs: float
    bic_per_obs: float
    conditional_variance: np.ndarray
    conditional_volatility: np.ndarray
    resid: np.ndarray
    std_resid: np.ndarray
    mean_fitted: np.ndarray
    nobs: int
    cov_matrix: Optional[np.ndarray] = None

    def parameter_frame(self) -> pd.DataFrame:
        rows = []
        for name, coef in self.params.items():
            rows.append(
                {
                    "parameter": name,
                    "coef": coef,
                    "std_err": self.stderr.get(name, np.nan),
                    "z_stat": self.zvalues.get(name, np.nan),
                    "p_value": self.pvalues.get(name, np.nan),
                }
            )
        return pd.DataFrame(rows)

    @property
    def summary_text(self) -> str:
        lines = [
            f"Модель: {self.model_name}",
            f"Распределение: {self.distribution}",
            f"Успешное завершение: {self.success}",
            f"Сообщение оптимизатора: {self.message}",
            f"Метод оптимизации: {self.optimizer_method}",
            f"Число наблюдений: {self.nobs}",
            f"LLF: {self.loglikelihood:.6f}",
            f"AIC: {self.aic:.6f}",
            f"BIC: {self.bic:.6f}",
            f"AIC / n: {self.aic_per_obs:.6f}",
            f"BIC / n: {self.bic_per_obs:.6f}",
            "",
            f"{'parameter':>14s} {'coef':>14s} {'std.err':>14s} {'z':>14s} {'p>|z|':>14s}",
            "-" * 78,
        ]
        for _, row in self.parameter_frame().iterrows():
            lines.append(
                f"{row['parameter']:>14s} "
                f"{row['coef']:14.6f} "
                f"{row['std_err']:14.6f} "
                f"{row['z_stat']:14.6f} "
                f"{row['p_value']:14.6f}"
            )
        return "\n".join(lines)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python requests"}
    )
    return session


def load_eurusd_ecb(
    start: str = ARTICLE_EURUSD_START,
    end: str = ARTICLE_EURUSD_END,
    align_business_days: bool = True,
) -> pd.DataFrame:
    prev_business_day = (pd.Timestamp(start) - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    url = (
        "https://data-api.ecb.europa.eu/service/data/EXR/"
        "D.USD.EUR.SP00.A"
        f"?startPeriod={prev_business_day}&endPeriod={end}&format=csvdata"
    )

    session = build_session()
    response = session.get(url, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    cols = {c.upper(): c for c in df.columns}
    time_col = cols.get("TIME_PERIOD", "TIME_PERIOD")
    value_col = cols.get("OBS_VALUE", "OBS_VALUE")

    out = df[[time_col, value_col]].copy()
    out.columns = ["date", "rate"]
    out["date"] = pd.to_datetime(out["date"])
    out["rate"] = pd.to_numeric(out["rate"], errors="coerce")
    out = out.dropna().sort_values("date").reset_index(drop=True)

    if align_business_days:
        out = (
            out.set_index("date")
            .reindex(pd.date_range(start=prev_business_day, end=end, freq="B"))
            .sort_index()
            .ffill()
            .rename_axis("date")
            .reset_index()
        )

    out = out[out["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    return out


def load_usdrub_cbr(start: str = "01/01/2016", end: str = "31/12/2019") -> pd.DataFrame:
    url = (
        "https://www.cbr.ru/scripts/XML_dynamic.asp"
        f"?date_req1={start}&date_req2={end}&VAL_NM_RQ=R01235"
    )

    session = build_session()
    response = session.get(url, timeout=60)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    rows: List[Dict[str, float]] = []
    for record in root.findall("Record"):
        date_str = record.attrib.get("Date")
        value = record.findtext("Value")
        nominal = record.findtext("Nominal")
        if value is None or nominal is None or date_str is None:
            continue
        value_num = float(value.replace(",", "."))
        nominal_num = float(nominal.replace(",", "."))
        rows.append(
            {
                "date": pd.to_datetime(date_str, format="%d.%m.%Y"),
                "rate": value_num / nominal_num,
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def prepare_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["log_return"] = np.log(out["rate"] / out["rate"].shift(1))
    return out.dropna().reset_index(drop=True)


def descriptive_stats(series: pd.Series) -> pd.Series:
    x = pd.Series(series, dtype=float).dropna()
    jb = jarque_bera(x)
    return pd.Series(
        {
            "count": int(x.count()),
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
            "min": float(x.min()),
            "max": float(x.max()),
            "skew": float(x.skew()),
            "kurtosis": float(kurtosis(x, fisher=False, bias=False)),
            "jarque_bera": float(jb.statistic),
            "jarque_bera_pvalue": float(jb.pvalue),
        }
    )


def adf_report(series: pd.Series, name: str) -> pd.DataFrame:
    stat, pvalue, usedlag, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    rows = [
        ["series", name],
        ["adf_stat", stat],
        ["pvalue", pvalue],
        ["used_lag", usedlag],
        ["nobs", nobs],
        ["crit_1pct", crit["1%"]],
        ["crit_5pct", crit["5%"]],
        ["crit_10pct", crit["10%"]],
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def ljung_box_suite(series: pd.Series, title: str) -> pd.DataFrame:
    frame = acorr_ljungbox(
        pd.Series(series, dtype=float).dropna(),
        lags=LJUNG_BOX_LAGS,
        return_df=True,
        boxpierce=True,
    )
    frame = frame.reset_index().rename(columns={"index": "lag"})
    frame["series"] = title
    return frame[["series", "lag", "lb_stat", "lb_pvalue", "bp_stat", "bp_pvalue"]]


def arch_lm_report(series: pd.Series, lags: int = 12) -> pd.DataFrame:
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(pd.Series(series, dtype=float).dropna(), nlags=lags)
    return pd.DataFrame(
        [
            ["lags", lags],
            ["LM_stat", lm_stat],
            ["LM_pvalue", lm_pvalue],
            ["F_stat", f_stat],
            ["F_pvalue", f_pvalue],
        ],
        columns=["metric", "value"],
    )


def save_basic_plots(df: pd.DataFrame, pair_name: str, outdir: Path) -> None:
    ensure_dir(outdir)

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["rate"])
    plt.title(f"Динамика обменного курса: {pair_name}")
    plt.xlabel("Дата")
    plt.ylabel("Курс")
    plt.tight_layout()
    plt.savefig(outdir / "rate.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["log_return"])
    plt.title(f"Логарифмические доходности: {pair_name}")
    plt.xlabel("Дата")
    plt.ylabel("log-return")
    plt.tight_layout()
    plt.savefig(outdir / "log_returns.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["log_return"].abs())
    plt.title(f"Абсолютные логарифмические доходности: {pair_name}")
    plt.xlabel("Дата")
    plt.ylabel("|log-return|")
    plt.tight_layout()
    plt.savefig(outdir / "abs_log_returns.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["log_return"] ** 2)
    plt.title(f"Квадраты логарифмических доходностей: {pair_name}")
    plt.xlabel("Дата")
    plt.ylabel("(log-return)^2")
    plt.tight_layout()
    plt.savefig(outdir / "squared_returns.png", dpi=300)
    plt.close()

    for name, values, fname in [
        ("ACF логарифмических доходностей", df["log_return"].dropna(), "acf_log_returns.png"),
        ("PACF логарифмических доходностей", df["log_return"].dropna(), "pacf_log_returns.png"),
        ("ACF абсолютных доходностей", df["log_return"].abs().dropna(), "acf_abs_returns.png"),
        ("PACF абсолютных доходностей", df["log_return"].abs().dropna(), "pacf_abs_returns.png"),
        ("ACF квадратов доходностей", (df["log_return"] ** 2).dropna(), "acf_sq_returns.png"),
        ("PACF квадратов доходностей", (df["log_return"] ** 2).dropna(), "pacf_sq_returns.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        if "PACF" in name:
            plot_pacf(values, ax=ax, lags=40, method="ywm")
        else:
            plot_acf(values, ax=ax, lags=40)
        ax.set_title(f"{name}: {pair_name}")
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=300)
        plt.close()


def arma_grid_search(series: pd.Series, max_p: int = 3, max_q: int = 3) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    y = pd.Series(series, dtype=float).dropna()

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                res = ARIMA(y, order=(p, 0, q), trend="c").fit()
                rows.append(
                    {
                        "p": p,
                        "q": q,
                        "AIC": float(res.aic),
                        "BIC": float(res.bic),
                        "LogLik": float(res.llf),
                        "converged": bool(getattr(res, "mle_retvals", {}).get("converged", True)),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "p": p,
                        "q": q,
                        "AIC": np.nan,
                        "BIC": np.nan,
                        "LogLik": np.nan,
                        "converged": False,
                        "fit_error": str(exc),
                    }
                )

    return pd.DataFrame(rows).sort_values(["BIC", "AIC"], na_position="last").reset_index(drop=True)


def fit_arma_11(series: pd.Series):
    y = pd.Series(series, dtype=float).dropna()
    res = ARIMA(y, order=ARTICLE_MEAN_ORDER, trend="c").fit()
    resid = pd.Series(res.resid, index=y.index, name="arma_resid")
    return res, resid


def _normal_logpdf(z: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2.0 * np.pi) + z ** 2)


def _std_t_logpdf(z: np.ndarray, nu: float) -> np.ndarray:
    if nu <= 2.0:
        return np.full_like(z, -np.inf, dtype=float)
    c = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(np.pi * (nu - 2.0))
    )
    return c - ((nu + 1.0) / 2.0) * np.log1p((z ** 2) / (nu - 2.0))


def _sstd_logpdf_rugarch_style(z: np.ndarray, nu: float, skew: float) -> np.ndarray:
    if nu <= 2.0 or skew <= 0.0:
        return np.full_like(z, -np.inf, dtype=float)

    m = (
        math.exp(gammaln((nu - 1.0) / 2.0) - gammaln(nu / 2.0))
        * math.sqrt((nu - 2.0) / math.pi)
        * (skew - 1.0 / skew)
    )
    s2 = (skew ** 2 + skew ** (-2) - 1.0) - m ** 2
    if s2 <= 0.0:
        return np.full_like(z, -np.inf, dtype=float)
    s = math.sqrt(s2)

    c = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(np.pi * (nu - 2.0))
    )
    log_k = np.log(2.0) + np.log(s) - np.log(skew + 1.0 / skew) + c

    y = m + s * z
    out = np.empty_like(z, dtype=float)
    left = y < 0.0
    right = ~left

    x_left = y[left] * skew
    x_right = y[right] / skew

    out[left] = log_k - ((nu + 1.0) / 2.0) * np.log1p((x_left ** 2) / (nu - 2.0))
    out[right] = log_k - ((nu + 1.0) / 2.0) * np.log1p((x_right ** 2) / (nu - 2.0))
    return out


def _distribution_logpdf(z: np.ndarray, distribution: str, params: Dict[str, float]) -> np.ndarray:
    distribution = distribution.lower()
    if distribution == "normal":
        return _normal_logpdf(z)
    if distribution == "t":
        return _std_t_logpdf(z, params["nu"])
    if distribution == "skewt":
        return _sstd_logpdf_rugarch_style(z, params["nu"], params["skew"])
    raise ValueError(f"Неизвестное распределение: {distribution}")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _variance_seed(y: np.ndarray) -> float:
    return max(float(np.var(y, ddof=1)), 1e-8)


def _expected_abs_z() -> float:
    return math.sqrt(2.0 / math.pi)


def _pack_param_names(model_name: str, distribution: str) -> List[str]:
    names = ["mu", "phi", "theta"]

    if model_name == "GARCH-M":
        names.append("lambda_m")

    if model_name in {"GARCH", "GARCH-M"}:
        names.extend(["omega", "alpha", "beta"])
    elif model_name == "EGARCH":
        names.extend(["omega", "alpha", "gamma", "beta"])
    elif model_name == "GJR-GARCH":
        names.extend(["omega", "alpha", "gamma", "beta"])
    elif model_name == "APARCH":
        names.extend(["omega", "alpha", "gamma", "beta", "delta"])
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    if distribution in {"t", "skewt"}:
        names.append("nu")
    if distribution == "skewt":
        names.append("skew")

    return names


def _extract_arma_starts(arma_res) -> Dict[str, float]:
    params = getattr(arma_res, "params", None)
    names = list(getattr(params, "index", [])) if hasattr(params, "index") else []
    values = np.asarray(params, dtype=float) if params is not None else np.array([])
    lookup = {name: value for name, value in zip(names, values)}

    mu = _safe_float(lookup.get("const"), default=0.0)
    phi = _safe_float(lookup.get("ar.L1"), default=0.05)
    theta = _safe_float(lookup.get("ma.L1"), default=-0.05)

    phi = float(np.clip(phi, -0.98, 0.98))
    theta = float(np.clip(theta, -0.98, 0.98))
    return {"mu": mu, "phi": phi, "theta": theta}


def _default_start_vector(model_name: str, distribution: str, y: np.ndarray, arma_res) -> np.ndarray:
    arma_start = _extract_arma_starts(arma_res)
    base_var = _variance_seed(y)
    start: List[float] = [arma_start["mu"], arma_start["phi"], arma_start["theta"]]

    if model_name == "GARCH-M":
        start.append(0.05)

    if model_name in {"GARCH", "GARCH-M"}:
        alpha0, beta0 = 0.08, 0.90
        start.extend([base_var * max(1.0 - alpha0 - beta0, 1e-3), alpha0, beta0])
    elif model_name == "EGARCH":
        beta0 = 0.95
        start.extend([math.log(base_var) * (1.0 - beta0), 0.08, -0.05, beta0])
    elif model_name == "GJR-GARCH":
        alpha0, gamma0, beta0 = 0.06, 0.08, 0.88
        start.extend(
            [
                base_var * max(1.0 - alpha0 - 0.5 * max(gamma0, 0.0) - beta0, 1e-3),
                alpha0,
                gamma0,
                beta0,
            ]
        )
    elif model_name == "APARCH":
        alpha0, gamma0, beta0, delta0 = 0.07, -0.05, 0.88, 1.5
        start.extend([base_var * max(1.0 - alpha0 - beta0, 1e-3), alpha0, gamma0, beta0, delta0])

    if distribution in {"t", "skewt"}:
        start.append(8.0)
    if distribution == "skewt":
        start.append(1.05)

    return np.asarray(start, dtype=float)


def _bounds_for_model(model_name: str, distribution: str, y: np.ndarray) -> List[Tuple[float, float]]:
    y_std = max(float(np.std(y, ddof=1)), 1e-4)
    base_var = _variance_seed(y)
    upper_omega = max(50.0 * base_var, 1e-4)
    mean_bound = max(20.0 * y_std, 0.05)
    bounds: List[Tuple[float, float]] = [(-mean_bound, mean_bound), (-0.99, 0.99), (-0.99, 0.99)]

    if model_name == "GARCH-M":
        bounds.append((-10.0, 10.0))

    if model_name in {"GARCH", "GARCH-M"}:
        bounds.extend([(1e-12, upper_omega), (1e-8, 0.999), (1e-8, 0.999)])
    elif model_name == "EGARCH":
        bounds.extend([(-50.0, 50.0), (-5.0, 5.0), (-5.0, 5.0), (-0.999, 0.999)])
    elif model_name == "GJR-GARCH":
        bounds.extend([(1e-12, upper_omega), (1e-8, 0.999), (-0.999, 0.999), (1e-8, 0.999)])
    elif model_name == "APARCH":
        bounds.extend([(1e-12, upper_omega), (1e-8, 0.999), (-0.999, 0.999), (1e-8, 0.999), (0.2, 4.0)])
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    if distribution in {"t", "skewt"}:
        bounds.append((2.05, 100.0))
    if distribution == "skewt":
        bounds.append((0.2, 5.0))

    return bounds


def _vector_to_param_dict(model_name: str, distribution: str, x: Sequence[float]) -> Dict[str, float]:
    names = _pack_param_names(model_name, distribution)
    return {name: float(value) for name, value in zip(names, x)}


def _clip_to_bounds(x: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    return np.asarray([np.clip(v, lo + 1e-8, hi - 1e-8) for v, (lo, hi) in zip(x, bounds)], dtype=float)


def _apply_penalty(model_name: str, distribution: str, params: Dict[str, float]) -> float:
    penalty = 0.0

    if abs(params["phi"]) >= 0.999:
        penalty += 1e8
    if abs(params["theta"]) >= 0.999:
        penalty += 1e8

    if model_name in {"GARCH", "GARCH-M"}:
        if params["alpha"] + params["beta"] >= 0.999:
            penalty += 1e8
    elif model_name == "EGARCH":
        if abs(params["beta"]) >= 0.999:
            penalty += 1e8
    elif model_name == "GJR-GARCH":
        if params["alpha"] + 0.5 * max(params["gamma"], 0.0) + params["beta"] >= 0.999:
            penalty += 1e8
    elif model_name == "APARCH":
        if params["alpha"] + params["beta"] >= 0.999:
            penalty += 1e8
        if abs(params["gamma"]) >= 0.999:
            penalty += 1e8
        if params["delta"] <= 0.0:
            penalty += 1e8

    if distribution in {"t", "skewt"} and params.get("nu", 999.0) <= 2.0:
        penalty += 1e8
    if distribution == "skewt" and params.get("skew", 1.0) <= 0.0:
        penalty += 1e8

    return penalty


def _compute_recursions(
    y: np.ndarray,
    model_name: str,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    eps = np.zeros(n, dtype=float)
    sigma2 = np.zeros(n, dtype=float)
    mean = np.zeros(n, dtype=float)

    mu = params["mu"]
    phi = params["phi"]
    theta = params["theta"]
    lambda_m = params.get("lambda_m", 0.0)

    seed_var = _variance_seed(y)
    sigma2[0] = seed_var
    mean[0] = mu + (lambda_m * math.sqrt(sigma2[0]) if model_name == "GARCH-M" else 0.0)
    eps[0] = y[0] - mean[0]

    e_abs = _expected_abs_z()

    for t in range(1, n):
        prev_eps = eps[t - 1]
        prev_sigma2 = max(sigma2[t - 1], EPS)
        prev_sigma = math.sqrt(prev_sigma2)
        prev_z = prev_eps / prev_sigma

        if model_name in {"GARCH", "GARCH-M"}:
            sigma2[t] = params["omega"] + params["alpha"] * prev_eps ** 2 + params["beta"] * prev_sigma2

        elif model_name == "EGARCH":
            log_sigma2 = (
                params["omega"]
                + params["alpha"] * (abs(prev_z) - e_abs)
                + params["gamma"] * prev_z
                + params["beta"] * math.log(prev_sigma2)
            )
            sigma2[t] = math.exp(log_sigma2)

        elif model_name == "GJR-GARCH":
            indicator = 1.0 if prev_eps < 0.0 else 0.0
            sigma2[t] = (
                params["omega"]
                + params["alpha"] * prev_eps ** 2
                + params["gamma"] * indicator * prev_eps ** 2
                + params["beta"] * prev_sigma2
            )

        elif model_name == "APARCH":
            delta = params["delta"]
            gamma = params["gamma"]
            sigma_delta = (
                params["omega"]
                + params["alpha"] * (abs(prev_eps) - gamma * prev_eps) ** delta
                + params["beta"] * (prev_sigma ** delta)
            )
            sigma2[t] = max(sigma_delta, EPS) ** (2.0 / delta)

        else:
            raise ValueError(f"Неизвестная модель: {model_name}")

        sigma2[t] = max(sigma2[t], EPS)
        mean[t] = mu + phi * (y[t - 1] - mu) + theta * prev_eps
        if model_name == "GARCH-M":
            mean[t] += lambda_m * math.sqrt(sigma2[t])
        eps[t] = y[t] - mean[t]

    return mean, eps, sigma2


def _negative_loglikelihood(x: np.ndarray, y: np.ndarray, model_name: str, distribution: str) -> float:
    params = _vector_to_param_dict(model_name, distribution, x)
    penalty = _apply_penalty(model_name, distribution, params)
    if penalty > 0:
        return penalty

    try:
        _, eps, sigma2 = _compute_recursions(y, model_name, params)
        sigma2 = np.maximum(sigma2, EPS)
        sigma = np.sqrt(sigma2)
        z = eps / sigma
        logpdf = _distribution_logpdf(z, distribution, params)

        if not np.all(np.isfinite(logpdf)):
            return 1e8

        ll = np.sum(logpdf - np.log(sigma))
        if not np.isfinite(ll):
            return 1e8
        return float(-ll)
    except Exception:
        return 1e8


def _touches_boundary(x: np.ndarray, bounds: Sequence[Tuple[float, float]], tol: float = 1e-7) -> bool:
    for value, (low, high) in zip(x, bounds):
        if abs(value - low) <= tol or abs(value - high) <= tol:
            return True
    return False


def _unstable_inference(cov: np.ndarray, stderr: np.ndarray, cond_limit: float = 1e12, min_stderr: float = 1e-8) -> bool:
    if cov is None:
        return True
    if not np.all(np.isfinite(cov)) or not np.all(np.isfinite(stderr)):
        return True
    if np.any(stderr < min_stderr):
        return True
    try:
        cond_number = np.linalg.cond(cov)
    except Exception:
        return True
    if not np.isfinite(cond_number) or cond_number > cond_limit:
        return True
    return False


def _compute_inference(
    x_hat: np.ndarray,
    y: np.ndarray,
    model_name: str,
    distribution: str,
    bounds: Sequence[Tuple[float, float]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Optional[np.ndarray]]:
    names = _pack_param_names(model_name, distribution)
    nan_dict = {n: float("nan") for n in names}

    if _touches_boundary(x_hat, bounds):
        return nan_dict, nan_dict.copy(), nan_dict.copy(), None

    try:
        f = lambda p: _negative_loglikelihood(np.asarray(p, dtype=float), y, model_name, distribution)
        hess = approx_hess(x_hat, f)
        if not np.all(np.isfinite(hess)):
            return nan_dict, nan_dict.copy(), nan_dict.copy(), None

        cov = np.linalg.pinv(hess)
        if not np.all(np.isfinite(cov)):
            return nan_dict, nan_dict.copy(), nan_dict.copy(), None

        diag = np.diag(cov).astype(float)
        if np.any(diag <= 0.0):
            return nan_dict, nan_dict.copy(), nan_dict.copy(), None

        stderr = np.sqrt(diag)

        if _unstable_inference(cov, stderr):
            return nan_dict, nan_dict.copy(), nan_dict.copy(), None

        zvalues = x_hat / stderr
        pvalues = 2.0 * (1.0 - norm.cdf(np.abs(zvalues)))
        if not np.all(np.isfinite(zvalues)) or not np.all(np.isfinite(pvalues)):
            return nan_dict, nan_dict.copy(), nan_dict.copy(), None

        stderr_dict = {n: float(v) for n, v in zip(names, stderr)}
        z_dict = {n: float(v) for n, v in zip(names, zvalues)}
        p_dict = {n: float(v) for n, v in zip(names, pvalues)}
        return stderr_dict, z_dict, p_dict, cov
    except Exception:
        return nan_dict, nan_dict.copy(), nan_dict.copy(), None


def _multi_start_vectors(base: np.ndarray, bounds: Sequence[Tuple[float, float]], n_random: int = 8) -> List[np.ndarray]:
    rng = np.random.default_rng(42)
    midpoint = np.asarray([(lo + hi) / 2.0 for lo, hi in bounds], dtype=float)
    starts = [base.copy(), 0.7 * base + 0.3 * midpoint, 0.5 * base + 0.5 * midpoint]
    for _ in range(n_random):
        starts.append(np.asarray([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float))
    return [_clip_to_bounds(x, bounds) for x in starts]


def _build_start_override_from_previous(
    model_name: str,
    distribution: str,
    prev_params: Optional[Dict[str, float]],
    arma_res,
    y: np.ndarray,
) -> Optional[np.ndarray]:
    if prev_params is None:
        return None

    names = _pack_param_names(model_name, distribution)
    base = _default_start_vector(model_name, distribution, y, arma_res)
    default_dict = _vector_to_param_dict(model_name, distribution, base)
    merged = default_dict.copy()

    for key, value in prev_params.items():
        if key in merged and np.isfinite(value):
            merged[key] = float(value)

    if distribution in {"t", "skewt"} and "nu" not in merged:
        merged["nu"] = 8.0
    if distribution == "skewt" and "skew" not in merged:
        merged["skew"] = 1.05

    return np.asarray([merged[name] for name in names], dtype=float)


def fit_single_model(
    y: pd.Series,
    model_name: str,
    distribution: str,
    arma_res,
    start_override: Optional[np.ndarray] = None,
) -> FitResult:
    display_name = model_name.upper()
    data = np.asarray(pd.Series(y, dtype=float).dropna(), dtype=float)
    if len(data) < 50:
        raise ValueError("Ряд слишком короткий для оценивания модели.")

    bounds = _bounds_for_model(display_name, distribution, data)
    x0 = _default_start_vector(display_name, distribution, data, arma_res)
    if start_override is not None and len(start_override) == len(x0):
        x0 = _clip_to_bounds(np.asarray(start_override, dtype=float), bounds)

    starts = _multi_start_vectors(x0, bounds, n_random=6)
    methods = [
        ("L-BFGS-B", {"maxiter": 4000, "ftol": 1e-12, "gtol": 1e-8}),
        ("SLSQP", {"maxiter": 4000, "ftol": 1e-12}),
        ("Powell", {"maxiter": 4000, "xtol": 1e-6, "ftol": 1e-6}),
    ]

    best_opt = None
    best_ll = -np.inf
    best_params = None
    best_resid = None
    best_sigma2 = None
    best_mean = None
    best_method = ""

    for start in starts:
        for method, options in methods:
            try:
                opt = minimize(
                    _negative_loglikelihood,
                    x0=start,
                    args=(data, display_name, distribution),
                    method=method,
                    bounds=bounds,
                    options=options,
                )
                params = _vector_to_param_dict(display_name, distribution, opt.x)
                mean, resid, sigma2 = _compute_recursions(data, display_name, params)
                ll = -_negative_loglikelihood(opt.x, data, display_name, distribution)

                valid = (
                    np.isfinite(ll)
                    and np.all(np.isfinite(opt.x))
                    and np.all(np.isfinite(sigma2))
                    and np.min(sigma2) > 0.0
                )
                if valid and ll > best_ll:
                    best_ll = float(ll)
                    best_opt = opt
                    best_params = params
                    best_resid = resid
                    best_sigma2 = sigma2
                    best_mean = mean
                    best_method = method
            except Exception:
                continue

    if best_opt is None or best_params is None or best_resid is None or best_sigma2 is None or best_mean is None:
        raise RuntimeError(f"Не удалось оценить {display_name} с распределением {distribution}")

    sigma = np.sqrt(best_sigma2)
    z = best_resid / sigma
    k = len(best_opt.x)
    n = len(data)
    aic = 2.0 * k - 2.0 * best_ll
    bic = math.log(n) * k - 2.0 * best_ll

    stderr, zvalues, pvalues, cov = _compute_inference(best_opt.x, data, display_name, distribution, bounds)

    return FitResult(
        model_name=display_name,
        distribution=distribution,
        success=bool(best_opt.success),
        message=str(best_opt.message),
        optimizer_method=best_method,
        params=best_params,
        stderr=stderr,
        zvalues=zvalues,
        pvalues=pvalues,
        loglikelihood=float(best_ll),
        aic=float(aic),
        bic=float(bic),
        aic_per_obs=float(aic / n),
        bic_per_obs=float(bic / n),
        conditional_variance=best_sigma2,
        conditional_volatility=sigma,
        resid=best_resid,
        std_resid=z,
        mean_fitted=best_mean,
        nobs=n,
        cov_matrix=cov,
    )


def fit_all_models(series: pd.Series, arma_res) -> Tuple[Dict[str, FitResult], pd.DataFrame]:
    fitted: Dict[str, FitResult] = {}
    rows: List[Dict[str, object]] = []
    data = np.asarray(pd.Series(series, dtype=float).dropna(), dtype=float)

    for model_name in MODEL_NAMES:
        prev_params: Optional[Dict[str, float]] = None
        for distribution in DIST_NAMES:
            print(f"Оценивается {model_name} с распределением {distribution}...", flush=True)
            key = f"{model_name} | {distribution}"
            try:
                start_override = _build_start_override_from_previous(model_name, distribution, prev_params, arma_res, data)
                res = fit_single_model(series, model_name, distribution, arma_res=arma_res, start_override=start_override)
                fitted[key] = res
                prev_params = res.params.copy()
                rows.append(
                    {
                        "Model": res.model_name,
                        "Distribution": res.distribution,
                        "Success": res.success,
                        "Optimizer": res.optimizer_method,
                        "Message": res.message,
                        "LogLik": res.loglikelihood,
                        "AIC": res.aic,
                        "BIC": res.bic,
                        "AIC_per_obs": res.aic_per_obs,
                        "BIC_per_obs": res.bic_per_obs,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "Model": model_name,
                        "Distribution": distribution,
                        "Success": False,
                        "Optimizer": "",
                        "Message": str(exc),
                        "LogLik": np.nan,
                        "AIC": np.nan,
                        "BIC": np.nan,
                        "AIC_per_obs": np.nan,
                        "BIC_per_obs": np.nan,
                    }
                )

    summary = pd.DataFrame(rows).sort_values(["AIC", "BIC"], na_position="last").reset_index(drop=True)
    return fitted, summary


def residual_diagnostics(std_resid: Sequence[float]) -> pd.DataFrame:
    series = pd.Series(std_resid, dtype=float).dropna()
    tables = []
    for label, obj in {
        "std_resid": series,
        "abs_std_resid": series.abs(),
        "sq_std_resid": series ** 2,
    }.items():
        tables.append(ljung_box_suite(obj, title=label))
    return pd.concat(tables, ignore_index=True)


def save_model_summaries(fitted: Dict[str, FitResult], outdir: Path) -> None:
    ensure_dir(outdir)
    for key, res in fitted.items():
        safe_name = (
            key.replace(" | ", "__")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
            .replace(" ", "_")
        )
        with open(outdir / f"{safe_name}_summary.txt", "w", encoding="utf-8") as handle:
            handle.write(res.summary_text)
        res.parameter_frame().to_csv(outdir / f"{safe_name}_coefficients.csv", index=False)


def save_combined_coefficients(fitted: Dict[str, FitResult], outdir: Path) -> pd.DataFrame:
    tables = []
    for key, res in fitted.items():
        table = res.parameter_frame().copy()
        table.insert(0, "model_key", key)
        table.insert(1, "model", res.model_name)
        table.insert(2, "distribution", res.distribution)
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    combined.to_csv(outdir / "coefficient_significance.csv", index=False)
    return combined


def save_residual_diagnostics(fitted: Dict[str, FitResult], outdir: Path) -> pd.DataFrame:
    tables = []
    for key, res in fitted.items():
        table = residual_diagnostics(res.std_resid)
        table.insert(0, "model_key", key)
        tables.append(table)
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    out.to_csv(outdir / "residual_diagnostics.csv", index=False)
    return out


def save_conditional_volatility_csv(fitted: Dict[str, FitResult], dates: pd.Series, outdir: Path) -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(dates).reset_index(drop=True)})
    for key, res in fitted.items():
        col_name = key.replace(" | ", "__").replace("-", "_").replace(" ", "_")
        values = pd.Series(res.conditional_volatility).reset_index(drop=True)
        frame[col_name] = values.values[: len(frame)]
    frame.to_csv(outdir / "conditional_volatility.csv", index=False)


def save_volatility_plot(fitted: Dict[str, FitResult], dates: pd.Series, pair_name: str, outdir: Path) -> None:
    plt.figure(figsize=(12, 6))
    x = pd.to_datetime(dates).to_numpy()
    for _, res in fitted.items():
        if res.distribution != "t":
            continue
        plt.plot(x[-len(res.conditional_volatility):], res.conditional_volatility, label=res.model_name, alpha=0.9)
    plt.title(f"Условная волатильность для {pair_name} (t-распределение)")
    plt.xlabel("Дата")
    plt.ylabel("Волатильность")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "conditional_volatility_models.png", dpi=300)
    plt.close()


def write_analysis_note(outdir: Path, pair_name: str) -> None:
    note = (
        f"Результаты анализа для {pair_name}.\n"
        "Логика приведена в соответствие со статьёй настолько, насколько это возможно в чистом Python:\n"
        "лог-доходности без *100, ARMA(1,1) как среднее уравнение статьи, "
        "GARCH/GARCH-M/EGARCH/GJR-GARCH/APARCH,\n"
        "распределения normal, Student t и skewed Student t в параметризации skew=1 как симметрия.\n"
    )
    with open(outdir / "analysis_note.txt", "w", encoding="utf-8") as handle:
        handle.write(note)


def run_article_style_analysis(
    df_raw: pd.DataFrame,
    pair_name: str,
    analysis_name: str,
    results_root: Path,
) -> None:
    outdir = ensure_dir(results_root / analysis_name)
    df = prepare_log_returns(df_raw)
    y = df["log_return"]

    print(f"\nАнализ: {analysis_name}")
    print(f"Валютная пара: {pair_name}")
    print(f"Число исходных наблюдений: {len(df_raw)}")
    print(f"Число наблюдений после построения доходностей: {len(df)}")

    write_analysis_note(outdir, pair_name)
    save_basic_plots(df, pair_name, outdir)

    stats = descriptive_stats(y)
    stats.to_csv(outdir / "descriptive_stats.csv")

    adf_rate = adf_report(df["rate"], "rate")
    adf_ret = adf_report(y, "log_return")
    adf_all = pd.concat([adf_rate.assign(block="rate"), adf_ret.assign(block="log_return")], ignore_index=True)
    adf_all.to_csv(outdir / "adf_tests.csv", index=False)

    jb_table = pd.DataFrame(
        {
            "metric": ["jarque_bera", "jarque_bera_pvalue"],
            "value": [stats["jarque_bera"], stats["jarque_bera_pvalue"]],
        }
    )
    jb_table.to_csv(outdir / "jarque_bera.csv", index=False)

    lb_ret = ljung_box_suite(y, "log_return")
    lb_abs = ljung_box_suite(y.abs(), "abs_log_return")
    lb_sq = ljung_box_suite(y ** 2, "sq_log_return")
    raw_lb = pd.concat([lb_ret, lb_abs, lb_sq], ignore_index=True)
    raw_lb.to_csv(outdir / "raw_ljungbox.csv", index=False)

    arma_grid = arma_grid_search(y, max_p=3, max_q=3)
    arma_grid.to_csv(outdir / "arma_grid_search.csv", index=False)

    arma_res, arma_resid = fit_arma_11(y)
    with open(outdir / "arma_11_summary.txt", "w", encoding="utf-8") as handle:
        handle.write(str(arma_res.summary()))

    arch_lm = arch_lm_report(arma_resid, lags=12)
    arch_lm.to_csv(outdir / "arch_lm_test.csv", index=False)

    fitted, model_table = fit_all_models(y, arma_res=arma_res)
    model_table.to_csv(outdir / "model_comparison.csv", index=False)

    symmetric = model_table[model_table["Model"].isin(["GARCH", "GARCH-M"])].copy()
    asymmetric = model_table[model_table["Model"].isin(["EGARCH", "GJR-GARCH", "APARCH"])].copy()
    symmetric.to_csv(outdir / "symmetric_models.csv", index=False)
    asymmetric.to_csv(outdir / "asymmetric_models.csv", index=False)

    save_model_summaries(fitted, outdir)
    save_combined_coefficients(fitted, outdir)
    save_residual_diagnostics(fitted, outdir)
    save_conditional_volatility_csv(fitted, df["date"], outdir)
    save_volatility_plot(fitted, df["date"], pair_name, outdir)

    print("\nЛучшие модели по AIC:")
    print(model_table.sort_values("AIC").head(10))


__all__ = [
    "load_eurusd_ecb",
    "load_usdrub_cbr",
    "run_article_style_analysis",
    "prepare_log_returns",
    "descriptive_stats",
    "adf_report",
    "ljung_box_suite",
    "arch_lm_report",
    "arma_grid_search",
    "fit_arma_11",
    "fit_all_models",
]


if __name__ == "__main__":
    results_root = ensure_dir(Path("results"))
    df_eurusd = load_eurusd_ecb()
    run_article_style_analysis(
        df_raw=df_eurusd,
        pair_name="EUR/USD",
        analysis_name="eurusd_article_style",
        results_root=results_root,
    )