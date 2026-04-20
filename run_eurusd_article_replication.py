from pathlib import Path

from common_replication import load_eurusd_ecb, run_article_style_analysis

RESULTS_ROOT = Path("results")


def main() -> None:
    df = load_eurusd_ecb(start="2010-01-01", end="2016-12-30", align_business_days=True)
    run_article_style_analysis(
        df_raw=df,
        pair_name="EUR/USD",
        analysis_name="eurusd_article_replication_2010_2016",
        results_root=RESULTS_ROOT,
    )


if __name__ == "__main__":
    main()