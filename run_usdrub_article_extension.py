from pathlib import Path

from common_replication import load_usdrub_cbr, run_article_style_analysis

RESULTS_ROOT = Path("results")


def main() -> None:
    df = load_usdrub_cbr(start="01/01/2016", end="31/12/2019")
    run_article_style_analysis(
        df_raw=df,
        pair_name="USD/RUB",
        analysis_name="usdrub_article_extension_2016_2019",
        results_root=RESULTS_ROOT,
    )


if __name__ == "__main__":
    main()