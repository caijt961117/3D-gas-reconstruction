from __future__ import annotations

from model import RESULT_ROOT, TWO_VIEW_MODEL, print_min_only, run_pipeline


TWO_VIEW_SCHEMES = {
    "scheme_1": [(0, 20, 40), (20, 0, 30)],
    "scheme_2": [(20, 20, 40), (20, 0, 30)],
    "scheme_3": [(40, 20, 40), (20, 0, 30)],
}


def main() -> None:
    summary = run_pipeline(
        TWO_VIEW_SCHEMES,
        RESULT_ROOT / "two-view",
        "two_view_E_field.csv",
        TWO_VIEW_MODEL,
        selection="mean",
    )
    print_min_only(summary)


if __name__ == "__main__":
    main()
