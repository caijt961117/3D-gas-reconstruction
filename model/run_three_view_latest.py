from __future__ import annotations

from model import RESULT_ROOT, THREE_VIEW_MODEL, THREE_VIEW_SCHEMES, print_min_only, run_pipeline


def main() -> None:
    summary = run_pipeline(
        THREE_VIEW_SCHEMES,
        RESULT_ROOT / "three-view",
        "three_view_E_field.csv",
        THREE_VIEW_MODEL,
    )
    print_min_only(summary)


if __name__ == "__main__":
    main()
