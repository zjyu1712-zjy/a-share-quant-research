"""Compatibility entry point for the canonical baseline grid search.

The former implementation duplicated execution logic and optimized a
price-derived pseudo-sentiment factor together with individual stop losses.
All callers now use :mod:`src.grid_search_1M`, which shares the production
selector and the canonical next-session-open execution engine.
"""

from src.grid_search_1M import main, run_grid_search

__all__ = ["main", "run_grid_search"]


if __name__ == "__main__":
    main()
