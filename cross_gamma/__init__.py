"""QIS Cross Gamma — book-level cross gamma analysis.

Reads daily per-contract cash gamma JSON files from the EDSLib network share,
aggregates across all trades with ticker normalisation, and serves the result
to the Analysis dashboard.
"""

from .ticker_map import normalize, normalize_pair, get_asset_class, sort_key
from .loader import find_latest_file, find_all_files, load_cross_gamma_data
from .aggregator import aggregate_book
