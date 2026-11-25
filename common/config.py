"""Project-wide constants shared by Parts 1, 2 and 3.

Every decision that must stay identical across the three parts lives here,
so SQL, the Python pipeline and the ML models can never drift apart.
Decision IDs (D1-D7) refer to the decision log in the root README.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the repository root, so the project runs anywhere)
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_CSV = DATA_DIR / "raw" / "Metro_Interstate_Traffic_Volume.csv"
CLEAN_CSV = DATA_DIR / "processed" / "traffic_clean.csv"        # D2: single source of truth
FEATURES_CSV = DATA_DIR / "processed" / "traffic_features.csv"
SQLITE_DB = ROOT_DIR / "part1_data_analytics" / "traffic.db"
FIGURES_DIR = ROOT_DIR / "part2_python" / "figures"
LOG_FILE = ROOT_DIR / "part2_python" / "pipeline.log"

# --------------------------------------------------------------------------
# Reproducibility (D7 also depends on this)
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Schema: the columns the raw file must contain (validated before anything else)
# --------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
]

# --------------------------------------------------------------------------
# D1: hourly grain. When one hour has several weather labels, keep the most
# severe. Lower number = more severe (more impact on driving conditions).
# --------------------------------------------------------------------------
WEATHER_SEVERITY = {
    "Thunderstorm": 1,
    "Squall": 2,
    "Snow": 3,
    "Rain": 4,
    "Drizzle": 5,
    "Fog": 6,
    "Smoke": 7,
    "Haze": 8,
    "Mist": 9,
    "Clouds": 10,
    "Clear": 11,
}

# --------------------------------------------------------------------------
# D4: physically plausible ranges. Values outside are treated as sensor errors
# and imputed with the median of the same calendar month.
# --------------------------------------------------------------------------
TEMP_MIN_K = 200.0        # about -73 C, far below any Minneapolis record
TEMP_MAX_K = 330.0        # about 57 C
RAIN_MAX_MM = 200.0       # one hour of rain; the raw file contains 9,831.3 mm
                          # (55.6 mm, a real "very heavy rain" hour, must stay valid)

# D11: rainfall is imputed with the median of the same weather_description.
# A group needs at least this many valid hours, otherwise the monthly median is used.
MIN_GROUP_SIZE = 5

# traffic_volume is the target, so it is FLAGGED, never imputed (see pipeline step 8)
DAYTIME_START_HOUR = 6
DAYTIME_END_HOUR = 21
MIN_DAYTIME_VOLUME = 100  # daytime hours below this look like road closures

# Gaps longer than this are reported in the log (L2: nine missing months)
GAP_WARNING_DAYS = 7

# --------------------------------------------------------------------------
# D5: three congestion definitions, each required by a different task
# --------------------------------------------------------------------------
CONGESTION_THRESHOLD = 5500                 # Part 1 Task 3: congested = volume > 5,500
CATEGORY_BANDS = (4500, 5500)               # Part 1 Task 4: Low < 4,500 <= Medium <= 5,500 < High
HIGH_TEMP_K = 292.0                         # Part 1 Task 3.2: high temperature > 292 K
PEAK_HOURS = [6, 7, 8, 15, 16, 17]          # hours where congestion actually occurs (>40% of hours)
# Part 3 uses data-driven quartiles of traffic_volume, computed at run time.

# --------------------------------------------------------------------------
# D6: weather sets for the proxy accident-risk label (the brief names them
# but does not define them)
# --------------------------------------------------------------------------
SEVERE_WEATHER = ["Thunderstorm", "Snow", "Squall"]
LOW_VISIBILITY = ["Fog", "Mist", "Haze", "Smoke"]

# --------------------------------------------------------------------------
# D7: chronological train/test split (hourly traffic is autocorrelated)
# --------------------------------------------------------------------------
TEST_START = "2018-01-01"

# --------------------------------------------------------------------------
# Part 3 association rules (Apriori)
# --------------------------------------------------------------------------
APRIORI_MIN_SUPPORT = 0.01      # an itemset must cover at least 1% of hours (~400)
APRIORI_MIN_CONFIDENCE = 0.5    # a rule must be right at least half the time
KMEANS_MAX_K = 6                # largest number of clusters still easy to interpret
