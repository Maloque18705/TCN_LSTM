# Configuration values used by the notebook and DataLoader/DataProcess classes

# Sequence lengths
INPUT_STEPS = 100
OUTPUT_STEPS = 5  # notebook uses 5

# Data folder (relative to project root)
FOLDER_PATH = r"./Data/Z24"

# Sensor selection (1-based in notebook)
SELECTED_SENSOR = 1  # 1..27

# Number of designed samples used for slicing
DESIGN_SAMPLES = 10000

# Default slice start and finish for extracting a sensor
STEP_START = 20000
# Compute finish like notebook: STEP_START + DESIGN_SAMPLES + (INPUT_STEPS + OUTPUT_STEPS) - 1
STEP_FINISH = STEP_START + DESIGN_SAMPLES + (INPUT_STEPS + OUTPUT_STEPS) - 1

__all__ = [
	"INPUT_STEPS",
	"OUTPUT_STEPS",
	"FOLDER_PATH",
	"SELECTED_SENSOR",
	"DESIGN_SAMPLES",
	"STEP_START",
	"STEP_FINISH",
]
