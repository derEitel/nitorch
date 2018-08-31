import os, sys
sys.path.insert(0, os.path.abspath("../"))
from nitorch.transforms import IntensityRescale
if __name__ == "__main__":
	intensity = IntensityRescale(masked=False, on_gpu=True)
	print("hello world")