from nitorch.transforms import IntensityRescale

if __name__ == "__main__":
    intensity = IntensityRescale(masked=False, on_gpu=True)
    print("hello world")
