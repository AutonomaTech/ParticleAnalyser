# this script will run a test image and verify the outupts


CIRCLE_IMAGE = "quality-test-files\\circles_image.png"
# IN PX^2
EXPECTED_AREAS = [
    196349.54,
    159043.13,
    125663.71,
    96211.28,
    70685.83,
    49087.39,
    31415.93,
    17671.46,
    7853.98,
    1963.50
]

EXPECTED_PSD = [ 
    25.97,
    21.04,
    16.62,
    12.73,
    9.35,
    6.49,
    4.16,
    2.34,
    1.04,
    0.26
]

BINS = [
    500,
    450,
    400,
    350,
    300,
    250,
    200,
    150,
    100,
    50
]

def run_test():
    from .. import ImageAnalysisModel as pa
    analyser = pa.ImageAnalysisModel("quality-test-files\\circle-checker", 1)
    

