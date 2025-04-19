import numpy as np
from navigation import create_histogram, find_clear_path, generate_action

# --- Configuration ---
CAMERA_FOV_HORIZONTAL = 53.5  # Example: Adjust based on your camera module
HISTOGRAM_ZONES = 30
IMAGE_WIDTH = 640 # Example, use the width of your camera image
MIN_SAFE_WIDTH_ZONES = 5  # Minimum width for a clear path

def test_histogram_generation():
    """
    Tests the histogram generation with several scenarios
    """
    print("\n--- Testing Histogram Generation ---")
    test_cases = [
        {
            "name": "Obstacle Left",
            "detections": [
                {'label': 'box', 'bbox': [100, 50, 250, 300], 'confidence': 0.95},
            ],
            "expected_histogram": [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
        },
        {
            "name": "Obstacle Right",
            "detections": [
                {'label': 'rock', 'bbox': [400, 100, 500, 200], 'confidence': 0.9},
            ],
            "expected_histogram": [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
        },
        {
            "name": "Obstacle Center",
            "detections": [
                {'label': 'wall', 'bbox': [250, 0, 350, 400], 'confidence': 0.98},
            ],
            "expected_histogram": [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
        },
        {
            "name": "Two Obstacles",
            "detections": [
                {'label': 'box', 'bbox': [50, 100, 150, 200], 'confidence': 0.85},
                {'label': 'chair', 'bbox': [450, 150, 550, 250], 'confidence': 0.92},
            ],
            "expected_histogram": [
                0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.92, 0.0
            ],
        },
        {
            "name": "No Obstacles",
            "detections": [],
            "expected_histogram": [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
        },
    ]

    for test_case in test_cases:
        name = test_case["name"]
        detections = test_case["detections"]
        expected_histogram = test_case["expected_histogram"]

        histogram = create_histogram(detections, IMAGE_WIDTH, CAMERA_FOV_HORIZONTAL, HISTOGRAM_ZONES)
        print(f"\nTest Case: {name}")
        print("  Generated Histogram:", histogram)
        print("  Expected Histogram:", expected_histogram)

        if np.allclose(histogram, expected_histogram):
            print("  Result: PASS")
        else:
            print("  Result: FAIL")



def test_path_finding():
    """Tests the path finding with several scenarios"""
    print("\n--- Testing Path Finding ---")
    test_cases = [
        {
            "name": "Clear Path Center",
            "histogram": [
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
            ],
            "expected_clear_path": (0, 29),
        },
        {
            "name": "Obstacle on Left",
            "histogram": [
                0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
            ],
            "expected_clear_path": (5, 29),
        },
        {
            "name": "Obstacle on Right",
            "histogram": [
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9
            ],
            "expected_clear_path": (0, 23),
        },
        {
            "name": "Two Obstacles",
            "histogram": [
                0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9,
                0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9,
                0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9
            ],
            "expected_clear_path": (3, 8),
        },
        {
            "name": "No Clear Path",
            "histogram": [
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9
            ],
            "expected_clear_path": None,
        },
    ]

    for test_case in test_cases:
        name = test_case["name"]
        histogram = test_case["histogram"]
        expected_clear_path = test_case["expected_clear_path"]

        clear_path = find_clear_path(histogram, MIN_SAFE_WIDTH_ZONES)
        print(f"\nTest Case: {name}")
        print("  Generated Clear Path:", clear_path)
        print("  Expected Clear Path:", expected_clear_path)

        if clear_path == expected_clear_path:
            print("  Result: PASS")
        else:
            print("  Result: FAIL")



def test_action_generation():
    """Tests action generation."""
    print("\n--- Testing Action Generation ---")
    test_cases = [
        {
            "name": "Straight Ahead",
            "clear_path": (10, 20),
            "expected_action": {'steering': 0.0, 'speed': 0.5},
        },
        {
            "name": "Turn Left",
            "clear_path": (0, 10),
            "expected_action": {'steering': -20.0, 'speed': 0.5},
        },
        {
            "name": "Turn Right",
            "clear_path": (20, 29),
            "expected_action": {'steering': 20.0, 'speed': 0.5},
        },
        {
            "name": "No Clear Path",
            "clear_path": None,
            "expected_action": {'steering': 0.0, 'speed': 0.0},
        },
    ]

    for test_case in test_cases:
        name = test_case["name"]
        clear_path = test_case["clear_path"]
        expected_action = test_case["expected_action"]

        action = generate_action(clear_path, HISTOGRAM_ZONES)
        print(f"\nTest Case: {name}")
        print("  Generated Action:", action)
        print("  Expected Action:", expected_action)

        if action == expected_action:
            print("  Result: PASS")
        else:
            print("  Result: FAIL")



if __name__ == "__main__":
    test_histogram_generation()
    test_path_finding()
    test_action_generation()