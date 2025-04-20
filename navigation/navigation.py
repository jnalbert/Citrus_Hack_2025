# take in the image, the bounding box, the ultra sonic sensor data
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# --- Configuration ---
CAMERA_FOV_HORIZONTAL = 53.5  # Example: Adjust based on your camera module
HISTOGRAM_ZONES = 30
IMAGE_WIDTH = 640 # Example, use the width of your camera image
IMAGE_HEIGHT = 480 # Typical height for 640x480 resolution

# Classes to ignore - adjust based on your YOLO model's class list
# Common floor/ground classes in COCO dataset
IGNORE_CLASSES = [
    'floor', 'ground', 'road', 'dirt', 'grass', 'pavement', 'asphalt',
    'carpet', 'mat', 'rug', 'sand', 'snow', 'earth', 'field'
]

# Position-based filtering
FLOOR_REGION_THRESHOLD = 0.35  # Bottom 65% of the frame might be floor

class HistogramBuffer:
    def __init__(self, buffer_size=20, num_zones=HISTOGRAM_ZONES):
        self.buffer_size = buffer_size
        self.histograms = []
        self.num_zones = num_zones
        self.count = 0
    
    def add_histogram(self, histogram):
        """Add a new histogram to the buffer"""
        if len(self.histograms) >= self.buffer_size:
            self.histograms.pop(0)  # Remove oldest histogram
        self.histograms.append(histogram)
        self.count += 1
    
    def get_average_histogram(self):
        """Get the average of all histograms in the buffer"""
        if not self.histograms:
            return np.zeros(self.num_zones)
        return np.mean(self.histograms, axis=0)
    
    def is_ready(self):
        """Check if we've collected enough histograms to make a decision"""
        return self.count >= self.buffer_size
    
    def reset_count(self):
        """Reset counter after sending action"""
        self.count = 0

# Create a global histogram buffer
histogram_buffer = HistogramBuffer(buffer_size=45)

def is_floor_detection(detection, image_height=IMAGE_HEIGHT):
    """
    Check if a detection is likely to be a floor or ground object.
    
    Args:
        detection: Dictionary with detection info
        image_height: Height of the frame
    
    Returns:
        bool: True if detection is likely floor/ground
    """
    # Check if class is in the ignore list
    if isinstance(detection.get('label'), str) and detection['label'].lower() in IGNORE_CLASSES:
        return True
        
    # Check if the bounding box is mostly in the bottom portion of the frame
    bbox = detection['bbox']
    y_min, y_max = bbox[1], bbox[3]
    
    # Calculate the center of the bounding box
    center_y = (y_min + y_max) / 2
    
    # Check if the center is in the bottom portion of the frame
    if center_y / image_height > FLOOR_REGION_THRESHOLD:
        # Additional check: is the box wider than tall? (floor objects typically are)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width > height * 1.5:  # Box is significantly wider than tall
            return True
    
    return False

def create_histogram(detections, image_width, camera_fov_horizontal, num_zones):
    """Creates a histogram representing obstacle distribution based on YOLO detections.

    Args:
        detections (list): A list of YOLO detection dictionaries.  Each dictionary
            should have the format:
            {'label': 'object_class', 'bbox': [x_min, y_min, x_max, y_max], 'confidence': 0.0-1.0}
        image_width (int): The width of the image in pixels.
        camera_fov_horizontal (float): The horizontal field of view of the camera in degrees.
        num_zones (int): The number of angular zones in the histogram.

    Returns:
        numpy.ndarray: A 1D numpy array representing the histogram.
    """
    histogram = np.zeros(num_zones)
    center_pixel = image_width / 2
    fov_rad_per_pixel = np.radians(camera_fov_horizontal) / image_width

    for detection in detections:
        # Skip floor/ground detections
        if is_floor_detection(detection):
            continue
            
        bbox = detection['bbox']
        x_min_pixel = bbox[0]
        x_max_pixel = bbox[2]

        angle_min_rad = (x_min_pixel - center_pixel) * fov_rad_per_pixel
        angle_max_rad = (x_max_pixel - center_pixel) * fov_rad_per_pixel
        angle_min_deg = np.degrees(angle_min_rad)
        angle_max_deg = np.degrees(angle_max_rad)

        # Map angles to histogram zones
        zone_min_index = int((angle_min_deg + camera_fov_horizontal / 2) / camera_fov_horizontal * num_zones)
        zone_max_index = int((angle_max_deg + camera_fov_horizontal / 2) / camera_fov_horizontal * num_zones)

        # Ensure indices are within bounds
        zone_min_index = max(0, zone_min_index)
        zone_max_index = min(num_zones - 1, zone_max_index)

        # Increment all zones the bounding box occupies
        for zone_index in range(zone_min_index, zone_max_index + 1):
            histogram[zone_index] += detection['confidence']
            
    kernel = np.array([0.05, 0.2, 0.5, 0.2, 0.05])  # Example Gaussian kernel
    histogram = convolve(histogram, kernel, mode='same')

    return histogram

def find_clear_path(histogram, min_width):
    """Finds the widest clear path in the histogram.

    Args:
        histogram (numpy.ndarray):  A 1D numpy array representing the histogram
        min_width (int):  The minimum number of zones required for a path to be considered clear
    Returns:
        A tuple (start_zone, end_zone) representing the indices of the widest clear path,
        or None if no clear path is found.
    """
    clear_paths = []
    start_index = -1
    for i, value in enumerate(histogram):
        if value < 0.5:  # Adjust threshold for "clear"
            if start_index == -1:
                start_index = i
        else:
            if start_index != -1:
                clear_paths.append((start_index, i - 1))
                start_index = -1
    if start_index != -1:
        clear_paths.append((start_index, len(histogram) - 1))

    # Find the widest clear path
    best_path = None
    max_width = 0
    for start, end in clear_paths:
        width = end - start + 1
        if width >= min_width and width > max_width:
            max_width = width
            best_path = (start, end)

    return best_path
    
def generate_action(clear_path, num_zones):
    """Generates steering and speed commands based on the clear path.

    Args:
        clear_path:  A tuple (start_zone, end_zone) of the clear path
        num_zones: The total number of zones
    Returns:
       A dictionary containing 'steering' and 'speed'
    """
    if clear_path:
        center_of_path = (clear_path[0] + clear_path[1]) / 2
        deviation_from_center = center_of_path - (num_zones / 2)
        normalized_deviation = deviation_from_center / (num_zones / 2)  # -1 to 1

        steering_angle = normalized_deviation * 20.0  # Adjust max steering angle.  Adjust this scaling factor!
        speed = 0.1  # Base speed, adjust as needed
        return {'steering': steering_angle, 'speed': speed}
    else:
        return {'steering': 0.0, 'speed': 0.0}  # Stop if no clear path

def visualize_histogram(histogram):
    """Visualizes the histogram as a bar chart.

    Args:
        histogram: A numpy array representing the histogram
    """
    # Visualize
    plt.bar(range(HISTOGRAM_ZONES), histogram)
    plt.xlabel("Angular Zone")
    plt.ylabel("Obstacle Density")
    plt.title(f"Histogram")
    plt.show(block=True)
    
def generate_action_from_bounding_boxes(bounding_boxes):
    """Generates steering and speed commands based on the bounding boxes.
    
    Now averages histograms over multiple frames for smoother control.
    Only returns an action after processing buffer_size (20) frames.
    Also filters out floor/ground detections.
    
    Args:
        bounding_boxes: A list of bounding boxes
    Returns:
        A dictionary containing 'steering' and 'speed', or None if not enough frames processed
    """
    global histogram_buffer
    
    # Log number of total detections vs filtered detections
    total_detections = len(bounding_boxes)
    filtered_bboxes = [box for box in bounding_boxes if not is_floor_detection(box)]
    filtered_count = total_detections - len(filtered_bboxes)
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} floor/ground detections out of {total_detections} total")
    
    # Create the histogram for current frame using filtered detections
    current_histogram = create_histogram(filtered_bboxes, IMAGE_WIDTH, CAMERA_FOV_HORIZONTAL, HISTOGRAM_ZONES)
    
    # Add to buffer
    histogram_buffer.add_histogram(current_histogram)
    
    # Return None until we have enough data
    if not histogram_buffer.is_ready():
        return None
    
    # Get the averaged histogram
    averaged_histogram = histogram_buffer.get_average_histogram()
    # visualize_histogram(averaged_histogram)
    
    # Find a clear path using the averaged histogram
    min_safe_width = 5
    clear_path = find_clear_path(averaged_histogram, min_safe_width)
    
    # Generate an action based on the averaged histogram
    action = generate_action(clear_path, HISTOGRAM_ZONES)
    
    # Reset counter to start fresh for next batch
    histogram_buffer.reset_count()
    
    return action