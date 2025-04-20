"""
PiCar-X Movement Controller
This module handles the basic movement operations for the PiCar-X Smart Service Dog project.
"""
import time
import cv2
import numpy as np
from robot_hat import TTS
from picamera2 import Picamera2, Preview
import libcamera
from picarx import Picarx
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MovementController:
    def __init__(self):
        """Initialize the PiCar-X movement controller."""
        try:
            self.px = Picarx()
            # Obstacle detection threshold (cm) and TTS setup
            self.obstacle_threshold = 20  # Distance in cm to trigger obstacle detection
            # Initialize straight-line correction parameters
            self.drift_correction = 0  # Default correction value
            self.last_error = 0
            self.straight_kp = 0.5  # Proportional gain
            self.straight_kd = 0.2  # Derivative gain
            
            # Initialize camera using Picamera2
            self.picam2 = Picamera2()
            self.camera_config = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                transform=libcamera.Transform(hflip=False, vflip=False)
            )
            self.picam2.configure(self.camera_config)
            self.camera_active = False
            
            logger.info("PiCar-X movement controller initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PiCar-X: {e}")
            raise

    def start_camera(self, display_preview=False):
        """
        Start the camera stream.
        
        Args:
            display_preview (bool): Whether to display a preview window
        """
        try:
            if not self.camera_active:
                if display_preview:
                    self.picam2.start_preview(Preview.QTGL)
                self.picam2.start()
                self.camera_active = True
                logger.info("Camera started")
            else:
                logger.info("Camera already active")
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            
    def stop_camera(self):
        """Stop the camera stream."""
        try:
            if self.camera_active:
                self.picam2.stop()
                self.camera_active = False
                logger.info("Camera stopped")
            else:
                logger.info("Camera already stopped")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
    
    def capture_image(self):
        """Capture a single image from the camera."""
        try:
            if not self.camera_active:
                self.start_camera()
                # Small delay to let the camera initialize
                time.sleep(0.5)
            
            # Capture and return an image
            image = self.picam2.capture_array()
            return image
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return None
    
    def detect_lines(self, image):
        """
        Detect lines in an image for path following.
        
        Args:
            image: Image captured from the camera
            
        Returns:
            error: Deviation from center path (positive is right, negative is left)
            processed_image: Image with detected lines drawn (for debugging)
        """
        try:
            if image is None:
                logger.error("No image provided for line detection")
                return 0, None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blur, 50, 150)
            
            # Define region of interest (bottom half of image)
            height, width = edges.shape
            mask = np.zeros_like(edges)
            polygon = np.array([[(0, height), (0, height//2), 
                                 (width, height//2), (width, height)]], np.int32)
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Use Hough transform to detect lines
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 20, 
                                   minLineLength=20, maxLineGap=300)
            
            # Create an image to visualize the result
            line_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Variables for calculating average line position and slope
            left_lines = []
            right_lines = []
            center_point = width // 2
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Calculate slope
                    if x2 - x1 == 0:  # Avoid division by zero
                        continue
                        
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Filter lines based on slope (horizontal lines are irrelevant)
                    if abs(slope) < 0.5:
                        continue
                        
                    # Determine if line is on left or right side
                    if slope < 0 and x1 < center_point:  # Right side (in image coordinates)
                        right_lines.append(line[0])
                    elif slope > 0 and x1 > center_point:  # Left side
                        left_lines.append(line[0])
                        
                    # Draw the line for visualization
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate center point between the average left and right line
            if len(left_lines) > 0 and len(right_lines) > 0:
                # Average left line position
                left_x = np.mean([line[0] for line in left_lines])
                
                # Average right line position
                right_x = np.mean([line[0] for line in right_lines])
                
                # Center between the lines
                path_center = (left_x + right_x) // 2
                
                # Calculate error from the center of the image
                error = center_point - path_center
                
                # Draw the center point for visualization
                cv2.circle(line_image, (int(path_center), height-30), 10, (0, 0, 255), -1)
            else:
                # If can't detect both lines, return 0 error (no correction)
                error = 0
            
            # Combine the original image with the line detections
            processed_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
            
            return error, processed_image
            
        except Exception as e:
            logger.error(f"Error detecting lines: {e}")
            return 0, None

    def move_forward(self, speed=50, duration=None, maintain_straight=True):
        """
        Move the PiCar-X forward.
        
        Args:
            speed (int): Speed value between 0-100.
            duration (float, optional): Time in seconds to move. If None, moves indefinitely.
            maintain_straight (bool): Whether to apply straight-line correction.
        """
        try:
            logger.info(f"Moving forward at speed {speed}")
            speed = min(max(speed, 0), 100)  # Clamp speed between 0-100
            
            start_time = time.time()
            while duration is None or time.time() - start_time < duration:
                # Ultrasonic-based obstacle detection
                dist = self.px.ultrasonic.read()
                logger.debug(f"Ultrasonic distance: {dist:.2f} cm")
                if dist is not None and dist < self.obstacle_threshold:
                    logger.info(f"Obstacle detected at {dist:.2f} cm, stopping")
                    # Speak the detection alert
                    return
                
                if maintain_straight:
                    self._apply_straight_line_correction(speed)
                else:
                    self.px.forward(speed)
                
                if duration is not None:
                    time.sleep(0.05)  # Small sleep for correction updates
                else:
                    break  # If no duration, just set and exit
                    
            if duration is not None:
                self.stop()
                
        except Exception as e:
            logger.error(f"Error moving forward: {e}")
            self.stop()

    def _apply_straight_line_correction(self, speed):
        """
        Apply straight-line correction using camera input to detect and follow a path.
        
        Uses Picamera2 library to detect lines and calculates the error
        from the center to adjust steering accordingly.
        """
        try:
            # Capture image from camera
            image = self.capture_image()
            if image is None:
                # If image capture failed, continue moving without correction
                self.px.forward(speed)
                return
                
            # Detect lines and get error from center
            current_error, _ = self.detect_lines(image)
            
            # PD controller calculation
            correction = (self.straight_kp * current_error + 
                         self.straight_kd * (current_error - self.last_error))
            
            self.last_error = current_error
            self.drift_correction = correction
            
            # Apply the correction - adjust steering angle while maintaining forward motion
            # Clamp the steering angle to avoid extreme corrections
            max_angle = 40
            steering_angle = max(min(self.drift_correction, max_angle), -max_angle)
            
            logger.debug(f"Line error: {current_error:.2f}, Steering angle: {steering_angle:.2f}")
            self.px.set_dir_servo_angle(steering_angle)
            self.px.forward(speed)
            
        except Exception as e:
            logger.error(f"Error in straight-line correction: {e}")
            # Continue moving forward without correction
            self.px.forward(speed)

    def reverse(self, speed=50, duration=None):
        """
        Move the PiCar-X in reverse.
        
        Args:
            speed (int): Speed value between 0-100.
            duration (float, optional): Time in seconds to move. If None, moves indefinitely.
        """
        try:
            logger.info(f"Moving in reverse at speed {speed}")
            speed = min(max(speed, 0), 100)
            self.px.backward(speed)
            
            if duration is not None:
                time.sleep(duration)
                self.stop()
                
        except Exception as e:
            logger.error(f"Error moving in reverse: {e}")
            self.stop()

    def turn_left(self, angle=30, speed=40):
        """
        Turn the PiCar-X to the left.
        
        Args:
            angle (int): Steering angle, typically between 0-40.
            speed (int): Forward speed while turning.
        """
        try:
            logger.info(f"Turning left with angle {angle}")
            # Ensure angle is positive for left turn
            angle = abs(angle)
            self.px.set_dir_servo_angle(angle)
            self.px.forward(speed)
        except Exception as e:
            logger.error(f"Error turning left: {e}")
            self.stop()

    def turn_right(self, angle=30, speed=40):
        """
        Turn the PiCar-X to the right.
        
        Args:
            angle (int): Steering angle, typically between 0-40.
            speed (int): Forward speed while turning.
        """
        try:
            logger.info(f"Turning right with angle {angle}")
            # Ensure angle is negative for right turn
            angle = -abs(angle)
            self.px.set_dir_servo_angle(angle)
            self.px.forward(speed)
        except Exception as e:
            logger.error(f"Error turning right: {e}")
            self.stop()

    def stop(self):
        """Stop all movement of the PiCar-X."""
        try:
            logger.info("Stopping movement")
            self.px.stop()
            # Reset steering to center
            self.px.set_dir_servo_angle(0)
            self.drift_correction = 0
            self.last_error = 0
        except Exception as e:
            logger.error(f"Error stopping: {e}")

    def cleanup(self):
        """Clean up and release resources."""
        try:
            logger.info("Cleaning up resources")
            self.stop()
            # Release Picamera2 resources
            # Picamera2 cleanup logic can be added here
            # Add any additional cleanup here
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage
if __name__ == "__main__":
    controller = MovementController()
    tts = TTS()
    tts.lang("en-US")
    
    # try:
    #     # Test sequence
    #     print("Moving forward with straight-line correction...")
    #     controller.move_forward(speed=200, duration=2, maintain_straight=True)
        
    #     print("Turning left...")
    #     controller.turn_left(angle=30, speed=10)
    #     time.sleep(1)
        
    #     print("Turning right...")
    #     controller.turn_right(angle=30, speed=10)
    #     time.sleep(1)
        
    #     print("Moving in reverse...")
    #     controller.reverse(speed=40, duration=2)
        
    #     print("Stopping...")
    #     controller.stop()


    try:
        while True:
            # Example of continuous movement
            controller.move_forward(speed=50, duration=1, maintain_straight=True)
            time.sleep(0.1)

            if controller.px.ultrasonic.read() < controller.obstacle_threshold:
                controller.stop()
                words = "Obstacle detected, stopping."
                tts.say(words)
                time.sleep(1)
                controller.reverse(speed=50, duration=1)
                controller.stop()
                time.sleep(1)
                break
        
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        controller.cleanup()