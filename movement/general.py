#!/usr/bin/env python3
"""
PiCar-X Movement Controller
This module handles the basic movement operations for the PiCar-X Smart Service Dog project.
"""
import time
from robot_hat import TTS
from vilib import Vilib
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
            
            # Initialize camera using Vilib
            Vilib.camera_start(vflip=False, hflip=False)  # Adjust flip parameters if needed
            Vilib.display(False)  # Don't display the camera stream by default
            
            logger.info("PiCar-X movement controller initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PiCar-X: {e}")
            raise

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
        
        Uses Vilib library to detect lines and calculates the error
        from the center to adjust steering accordingly.
        """
        try:
            # Get line detection results from Vilib
            line_status = Vilib.line_detect_get_status()
            
            if line_status:  # If a line is detected
                line_info = Vilib.line_detect_get_result()
                
                # Get the center of line
                if 'center_x' in line_info:
                    cx = line_info['center_x']
                    frame_width = 320  # Using the width we configured for the camera
                    
                    # Calculate error (distance from center)
                    current_error = cx - (frame_width / 2)
                    
                    # Scale the error to be more manageable
                    # Higher values mean stronger steering corrections
                    current_error = current_error / (frame_width / 2) * 20
                else:
                    current_error = 0
                    logger.debug("Line detected but center_x not found")
            else:
                # No line detected
                current_error = 0
                logger.debug("No line detected in camera frame")
            
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
            # Release Vilib camera resources
            Vilib.camera_close()
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