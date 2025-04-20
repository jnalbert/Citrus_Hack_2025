#!/usr/bin/env python3
"""
PiCar-X Movement Controller
This module handles the basic movement operations for the PiCar-X Smart Service Dog project.
"""
import time
from picarx import Picarx
from robot_hat import TTS
import logging
import random


# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovementController:
    def __init__(self, picar=None):
        """Initialize the PiCar-X movement controller."""
        try:
            if picar is None:
                self.px = Picarx()
            else:
                self.px = picar
            # Initialize straight-line correction parameters
            self.drift_correction = 0  # Default correction value
            self.obstacle_threshold = 20  # Distance in cm to trigger obstacle avoidance
            self.tts = TTS()
            self.last_error = 0
            self.straight_kp = 0.5  # Proportional gain
            self.straight_kd = 0.2  # Derivative gain
            self.ultrasonic = self.px.ultrasonic
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
        Apply straight-line correction using gyroscope/sensor data.
        
        In a full implementation, this would use gyro data from an IMU or
        analyze camera data for lane/path following.
        """
        try:
            # This is where you would get actual sensor readings
            # For example: current_angle = self.px.get_imu_data()
            # For now, we'll simulate with a placeholder
            
            # Placeholder - in a real implementation, get actual drift
            # Example with ultrasonic or IR sensors on both sides
            # left_distance = self.px.get_left_sensor()
            # right_distance = self.px.get_right_sensor()
            # current_error = left_distance - right_distance
            
            # Simulated error (you'll replace this with real sensor data)
            current_error = 0  # Replace with actual sensor reading
            
            # PD controller calculation
            correction = (self.straight_kp * current_error + 
                         self.straight_kd * (current_error - self.last_error))
            
            self.last_error = current_error
            self.drift_correction = correction
            
            # Apply the correction - adjust steering angle while maintaining forward motion
            self.px.set_dir_servo_angle(self.drift_correction)
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

    def reverse_left(self, angle=30, speed=40, duration=None):
        """
        Reverse the PiCar-X while steering left.
        
        Args:
            angle (int): Steering angle, typically between 0-40.
            speed (int): Reverse speed while turning.
            duration (float, optional): Time in seconds to move. If None, moves indefinitely.
        """
        try:
            logger.info(f"Reversing left with angle {angle} at speed {speed}")
            # Ensure angle is positive for left turn (same direction as forward left turn)
            angle = abs(angle)
            self.px.set_dir_servo_angle(angle)
            self.px.backward(speed)
            
            if duration is not None:
                time.sleep(duration)
                self.stop()
                
        except Exception as e:
            logger.error(f"Error reversing left: {e}")
            self.stop()
            
    def reverse_right(self, angle=30, speed=40, duration=None):
        """
        Reverse the PiCar-X while steering right.
        
        Args:
            angle (int): Steering angle, typically between 0-40.
            speed (int): Reverse speed while turning.
            duration (float, optional): Time in seconds to move. If None, moves indefinitely.
        """
        try:
            logger.info(f"Reversing right with angle {angle} at speed {speed}")
            # Ensure angle is negative for right turn (same direction as forward right turn)
            angle = -abs(angle)
            self.px.set_dir_servo_angle(angle)
            self.px.backward(speed)
            
            if duration is not None:
                time.sleep(duration)
                self.stop()
                
        except Exception as e:
            logger.error(f"Error reversing right: {e}")
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
            # self.px.forward(speed)
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
            # self.px.forward(speed)
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
            # Add any additional cleanup here
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage
if __name__ == "__main__":
    controller = MovementController()
    controller.tts.lang("en-US")
    controller.tts.say("Hello, I am your PiCar-X. Let's start moving!")
    time.sleep(2)
    try:
        print("Starting autonomous navigation. Press Ctrl+C to exit.")
        # --- parameters you can tweak ---
        forward_speed   = 40    # speed when moving forward
        reverse_speed   = 30    # speed when backing up
        turn_angle      = 30    # steering angle for turns
        step_duration   = 0.3   # seconds to move forward each cycle

        while True:
            # 1) Forward in short steps
            controller.move_forward(
                speed=forward_speed,
                duration=step_duration,
                maintain_straight=True
            )

            # 2) Read distance sensor
            dist = controller.ultrasonic.read()
            print(f"[Sensor] Distance = {dist:.1f} cm")

            # 3) If too close, stop and evade
            if 0 < dist < controller.obstacle_threshold:
                print("[Alert] Obstacle detected—evading!")
                controller.tts.say("Obstacle detected, evading!")
                controller.stop()
                time.sleep(0.5)

                # 3a) Back up with a random steering bias
                if random.random() < 0.5:
                    controller.reverse_left(
                        angle=turn_angle,
                        speed=reverse_speed,
                        duration=1.0
                    )
                else:
                    controller.reverse_right(
                        angle=turn_angle,
                        speed=reverse_speed,
                        duration=1.0
                    )

                # 3b) Pivot turn in place (forward) randomly
                if random.random() < 0.5:
                    controller.turn_left(
                        angle=turn_angle,
                        speed=forward_speed
                    )
                else:
                    controller.turn_right(
                        angle=turn_angle,
                        speed=forward_speed
                    )

                # small pause before next cycle
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nNavigation stopped by user")
    finally:
        controller.cleanup()
        print("Clean exit")