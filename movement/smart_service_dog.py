#!/usr/bin/env python3
"""
smart_service_dog.py - Main controller for the PiCar-X Smart Service Dog project
Integrates voice control with movement control for a complete system.
"""
import time
import logging
from general import MovementController
from voice_controller import VoiceController

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartServiceDog:
    def __init__(self):
        """Initialize the Smart Service Dog system."""
        try:
            logger.info("Initializing Smart Service Dog system...")
            
            # Initialize movement controller
            self.movement = MovementController()
            
            # Initialize voice controller
            self.voice = VoiceController()
            
            # Set the command handler
            self.voice.set_command_callback(self.handle_voice_command)
            
            logger.info("Smart Service Dog system initialized successfully")
            self.voice.speak("Smart Service Dog system ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Smart Service Dog: {e}")
            raise

    def handle_voice_command(self, command):
        """
        Process a voice command and execute the corresponding movement.
        
        Args:
            command (dict): Command dictionary containing action and parameters
        """
        try:
            action = command['action']
            params = command['parameters']
            
            if action == 'forward':
                speed = params.get('speed', 50)  # Default speed 50
                self.voice.speak(f"Moving forward at speed {speed}")
                self.movement.move_forward(speed=speed)
                
            elif action == 'reverse':
                speed = params.get('speed', 50)  # Default speed 50
                self.voice.speak(f"Moving backward at speed {speed}")
                self.movement.reverse(speed=speed)
                
            elif action == 'left':
                angle = params.get('angle', 30)  # Default angle 30
                self.voice.speak(f"Turning left {angle} degrees")
                self.movement.turn_left(angle=angle)
                
            elif action == 'right':
                angle = params.get('angle', 30)  # Default angle 30
                self.voice.speak(f"Turning right {angle} degrees")
                self.movement.turn_right(angle=angle)
                
            elif action == 'stop':
                self.voice.speak("Stopping")
                self.movement.stop()
                
            elif action == 'help':
                self.provide_help()
                
            elif action == 'quit':
                self.voice.speak("Shutting down voice control")
                self.voice.stop_listening()
                
        except Exception as e:
            logger.error(f"Error handling command: {e}")
            self.voice.speak("I had a problem processing that command")

    def provide_help(self):
        """Provide voice instructions on available commands."""
        help_text = """
        Available commands include: 
        Move forward, 
        Reverse, 
        Turn left, 
        Turn right, 
        Stop, 
        Help, 
        and Quit. 
        You can specify speed or angle by including a number.
        """
        self.voice.speak(help_text)

    def start(self):
        """Start the Smart Service Dog system."""
        try:
            logger.info("Starting Smart Service Dog system")
            self.voice.speak("Starting Smart Service Dog system")
            
            # Start voice control
            self.voice.start_listening()
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.voice.speak("Error starting system")

    def stop(self):
        """Stop the Smart Service Dog system."""
        try:
            logger.info("Stopping Smart Service Dog system")
            self.voice.speak("Stopping Smart Service Dog system")
            
            # Stop voice control
            self.voice.stop_listening()
            
            # Stop movement
            self.movement.stop()
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")

    def cleanup(self):
        """Clean up resources before shutdown."""
        try:
            logger.info("Cleaning up Smart Service Dog resources")
            self.voice.speak("Shutting down")
            
            # Cleanup subsystems
            self.voice.cleanup()
            self.movement.cleanup()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Main entry point
if __name__ == "__main__":
    smart_dog = SmartServiceDog()
    
    try:
        print("Starting Smart Service Dog. Use voice commands to control.")
        smart_dog.start()
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        smart_dog.cleanup()