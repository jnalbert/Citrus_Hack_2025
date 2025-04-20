#!/usr/bin/env python3
"""
voice_control.py - Voice Control Module for PiCar-X Smart Service Dog
Provides speech recognition for commands and text-to-speech for responses.
"""
import speech_recognition as sr
import threading
import time
import logging
from robot_hat import TTS

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceController:
    def __init__(self):
        """Initialize the voice control system."""
        try:
            # Initialize the speech recognizer
            self.recognizer = sr.Recognizer()
            
            # Initialize text-to-speech using robot_hat
            self.tts = TTS()
            
            # Flag to control recognition loop
            self.is_listening = False
            self.listener_thread = None
            
            # Command callback function (to be set by the main program)
            self.command_callback = None
            
            logger.info("Voice control system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice control: {e}")
            raise

    def set_command_callback(self, callback_function):
        """
        Set the callback function that will be called when a command is recognized.
        
        Args:
            callback_function: Function that takes a command string as parameter
        """
        self.command_callback = callback_function

    def speak(self, text):
        """
        Convert text to speech using robot_hat TTS.
        
        Args:
            text (str): The text to be spoken.
        """
        try:
            logger.info(f"Speaking: {text}")
            self.tts.say(text)
        except Exception as e:
            logger.error(f"TTS error: {e}")

    def recognize_speech(self):
        """
        Capture audio from microphone and convert to text.
        
        Returns:
            str: Recognized text or empty string if recognition fails.
        """
        try:
            with sr.Microphone() as source:
                logger.info("Listening for commands...")
                self.speak("Listening")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                # Try to recognize the speech
                logger.info("Processing speech...")
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Recognized: {text}")
                return text.lower()
                
        except sr.WaitTimeoutError:
            logger.info("No speech detected within timeout")
            return ""
        except sr.UnknownValueError:
            logger.info("Could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return ""

    def parse_command(self, text):
        """
        Parse the recognized text into a structured command.
        
        Args:
            text (str): The recognized speech text
            
        Returns:
            dict: A dictionary containing command type and parameters
        """
        command = {
            'action': None,
            'parameters': {}
        }
        
        # Check for basic command types
        if "forward" in text or "go forward" in text or "move forward" in text:
            command['action'] = 'forward'
        elif "reverse" in text or "backward" in text or "go back" in text:
            command['action'] = 'reverse'
        elif "turn left" in text or "go left" in text:
            command['action'] = 'left'
        elif "turn right" in text or "go right" in text:
            command['action'] = 'right'
        elif "stop" in text or "halt" in text:
            command['action'] = 'stop'
        elif "help" in text:
            command['action'] = 'help'
        elif "quit" in text or "exit" in text:
            command['action'] = 'quit'
        
        # Extract parameters (speed or angle)
        if command['action'] in ['forward', 'reverse']:
            # Look for a speed value
            for word in text.split():
                if word.isdigit():
                    command['parameters']['speed'] = min(int(word), 100)
                    break
        
        if command['action'] in ['left', 'right']:
            # Look for an angle value
            for word in text.split():
                if word.isdigit():
                    command['parameters']['angle'] = min(int(word), 40)
                    break
                    
        return command

    def continuous_listening(self):
        """Background thread function for continuous command recognition."""
        try:
            while self.is_listening:
                speech_text = self.recognize_speech()
                if speech_text and self.command_callback:
                    command = self.parse_command(speech_text)
                    if command['action']:
                        self.command_callback(command)
                    else:
                        self.speak("Command not recognized. Please try again.")
                time.sleep(0.1)  # Small delay to prevent CPU hogging
        except Exception as e:
            logger.error(f"Error in listening thread: {e}")
        finally:
            logger.info("Listening thread terminated")

    def start_listening(self):
        """Start listening for voice commands in a background thread."""
        if not self.is_listening:
            self.is_listening = True
            self.listener_thread = threading.Thread(target=self.continuous_listening)
            self.listener_thread.daemon = True
            self.listener_thread.start()
            logger.info("Voice command listening started")
            self.speak("Voice control activated")

    def stop_listening(self):
        """Stop listening for voice commands."""
        self.is_listening = False
        if self.listener_thread:
            self.listener_thread.join(timeout=2.0)
            self.listener_thread = None
        logger.info("Voice command listening stopped")
        self.speak("Voice control deactivated")

    def cleanup(self):
        """Clean up resources before shutdown."""
        try:
            logger.info("Cleaning up voice control resources")
            self.stop_listening()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example standalone usage (for testing)
if __name__ == "__main__":
    # Simple test callback function
    def test_callback(command):
        print(f"Command received: {command['action']}")
        print(f"Parameters: {command['parameters']}")
        
    # Create controller
    voice_controller = VoiceController()
    voice_controller.set_command_callback(test_callback)
    
    try:
        print("Starting voice control test. Speak commands...")
        voice_controller.speak("Voice control test started")
        voice_controller.start_listening()
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        voice_controller.cleanup()