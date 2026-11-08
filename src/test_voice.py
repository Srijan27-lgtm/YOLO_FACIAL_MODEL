import pyttsx3

print("🔊 Testing Text-to-Speech Engine...")

# Initialize the engine
engine = pyttsx3.init()

# Set volume to max and speed to normal
engine.setProperty('volume', 1.0)
engine.setProperty('rate', 150)

# Optional: Force a specific Windows voice (usually 'Microsoft David' or 'Microsoft Zira')
voices = engine.getProperty('voices')
if len(voices) > 0:
    engine.setProperty('voice', voices[0].id)

print("🗣️ Attempting to speak... (Check your speakers/headphones!)")
engine.say("Hello! This is a test of the text to speech engine. If you hear this, it is working perfectly.")
engine.runAndWait()

print("✅ Test complete.")