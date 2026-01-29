"""
graph/static_responses.py

Canned responses for static intents (Greetings, Safety, Errors).
"""

GREETING_RESPONSE = (
    "Hello! I'm your Fitbit Health Assistant. "
    "I can help you analyze your activity patterns, sleep quality, "
    "heart rate trends, and overall wellness goals.\n\n"
    "What would you like to check today?"
)

OUT_OF_SCOPE_RESPONSE = (
    "I'm designed to focus specifically on your health, fitness, and physiological data. "
    "I can't help with that particular request, but I'm ready to answer questions "
    "about your steps, sleep, or workout trends!"
)

ERROR_RESPONSE = (
    "I apologize, but I ran into an issue processing your request. "
    "Could you try asking again, perhaps rephrasing your question?"
)
