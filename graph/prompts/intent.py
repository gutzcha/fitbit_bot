"""
graph/prompts/intent.py

System prompt and finalized ChatPromptTemplate for the Intent Classifier.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from graph.data_config import INTENT_DEFINITIONS
from graph.schemas import DataSource, ResponseType

# ─────────────────────────────────────────────────────────────────────────────
# 1. DYNAMIC DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────


def _format_definitions() -> str:
    return "\n".join([f"- {k}: {v}" for k, v in INTENT_DEFINITIONS.items()])


def _format_sources() -> str:
    return "\n".join([f"- {x}" for x in list(DataSource.__args__)])  # type: ignore


def _format_responses() -> str:
    return "\n".join([f"- {x}" for x in list(ResponseType.__args__)])  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEW-SHOT EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

EX_SIMPLE = """
<example_1>
[CONVERSATION HISTORY]
User: "How many steps did I take today?"

[EXPECTED OUTPUT]
{{
    "intent": "METRIC_RETRIEVAL",
    "confidence": 0.99,
    "suggested_sources": ["USER_METRICS"],
    "response_type": "DATA_LOOKUP",
    "mentioned_metrics": ["steps"],
    "current_topic": "activity",
    "is_followup": false,
    "needs_clarification": false
}}
</example_1>
"""

EX_CONTEXTUAL = """
<example_2>
[CONVERSATION HISTORY]
User: "How am I doing?"
Assistant: "Are you asking about your sleep, activity, or heart rate?"
User: "hr"

[EXPECTED OUTPUT]
{{
    "intent": "METRIC_RETRIEVAL",
    "confidence": 0.98,
    "suggested_sources": ["USER_METRICS"],
    "response_type": "DATA_LOOKUP",
    "mentioned_metrics": ["heart_rate"],
    "current_topic": "heart_health",
    "is_followup": true,
    "needs_clarification": false
}}
</example_2>
"""

EX_AMBIGUOUS = """
<example_3>
[CONVERSATION HISTORY]
User: "Why is it like that?"

[EXPECTED OUTPUT]
{{
    "intent": "CORRELATION_ANALYSIS",
    "confidence": 0.65,
    "suggested_sources": ["USER_METRICS", "KNOWLEDGE_BASE"],
    "response_type": "TREND_ANALYSIS",
    "mentioned_metrics": [],
    "current_topic": "general",
    "is_followup": true,
    "needs_clarification": true
}}
</example_3>
"""

EX_GREETING = """
<example_4>
[CONVERSATION HISTORY]
User: "Hello, good morning!"

[EXPECTED OUTPUT]
{{
    "intent": "GREETING",
    "confidence": 1.0,
    "suggested_sources": ["NONE"],
    "response_type": "HELP_MESSAGE",
    "mentioned_metrics": [],
    "current_topic": "general",
    "is_followup": false,
    "needs_clarification": false
}}
</example_4>
"""

EX_CAPABILITY = """
<example_5>
[CONVERSATION HISTORY]
User: "how am i"
Assistant: "Are you checking your sleep or steps?"
User: "what else can you do?"

[EXPECTED OUTPUT]
{{
    "intent": "DATA_AVAILABILITY",
    "confidence": 0.99,
    "suggested_sources": ["KNOWLEDGE_BASE"],
    "response_type": "HELP_MESSAGE",
    "mentioned_metrics": [],
    "current_topic": "capability_check",
    "is_followup": true,
    "needs_clarification": false
}}
</example_5>
"""

ALL_EXAMPLES = (
    f"{EX_SIMPLE}\n{EX_CONTEXTUAL}\n{EX_AMBIGUOUS}\n{EX_GREETING}\n{EX_CAPABILITY}"
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SYSTEM STRING CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_SYSTEM_STRING = f"""
You are an expert Intent Classifier for a Fitbit Health Assistant.
Your task is to analyze the [CONVERSATION HISTORY] and extract structured metadata about the user's latest message.

### 1. DEFINITIONS
**INTENTS:**
{_format_definitions()}

**DATA SOURCES:**
{_format_sources()}

**RESPONSE TYPES:**
{_format_responses()}

### 2. LOGIC & RULES
- **Analyze Context:** Look closely at the previous messages in the history.
- **Extraction:**
    - `mentioned_metrics`: Extract specific metrics (e.g., 'steps', 'resting_heart_rate').
    - `current_topic`: Classify the high-level topic (e.g., 'activity', 'sleep', 'weight').
- **Confidence:** If the request is vague (e.g., "Is it bad?"), set `needs_clarification=True`.

### 3. FOLLOW-UP DETECTION RULES (Critical)
Set `is_followup=True` if ANY of the following are true:
1. **Direct Answer:** The Assistant's last message was a question (e.g., "Did you mean sleep or steps?"), and the User's input provides the answer.
2. **Short Keyword:** The User's input is very short (1-2 words) and only makes sense in the context of the previous turn (e.g., "steps", "yes", "hr", "slee").
3. **Pronoun Reference:** The User uses words like "it", "that", "this", or "previous" to refer to prior data.

### 4. EXAMPLES
{ALL_EXAMPLES}

### 5. YOUR TASK
Analyze the conversation below and output the JSON object.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 4. FINAL EXPORTED PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _INTENT_SYSTEM_STRING),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
