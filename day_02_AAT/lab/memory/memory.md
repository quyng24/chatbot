# Lab 03 - Agent Memory: Short-term & Long-term Memory

## Scenario

You are building **Smart-Trip AI** - a Personalized AI Travel Planning Assistant.

Travel planning requires handling two distinct memory challenges:

1. **Short-term Memory (Within a single Trip)**: Users chat back and forth to refine itineraries for a specific trip. The agent must maintain context across turns, manage conversation history without exploding token usage via windowing & summarization, and allow users to "time-travel" or "branch out" alternative plans.
2. **Long-term Memory (Across Multiple Trips)**: When the user returns a month later for a completely new trip, short-term memory is wiped. The agent should automatically recall key user preferences (e.g., dietary restrictions, preferred accommodation style, budget level) from cross-session storage without forcing the user to repeat themselves.

---

### Exercises at a glance

| #     | Topic                          | Core Concept                                                                        | Duration |
| ----- | ------------------------------ | ---------------------------------------------------------------------------------- | -------- |
| **1** | The Memory Bug & Checkpointer  | Observe stateless bug with basic agent and fix with `MemorySaver` (`danang_01`)     | 5 min   |
| **2** | Plan Branching & Time Travel   | Branch out alternative options using `checkpoint_id` (`thread_id: "danang_01"`)   | 10 min   |
| **3** | Context Windowing              | Trim old messages keeping $N$ most recent messages (`RemoveMessage` + `before_model`) | 5 min   |
| **4** | Conversation Summarization     | Summarize history and update checkpoint state via `before_model` middleware        | 10 min   |
| **5** | Cross-Trip Long-term Memory    | Persist and recover user profile across trips with `InMemoryStore` (`danang_02`)    | 10 min   |

---

## Setup

Make sure Ollama is running and required models are available.

Large language model: `gpt-oss:20b-cloud`

```python
%pip install langchain langchain-ollama langchain-community langgraph pydantic
```

or
```bash
uv add langchain langchain-ollama langchain-community langgraph pydantic
```

```python
from langchain_ollama import ChatOllama
from config import LLM_ENDPOINT, LLM_MODEL, LLM_API_KEY

# Setup LLM
llm = ChatOllama(
    base_url=LLM_ENDPOINT,
    model=LLM_MODEL,
    temperature=0.3,
    client_kwargs={
        "headers": {'Authorization': 'Bearer ' + LLM_API_KEY} if LLM_API_KEY else None
    },
)

system_prompt = """You are Smart-Trip AI, a helpful travel planning assistant. Keep response concise.

Da Nang Options & Pricing:
- Hotels:
  1. TMS Hotel Beach ($100/night - Luxury beachfront)
  2. Muong Thanh Grand ($50/night - Mid-range modern)
  3. Sanouva Hotel ($30/night - Budget central)
- Cafes (Avg per drink):
  1. À La Carte Rooftop Cafe ($10/drink - Scenic rooftop view)
  2. 43 Factory Coffee Roaster ($5/drink - Specialty coffee)
  3. The Cups Coffee ($3/drink - Local casual coffee)
- Attractions:
  1. Ba Na Hills ($40/person - Amusement park)
  2. Mikazuki Water Park ($25/person - Water park)
  3. Ngu Hanh Son Tour ($15/person - Cultural cave tour)
  
Nha Trang Options & Pricing:
- Hotels:
  1. Vinpearl Beachfront Nha Trang ($100/night - Luxury beachfront)
  2. Muong Thanh Luxury Nha Trang ($50/night - Mid-range modern)
  3. Aaron Hotel Nha Trang ($30/night - Budget central)
- Cafes (Avg per drink):
  1. Skylight Rooftop Bar ($10/drink - Panoramic 45th-floor ocean view)
  2. Rain Forest Cafe ($5/drink - Specialty jungle-themed coffee)
  3. CCCP Coffee ($3/drink - Local casual coffee)
- Attractions:
  1. VinWonders Nha Trang ($40/person - Theme park & cable car)
  2. I-Resort Mud Bath & Water Park ($25/person - Mineral mud bath & water park)
  3. Po Nagar Cham Towers Tour ($15/person - Cultural temple tour)  
"""

print("llm model:", llm.model)
```

**Note:** The system prompt intentionally includes both Da Nang **and** Nha Trang, even though every exercise below only travels to Da Nang. With only one destination in its knowledge, the LLM could "guess" the city correctly on memory-testing questions like *"What city am I visiting?"* without actually relying on conversation history. Adding a second option forces the agent to genuinely recall context instead of defaulting to the only city it knows.

---

## Exercise 1 - The Memory Bug & Checkpointer (5 min)

### Objective
First observe how a basic agent without a checkpointer forgets previous conversation turns. Then, fix this bug by equipping the agent with a `MemorySaver` checkpointer.

```python
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage

# Pre-built agent WITHOUT memory to demonstrate the Memory Bug
agent_without_memory = create_agent(
    model=llm,
    system_prompt=system_prompt
)

def chat_without_memory(query: str):
    print(f"\n{"="*30} Stateless Agent {"="*30}\n")
    print(f"[User]: {query}")

    result = agent_without_memory.invoke(
        {"messages": [HumanMessage(content=query)]}
    )

    print(f"[Assistant]: {result['messages'][-1].content}")

# Observe the Memory Bug!
chat_without_memory("Hi! I am planning a 3-day trip to Da Nang.")
chat_without_memory("What city did I just say I'm planning to visit?")
```

Now, equip the agent with `MemorySaver` to enable stateful conversation:

```python
from langgraph.checkpoint.memory import MemorySaver

## TODO: Initialize MemorySaver checkpointer and attach it to create_agent to fix the Memory Bug
# --- BEGIN ---

checkpointer = ...

agent_with_memory = create_agent(
    model=llm,
    system_prompt=system_prompt,
    checkpointer=...
)

def chat_with_memory(query: str, thread_id: str):
    print(f"\n{"="*30} Thread {thread_id} {"="*30}\n")
    print(f"[User]: {query}")
    
    config = {...} # Set up configurable with thread_id
    
    result = agent_with_memory.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
    
    ai_response = result["messages"][-1].content

    print(f"[Assistant]: {ai_response}")
# --- END ---
```

```python
thread_danang = "danang_01"

chat_with_memory("Hi! I am planning a 3-day trip to Da Nang.", thread_id=thread_danang)

chat_with_memory("What city did I just say I'm planning to visit?", thread_id=thread_danang)

# Capture Checkpoint ID after destination selection
current_state = agent_with_memory.get_state({"configurable": {"thread_id": thread_danang}})
fork_checkpoint_id = current_state.config["configurable"]["checkpoint_id"]

# Specify hotel choice
chat_with_memory("For my hotel, I book Khach san Muong Thanh.", thread_id=thread_danang)
```

---

## Exercise 2 - Plan Branching & Time Travel (10 min)

### Objective
Rewind back to `fork_checkpoint_id` (Turn 1) to branch off an alternative path.

Verify state isolation on the branched path: the agent should remember the destination (`Da Nang`), but must **NOT** know about the hotel (`Khach san Muong Thanh`) booked in Branch A after the checkpoint!

```python
# We print the Checkpoint Tree to inspect where the new branch will start from:
# - **Branch Starting Point**: Identifies the exact `fork_checkpoint_id` where execution will branch off.
# - **Included Context**: Messages up to Turn 1 (e.g. destination `Da Nang`) are present in this checkpoint.
# - **Excluded Context**: Subsequent messages from Branch A (e.g. hotel `Khach san Muong Thanh`) occurred after this checkpoint and are NOT included.

from utils import print_langgraph_checkpoint_tree

print(f"\nFork checkpoint id: {fork_checkpoint_id}")

print("\n--- Checkpoint Tree BEFORE Branching ---")
print_langgraph_checkpoint_tree(checkpointer)
```

__Helper Function Update for Branching__

To support branching and rewinding to past checkpoints seamlessly, we update our helper function to accept an optional `checkpoint_id` parameter. When `checkpoint_id` is provided, it is passed inside the `config` dictionary alongside `thread_id`, instructing LangGraph to rewind execution to that target checkpoint and fork a new branch.

```python
def chat_with_memory(query: str, thread_id: str, checkpoint_id: str | None = None):
    print(f"\n{'='*30} Thread {thread_id} {'='*30}\n")
    print(f"[User]: {query}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    if checkpoint_id:

        ## TODO: pass checkpoint_id to the config
        config["configurable"][...] = ...
    
    result = agent_with_memory.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
    
    ai_response = result["messages"][-1].content
    print(f"[Assistant]: {ai_response}")
```

Now we use the saved `fork_checkpoint_id` alongside `thread_danang` to configure and invoke the agent on the branched path.

Notice the state isolation in action: Assistant remembers the trip destination (`Da Nang`), but has no knowledge of the hotel (`Khach san Muong Thanh`) booked later!

```python
## TODO: Call chat_with_memory with thread_id and forked checkpoint_id
# --- BEGIN ---

query = "What city am I visiting, and have I booked a hotel yet?"

chat_with_memory(query, ..., ...)
# --- END ---
```

```python
from utils import print_langgraph_checkpoint_tree
print("\n--- Checkpoint Tree AFTER Branching ---")
print_langgraph_checkpoint_tree(checkpointer)
```

---

## Exercise 3 - Context Windowing (5 min)

### Objective
Build up a multi-turn conversation on the new branch, apply `@before_model` trimming middleware to restrict the LLM's context window (`max_window = 3`), and verify that old messages (such as the hotel choice) are trimmed out while full history remains in the checkpointer.

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

current_state = agent_with_memory.get_state({"configurable": {"thread_id": "danang_01"}})
current_messages = current_state.values.get("messages", [])

print(f"Latest message: \"{current_messages[-1].content}\"")
print(f"Current total messages in checkpointer: {len(current_messages)}")
```

```python
from langchain_core.messages import RemoveMessage
from langchain.agents.middleware import before_model

## TODO: Implement message trimming (windowing) to keep SystemMessage + the last 3 Human/AI messages
# --- BEGIN ---

@before_model
def window_messages(state, runtime):
    messages = state["messages"]
    max_window = 3 # Keep only the last 3 messages
    
    if len(messages) <= max_window:
        return None
        
    # 1. Extract older messages that fall outside the max_window
    old_messages = ...
    
    # 2. Create a list of RemoveMessage operations using each message's ID
    remove_messages = [...]
    
    # 3. Return the update dictionary for the state reducer to apply changes
    return {"messages": remove_messages}

# Create a new agent equipped with the middleware
windowed_agent = create_agent(
    model=llm,
    system_prompt=system_prompt,
    checkpointer=checkpointer,
    middleware=[...], # Add middleware to agent
)

# --- END ---

def chat_with_windowed_memory(query: str, *, thread_id: str, checkpoint_id: str | None = None):
    print(f"\n{'='*30} Thread {thread_id} {'='*30}\n")
    print(f"[User]: {query}")

    config = {"configurable": {"thread_id": thread_id}}

    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id
    
    res = windowed_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
    
    print(f"[Assistant]: {res['messages'][-1].content}")
```

```python
# Build up conversation history on Branch B
chat_with_windowed_memory("Ok, on this plan I booked TMS Hotel Beach.", thread_id=thread_danang, checkpoint_id=fork_checkpoint_id)
chat_with_windowed_memory("Suggest 2 cafes near my hotel.", thread_id=thread_danang)

# Verify trimming effect: Old messages were trimmed from the context window, so the assistant should no longer know which hotel was booked.
chat_with_windowed_memory("What city am I visiting, and have I booked a hotel yet?", thread_id=thread_danang)
```

```python
print("\n--- Checkpoint Tree AFTER Trimming on forked branch ---")
print_langgraph_checkpoint_tree(checkpointer)
```

---

## Exercise 4 - Conversation Summarization Middleware (10 min)

### Objective
Trimming drops old context completely. A better approach for long discussions is **Conversation Summarization**, which compresses past dialogue into a rolling summary once the message count grows.

Instead of manually updating the checkpointer state, we can use a `@before_model` middleware to automatically monitor the thread and summarize older messages on the fly.

Remember how LangGraph's `add_messages` Reducer processes message updates and removals when returning state changes from your middleware.

```python
from langchain.agents.middleware import before_model
from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()


# We will use this helper function inside the middleware
def generate_summary(old_messages: list) -> str:
    prompt = (
        "You are a conversation summarizer. "
        "Compress the following conversation turn(s) between the AI assistant and the user into a concise summary:\n"
    )
    for msg in old_messages:
        prompt += f"\n{msg.type.capitalize()}: {msg.content}"
    prompt += "\n\nSummary:"
    return llm.invoke(prompt).content

## TODO: Implement the summarize_messages middleware
# --- BEGIN ---

@before_model
def summarize_messages(state, runtime):
    messages = state["messages"]
    max_window = 3  # Keep the last 3 messages, summarize anything older
    
    # 1. If we haven't exceeded the max window, return None to do nothing
    if len(messages) <= max_window:
        return None
        
    # 2. Extract older messages to compress (all messages except the last max_window)
    old_messages = ...
        
    # 3. Generate summary for old messages with generate_summary
    summary_text = ...

    print(f" [✨ Summary]: {summary_text}\n")
    
    # 4. OVERWRITE the first message in-place to keep the summary at the TOP of context window
    # Reusing first old message, tells LangGraph Reducer to update this message's content
    summarized_message = SystemMessage(
        id=old_messages[0].id,
        content=f"✨ Summary of previous discussion: {summary_text}"
    )
    
    # 5. Remove the REST of the old messages (from index 1 onwards)
    remove_messages = [...]
    
    return {"messages": [summarized_message] + remove_messages}

# Create a new agent equipped with the summarize_messages middleware
summarizing_agent = create_agent(
    model=llm,
    system_prompt=system_prompt,
    checkpointer=checkpointer,
    middleware=[...], # Add middleware to agent
)

# --- END ---

def chat_summary(query: str, thread_id: str):
    print(f"\n{'='*30} Thread {thread_id} {'='*30}\n")
    print(f"[User]: {query}\n")

    res = summarizing_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}}
    )

    print(f"[Assistant]: {res['messages'][-1].content}\n")
```

Create a new thread and run the automatic summarization:

```python
summary_thread = "thread_summary_01"

# We feed it a series of messages to trigger summarization
chat_summary("Hi, I want to travel to Da Nang.", summary_thread)
chat_summary("I prefer beachfront hotels.", summary_thread)

# At this point, the middleware will summarize the old messages because we exceeded max_window!
chat_summary("I like going to water park.", summary_thread)

# Test if Assistant still remembers prefered hotel
chat_summary("What hotel did I say I prefer?", summary_thread)
```

```python
# Let's inspect the checkpointer to see the SystemMessage summary replacing the old messages
print("\n--- Checkpoint Tree AFTER Summarization ---")
print_langgraph_checkpoint_tree(checkpointer)
```

---

## Exercise 5 - Cross-Trip Long-term Memory (10 min)

### Objective
Persist and retrieve user preferences across different trips using Memory Tools (`save_user_preference` & `get_user_preference`) equipped with `InjectedStore`.

> 💡 This is LangGraph's standard design pattern for Long-term Memory. If you are not yet familiar with Tool Calling or writing Custom Tools, don't worry! For now, you only need to understand how the 2 Tool functions below are declared to read from & write to `InMemoryStore`. We will dive deep into Tool Architecture in **Day 03**.

```python
from typing import Literal
from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

# TODO: Create checkpointer for short-term memory and store for long-term memory
store = ...
checkpointer = ...

# Define save_user_preference Tool with category & InjectedStore annotation

@tool
def save_user_preference(
    preference: str,
    category: Literal["budget", "drink", "attraction", "general"],
    store: Annotated[BaseStore, InjectedStore()],
    config: RunnableConfig
) -> str:
    """
    Use this tool to save a new preference, budget, or travel detail about the user into long-term memory.
    
    Args:
        category: Literal["budget", "drink", "attraction", "general"]
    """
    # Get user_id from config["configurable"] at runtime, fallback to `default_user`` if not provided
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("users", user_id)
    
    existing_item = store.get(namespace, "preferences")
    data = existing_item.value if existing_item else {}
    data[category] = data.get(category, [])
    if preference not in data[category]:
        data[category].append(preference)
    store.put(namespace, "preferences", data)
    
    print(f"  [Tool Execution] 💾 Saved preference for [{user_id}]: [{category}] {preference}")
    return f"Successfully saved: [{category}] {preference}"


# Define get_user_preference Tool with category & InjectedStore annotation

@tool
def get_user_preference(
    category: Literal["budget", "drink", "attraction", "general"],
    store: Annotated[BaseStore, InjectedStore()],
    config: RunnableConfig
) -> str:
    """
    Use this tool to extract/retrieve all saved preferences and budget from long-term memory.
    
    Args:
        category: Literal["budget", "drink", "attraction", "general"]
    """
    # Get user_id from config["configurable"] at runtime, fallback to `default_user`` if not provided
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("users", user_id)
    existing_item = store.get(namespace, "preferences")
    
    print(f"  [Tool Execution] 🔍 Retrieving profile for [{user_id}] with category `{category}`...")
    
    if not existing_item or not existing_item.value.get(category):
        print(f"  [Result] No preferences found for category `{category}`.")
        return "No preferences found for this category."
        
    preferences = existing_item.value[category]

    print(f"  [Result] {preferences}")
    return f"User preferences for {category}: " + ", ".join(preferences)

# TODO: Create Agent equipped with BOTH Memory Tools, Checkpointer, and Store
agent_with_memory = create_agent(
    model=llm,
    system_prompt=system_prompt + "Use provided tools to interact with user's preferences.",
    tools=[..., ...], # Add tools to agent
    checkpointer=checkpointer,
    store=store
)

# Helper function to chat with memory agent, accepting user_id dynamically
def chat_with_memory(query: str, *, thread_id: str, user_id: str):
    print(f"\n{"="*30} Thread {thread_id} | User {user_id} {"="*30}\n")
    print(f"[User]: {query}")

    result = agent_with_memory.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id, "user_id": user_id}}
    )

    ai_response = result["messages"][-1].content

    print(f"[Assistant]: {ai_response}")
```

```python
# TODO: Define your own user_id
user_id = ...

# Trip 1 (thread_id: "danang_01"): Agent autonomously detects preferences and saves to user profile
chat_with_memory(
    "Hi! I am planning a trip to Da Nang. My budget is around $85/day, I love specialty coffee, and I prefer water parks.", 
    thread_id="danang_01", 
    user_id=user_id
)

# Trip 2 (New thread_id: "danang_02"): Agent autonomously retrieves user profile from long-term memory
chat_with_memory(
    "Suggest a combo of 1 hotel, 1 cafe, and 1 attraction for a 1-day itinerary in Da Nang that fits my saved budget and preferences.", 
    thread_id="danang_02", 
    user_id=user_id
)
```

> ⚠️ **Probabilistic Nature of Autonomous Tool Calling**
>
> Keep in mind that LLM tool calling is inherently **probabilistic**, not deterministic. Depending on the model size and capabilities (especially with smaller local LLMs), the agent may:
> - Skip calling `save_user_preference` or `get_user_preference` tools entirely.
> - Query `get_user_preference` with an unexpected or incorrect `category`.
> - Retrieve only a subset of the saved preferences.
> 
> If the agent does not trigger the tools automatically during testing, you can make your prompts more explicit (e.g., *"Check my saved budget and preferences..."*) or enhance the tool docstrings to give clearer guidance to the model.

---

## Summary

In this lab you learned how to:

- **Fix stateless memory bugs** using LangGraph checkpointers (`MemorySaver`).
- **Perform Time Travel & Branching** by specifying `checkpoint_id`.
- **Manage context length** with trimming and rolling summarization.
- **Persist long-term user preferences** across isolated threads using `tools` and `Store`.
