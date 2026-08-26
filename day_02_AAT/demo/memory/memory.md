# Lesson 03 - Agent Memory: Short-term & Long-term Memory

Welcome to **Day 2** of the LangChain Agent Development course!

Today you will give your agent **Memory**—allowing it to maintain context across conversation turns and remember user preferences across sessions. 

Without memory, each interaction with an LLM is a blank slate. If you say "Hi, my name is Alice", and then ask "What is my name?", the LLM will not know. In this lab, we will solve this problem using LangGraph.

In this demo you will:

1. **Part 1 - Short-term Memory**: Maintain state across chat turns within a single thread using LangGraph checkpointers, branch conversation history (time travel), and control context windows.
2. **Part 2 - Long-term Memory**: Persist user profile preferences across isolated threads using Pydantic structured output and `InMemoryStore`.

## Setup

Make sure Ollama is running and required models are available.

Large language model: `gpt-oss:20b-cloud`

`gpt-oss:20b-cloud` is running on Ollama cloud server so we don't need to pull it locally.

```python
%pip install langchain langchain-ollama langchain-community langgraph pydantic
```

or
```bash
uv add langchain langchain-ollama langchain-community langgraph pydantic
```

Let's initialize our LLM.

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

print("llm model:", llm.model)
```

## Part 1 - Short-term Memory

Short-term memory allows an agent to track conversation history within a specific thread (`thread_id`). Think of a thread like a single chat window in ChatGPT or WhatsApp. As long as you stay in that window, the assistant remembers what was said earlier.

### 1. Simple Flat Memory Store

To understand how memory works under the hood, let's first build a naive memory system using a simple Python list.

**The Concept:**
Every time a user sends a message, we append it to a list (our "database"). When the AI replies, we append its response to the same list. 
Crucially, every message must be tagged with a `thread_id` so we know which conversation it belongs to. When calling the LLM, we filter the list to only include messages from the current thread.

```python
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage

# Our naive in-memory database
db = []

def chat(query, *, thread_id):
    print("User:", query)

    # 1. Save user message to database
    db.append({"id": str(uuid4()), "role": "user", "content": query, "thread_id": thread_id})

    # 2. Retrieve conversation history for this specific thread
    messages = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in db if m["thread_id"] == thread_id
    ]

    # 3. Pass full history to the LLM
    ai_message = llm.invoke(messages)
    print("AI:", ai_message.content)
    print("-" * 60)

    # 4. Save AI response to database
    db.append({"id": str(uuid4()), "role": "assistant", "content": ai_message.content, "thread_id": thread_id})
```

Let's test this naive implementation:

```python
thread_id = "thread_1"

# The agent will remember the name because the previous message is passed in the list
chat("My name is Son. I'm 24 years old.", thread_id=thread_id)
chat("What's my name?", thread_id=thread_id)
```

### 2. State Persistence with LangGraph Checkpointer

While our flat list works conceptually, it is not scalable or robust. What happens if the server crashes? How do we handle complex agent workflows?

In production, LangGraph handles state persistence automatically using a **Checkpointer**. A checkpointer saves the state of a graph at every super-step. 
- `MemorySaver` saves state in RAM (useful for testing).
- `PostgresSaver` or `SqliteSaver` saves state in a real database.

By passing a checkpointer to `create_agent`, LangGraph intercepts the conversation, saves the messages, and retrieves them automatically based on the `thread_id` we provide in the configuration.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Initialize an in-memory checkpointer
checkpointer = MemorySaver()

# Create an agent powered by LangGraph, equipped with the checkpointer
agent = create_agent(
    model=llm,
    checkpointer=checkpointer,
)

def chat_with_memory(query: str, *, thread_id: str) -> str:
    # We pass the thread_id via the config object
    config = {"configurable": {"thread_id": thread_id}}
    print(f"User ({thread_id}): {query}")

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )

    ai_response = result["messages"][-1].content
    print(f"Agent: {ai_response}")
    print(f"  [Usage metadata]: {result["messages"][-1].usage_metadata}")
    print("-" * 60)
```

Let's see how this automatically isolates conversations:

```python
# Session 1: thread_1
chat_with_memory("I like playing badminton.", thread_id="thread_1")
chat_with_memory("What is my favorite sport?", thread_id="thread_1")

# Session 2: A completely isolated thread (thread_2)
# The agent will NOT know the favorite sport here.
chat_with_memory("What is my favorite sport?", thread_id="thread_2")
```

We can visualize how LangGraph saves these states using a utility function:

```python
from utils import print_langgraph_checkpoint_tree

# Visualize Checkpoint Tree
print_langgraph_checkpoint_tree(checkpointer)
```

### 3. Plan Branching (Time Travel)

Because LangGraph checkpointers save the state at *every* step, they provide a powerful feature: **Time Travel**.

Imagine a user is chatting with a bot and wants to undo their last message, or explore a different conversational path without losing the original history. By supplying a past `checkpoint_id` in the `config` alongside the original `thread_id`, LangGraph rewinds execution to that exact state and forks a new branch within the thread tree.

> 💡 **Understanding Checkpoint Scoping in LangGraph:**
> Checkpoints in LangGraph are stored hierarchically under a specific `thread_id`. 
> - To branch off from a past state within the same thread tree, pass the original `thread_id` and the target `checkpoint_id`.
> - If you want to copy a past checkpoint state to a completely separate `thread_id`, use `agent.get_state(...)` followed by `agent.update_state(...)`.

**Steps:**
1. Chat on a thread (`thread_1`).
2. Retrieve the `checkpoint_id` after Turn 1.
3. Continue conversation on Turn 2 & Turn 3.
4. Pass `fork_checkpoint_id` with `thread_id` to rewind back to Turn 1 and fork a new branch.

```python
# Reset our environment
checkpointer = MemorySaver()
agent = create_agent(model=llm, checkpointer=checkpointer)

def chat_with_memory(query: str, *, thread_id: str, checkpoint_id: str | None = None) -> str:
    # We pass the thread_id and checkpoint_id via the config object
    config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    print(f"User ({thread_id}): {query}")

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )

    ai_response = result["messages"][-1].content
    print(f"Agent: {ai_response}")
    print(f"  [Usage metadata]: {result["messages"][-1].usage_metadata}")
    print("-" * 60)
```

```python
thread_id = "thread_1"

# Turn 1: Introduce name
chat_with_memory("Hi, my name is Son.", thread_id=thread_id)

# Save the checkpoint ID exactly after Turn 1
current_state = agent.get_state({"configurable": {"thread_id": thread_id}})
fork_checkpoint_id = current_state.config["configurable"]["checkpoint_id"]

# Turn 2 & 3 on original thread
chat_with_memory("I am 24 years old.", thread_id=thread_id)
chat_with_memory("How old am I?", thread_id=thread_id)

# Now, we Branch off from Turn 1 using fork_checkpoint_id
# The agent at fork_checkpoint_id knows the name "Son", but NOT the age "24".
user_query = "What's my name and how old am I?"

print(f"\n{"="*30} [Branched Path] {"="*30}")
chat_with_memory(user_query, thread_id=thread_id, checkpoint_id=fork_checkpoint_id)
```

We can visualize this branching path:

```python
from utils import print_langgraph_checkpoint_tree

print_langgraph_checkpoint_tree(checkpointer)
```

### 4. Context Windowing (Message Trimming)

LLMs have a maximum context window (the number of tokens they can process at once). If a conversation goes on for hours, the list of messages will eventually exceed this limit, causing an error. Furthermore, processing long message lists is slow and expensive.

To fix this, we need to **trim** the messages before sending them to the LLM. However, we want to keep the *full* history in our database (the checkpointer) for record-keeping.

We solve this using a `@before_model` middleware. This middleware intercepts the state right before the LLM sees it, and instructs LangGraph to remove older messages from the LLM's view.

```python
from langchain.agents.middleware import before_model
from langchain_core.messages import RemoveMessage

checkpointer = MemorySaver()

# This middleware runs right before the LLM is invoked
@before_model
def window_messages(state, runtime):
    messages = state["messages"]
    max_window = 3 # Keep only the last 3 messages
    
    if len(messages) <= max_window:
        return None
        
    # We return a list of RemoveMessage objects for the older messages.
    # This only removes them from the LLM's context window, NOT from the checkpointer.
    return {"messages": [RemoveMessage(id=m.id) for m in messages[:-max_window]]}

# Create a new agent equipped with the middleware
windowed_agent = create_agent(
    model=llm,
    system_prompt="You are a helpful assistant. Keep responses brief. If you don't know, just say so.",
    checkpointer=checkpointer,
    middleware=[window_messages],
)

def chat_window(query: str, *, thread_id: str):
    print(f"User: {query}")
    res = windowed_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    print(f"Agent: {res['messages'][-1].content}")
    print("-" * 60)
```

> 💡 **How it works (LangGraph's `add_messages` Reducer)**:
> When middleware returns `{"messages": ...}`, LangGraph passes this update to its built-in `add_messages` Reducer:
> - **`RemoveMessage(id=...)`**: Instructs the Reducer to remove specified messages from the active context.
> - **Reusing `id`**: Instructs the Reducer to overwrite an existing message in-place.
> - **New messages**: Appends them to the end of the conversation.
> 
> 📌 Pay close attention to how the Reducer handles in-place updates vs removals—you will need this exact mechanism to implement conversation summarization in the Lab exercise!*

Let's test the windowing:

```python
window_thread = "thread_window"

# Turn 1: Introduce food and sport
chat_window("My favorite food is bread, and my favorite sport is badminton.", thread_id=window_thread)

# Turn 2: Ask about sport (This is within the window)
chat_window("What is my favorite sport?", thread_id=window_thread)

# Turn 3: Ask about food
# Because max_window is 3, Turn 1 has now been trimmed out of the LLM's context window!
# The agent will NOT remember the favorite food.
chat_window("What is my favorite food?", thread_id=window_thread)
```

Even though the LLM forgot the food, the checkpointer still holds the full history:

```python
from utils import print_langgraph_checkpoint_tree

# The tree will show all messages, confirming data is not lost permanently.
print_langgraph_checkpoint_tree(checkpointer)
```

## Part 2 - Long-term Memory

Short-term memory (`thread_id`) is great for an ongoing conversation. But what if the user starts a *new* chat window tomorrow? The `thread_id` will be different, and the agent will forget everything.

**Long-term memory** solves this by persisting user facts and preferences across entirely isolated threads. Instead of saving raw conversation logs, we extract key facts and store them centrally under a `user_id`.

In LangGraph, the standard production design pattern for Long-term Memory uses **Tools equipped with `InjectedStore`**.

> 💡 If you are not yet familiar with Tool Calling or writing Custom Tools, don't worry! For now, you only need to understand how the Tool functions below are declared and how they automatically interact with `InMemoryStore`. We will dive deep into Tool Architecture and Custom Tools in **Day 03**.

### How Tool-based Long-term Memory Works:

1. **Active Saving via Tool (`save_preference`)**: When the user shares preferences during conversation (e.g., "I am strictly vegetarian"), the Agent decides to call `save_preference`.
2. **Active Recall via Tool (`get_preference`)**: When starting a new chat thread (`thread_2`), the Agent dynamically calls `get_preference` to fetch stored user facts from `InMemoryStore` whenever it needs to answer personalized questions!
3. **Under the Hood (`InjectedStore`)**: LangGraph automatically hides the `store` argument from the LLM tool schema so the LLM doesn't pass it. When either Tool executes, LangGraph Runtime injects the `InMemoryStore` instance into the tool.

```python
from typing import Literal
from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig  # Import RunnableConfig to access runtime config
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 1. Define Memory Save Tool with InjectedStore
@tool
def save_user_preference(
    preference: str,
    category: Literal["sport", "food", "general"],
    store: Annotated[BaseStore, InjectedStore()],
    config: RunnableConfig
) -> str:
    """
    Use this tool to save a new preference, fact, or personal detail about the user into long-term memory.
    
    Args:
        preference: The preference, fact, or personal detail to save.
        category: Literal["sport", "food", "general"]
    """
    # Dynamically extract user_id from runtime config
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

# 2. Define Memory Retrieve Tool with InjectedStore
@tool
def get_user_preference(
    category: Literal["sport", "food", "general"],
    store: Annotated[BaseStore, InjectedStore()],
    config: RunnableConfig
) -> str:
    """
    Use this tool to extract/retrieve all saved preferences and facts about the user by category from long-term memory.
    
    Args:
        category: Literal["sport", "food", "general"]
    """
    # Dynamically extract user_id from runtime config
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
```

```python
# 3. Create Store & Checkpointer
store = InMemoryStore()
checkpointer = MemorySaver()

# 4. Create Agent equipped with BOTH Memory Tools, Checkpointer, and Store
agent_with_memory = create_agent(
    model=llm,
    system_prompt = (
        "You are a helpful assistant with long-term memory capabilities.\n"
        "- When the user shares personal details, facts, or preferences, use the `save_user_preference` tool to store them.\n"
        "- When answering questions about the user's background or preferences, use the `get_user_preference` tool to retrieve relevant details."
    ),
    tools=[save_user_preference, get_user_preference],
    checkpointer=checkpointer,
    store=store
)

def chat_with_memory(query: str, thread_id: str, user_id: str = "user_123"):
    print(f"\n[Thread: {thread_id} | User: {user_id}] User: {query}")
    result = agent_with_memory.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id, "user_id": user_id}}
    )
    ai_response = result["messages"][-1].content
    print(f"AI Assistant: {ai_response}")
    print("-" * 60)
```

```python
# Session 1: thread_1 (User shares preferences)
# The Agent automatically invokes save_user_preference Tool!
chat_with_memory(
    "Hi! My name is Son. I like playing badminton and I like eating bread.", 
    thread_id="thread_1", 
    user_id="user_123"
)

# Session 2: thread_2 (A completely clean new thread)
# When asked a question, the Agent automatically invokes get_user_preference Tool to recall memory!
chat_with_memory(
    "What is my favorite sport and my favorite food?", 
    thread_id="thread_2", 
    user_id="user_123"
)
```

## Summary

In this lesson you learned how to:

- **Short-term Memory**: Use LangGraph checkpointers (`MemorySaver`) to automatically track conversation history by `thread_id`.
- **Time Travel**: Fork conversations from past states using `checkpoint_id`.
- **Message Trimming**: Manage LLM context limits cleanly using `@before_model` middleware.
- **Long-term Memory**: Use LangGraph Tools with `InjectedStore` to actively persist user preferences in `InMemoryStore` across isolated threads. 

By combining short-term and long-term memory, you can build agents that hold context-rich conversations today, and remember user preferences tomorrow!
