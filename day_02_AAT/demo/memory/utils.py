from collections import defaultdict

def print_checkpoint_tree(checkpointer, checkpoint_id: str | None = None):
    """Builds and prints an ASCII tree visualization of all checkpoint branches and messages,

    or a specific subtree starting from checkpoint_id if provided.
    """
    # 1. Collect all unique checkpoints and map them to their corresponding threads
    all_cps = {}
    id_to_threads = defaultdict(set)
    
    for thread_id, cps_dict in checkpointer.storage.items():
        for cp_id, cp in cps_dict.items():
            all_cps[cp_id] = cp
            id_to_threads[cp_id].add(thread_id)
            
    # 2. Build parent-child relationships
    parent_to_children = defaultdict(list)
    for cp_id, cp in all_cps.items():
        if cp.parent_id and cp.parent_id in all_cps:
            parent_to_children[cp.parent_id].append(cp_id)
            
    # Determine roots to print
    if checkpoint_id:
        if checkpoint_id not in all_cps:
            print(f"Checkpoint ID '{checkpoint_id}' not found in checkpointer storage.")
            return
        roots = [checkpoint_id]
    else:
        roots = []
        for cp_id, cp in all_cps.items():
            if not cp.parent_id or cp.parent_id not in all_cps:
                roots.append(cp_id)
            
    # 3. Recursive helper to print nodes in ASCII format
    def print_node(cp_id, indent="", is_last=True):
        cp = all_cps[cp_id]
        threads_str = ", ".join(sorted(id_to_threads[cp_id]))
        
        # ASCII connectors
        marker = "└── " if is_last else "├── "
        print(f"{indent}{marker}CP: {cp_id[:8]} [Threads: {threads_str}]")
        
        # Format and display truncated message list inside this checkpoint
        msg_indent = indent + ("    " if is_last else "│   ")
        messages = cp.values.get("messages", [])
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "").replace("\n", " ") # Clean newlines for single-line display
            
            # Trim message content for clean layout
            trimmed_content = content[:30] + "..." if len(content) > 30 else content
            role_display = "user" if role in ["user", "human"] else "assistant"
            
            print(f"{msg_indent}  [{role_display}] {trimmed_content}")
            
        # Recursive print children
        children = parent_to_children[cp_id]
        child_indent = indent + ("    " if is_last else "│   ")
        for idx, child_id in enumerate(children):
            print_node(child_id, child_indent, is_last=(idx == len(children) - 1))

    # 4. Print the tree
    print("\n" + "="*10 + " CHECKPOINT BRANCHING TREE " + "="*10)
    for idx, root_id in enumerate(roots):
        print_node(root_id, is_last=(idx == len(roots) - 1))
    print("="*60 + "\n")


def print_langgraph_checkpoint_tree(checkpointer, checkpoint_id: str | None = None):
    """Builds and prints an ASCII tree visualization of checkpoints inside a LangGraph checkpointer."""
    # 1. Collect all checkpoints from checkpointer
    all_cps = {}
    id_to_threads = defaultdict(set)
    parent_to_children = defaultdict(list)
    
    # Query all checkpoints by passing config=None
    for cp_tuple in checkpointer.list(config=None):
        cp_id = cp_tuple.config["configurable"]["checkpoint_id"]
        thread_id = cp_tuple.config["configurable"]["thread_id"]
        
        id_to_threads[cp_id].add(thread_id)
        all_cps[cp_id] = cp_tuple
        
        if cp_tuple.parent_config:
            parent_id = cp_tuple.parent_config["configurable"]["checkpoint_id"]
            parent_to_children[parent_id].append(cp_id)
            
    # Determine root checkpoints
    if checkpoint_id:
        if checkpoint_id not in all_cps:
            print(f"Checkpoint ID '{checkpoint_id}' not found in LangGraph checkpointer storage.")
            return
        roots = [checkpoint_id]
    else:
        roots = []
        for cp_id, cp_tuple in all_cps.items():
            parent_id = cp_tuple.parent_config["configurable"]["checkpoint_id"] if cp_tuple.parent_config else None
            if not parent_id or parent_id not in all_cps:
                roots.append(cp_id)
                
    # Recursive helper to print nodes in ASCII format
    def print_node(cp_id, indent="", is_last=True):
        cp_tuple = all_cps[cp_id]
        threads_str = ", ".join(sorted(id_to_threads[cp_id]))
        parent_id = cp_tuple.parent_config["configurable"]["checkpoint_id"] if cp_tuple.parent_config else None
        
        # ASCII connectors
        marker = "└── " if is_last else "├── "
        print(f"{indent}{marker}CP: ...{cp_id[-8:]} [Threads: {threads_str}] (Parent: ...{parent_id[-8:] if parent_id else 'None'})")
        
        # Display messages inside the checkpoint
        msg_indent = indent + ("    " if is_last else "│   ")
        messages = cp_tuple.checkpoint["channel_values"].get("messages", [])
        for msg in messages:
            role = msg.type
            content = msg.content.replace("\n", " ") # Clean newlines for single-line display
            
            # Trim message content for clean layout
            trimmed_content = content[:30] + "..." if len(content) > 30 else content
            role_display = "user" if role in ["user", "human"] else "assistant"
            
            print(f"{msg_indent}  [{role_display}] {trimmed_content}")
            
        # Recursive print children, sorted by timestamp to ensure chronological order
        children = sorted(parent_to_children[cp_id], key=lambda cid: all_cps[cid].checkpoint["ts"])
        child_indent = indent + ("    " if is_last else "│   ")
        for idx, child_id in enumerate(children):
            print_node(child_id, child_indent, is_last=(idx == len(children) - 1))

    # Print the tree
    print("\n" + "="*10 + " LANGGRAPH CHECKPOINT TREE " + "="*10)
    for idx, root_id in enumerate(roots):
        print_node(root_id, is_last=(idx == len(roots) - 1))
    print("="*60 + "\n")

