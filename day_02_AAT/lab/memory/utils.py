from collections import defaultdict

def print_langgraph_checkpoint_tree(checkpointer, checkpoint_id: str | None = None):
    """Builds and prints an ASCII tree visualization of checkpoints inside a LangGraph checkpointer."""
    all_cps = {}
    id_to_threads = defaultdict(set)
    parent_to_children = defaultdict(list)
    
    for cp_tuple in checkpointer.list(config=None):
        cp_id = cp_tuple.config["configurable"]["checkpoint_id"]
        thread_id = cp_tuple.config["configurable"]["thread_id"]
        
        id_to_threads[cp_id].add(thread_id)
        all_cps[cp_id] = cp_tuple
        
        if cp_tuple.parent_config:
            parent_id = cp_tuple.parent_config["configurable"]["checkpoint_id"]
            parent_to_children[parent_id].append(cp_id)
            
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
                
    def print_node(cp_id, indent="", is_last=True):
        cp_tuple = all_cps[cp_id]
        threads_str = ", ".join(sorted(id_to_threads[cp_id]))
        parent_id = cp_tuple.parent_config["configurable"]["checkpoint_id"] if cp_tuple.parent_config else None
        
        marker = "└── " if is_last else "├── "
        print(f"{indent}{marker}CP: ...{cp_id[-8:]} [Threads: {threads_str}] (Parent: ...{parent_id[-8:] if parent_id else 'None'})")
        
        msg_indent = indent + ("    " if is_last else "│   ")
        messages = cp_tuple.checkpoint["channel_values"].get("messages", [])
        for msg in messages:
            role = msg.type
            content = msg.content.replace("\n", " ")
            trimmed_content = content[:30] + "..." if len(content) > 30 else content
            role_display = "user" if role in ["user", "human"] else "assistant"
            print(f"{msg_indent}  [{role_display}] {trimmed_content}")
            
        children = sorted(parent_to_children[cp_id], key=lambda cid: all_cps[cid].checkpoint["ts"])
        child_indent = indent + ("    " if is_last else "│   ")
        for idx, child_id in enumerate(children):
            print_node(child_id, child_indent, is_last=(idx == len(children) - 1))

    print("\n" + "="*10 + " LANGGRAPH CHECKPOINT TREE " + "="*10)
    for idx, root_id in enumerate(roots):
        print_node(root_id, is_last=(idx == len(roots) - 1))
    print("="*60 + "\n")
