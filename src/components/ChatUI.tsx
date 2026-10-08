"use client";

import { useChat } from "@ai-sdk/react";
import { useState } from "react";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { AITypingIndicator } from "./loading/AITypingIndicator";

export default function ChatUI() {
  const { messages, sendMessage, status } = useChat();
  const [input, setInput] = useState("");
  return (
    <main className="mx-auto flex h-screen max-w-2xl flex-col p-4">
      <div className="flex-1 space-y-3 overflow-y-auto">
        {messages.map((m) => (
          <div
            key={m.id}
            className={m.role === "user" ? "text-right" : "text-left"}
          >
            <span className="inline-block rounded-lg bg-muted px-3 py-2 whitespace-pre-wrap">
              {m.parts.map((p, i) =>
                p.type === "text" ? <span key={i}>{p.text}</span> : null,
              )}
            </span>
          </div>
        ))}
        {status === "submitted" && <AITypingIndicator />}
      </div>
      <form
        className="flex gap-2 pt-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (!input.trim()) return;
          sendMessage({ text: input });
          setInput("");
        }}
      >
        <Input value={input} onChange={e => setInput(e.target.value)} type="text" placeholder="Nhập tin nhắn ..."/>
        <Button type="submit" disabled={status !== "ready"}>Gửi</Button>
      </form>
    </main>
  );
}
