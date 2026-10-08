"use client";

import { Sparkles } from "lucide-react";

export function AITypingIndicator() {
  return (
     <div role="status" aria-label="AI đang trả lời" className="flex items-center gap-3 py-1 text-left">
      {/* Avatar AI */}
      <div className="relative flex size-8 shrink-0 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 text-primary">
        <Sparkles className="size-4 animate-pulse" />
      </div>

      {/* 3 chấm nhảy đan xen (Staggered Effect) */}
      <div className="ml-1 flex h-8 items-center gap-1">
        <span 
          className="size-1.5 rounded-full bg-muted-foreground animate-typing-dot" 
          style={{ animationDelay: '0s' }} 
        />
        <span 
          className="size-1.5 rounded-full bg-muted-foreground animate-typing-dot" 
          style={{ animationDelay: '0.2s' }} 
        />
        <span 
          className="size-1.5 rounded-full bg-muted-foreground animate-typing-dot" 
          style={{ animationDelay: '0.4s' }} 
        />
      </div>
    </div>
  );
}