"use client";

import { Sparkles } from "lucide-react";

interface AITypingIndicatorProps {
  text?: string;
}

export function AITypingIndicator({
  text = "AI đang suy nghĩ...",
}: AITypingIndicatorProps) {
  return (
    <div className="flex items-start gap-3 py-2 text-left">
      {/* Avatar AI nhỏ với hiệu ứng phát sáng nhẹ */}
      <div className="relative flex size-8 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20 shadow-xs">
        <Sparkles className="size-4 animate-pulse motion-reduce:animate-none" />
        <span className="absolute -top-0.5 -right-0.5 flex size-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60 opacity-75" />
          <span className="relative inline-flex size-2 rounded-full bg-primary" />
        </span>
      </div>

      {/* Bong bóng tin nhắn dạng Loading */}
      <div className="flex flex-col gap-1.5 max-w-[80%]">
        <div className="inline-flex items-center gap-3 rounded-2xl rounded-tl-sm border border-border/50 bg-muted/70 px-4 py-2.5 shadow-xs backdrop-blur-sm">
          <span className="text-sm font-medium text-muted-foreground animate-pulse">
            {text}
          </span>

          {/* 3 chấm nhảy (Typing Dots) đồng bộ màu sắc */}
          <div aria-hidden="true" className="flex items-center gap-1">
            <span className="size-1.5 animate-bounce rounded-full bg-primary/60 [animation-duration:0.8s] motion-reduce:animate-none" />
            <span className="size-1.5 animate-bounce rounded-full bg-primary/80 [animation-delay:0.15s] [animation-duration:0.8s] motion-reduce:animate-none" />
            <span className="size-1.5 animate-bounce rounded-full bg-primary [animation-delay:0.3s] [animation-duration:0.8s] motion-reduce:animate-none" />
          </div>
        </div>
      </div>
    </div>
  );
}