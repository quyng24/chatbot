import { Bot } from "lucide-react";

export function LoadingSkeleton() {
  return (
    <main className="mx-auto flex h-screen max-w-2xl flex-col p-4 pointer-events-none">
      <div className="flex justify-center pt-6 pb-4">
        <div className="flex items-center gap-2 rounded-full border border-border/50 bg-muted/30 px-4 py-1.5 backdrop-blur-sm shadow-sm">
          <Bot className="size-4 animate-bounce text-muted-foreground [animation-duration:1.5s]" />
          <span className="text-sm font-medium text-muted-foreground/80 animate-pulse">
            Đang khởi tạo không gian làm việc...
          </span>
        </div>
      </div>

      <div className="flex-1 space-y-6 overflow-hidden pt-4">
        <div className="flex items-start gap-3 w-full">
          <div className="size-8 shrink-0 rounded-xl bg-muted/60 animate-pulse" />
          <div className="flex flex-col gap-2 w-full max-w-[80%]">
            <div className="h-12 w-3/4 rounded-2xl rounded-tl-sm bg-muted/50 animate-pulse" />
            <div className="h-4 w-1/3 rounded-md bg-muted/50 animate-pulse" />
          </div>
        </div>

        <div className="flex flex-col items-end gap-2 w-full mt-8">
          <div className="h-10 w-2/3 rounded-2xl rounded-tr-sm bg-primary/10 animate-pulse" />
        </div>

        <div className="flex items-start gap-3 w-full mt-8">
          <div className="size-8 shrink-0 rounded-xl bg-muted/60 animate-pulse" />
          <div className="flex flex-col gap-2 w-full max-w-[80%]">
            <div className="h-24 w-full rounded-2xl rounded-tl-sm bg-muted/50 animate-pulse" />
          </div>
        </div>
      </div>

      <div className="flex gap-2 pt-4 border-t border-border/40 mt-2">
        <div className="h-[42px] flex-1 rounded-lg bg-muted/40 animate-pulse" />
        <div className="h-[42px] w-16 rounded-lg bg-primary/20 animate-pulse" />
      </div>
    </main>
  );
}