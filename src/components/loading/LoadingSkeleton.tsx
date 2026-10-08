import { Bot } from "lucide-react";

export function LoadingSkeleton() {
  return (
    <main className="flex h-dvh flex-col overflow-hidden bg-background">
      <header className="shrink-0 border-b border-border/60 px-4 sm:px-6">
        <div className="mx-auto flex h-[4.25rem] w-full max-w-6xl items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <Bot className="size-5 animate-pulse" />
          </div>
          <div className="space-y-1">
            <div className="h-4 w-24 animate-pulse rounded bg-muted" />
            <div className="h-3 w-36 animate-pulse rounded bg-muted/70" />
          </div>
        </div>
      </header>

      <div className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col justify-end gap-8 overflow-hidden px-4 py-6 sm:px-6 sm:py-8">
        <div className="flex items-start gap-3">
          <div className="size-8 shrink-0 animate-pulse rounded-xl bg-muted/60" />
          <div className="flex w-full max-w-[80%] flex-col gap-2">
            <div className="h-12 w-3/4 animate-pulse rounded-2xl bg-muted/50" />
            <div className="h-4 w-1/3 animate-pulse rounded-md bg-muted/50" />
          </div>
        </div>

        <div className="flex w-full flex-col items-end gap-2">
          <div className="h-10 w-2/3 animate-pulse rounded-2xl bg-primary/10" />
        </div>

        <div className="flex items-start gap-3">
          <div className="size-8 shrink-0 animate-pulse rounded-xl bg-muted/60" />
          <div className="flex w-full max-w-[80%] flex-col gap-2">
            <div className="h-24 w-full animate-pulse rounded-2xl bg-muted/50" />
          </div>
        </div>
      </div>

      <footer className="shrink-0 border-t border-border/60 px-4 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-4 sm:px-6">
        <div className="mx-auto flex w-full max-w-3xl gap-2">
          <div className="h-12 flex-1 animate-pulse rounded-2xl bg-muted/40" />
          <div className="size-12 animate-pulse rounded-xl bg-primary/20" />
        </div>
      </footer>
    </main>
  );
}