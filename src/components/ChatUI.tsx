"use client";

import { useChat } from "@ai-sdk/react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import { ArrowUp, Bot, Sparkles } from "lucide-react";
import { AITypingIndicator } from "./loading/AITypingIndicator";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

export default function ChatUI() {
  const { messages, sendMessage, status, error } = useChat();
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: status === "streaming" ? "auto" : "smooth",
    });
  }, [messages, status]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = input.trim();

    if (!message || status !== "ready") return;

    sendMessage({ text: message });
    setInput("");
  }

  return (
    <main className="relative flex h-dvh flex-col overflow-hidden bg-background">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -left-40 -top-48 size-96 rounded-full bg-primary/5 blur-3xl"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -bottom-48 -right-40 size-96 rounded-full bg-primary/5 blur-3xl"
      />

      <header className="relative z-10 shrink-0 border-b border-border/60 bg-background/80 backdrop-blur-xl">
        <div className="mx-auto flex h-[4.25rem] w-full max-w-6xl items-center gap-3 px-4 sm:px-6">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-sm shadow-primary/20">
            <Sparkles className="size-5" />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold tracking-tight text-foreground sm:text-base">
              Trợ lý AI
            </h1>
            <p className="text-xs text-muted-foreground">
              Hỏi đáp thông minh, rõ ràng
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2 rounded-full border border-border/70 bg-card/70 px-3 py-1.5 text-xs text-muted-foreground">
            <span className="size-1.5 rounded-full bg-emerald-500" />
            <span className="hidden sm:inline">Đang hoạt động</span>
            <span className="sm:hidden">Sẵn sàng</span>
          </div>
        </div>
      </header>

      <section
        aria-label="Cuộc trò chuyện"
        className="relative flex min-h-0 flex-1 flex-col"
      >
        <div
          aria-live="polite"
          className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
        >
          <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col justify-end gap-6 px-4 py-6 sm:gap-8 sm:px-6 sm:py-8">
            {messages.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center py-12 text-center">
                <div className="mb-5 flex size-16 items-center justify-center rounded-3xl border border-primary/10 bg-primary/5 text-primary shadow-sm shadow-primary/5 sm:size-20">
                  <Bot className="size-8 sm:size-10" />
                </div>
                <h2 className="text-xl font-semibold tracking-tight text-foreground sm:text-2xl">
                  Bắt đầu cuộc trò chuyện
                </h2>
                <p className="mt-2 max-w-sm text-sm leading-6 text-muted-foreground sm:text-base">
                  Bạn đang cần tìm hiểu điều gì? Hãy nhập câu hỏi bên dưới để
                  bắt đầu.
                </p>
              </div>
            ) : (
              messages.map((message) => {
                const text = message.parts
                  .flatMap((part) => (part.type === "text" ? [part.text] : []))
                  .join("");
                const isUser = message.role === "user";

                return (
                  <article
                    key={message.id}
                    className={`flex min-w-0 items-start gap-3 sm:gap-4 ${
                      isUser ? "justify-end" : "justify-start"
                    }`}
                  >
                    {!isUser && (
                      <div
                        aria-hidden="true"
                        className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary"
                      >
                        <Sparkles className="size-4" />
                      </div>
                    )}
                    <div
                      className={`min-w-0 max-w-[88%] sm:max-w-[80%] ${
                        isUser
                          ? "rounded-2xl rounded-tr-md bg-primary px-4 py-3 text-primary-foreground shadow-sm sm:px-5"
                          : "pt-1"
                      }`}
                    >
                      {isUser ? (
                        <p className="whitespace-pre-wrap break-words text-sm leading-6 sm:text-[0.9375rem]">
                          {text}
                        </p>
                      ) : (
                        <div className="min-w-0 break-words text-sm leading-7 text-foreground sm:text-[0.9375rem] [&_a]:text-primary [&_a]:underline [&_a]:underline-offset-4 [&_blockquote]:border-l-2 [&_blockquote]:border-primary/40 [&_blockquote]:pl-4 [&_blockquote]:text-muted-foreground [&_code]:rounded [&_code]:bg-muted [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.85em] [&_h1]:mb-3 [&_h1]:mt-5 [&_h1]:text-xl [&_h1]:font-semibold [&_h2]:mb-2 [&_h2]:mt-4 [&_h2]:text-lg [&_h2]:font-semibold [&_h3]:mb-2 [&_h3]:mt-4 [&_h3]:font-semibold [&_li]:pl-1 [&_ol]:my-3 [&_ol]:list-decimal [&_ol]:space-y-1 [&_ol]:pl-5 [&_p+p]:mt-3 [&_pre]:my-3 [&_pre]:overflow-x-auto [&_pre]:rounded-xl [&_pre]:bg-muted [&_pre]:p-4 [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_strong]:font-semibold [&_strong]:text-foreground [&_ul]:my-3 [&_ul]:list-disc [&_ul]:space-y-1 [&_ul]:pl-5">
                          <ReactMarkdown>{text}</ReactMarkdown>
                        </div>
                      )}
                    </div>
                  </article>
                );
              })
            )}
            {status === "submitted" && (
              <AITypingIndicator />
            )}
            <div ref={messagesEndRef} aria-hidden="true" />
          </div>
        </div>

        <footer className="relative z-10 shrink-0 border-t border-border/60 bg-background/85 px-4 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3 backdrop-blur-xl sm:px-6 sm:pt-4">
          <div className="mx-auto w-full max-w-3xl">
            {error && (
              <div className="flex flex-col items-center justify-center gap-2 rounded-xl bg-destructive/10 p-4 text-destructive border border-destructive/20 mt-4 mx-12 shadow-sm">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-alert-circle size-6 opacity-80">
                  <circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>
                </svg>
                <p className="text-sm font-medium text-center leading-relaxed">
                  {/* Lấy trực tiếp câu lỗi đã được format từ backend */}
                  {error.message || "Đã có lỗi xảy ra khi kết nối với AI."}
                </p>
              </div>
            )}
            <form
              onSubmit={handleSubmit}
              className="flex items-center gap-2 rounded-2xl border border-border/80 bg-card p-1.5 shadow-lg shadow-foreground/[0.03] transition-colors focus-within:border-ring/70 focus-within:ring-4 focus-within:ring-ring/10 sm:gap-3 sm:p-2"
            >
              <Input
                aria-label="Tin nhắn"
                autoComplete="off"
                className="h-10 border-0 bg-transparent px-3 text-sm shadow-none focus-visible:border-transparent focus-visible:ring-0 sm:h-11 sm:px-4 sm:text-base"
                disabled={status !== "ready"}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Nhập câu hỏi của bạn..."
                type="text"
                value={input}
              />
              <Button
                aria-label="Gửi tin nhắn"
                className="size-10 rounded-xl px-0 shadow-sm sm:size-11"
                disabled={status !== "ready" || !input.trim()}
                type="submit"
              >
                <ArrowUp className="size-5" />
              </Button>
            </form>
            <p className="mt-2 text-center text-[11px] text-muted-foreground sm:text-xs">
              AI có thể mắc lỗi. Hãy kiểm tra lại những thông tin quan trọng.
            </p>
          </div>
        </footer>
      </section>
    </main>
  );
}
