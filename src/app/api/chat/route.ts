import { SYSTEM_PROMPT } from "@/constants/message";
import { getPublicErrorMessage, getQuotaErrorMessage } from "@/lib/chat-error";
import { google } from "@ai-sdk/google";
import { convertToModelMessages, smoothStream, streamText, type UIMessage } from "ai";

export const maxDuration = 30;

export async function POST(req: Request) {
    try {
        const { messages }: {messages: UIMessage[]} = await req.json();
        const result = await streamText({
            model: google("gemini-3.1-pro-preview"),
            system: SYSTEM_PROMPT,
            messages: await convertToModelMessages(messages),
            experimental_transform: smoothStream({ chunking: "word"}),
        })
        return result.toUIMessageStreamResponse({
            onError: getPublicErrorMessage,
        });
    } catch (error: unknown) {
        const quotaError = getQuotaErrorMessage(error);
        if (quotaError) {
            return new Response(quotaError, { status: 429 });
        }

        console.error("AI Error:", error);
        return new Response(getPublicErrorMessage(error), { status: 500 });
    }
}