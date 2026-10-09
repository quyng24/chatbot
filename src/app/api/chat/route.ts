import { SYSTEM_PROMPT } from "@/constants/message";
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
        return result.toUIMessageStreamResponse();
    } catch (error: any) {
        const errorMessage = error?.message || "Unknown error";
        if(errorMessage.includes("Quota exceeded")) {
            const match = errorMessage.match(/Please retry in (.*?)\./);
            if(match && match[1]) {
                let retryTime = match[1];
                retryTime = retryTime
                    .replace(/(\d+)h/, '$1 giờ ')
                    .replace(/(\d+)m/, '$1 phút ')
                    .replace(/(\d+)\.\d+s/, '$1 giây')
                    .replace(/(\d+)s/, '$1 giây');
                return new Response( `Bạn đã dùng hết lượt AI miễn phí. Vui lòng thử lại sau: ${retryTime.trim()}`, { status: 429 });
            }
            return new Response( `Bạn đã dùng hết lượt AI miễn phí. Vui lòng thử lại sau.`, { status: 429 });
        }
        console.error("AI Error:", error);
        return new Response("Có lỗi xảy ra khi kết nối máy chủ AI.", { status: 500 });
    }
}