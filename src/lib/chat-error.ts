function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null;
}

function getErrorMessages(error: unknown): string[] {
    const messages: string[] = [];

    if (error instanceof Error) {
        messages.push(error.message);
    }

    if (!isRecord(error)) {
        if (typeof error === "string") messages.push(error);
        return messages;
    }

    if (typeof error.message === "string") {
        messages.push(error.message);
    }

    const responseBody = error.responseBody;
    if (typeof responseBody === "string") {
        messages.push(responseBody);
        try {
            const body = JSON.parse(responseBody) as unknown;
            if (isRecord(body)) {
                if (typeof body.message === "string") messages.unshift(body.message);
                if (isRecord(body.error) && typeof body.error.message === "string") {
                    messages.unshift(body.error.message);
                }
            }
        } catch {
            // The provider may return plain text instead of a JSON response body.
        }
    } else if (isRecord(responseBody)) {
        if (typeof responseBody.message === "string") {
            messages.unshift(responseBody.message);
        }
        if (
            isRecord(responseBody.error) &&
            typeof responseBody.error.message === "string"
        ) {
            messages.unshift(responseBody.error.message);
        }
    }

    return messages;
}

export function getQuotaErrorMessage(error: unknown): string | undefined {
    const errorMessages = getErrorMessages(error);
    const quotaMessage = errorMessages.find((message) =>
        /quota exceeded|exceeded your current quota/i.test(message),
    );

    if (!quotaMessage) return undefined;

    const retryMatch = errorMessages
        .map((message) =>
            message.match(/Please retry in\s+((?:\d+(?:\.\d+)?\s*[hms]\s*)+)/i),
        )
        .find((match) => match?.[1]);

    if (!retryMatch?.[1]) {
        return "Bạn đã dùng hết lượt AI miễn phí. Vui lòng thử lại sau.";
    }

    const retryDuration = retryMatch[1]
        .trim()
        .replace(/(\d+(?:\.\d+)?)\s*h/gi, "$1 giờ ")
        .replace(/(\d+(?:\.\d+)?)\s*m/gi, "$1 phút ")
        .replace(/(\d+(?:\.\d+)?)\s*s/gi, (_, seconds: string) =>
            `${Math.ceil(Number(seconds))} giây`,
        )
        .trim();

    return `Bạn đã dùng hết lượt AI miễn phí. Vui lòng thử lại sau ${retryDuration}.`;
}

export function getPublicErrorMessage(error: unknown): string {
    return (
        getQuotaErrorMessage(error) ??
        "Có lỗi xảy ra khi kết nối máy chủ AI. Vui lòng thử lại."
    );
}
