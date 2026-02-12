import { randomUUID } from "crypto";

type LogLevel = "info" | "error";

type LogMeta = Record<string, unknown>;

function emitLog(level: LogLevel, event: string, meta: LogMeta = {}) {
    const payload = {
        ts: new Date().toISOString(),
        level,
        event,
        ...meta,
    };
    const line = JSON.stringify(payload);
    if (level === "error") {
        console.error(line);
    } else {
        console.info(line);
    }
}

export function createRequestId(): string {
    return randomUUID();
}

export function logInfo(event: string, meta: LogMeta = {}) {
    emitLog("info", event, meta);
}

export function logError(event: string, meta: LogMeta = {}) {
    emitLog("error", event, meta);
}

export function safeErrorMessage(error: unknown): string {
    if (error instanceof Error && error.message) {
        return error.message;
    }
    if (typeof error === "string" && error.trim()) {
        return error.trim();
    }
    return "Unknown error";
}
