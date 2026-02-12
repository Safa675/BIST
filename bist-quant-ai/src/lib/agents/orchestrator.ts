/**
 * Multi-Agent Orchestrator for Quant AI Platform
 *
 * Coordinates three specialized AI agents:
 * - Portfolio Manager: Factor allocations & rebalancing
 * - Risk Manager: Drawdowns, vol-targeting, regime shifts
 * - Market Analyst: BIST trends, sectors, macro analysis
 */

import { logError, logInfo } from "@/lib/agents/logging";

export type AgentType = "portfolio" | "risk" | "analyst";

export interface AgentMessage {
    role: "user" | "agent";
    agent: AgentType;
    content: string;
    timestamp: Date;
}

export interface AgentContext {
    regime: string;
    signals: {
        name: string;
        cagr: number;
        sharpe: number;
        max_dd: number;
        ytd: number;
    }[];
    holdings: Record<string, string[]>;
}

interface AzureOpenAIConfig {
    endpoint: string;
    apiKey: string;
    deployment: string;
    apiVersion: string;
}

interface AzureUsage {
    promptTokens: number | null;
    completionTokens: number | null;
    totalTokens: number | null;
}

interface AgentResponseOptions {
    requestId?: string;
    maxRetries?: number;
}

// Configuration constants
const MAX_QUERY_LENGTH = 2000;
const MIN_QUERY_LENGTH = 1;
const DEFAULT_MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;
const RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504];

/**
 * Sleep utility for retry delays with exponential backoff
 */
function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Validate user query input
 */
export function validateQuery(query: unknown): { valid: boolean; error?: string; sanitized?: string } {
    if (query === null || query === undefined) {
        return { valid: false, error: "Query is required." };
    }

    if (typeof query !== "string") {
        return { valid: false, error: "Query must be a string." };
    }

    const trimmed = query.trim();

    if (trimmed.length < MIN_QUERY_LENGTH) {
        return { valid: false, error: "Query cannot be empty." };
    }

    if (trimmed.length > MAX_QUERY_LENGTH) {
        return { valid: false, error: `Query exceeds maximum length of ${MAX_QUERY_LENGTH} characters.` };
    }

    return { valid: true, sanitized: trimmed };
}

export interface AgentHealthCheckResult {
    ok: boolean;
    status: number;
    latencyMs: number;
    endpointHost: string;
    deployment: string;
    apiVersion: string;
    azureRequestId: string | null;
    usage: AzureUsage;
    sample: string;
}

const SYSTEM_PROMPTS: Record<AgentType, string> = {
    portfolio: `You are the Portfolio Manager AI agent for Quant AI Platform.
You manage factor allocations across 34+ quantitative signals on Borsa Istanbul (BIST).
Your responsibilities:
- Explain portfolio construction decisions
- Analyze factor signal performance
- Discuss rebalancing strategy (monthly, inverse downside vol weights)
- Identify cross-signal conviction (stocks held by multiple factors)
Use data from the provided context. Be specific with numbers and signal names.
Speak with authority but accessible language for retail investors.`,

    risk: `You are the Risk Manager AI agent for Quant AI Platform.
You monitor risk metrics and protect capital across all factor strategies.
Your responsibilities:
- Monitor regime changes (Bull/Bear/Recovery/Stress) via ML ensemble
- Implement volatility targeting (20% annualized downside vol)
- Track max drawdowns and stop-losses
- Explain gold rotation during Bear/Stress regimes
Use data from the provided context. Be specific about risk metrics.
Alert proactively about potential risk events.`,

    analyst: `You are the Market Analyst AI agent for Quant AI Platform.
You analyze BIST market conditions, sector rotations, and macro drivers.
Your responsibilities:
- Track XU100 performance and key market drivers
- Analyze sector rotation and relative strength
- Monitor USD/TRY impact on equity valuations
- Identify earnings catalysts and fundamental signals
Use data from the provided context. Provide actionable market intelligence.
Focus on BIST-specific dynamics and emerging market factors.`,
};

const REQUIRED_AZURE_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
] as const;

function _env(name: string): string {
    return (process.env[name] || "").trim();
}

export function getAzureOpenAIConfig(): AzureOpenAIConfig {
    const missing = REQUIRED_AZURE_ENV.filter((name) => !_env(name));
    if (missing.length > 0) {
        throw new Error(
            `Missing required Azure OpenAI environment variable(s): ${missing.join(", ")}. ` +
            `Set them in your runtime environment (e.g. .env.local or Vercel project settings).`
        );
    }

    return {
        endpoint: _env("AZURE_OPENAI_ENDPOINT").replace(/\/$/, ""),
        apiKey: _env("AZURE_OPENAI_API_KEY"),
        deployment: _env("AZURE_OPENAI_DEPLOYMENT"),
        apiVersion: _env("AZURE_OPENAI_API_VERSION") || "2024-10-21",
    };
}

function _extractAssistantText(payload: unknown): string {
    const candidate = payload as {
        choices?: Array<{
            message?: {
                content?: string | Array<{ text?: string; type?: string }>;
            };
        }>;
    };

    const content = candidate?.choices?.[0]?.message?.content;
    if (typeof content === "string") {
        return content.trim();
    }
    if (Array.isArray(content)) {
        return content
            .map((item) => (typeof item?.text === "string" ? item.text : ""))
            .join("\n")
            .trim();
    }
    return "";
}

function _extractUsage(payload: unknown): AzureUsage {
    const usage = (payload as { usage?: { prompt_tokens?: unknown; completion_tokens?: unknown; total_tokens?: unknown } })?.usage;
    const toNumOrNull = (value: unknown) => {
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
    };
    return {
        promptTokens: toNumOrNull(usage?.prompt_tokens),
        completionTokens: toNumOrNull(usage?.completion_tokens),
        totalTokens: toNumOrNull(usage?.total_tokens),
    };
}

function _azureRequestIdFromHeaders(headers: Headers): string | null {
    return headers.get("x-ms-request-id")
        || headers.get("apim-request-id")
        || headers.get("x-request-id")
        || null;
}

function _endpointHost(endpoint: string): string {
    try {
        return new URL(endpoint).host;
    } catch {
        return endpoint;
    }
}

/**
 * Build the context string for the AI agent from real data
 */
export function buildContext(ctx: AgentContext): string {
    const topSignals = [...ctx.signals]
        .sort((a, b) => b.cagr - a.cagr)
        .slice(0, 10);

    const topYtd = [...ctx.signals]
        .sort((a, b) => b.ytd - a.ytd)
        .slice(0, 5);

    const avgCagr = ctx.signals.length > 0
        ? (ctx.signals.reduce((a, s) => a + s.cagr, 0) / ctx.signals.length).toFixed(1)
        : "0.0";

    const avgSharpe = ctx.signals.length > 0
        ? (ctx.signals.reduce((a, s) => a + s.sharpe, 0) / ctx.signals.length).toFixed(2)
        : "0.00";

    return `
Current Market Regime: ${ctx.regime}
Total Active Signals: ${ctx.signals.length}

Top Signals by CAGR:
${topSignals.map((s) => `- ${s.name}: CAGR=${s.cagr}%, Sharpe=${s.sharpe}, MaxDD=${s.max_dd}%, YTD=${s.ytd}%`).join("\n")}

Best YTD Performers:
${topYtd.map((s) => `- ${s.name}: YTD=${s.ytd}%`).join("\n")}

Average CAGR: ${avgCagr}%
Average Sharpe: ${avgSharpe}

Current Holdings (Breakout Value): ${ctx.holdings.breakout_value?.join(", ") || "None loaded"}
Total factors with holdings: ${Object.keys(ctx.holdings).length}
  `.trim();
}

/**
 * Get the system prompt for a specific agent
 */
export function getSystemPrompt(agent: AgentType): string {
    return SYSTEM_PROMPTS[agent];
}

export async function generateAgentResponse(
    agent: AgentType,
    query: string,
    context: AgentContext,
    options: AgentResponseOptions = {}
): Promise<string> {
    // Validate query input
    const validation = validateQuery(query);
    if (!validation.valid) {
        throw new Error(validation.error || "Invalid query.");
    }
    const sanitizedQuery = validation.sanitized!;

    const cfg = getAzureOpenAIConfig();
    const systemPrompt = getSystemPrompt(agent);
    const contextStr = buildContext(context);
    const url = `${cfg.endpoint}/openai/deployments/${encodeURIComponent(cfg.deployment)}/chat/completions` +
        `?api-version=${encodeURIComponent(cfg.apiVersion)}`;

    const maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;

    logInfo("agent.azure.request", {
        requestId: options.requestId || null,
        agent,
        endpointHost: _endpointHost(cfg.endpoint),
        deployment: cfg.deployment,
        apiVersion: cfg.apiVersion,
        queryChars: sanitizedQuery.length,
        signalCount: context.signals.length,
        holdingFactorCount: Object.keys(context.holdings).length,
    });

    const reqBody = {
        messages: [
            { role: "system", content: systemPrompt },
            {
                role: "user",
                content: `Context:\n${contextStr}\n\nUser question:\n${sanitizedQuery}`,
            },
        ],
        max_completion_tokens: 700,
    };

    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        const startedAt = Date.now();

        try {
            const res = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "api-key": cfg.apiKey,
                },
                body: JSON.stringify(reqBody),
                cache: "no-store",
            });

            const raw = await res.text();
            let parsed: unknown = null;
            try {
                parsed = JSON.parse(raw);
            } catch {
                parsed = null;
            }

            const usage = _extractUsage(parsed);
            const latencyMs = Date.now() - startedAt;
            const azureRequestId = _azureRequestIdFromHeaders(res.headers);

            if (!res.ok) {
                const errPayload = parsed as { error?: { message?: string } } | null;
                const upstreamMessage = errPayload?.error?.message || raw.slice(0, 400) || "Unknown Azure OpenAI error";

                // Check if error is retryable
                const isRetryable = RETRYABLE_STATUS_CODES.includes(res.status);

                logError("agent.azure.error", {
                    requestId: options.requestId || null,
                    agent,
                    status: res.status,
                    latencyMs,
                    azureRequestId,
                    endpointHost: _endpointHost(cfg.endpoint),
                    deployment: cfg.deployment,
                    apiVersion: cfg.apiVersion,
                    promptTokens: usage.promptTokens,
                    completionTokens: usage.completionTokens,
                    totalTokens: usage.totalTokens,
                    message: upstreamMessage,
                    attempt,
                    maxRetries,
                    willRetry: isRetryable && attempt < maxRetries,
                });

                if (isRetryable && attempt < maxRetries) {
                    const delayMs = RETRY_DELAY_MS * Math.pow(2, attempt - 1); // Exponential backoff
                    await sleep(delayMs);
                    continue;
                }

                throw new Error(`Azure OpenAI request failed (${res.status}): ${upstreamMessage}`);
            }

            const text = _extractAssistantText(parsed);
            if (!text) {
                throw new Error("Azure OpenAI returned an empty assistant response.");
            }

            logInfo("agent.azure.response", {
                requestId: options.requestId || null,
                agent,
                status: res.status,
                latencyMs,
                azureRequestId,
                endpointHost: _endpointHost(cfg.endpoint),
                deployment: cfg.deployment,
                apiVersion: cfg.apiVersion,
                promptTokens: usage.promptTokens,
                completionTokens: usage.completionTokens,
                totalTokens: usage.totalTokens,
                responseChars: text.length,
                attempt,
            });

            return text;
        } catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));

            // Network errors are retryable
            const isNetworkError = lastError.message.includes("fetch") ||
                                   lastError.message.includes("network") ||
                                   lastError.message.includes("ECONNRESET");

            if (isNetworkError && attempt < maxRetries) {
                logInfo("agent.azure.retry", {
                    requestId: options.requestId || null,
                    agent,
                    attempt,
                    maxRetries,
                    reason: lastError.message,
                });
                const delayMs = RETRY_DELAY_MS * Math.pow(2, attempt - 1);
                await sleep(delayMs);
                continue;
            }

            throw lastError;
        }
    }

    throw lastError || new Error("Agent response generation failed after retries.");
}

export async function runAzureOpenAIHealthCheck(options: AgentResponseOptions = {}): Promise<AgentHealthCheckResult> {
    const cfg = getAzureOpenAIConfig();
    const url = `${cfg.endpoint}/openai/deployments/${encodeURIComponent(cfg.deployment)}/chat/completions` +
        `?api-version=${encodeURIComponent(cfg.apiVersion)}`;

    const requestBody = {
        messages: [
            { role: "system", content: "You are a health-check assistant." },
            { role: "user", content: "Reply with exactly: OK" },
        ],
        max_completion_tokens: 8,
    };

    const startedAt = Date.now();
    logInfo("agent.health.azure.request", {
        requestId: options.requestId || null,
        endpointHost: _endpointHost(cfg.endpoint),
        deployment: cfg.deployment,
        apiVersion: cfg.apiVersion,
    });

    const res = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "api-key": cfg.apiKey,
        },
        body: JSON.stringify(requestBody),
        cache: "no-store",
    });

    const raw = await res.text();
    let parsed: unknown = null;
    try {
        parsed = JSON.parse(raw);
    } catch {
        parsed = null;
    }

    const usage = _extractUsage(parsed);
    const latencyMs = Date.now() - startedAt;
    const azureRequestId = _azureRequestIdFromHeaders(res.headers);
    const sample = _extractAssistantText(parsed);

    if (!res.ok) {
        const errPayload = parsed as { error?: { message?: string } } | null;
        const upstreamMessage = errPayload?.error?.message || raw.slice(0, 400) || "Unknown Azure OpenAI error";
        logError("agent.health.azure.error", {
            requestId: options.requestId || null,
            status: res.status,
            latencyMs,
            azureRequestId,
            endpointHost: _endpointHost(cfg.endpoint),
            deployment: cfg.deployment,
            apiVersion: cfg.apiVersion,
            promptTokens: usage.promptTokens,
            completionTokens: usage.completionTokens,
            totalTokens: usage.totalTokens,
            message: upstreamMessage,
        });
        throw new Error(`Azure OpenAI health check failed (${res.status}): ${upstreamMessage}`);
    }

    logInfo("agent.health.azure.response", {
        requestId: options.requestId || null,
        status: res.status,
        latencyMs,
        azureRequestId,
        endpointHost: _endpointHost(cfg.endpoint),
        deployment: cfg.deployment,
        apiVersion: cfg.apiVersion,
        promptTokens: usage.promptTokens,
        completionTokens: usage.completionTokens,
        totalTokens: usage.totalTokens,
        sampleChars: sample.length,
    });

    return {
        ok: true,
        status: res.status,
        latencyMs,
        endpointHost: _endpointHost(cfg.endpoint),
        deployment: cfg.deployment,
        apiVersion: cfg.apiVersion,
        azureRequestId,
        usage,
        sample,
    };
}
