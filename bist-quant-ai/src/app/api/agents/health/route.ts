import { NextRequest, NextResponse } from "next/server";
import { getAzureOpenAIConfig, runAzureOpenAIHealthCheck } from "@/lib/agents/orchestrator";
import { createRequestId, logError, logInfo, safeErrorMessage } from "@/lib/agents/logging";

export async function GET(request: NextRequest) {
    const requestId = createRequestId();
    const startedAt = Date.now();

    try {
        const cfg = getAzureOpenAIConfig();
        logInfo("agent.health.request.received", {
            requestId,
            path: request.nextUrl.pathname,
            deployment: cfg.deployment,
            apiVersion: cfg.apiVersion,
        });

        const result = await runAzureOpenAIHealthCheck({ requestId });
        const latencyMs = Date.now() - startedAt;

        logInfo("agent.health.request.completed", {
            requestId,
            path: request.nextUrl.pathname,
            latencyMs,
            status: result.status,
            endpointHost: result.endpointHost,
            deployment: result.deployment,
            azureRequestId: result.azureRequestId,
            totalTokens: result.usage.totalTokens,
        });

        return NextResponse.json({
            mode: "llm_live",
            request_id: requestId,
            ...result,
            checked_at: new Date().toISOString(),
        });
    } catch (error) {
        const latencyMs = Date.now() - startedAt;
        const message = safeErrorMessage(error);

        logError("agent.health.request.failed", {
            requestId,
            path: request.nextUrl.pathname,
            latencyMs,
            message,
        });

        return NextResponse.json(
            {
                ok: false,
                request_id: requestId,
                error: message,
                checked_at: new Date().toISOString(),
            },
            { status: 500 }
        );
    }
}
