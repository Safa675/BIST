import { NextRequest, NextResponse } from "next/server";
import { type AgentType, generateAgentResponse } from "@/lib/agents/orchestrator";
import { createRequestId, logError, logInfo, safeErrorMessage } from "@/lib/agents/logging";
import { buildFallbackContext, loadDashboardData, resolveContext } from "./_shared";

function prettyAgentName(agent: AgentType): string {
    return `${agent.charAt(0).toUpperCase()}${agent.slice(1)}`;
}

export function createAgentPostHandler(agent: AgentType) {
    return async function POST(request: NextRequest) {
        const requestId = createRequestId();
        const startedAt = Date.now();
        const agentLabel = prettyAgentName(agent);

        try {
            let body: unknown = null;
            try {
                body = await request.json();
            } catch {
                return NextResponse.json(
                    {
                        error: "Request body must be valid JSON.",
                        request_id: requestId,
                    },
                    { status: 400 }
                );
            }

            const payload = (body || {}) as { query?: unknown; context?: unknown };
            const query = typeof payload.query === "string" ? payload.query.trim() : "";

            if (!query) {
                return NextResponse.json(
                    {
                        error: "Missing 'query' in request body.",
                        request_id: requestId,
                    },
                    { status: 400 }
                );
            }

            const data = await loadDashboardData();
            const fallbackContext = buildFallbackContext(data);
            const context = resolveContext(payload.context, fallbackContext);

            logInfo("agent.request.received", {
                requestId,
                agent,
                path: request.nextUrl.pathname,
                queryChars: query.length,
                regime: context.regime,
                signalCount: context.signals.length,
                holdingsFactorCount: Object.keys(context.holdings).length,
            });

            const responseText = await generateAgentResponse(agent, query, context, { requestId });
            const latencyMs = Date.now() - startedAt;

            logInfo("agent.request.completed", {
                requestId,
                agent,
                path: request.nextUrl.pathname,
                latencyMs,
                responseChars: responseText.length,
            });

            return NextResponse.json({
                mode: "llm_live",
                request_id: requestId,
                agent,
                query,
                response: responseText,
                context: {
                    regime: context.regime,
                    signalCount: context.signals.length,
                    holdingsFactorCount: Object.keys(context.holdings).length,
                },
                timestamp: new Date().toISOString(),
            });
        } catch (error) {
            const message = safeErrorMessage(error);
            const latencyMs = Date.now() - startedAt;

            logError("agent.request.failed", {
                requestId,
                agent,
                path: request.nextUrl.pathname,
                latencyMs,
                message,
            });

            return NextResponse.json(
                {
                    error: `${agentLabel} agent failed: ${message}`,
                    request_id: requestId,
                },
                { status: 500 }
            );
        }
    };
}
