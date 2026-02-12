import type { AgentContext } from "@/lib/agents/orchestrator";
import { loadDashboardData as loadDashboardDataFromSource } from "@/lib/server/dashboardData";

interface SignalLike {
    name?: unknown;
    cagr?: unknown;
    sharpe?: unknown;
    max_dd?: unknown;
    ytd?: unknown;
}

function toNumber(value: unknown, fallback = 0): number {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeSignal(raw: SignalLike) {
    return {
        name: String(raw?.name || "unknown"),
        cagr: toNumber(raw?.cagr, 0),
        sharpe: toNumber(raw?.sharpe, 0),
        max_dd: toNumber(raw?.max_dd, 0),
        ytd: toNumber(raw?.ytd, 0),
    };
}

function normalizeHoldings(raw: unknown): Record<string, string[]> {
    if (!raw || typeof raw !== "object") {
        return {};
    }

    const output: Record<string, string[]> = {};
    for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
        if (!Array.isArray(value)) {
            continue;
        }
        output[String(key)] = value
            .map((item) => String(item || "").trim())
            .filter(Boolean);
    }
    return output;
}

export async function loadDashboardData() {
    try {
        const { data } = await loadDashboardDataFromSource();
        return data;
    } catch {
        return null;
    }
}

export function buildFallbackContext(data: unknown): AgentContext {
    const payload = (data || {}) as {
        current_regime?: unknown;
        signals?: SignalLike[];
        holdings?: unknown;
    };

    const signals = Array.isArray(payload.signals)
        ? payload.signals.map(normalizeSignal)
        : [];

    return {
        regime: String(payload.current_regime || "Unknown"),
        signals,
        holdings: normalizeHoldings(payload.holdings),
    };
}

export function resolveContext(rawContext: unknown, fallback: AgentContext): AgentContext {
    if (!rawContext || typeof rawContext !== "object") {
        return fallback;
    }

    const ctx = rawContext as {
        regime?: unknown;
        signals?: SignalLike[];
        holdings?: unknown;
    };

    const signals = Array.isArray(ctx.signals) ? ctx.signals.map(normalizeSignal) : fallback.signals;
    const holdings = normalizeHoldings(ctx.holdings);

    return {
        regime: String(ctx.regime || fallback.regime || "Unknown"),
        signals,
        holdings: Object.keys(holdings).length > 0 ? holdings : fallback.holdings,
    };
}
