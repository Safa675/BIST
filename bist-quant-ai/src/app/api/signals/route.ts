import { NextResponse } from "next/server";
import { loadDashboardData } from "@/lib/server/dashboardData";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
    try {
        const { data } = await loadDashboardData();
        const signals = Array.isArray(data.signals) ? data.signals : [];

        return NextResponse.json({
            signals,
            current_regime: data.current_regime,
            xu100_ytd: data.xu100_ytd,
            last_update: data.last_update,
            active_signals: data.active_signals,
        });
    } catch (error) {
        console.error("Signals API error:", error);
        return NextResponse.json(
            { error: "Failed to load signal data" },
            { status: 500 }
        );
    }
}
