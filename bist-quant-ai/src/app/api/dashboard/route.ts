import { NextResponse } from "next/server";
import { loadDashboardData } from "@/lib/server/dashboardData";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
    const url = new URL(request.url);
    const forceRefresh = url.searchParams.get("refresh") === "1";

    try {
        const { data, refreshed, refreshError } = await loadDashboardData({
            refresh: forceRefresh,
            force: forceRefresh,
        });

        const payload: Record<string, unknown> = { ...data };
        if (refreshError) {
            payload.refresh_warning = refreshError;
        }

        return NextResponse.json(payload, {
            headers: {
                "Cache-Control": "no-store",
                "X-Dashboard-Refreshed": String(refreshed),
            },
        });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to load dashboard data.";
        return NextResponse.json(
            { error: message },
            {
                status: 500,
                headers: { "Cache-Control": "no-store" },
            }
        );
    }
}
