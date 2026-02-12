"use client";

import { useEffect, useState, useMemo } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    CartesianGrid,
} from "recharts";

interface CurvePoint {
    date: string;
    value: number;
}

type CurveData = Record<string, CurvePoint[]>;

const FACTOR_COLORS: Record<string, string> = {
    breakout_value: "#10b981",
    small_cap_momentum: "#06b6d4",
    trend_value: "#8b5cf6",
    donchian: "#f59e0b",
    value: "#ec4899",
    momentum: "#3b82f6",
    xu100: "#64748b",
};

const FACTOR_LABELS: Record<string, string> = {
    breakout_value: "Breakout Value",
    small_cap_momentum: "Small Cap Mom.",
    trend_value: "Trend Value",
    donchian: "Donchian",
    value: "Value",
    momentum: "Momentum",
    xu100: "XU100 (Bench.)",
};

export default function EquityChart() {
    const [curves, setCurves] = useState<CurveData | null>(null);
    const [visible, setVisible] = useState<Set<string>>(
        new Set(["breakout_value", "trend_value", "xu100"])
    );
    const [logScale, setLogScale] = useState(true);

    useEffect(() => {
        fetch("/data/equity_curves.json")
            .then((r) => r.json())
            .then(setCurves)
            .catch(console.error);
    }, []);

    // Merge all curves into one unified dataset keyed by date
    const merged = useMemo(() => {
        if (!curves) return [];
        const dateMap: Record<string, Record<string, number>> = {};

        for (const [factor, points] of Object.entries(curves)) {
            for (const pt of points) {
                if (!dateMap[pt.date]) dateMap[pt.date] = {};
                dateMap[pt.date][factor] = pt.value;
            }
        }

        return Object.entries(dateMap)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([date, values]) => ({ date, ...values }));
    }, [curves]);

    if (!curves) {
        return (
            <div className="glass-card" style={{ padding: 40, textAlign: "center" }}>
                <p style={{ color: "var(--text-muted)" }}>Loading equity curves...</p>
            </div>
        );
    }

    const factors = Object.keys(curves);

    const toggleFactor = (f: string) => {
        setVisible((prev) => {
            const next = new Set(prev);
            if (next.has(f)) next.delete(f);
            else next.add(f);
            return next;
        });
    };

    return (
        <div className="glass-card" style={{ padding: 24 }}>
            {/* Controls */}
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 20,
                    flexWrap: "wrap",
                    gap: 12,
                }}
            >
                <h3 style={{ fontSize: "1rem", fontWeight: 700 }}>
                    Equity Curves
                </h3>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <button
                        onClick={() => setLogScale(!logScale)}
                        style={{
                            padding: "4px 12px",
                            borderRadius: "var(--radius-sm)",
                            border: "1px solid var(--border-subtle)",
                            background: logScale ? "var(--accent-emerald-dim)" : "transparent",
                            color: logScale ? "var(--accent-emerald)" : "var(--text-muted)",
                            fontSize: "0.75rem",
                            fontWeight: 600,
                            cursor: "pointer",
                        }}
                    >
                        {logScale ? "Log Scale" : "Linear"}
                    </button>
                    <button
                        onClick={() => setVisible(new Set(factors))}
                        style={{
                            padding: "4px 12px",
                            borderRadius: "var(--radius-sm)",
                            border: "1px solid var(--border-subtle)",
                            background: "transparent",
                            color: "var(--text-muted)",
                            fontSize: "0.75rem",
                            fontWeight: 600,
                            cursor: "pointer",
                        }}
                    >
                        Show All
                    </button>
                </div>
            </div>

            {/* Factor toggles */}
            <div style={{ display: "flex", gap: 6, marginBottom: 20, flexWrap: "wrap" }}>
                {factors.map((f) => (
                    <button
                        key={f}
                        onClick={() => toggleFactor(f)}
                        style={{
                            padding: "4px 10px",
                            borderRadius: "var(--radius-sm)",
                            border: `1px solid ${visible.has(f) ? FACTOR_COLORS[f] : "var(--border-subtle)"}`,
                            background: visible.has(f) ? `${FACTOR_COLORS[f]}15` : "transparent",
                            color: visible.has(f) ? FACTOR_COLORS[f] : "var(--text-muted)",
                            fontSize: "0.75rem",
                            fontWeight: 600,
                            cursor: "pointer",
                            transition: "all var(--transition-fast)",
                        }}
                    >
                        {FACTOR_LABELS[f] || f}
                    </button>
                ))}
            </div>

            {/* Chart */}
            <ResponsiveContainer width="100%" height={400}>
                <LineChart data={merged}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                    <XAxis
                        dataKey="date"
                        tick={{ fill: "#64748b", fontSize: 11 }}
                        tickFormatter={(v: string) => v.substring(0, 7)}
                        interval={60}
                    />
                    <YAxis
                        scale={logScale ? "log" : "auto"}
                        domain={logScale ? [1, "auto"] : ["auto", "auto"]}
                        tick={{ fill: "#64748b", fontSize: 11 }}
                        tickFormatter={(v: number) =>
                            v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 1 ? v.toFixed(0) : v.toFixed(2)
                        }
                    />
                    <Tooltip
                        contentStyle={{
                            background: "rgba(15,23,42,0.95)",
                            border: "1px solid rgba(148,163,184,0.15)",
                            borderRadius: 8,
                            fontSize: "0.8rem",
                            color: "#f1f5f9",
                        }}
                        labelFormatter={(l) => String(l)}
                        formatter={(value, name) => [
                            `${Number(value).toFixed(2)}x`,
                            FACTOR_LABELS[String(name)] || String(name),
                        ]}
                    />
                    {factors.map((f) => (
                        visible.has(f) && (
                            <Line
                                key={f}
                                type="monotone"
                                dataKey={f}
                                stroke={FACTOR_COLORS[f]}
                                strokeWidth={f === "xu100" ? 1.5 : 2}
                                dot={false}
                                strokeDasharray={f === "xu100" ? "6 3" : undefined}
                                connectNulls
                            />
                        )
                    ))}
                </LineChart>
            </ResponsiveContainer>

            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 12, textAlign: "center" }}>
                Equity curves start at 1.0 and show cumulative growth multiple. Log scale recommended for comparing different magnitudes.
            </p>
        </div>
    );
}
