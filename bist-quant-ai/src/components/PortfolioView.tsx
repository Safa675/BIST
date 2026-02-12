"use client";

import { useState } from "react";

interface Props {
    holdings: Record<string, string[]>;
    selectedSignal: string;
    onSelectSignal: (name: string) => void;
}

const formatName = (n: string) =>
    n.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

export default function PortfolioView({ holdings, selectedSignal, onSelectSignal }: Props) {
    const signals = Object.keys(holdings).sort();
    const currentHoldings = holdings[selectedSignal] || [];

    // Count ticker frequency across all signals
    const tickerFrequency: Record<string, number> = {};
    for (const tickers of Object.values(holdings)) {
        for (const t of tickers) {
            tickerFrequency[t] = (tickerFrequency[t] || 0) + 1;
        }
    }

    return (
        <div className="glass-card" style={{ overflow: "hidden" }}>
            {/* Signal selector */}
            <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--border-subtle)" }}>
                <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    Select Factor Signal
                </label>
                <select
                    value={selectedSignal}
                    onChange={(e) => onSelectSignal(e.target.value)}
                    style={{
                        width: "100%",
                        padding: "8px 14px",
                        borderRadius: "var(--radius-sm)",
                        border: "1px solid var(--border-subtle)",
                        background: "rgba(0,0,0,0.3)",
                        color: "var(--text-primary)",
                        fontSize: "0.9rem",
                        outline: "none",
                        cursor: "pointer",
                    }}
                >
                    {signals.map((s) => (
                        <option key={s} value={s}>
                            {formatName(s)} ({holdings[s]?.length || 0} holdings)
                        </option>
                    ))}
                </select>
            </div>

            {/* Holdings grid */}
            <div style={{ padding: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                    <h3 style={{ fontSize: "0.95rem", fontWeight: 700 }}>
                        {formatName(selectedSignal)} <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>· {currentHoldings.length} holdings</span>
                    </h3>
                </div>

                {currentHoldings.length === 0 ? (
                    <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
                        No holdings for this signal.
                    </p>
                ) : (
                    <div
                        style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(auto-fill, minmax(100px, 1fr))",
                            gap: 8,
                        }}
                    >
                        {currentHoldings.map((ticker) => {
                            const freq = tickerFrequency[ticker] || 1;
                            const isMulti = freq > 3;
                            return (
                                <div
                                    key={ticker}
                                    style={{
                                        padding: "10px 12px",
                                        borderRadius: "var(--radius-sm)",
                                        background: isMulti ? "var(--accent-emerald-dim)" : "rgba(0,0,0,0.2)",
                                        border: `1px solid ${isMulti ? "rgba(16,185,129,0.2)" : "var(--border-subtle)"}`,
                                        textAlign: "center",
                                        transition: "all var(--transition-fast)",
                                        cursor: "default",
                                    }}
                                    title={`${ticker} — held by ${freq} signals`}
                                >
                                    <div
                                        style={{
                                            fontFamily: "var(--font-mono)",
                                            fontWeight: 700,
                                            fontSize: "0.85rem",
                                            color: isMulti ? "var(--accent-emerald)" : "var(--text-primary)",
                                        }}
                                    >
                                        {ticker}
                                    </div>
                                    {isMulti && (
                                        <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: 2 }}>
                                            {freq} signals
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}

                {/* Legend */}
                <div style={{ marginTop: 16, display: "flex", gap: 16, fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                        <span style={{ width: 8, height: 8, borderRadius: 2, background: "var(--accent-emerald)" }} />
                        Cross-signal conviction (4+ factors)
                    </span>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                        <span style={{ width: 8, height: 8, borderRadius: 2, background: "var(--text-muted)" }} />
                        Single-signal holding
                    </span>
                </div>
            </div>
        </div>
    );
}
