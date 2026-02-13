"use client";

import { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import AgentChat from "@/components/AgentChat";
import { Target, Shield, Brain, Search, Sparkles, Zap, TrendingUp, Database } from "lucide-react";

interface DashboardData {
    current_regime: string;
    signals: { name: string; cagr: number; sharpe: number; max_dd: number; ytd: number }[];
    holdings: Record<string, string[]>;
}

const AGENTS = [
    {
        key: "portfolio",
        name: "Portfolio Manager",
        icon: Target,
        color: "#10b981",
        shortDesc: "Factor allocations & rebalancing",
    },
    {
        key: "risk",
        name: "Risk Manager",
        icon: Shield,
        color: "#06b6d4",
        shortDesc: "Drawdowns & regime monitoring",
    },
    {
        key: "analyst",
        name: "Market Analyst",
        icon: Brain,
        color: "#8b5cf6",
        shortDesc: "BIST trends & macro analysis",
    },
    {
        key: "research",
        name: "Research Analyst",
        icon: Search,
        color: "#f59e0b",
        shortDesc: "Live screening & fundamentals",
    },
];

const CAPABILITIES = [
    { icon: Zap, label: "34+ Factor Models", desc: "Value, momentum, quality signals" },
    { icon: TrendingUp, label: "Technical Scans", desc: "RSI, MACD, Bollinger, Supertrend" },
    { icon: Database, label: "Live BIST Data", desc: "758 stocks via Borsa MCP" },
    { icon: Shield, label: "Regime Detection", desc: "XGBoost + LSTM + HMM ensemble" },
];

export default function AgentsPage() {
    const [data, setData] = useState<DashboardData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch("/api/dashboard", { cache: "no-store" })
            .then((r) => {
                if (!r.ok) throw new Error(`Dashboard API failed (${r.status})`);
                return r.json();
            })
            .then(setData)
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    return (
        <>
            <Navbar />

            <main className="page-compact">
                <div className="page-container">
                    {/* Header */}
                    <div style={{ marginBottom: 16, display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
                        <div>
                            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                                <Sparkles size={18} color="var(--accent-emerald)" />
                                <h1 style={{ margin: 0, fontSize: "1.4rem", fontWeight: 800 }}>
                                    AI <span className="gradient-text">Agents</span>
                                </h1>
                            </div>
                            <p style={{ margin: 0, color: "var(--text-muted)", fontSize: "0.85rem" }}>
                                Multi-agent system for portfolio analysis, risk management, and live market research
                            </p>
                        </div>
                        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                            {CAPABILITIES.map((cap) => (
                                <div
                                    key={cap.label}
                                    style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 6,
                                        padding: "4px 10px",
                                        background: "var(--bg-card)",
                                        border: "1px solid var(--border-subtle)",
                                        borderRadius: "var(--radius-sm)",
                                        fontSize: "0.72rem",
                                    }}
                                >
                                    <cap.icon size={12} color="var(--accent-emerald)" />
                                    <span style={{ color: "var(--text-secondary)" }}>{cap.label}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Main Content Grid */}
                    <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 12, minHeight: "calc(100vh - 140px)" }}>
                        {/* Left Sidebar - Agent Info */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            <div className="glass-card" style={{ padding: 12 }}>
                                <h3 style={{ margin: "0 0 10px", fontSize: "0.85rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                                    Available Agents
                                </h3>
                                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                                    {AGENTS.map((agent) => {
                                        const Icon = agent.icon;
                                        return (
                                            <div
                                                key={agent.key}
                                                style={{
                                                    display: "flex",
                                                    alignItems: "center",
                                                    gap: 10,
                                                    padding: "10px 12px",
                                                    background: "var(--bg-tertiary)",
                                                    borderRadius: "var(--radius-sm)",
                                                    border: "1px solid var(--border-subtle)",
                                                }}
                                            >
                                                <div
                                                    style={{
                                                        width: 32,
                                                        height: 32,
                                                        borderRadius: "var(--radius-sm)",
                                                        background: `${agent.color}20`,
                                                        display: "flex",
                                                        alignItems: "center",
                                                        justifyContent: "center",
                                                        flexShrink: 0,
                                                    }}
                                                >
                                                    <Icon size={16} color={agent.color} />
                                                </div>
                                                <div style={{ minWidth: 0 }}>
                                                    <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-primary)" }}>
                                                        {agent.name}
                                                    </div>
                                                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                                                        {agent.shortDesc}
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>

                            {/* Context Info */}
                            {data && (
                                <div className="glass-card" style={{ padding: 12 }}>
                                    <h3 style={{ margin: "0 0 10px", fontSize: "0.85rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                                        Current Context
                                    </h3>
                                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                                            <span style={{ color: "var(--text-muted)" }}>Regime</span>
                                            <span className={`badge badge-${data.current_regime.toLowerCase()}`} style={{ padding: "2px 8px", fontSize: "0.7rem" }}>
                                                {data.current_regime}
                                            </span>
                                        </div>
                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                                            <span style={{ color: "var(--text-muted)" }}>Active Signals</span>
                                            <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{data.signals.length}</span>
                                        </div>
                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                                            <span style={{ color: "var(--text-muted)" }}>Factors w/ Holdings</span>
                                            <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{Object.keys(data.holdings).length}</span>
                                        </div>
                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                                            <span style={{ color: "var(--text-muted)" }}>Top Signal CAGR</span>
                                            <span style={{ color: "var(--accent-emerald)", fontWeight: 600 }}>
                                                {data.signals.length > 0 ? `${Math.max(...data.signals.map(s => s.cagr)).toFixed(0)}%` : "—"}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Research Tools */}
                            <div className="glass-card" style={{ padding: 12 }}>
                                <h3 style={{ margin: "0 0 10px", fontSize: "0.85rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                                    Research Tools
                                </h3>
                                <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                                    <p style={{ margin: "0 0 6px" }}>The <strong style={{ color: "#f59e0b" }}>Research Analyst</strong> can:</p>
                                    <ul style={{ margin: 0, paddingLeft: 16 }}>
                                        <li>Screen stocks by P/E, P/B, dividend yield</li>
                                        <li>Fetch financial statements & ratios</li>
                                        <li>Run technical scans (RSI, MACD)</li>
                                        <li>Compare sector metrics</li>
                                        <li>Access TEFAS fund data (836+ funds)</li>
                                    </ul>
                                </div>
                            </div>
                        </div>

                        {/* Right Side - Chat Interface */}
                        <div className="glass-card" style={{ padding: 0, display: "flex", flexDirection: "column", minHeight: 600 }}>
                            {loading ? (
                                <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
                                    Loading agent context...
                                </div>
                            ) : data ? (
                                <AgentChat
                                    holdings={data.holdings}
                                    signals={data.signals}
                                    regime={data.current_regime}
                                />
                            ) : (
                                <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", padding: 24, textAlign: "center" }}>
                                    <div>
                                        <p style={{ margin: "0 0 8px" }}>Failed to load dashboard context.</p>
                                        <p style={{ margin: 0, fontSize: "0.8rem" }}>Agents still work but won&apos;t have portfolio context.</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>
        </>
    );
}
