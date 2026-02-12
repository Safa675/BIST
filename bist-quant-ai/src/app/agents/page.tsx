"use client";

import { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import AgentChat from "@/components/AgentChat";
import { Target, Shield, Brain, Sparkles, Zap, GitBranch, MessageCircle, ArrowRight } from "lucide-react";

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
        gradient: "linear-gradient(135deg, #10b981, #059669)",
        description: "Manages factor allocations, rebalancing decisions, and portfolio construction across 34+ quantitative signals.",
        capabilities: [
            "Factor signal selection & weighting",
            "Monthly rebalancing optimization",
            "Position sizing via inverse downside vol",
            "Cross-signal conviction analysis",
        ],
        example: "\"We're overweight Breakout Value this month due to strong momentum confirmation signals. The factor's 102% CAGR is supported by low beta (0.36) and a robust 2.92 Sharpe ratio.\"",
    },
    {
        key: "risk",
        name: "Risk Manager",
        icon: Shield,
        color: "#06b6d4",
        gradient: "linear-gradient(135deg, #06b6d4, #0891b2)",
        description: "Monitors drawdowns, volatility targeting, regime shifts, and implements stop-loss protocols to protect capital.",
        capabilities: [
            "Regime detection (XGBoost + LSTM + HMM)",
            "Volatility targeting at 20% annualized",
            "Max drawdown monitoring & alerts",
            "Automatic gold rotation in Bear/Stress",
        ],
        example: "\"Current regime: Bull — maintaining full equity exposure. Realized downside vol is 17.2%, below our 20% target, so leverage is scaled to 1.0x. No risk alerts active.\"",
    },
    {
        key: "analyst",
        name: "Market Analyst",
        icon: Brain,
        color: "#8b5cf6",
        gradient: "linear-gradient(135deg, #8b5cf6, #7c3aed)",
        description: "Analyzes BIST market trends, sector rotations, macro drivers, and provides actionable market intelligence.",
        capabilities: [
            "Sector rotation & relative strength",
            "USD/TRY & macro impact analysis",
            "Earnings & fundamental catalysts",
            "BIST-specific market intelligence",
        ],
        example: "\"Banking sector momentum is accelerating with AKBNK and VAKBN breaking 20-day highs. USD/TRY stability supports equity valuations. Our Breakout Value factor is capturing this rotation.\"",
    },
];

const ARCHITECTURE_STEPS = [
    {
        icon: Zap,
        title: "Data Ingestion",
        description: "Real-time market data, fundamentals, and macro indicators flow into the quant engine.",
    },
    {
        icon: GitBranch,
        title: "Signal Generation",
        description: "34+ factor models generate buy/sell signals based on value, momentum, quality, and breakout patterns.",
    },
    {
        icon: Shield,
        title: "Regime Detection",
        description: "ML ensemble (XGBoost + LSTM + HMM) classifies the current market regime and adjusts risk exposure.",
    },
    {
        icon: MessageCircle,
        title: "Agent Orchestration",
        description: "Three AI agents coordinate to analyze signals, manage risk, and explain decisions in natural language.",
    },
];

export default function AgentsPage() {
    const [data, setData] = useState<DashboardData | null>(null);

    useEffect(() => {
        fetch("/api/dashboard", { cache: "no-store" })
            .then((r) => {
                if (!r.ok) {
                    throw new Error(`Dashboard API failed (${r.status})`);
                }
                return r.json();
            })
            .then(setData)
            .catch(console.error);
    }, []);

    return (
        <>
            <Navbar />

            <main
                style={{
                    paddingTop: 100,
                    paddingBottom: 80,
                    minHeight: "100vh",
                    background: "var(--bg-primary)",
                }}
            >
                <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px" }}>

                    {/* ===== HEADER ===== */}
                    <div className="animate-fade-in-up" style={{ textAlign: "center", marginBottom: 64 }}>
                        <div className="badge badge-bull" style={{ marginBottom: 16 }}>
                            <Sparkles size={14} />
                            MULTI-AGENT AI SYSTEM
                        </div>
                        <h1
                            style={{
                                fontSize: "clamp(2rem, 5vw, 3rem)",
                                fontWeight: 800,
                                letterSpacing: "-0.03em",
                                lineHeight: 1.15,
                                marginBottom: 16,
                            }}
                        >
                            Three AI Agents,{" "}
                            <span className="gradient-text">One Mission</span>
                        </h1>
                        <p style={{ fontSize: "1.1rem", color: "var(--text-secondary)", maxWidth: 640, margin: "0 auto", lineHeight: 1.7 }}>
                            Our multi-agent architecture combines portfolio management, risk monitoring, and market analysis
                            into a collaborative intelligence system that explains every decision.
                        </p>
                    </div>

                    {/* ===== AGENT CARDS ===== */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 24, marginBottom: 80 }}>
                        {AGENTS.map((agent, i) => {
                            const Icon = agent.icon;
                            return (
                                <div
                                    key={agent.key}
                                    className="glass-card animate-fade-in-up"
                                    style={{ padding: 0, animationDelay: `${i * 0.15}s`, opacity: 0 }}
                                >
                                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", minHeight: 300 }}>
                                        {/* Left Info */}
                                        <div style={{ padding: "32px 40px", display: "flex", flexDirection: "column", justifyContent: "center" }}>
                                            <div
                                                style={{
                                                    width: 56,
                                                    height: 56,
                                                    borderRadius: "var(--radius-md)",
                                                    background: agent.gradient,
                                                    display: "flex",
                                                    alignItems: "center",
                                                    justifyContent: "center",
                                                    marginBottom: 20,
                                                    boxShadow: `0 8px 32px ${agent.color}40`,
                                                }}
                                            >
                                                <Icon size={28} color="#fff" />
                                            </div>
                                            <h2 style={{ fontSize: "1.75rem", fontWeight: 800, marginBottom: 8 }}>{agent.name}</h2>
                                            <p style={{ color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 20 }}>
                                                {agent.description}
                                            </p>
                                            <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 8 }}>
                                                {agent.capabilities.map((cap) => (
                                                    <li
                                                        key={cap}
                                                        style={{
                                                            display: "flex",
                                                            alignItems: "center",
                                                            gap: 10,
                                                            fontSize: "0.9rem",
                                                            color: "var(--text-secondary)",
                                                        }}
                                                    >
                                                        <span style={{ width: 6, height: 6, borderRadius: "50%", background: agent.color, flexShrink: 0 }} />
                                                        {cap}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>

                                        {/* Right Example */}
                                        <div
                                            style={{
                                                padding: "32px 40px",
                                                borderLeft: "1px solid var(--border-subtle)",
                                                background: "rgba(0,0,0,0.15)",
                                                display: "flex",
                                                flexDirection: "column",
                                                justifyContent: "center",
                                            }}
                                        >
                                            <div style={{ fontSize: "0.75rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 16 }}>
                                                Sample Output (Static)
                                            </div>
                                            <div
                                                style={{
                                                    padding: "20px 24px",
                                                    borderRadius: "var(--radius-md)",
                                                    background: "var(--bg-card)",
                                                    border: `1px solid ${agent.color}20`,
                                                    fontStyle: "italic",
                                                    color: "var(--text-secondary)",
                                                    lineHeight: 1.7,
                                                    fontSize: "0.9rem",
                                                }}
                                            >
                                                {agent.example}
                                            </div>
                                            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 16, fontSize: "0.8rem", color: agent.color }}>
                                                <span style={{ width: 8, height: 8, borderRadius: "50%", background: agent.color }} />
                                                Card Preview · Not live chat
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* ===== HOW IT WORKS ===== */}
                    <div style={{ marginBottom: 80 }}>
                        <h2
                            className="animate-fade-in-up"
                            style={{
                                fontSize: "1.75rem",
                                fontWeight: 800,
                                textAlign: "center",
                                marginBottom: 48,
                                letterSpacing: "-0.02em",
                            }}
                        >
                            How the <span className="gradient-text">Pipeline</span> Works
                        </h2>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 24 }}>
                            {ARCHITECTURE_STEPS.map((step, i) => {
                                const StepIcon = step.icon;
                                return (
                                    <div
                                        key={step.title}
                                        className="glass-card animate-fade-in-up"
                                        style={{ padding: 24, textAlign: "center", animationDelay: `${i * 0.1}s`, opacity: 0 }}
                                    >
                                        <div
                                            style={{
                                                width: 48,
                                                height: 48,
                                                borderRadius: "50%",
                                                background: "var(--accent-emerald-dim)",
                                                display: "flex",
                                                alignItems: "center",
                                                justifyContent: "center",
                                                margin: "0 auto 16px",
                                            }}
                                        >
                                            <StepIcon size={22} color="var(--accent-emerald)" />
                                        </div>
                                        <div
                                            style={{
                                                fontSize: "0.7rem",
                                                fontWeight: 700,
                                                fontFamily: "var(--font-mono)",
                                                color: "var(--accent-emerald)",
                                                marginBottom: 8,
                                            }}
                                        >
                                            STEP {i + 1}
                                        </div>
                                        <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: 8 }}>{step.title}</h3>
                                        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", lineHeight: 1.6 }}>
                                            {step.description}
                                        </p>
                                        {i < ARCHITECTURE_STEPS.length - 1 && (
                                            <ArrowRight
                                                size={16}
                                                color="var(--text-muted)"
                                                style={{
                                                    position: "absolute",
                                                    right: -20,
                                                    top: "50%",
                                                    transform: "translateY(-50%)",
                                                }}
                                            />
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* ===== LIVE CHAT DEMO ===== */}
                    {data && (
                        <div style={{ marginBottom: 64 }}>
                            <h2
                                className="animate-fade-in-up"
                                style={{
                                    fontSize: "1.75rem",
                                    fontWeight: 800,
                                    textAlign: "center",
                                    marginBottom: 12,
                                    letterSpacing: "-0.02em",
                                }}
                            >
                                Try the <span className="gradient-text">Agents</span> Live
                            </h2>
                            <p style={{ textAlign: "center", color: "var(--text-muted)", marginBottom: 32 }}>
                                Switch between agents and ask questions about the portfolio, risk, or market conditions
                            </p>
                            <div style={{ maxWidth: 500, margin: "0 auto" }}>
                                <AgentChat
                                    holdings={data.holdings}
                                    signals={data.signals}
                                    regime={data.current_regime}
                                />
                            </div>
                        </div>
                    )}

                </div>
            </main>
        </>
    );
}
