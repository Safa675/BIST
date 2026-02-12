import Navbar from "@/components/Navbar";
import Link from "next/link";
import {
  TrendingUp,
  Shield,
  Brain,
  BarChart3,
  Zap,
  Globe,
  ArrowRight,
  ChevronRight,
  Activity,
  Target,
  Bot,
  LineChart,
} from "lucide-react";

/* ---------- static data for the landing page ---------- */

const topSignals = [
  { name: "Breakout Value", cagr: "102.18%", sharpe: "2.92", dd: "-31.47%" },
  { name: "Small Cap Momentum", cagr: "94.47%", sharpe: "2.46", dd: "-26.92%" },
  { name: "Trend Value", cagr: "88.24%", sharpe: "2.66", dd: "-35.96%" },
  { name: "Five Factor Rotation", cagr: "87.43%", sharpe: "2.45", dd: "-29.09%" },
  { name: "Donchian", cagr: "81.79%", sharpe: "2.49", dd: "-26.90%" },
  { name: "Value", cagr: "81.10%", sharpe: "2.52", dd: "-35.21%" },
];

const agents = [
  {
    icon: Target,
    name: "Portfolio Manager",
    desc: "Optimizes factor allocation, explains holdings, and adjusts exposure based on real-time regime detection.",
    color: "var(--accent-emerald)",
    bg: "var(--accent-emerald-dim)",
  },
  {
    icon: Shield,
    name: "Risk Manager",
    desc: "Monitors drawdown, volatility targeting, stop-losses and regime shifts. Autonomously reduces exposure in stress.",
    color: "var(--accent-cyan)",
    bg: "var(--accent-cyan-dim)",
  },
  {
    icon: Brain,
    name: "Market Analyst",
    desc: "Analyzes BIST trends, sector rotations, and macro indicators. Generates natural-language market commentary.",
    color: "var(--accent-violet)",
    bg: "var(--accent-violet-dim)",
  },
];

const features = [
  { icon: BarChart3, title: "34+ Factor Models", desc: "Value, momentum, quality, size, sector rotation, macro hedge and more — all backtested on 10+ years of BIST data." },
  { icon: Activity, title: "Regime Detection", desc: "XGBoost + LSTM + HMM ensemble detects Bull, Bear, Recovery, and Stress regimes to dynamically allocate capital." },
  { icon: Zap, title: "Volatility Targeting", desc: "Downside-vol targeting and inverse-vol position sizing keep risk constant across market conditions." },
  { icon: Globe, title: "Emerging Markets First", desc: "Built for the unique dynamics of BIST and expanding to Brazil, India, and Southeast Asia." },
  { icon: Bot, title: "Autonomous Agents", desc: "AI agents collaborate to manage your portfolio, explain decisions, and alert you to risks — 24/7." },
  { icon: LineChart, title: "Institutional Analytics", desc: "CAPM alpha, rolling beta, Sharpe/Sortino, monthly rebalancing — hedge-fund-grade analytics for everyone." },
];

const stats = [
  { value: "102%", label: "Top CAGR", sub: "Breakout Value" },
  { value: "2.92", label: "Best Sharpe", sub: "Risk-adjusted" },
  { value: "34+", label: "Factor Models", sub: "Live signals" },
  { value: "10+", label: "Years Data", sub: "BIST 2013-2026" },
];

/* ---------- Component ---------- */

export default function LandingPage() {
  return (
    <>
      <Navbar />

      {/* ===== HERO ===== */}
      <section
        style={{
          position: "relative",
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "var(--gradient-hero)",
          overflow: "hidden",
          paddingTop: 64,
        }}
      >
        <div className="grid-bg" />

        {/* Radial glow */}
        <div
          style={{
            position: "absolute",
            top: "15%",
            left: "50%",
            transform: "translateX(-50%)",
            width: 800,
            height: 800,
            background: "radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%)",
            pointerEvents: "none",
          }}
        />

        <div
          style={{
            position: "relative",
            zIndex: 1,
            maxWidth: 1000,
            margin: "0 auto",
            padding: "80px 24px",
            textAlign: "center",
          }}
        >
          {/* Pre-headline badge */}
          <div
            className="animate-fade-in"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              padding: "6px 16px",
              borderRadius: 100,
              background: "var(--accent-emerald-dim)",
              border: "1px solid rgba(16,185,129,0.2)",
              fontSize: "0.8rem",
              fontWeight: 600,
              color: "var(--accent-emerald)",
              marginBottom: 32,
            }}
          >
            <Activity size={14} />
            Current Regime: Bull · XU100 YTD +20.35%
          </div>

          <h1
            className="animate-fade-in-up stagger-1"
            style={{
              fontSize: "clamp(2.5rem, 6vw, 4.2rem)",
              fontWeight: 800,
              lineHeight: 1.1,
              letterSpacing: "-0.03em",
              marginBottom: 24,
            }}
          >
            Institutional-Grade{" "}
            <span className="gradient-text">Quant Trading</span>
            <br />
            Powered by AI Agents
          </h1>

          <p
            className="animate-fade-in-up stagger-2"
            style={{
              fontSize: "1.15rem",
              color: "var(--text-secondary)",
              maxWidth: 640,
              margin: "0 auto 40px",
              lineHeight: 1.7,
            }}
          >
            34+ proven factor models with 10 years of backtested performance on
            emerging markets. Multi-agent AI that autonomously manages portfolios,
            detects regime changes, and explains every decision.
          </p>

          <div
            className="animate-fade-in-up stagger-3"
            style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}
          >
            <Link href="/dashboard" className="btn-primary" style={{ padding: "14px 32px", fontSize: "1rem" }}>
              Launch Dashboard
              <ArrowRight size={18} />
            </Link>
            <a href="#performance" className="btn-secondary" style={{ padding: "14px 32px", fontSize: "1rem" }}>
              View Performance
            </a>
          </div>

          {/* Stats Row */}
          <div
            className="animate-fade-in-up stagger-4"
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 1,
              marginTop: 72,
              background: "var(--border-subtle)",
              borderRadius: "var(--radius-lg)",
              overflow: "hidden",
            }}
          >
            {stats.map((s) => (
              <div
                key={s.label}
                style={{
                  background: "var(--bg-secondary)",
                  padding: "28px 16px",
                  textAlign: "center",
                }}
              >
                <div className="metric-value gradient-text" style={{ fontSize: "2rem" }}>
                  {s.value}
                </div>
                <div className="metric-label" style={{ marginTop: 4, marginBottom: 2 }}>
                  {s.label}
                </div>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  {s.sub}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== TICKER BAR ===== */}
      <section
        style={{
          borderTop: "1px solid var(--border-subtle)",
          borderBottom: "1px solid var(--border-subtle)",
          padding: "12px 0",
          overflow: "hidden",
          background: "var(--bg-secondary)",
        }}
      >
        <div className="animate-ticker" style={{ display: "flex", gap: 48, whiteSpace: "nowrap", width: "max-content" }}>
          {[...topSignals, ...topSignals].map((s, i) => (
            <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 12, fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}>
              <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{s.name}</span>
              <span style={{ color: "var(--accent-emerald)" }}>CAGR {s.cagr}</span>
              <span style={{ color: "var(--text-muted)" }}>Sharpe {s.sharpe}</span>
            </span>
          ))}
        </div>
      </section>

      {/* ===== PERFORMANCE TABLE ===== */}
      <section
        id="performance"
        style={{
          padding: "120px 24px",
          maxWidth: 1200,
          margin: "0 auto",
        }}
      >
        <div style={{ textAlign: "center", marginBottom: 64 }}>
          <div className="badge badge-bull" style={{ marginBottom: 16 }}>Backtested Results</div>
          <h2
            style={{
              fontSize: "clamp(2rem, 4vw, 2.8rem)",
              fontWeight: 800,
              letterSpacing: "-0.02em",
              marginBottom: 16,
            }}
          >
            Proven Factor{" "}
            <span className="gradient-text">Performance</span>
          </h2>
          <p style={{ color: "var(--text-secondary)", maxWidth: 560, margin: "0 auto", fontSize: "1.05rem" }}>
            All figures from live backtests on Borsa Istanbul data, 2013-2026.
            Includes transaction costs, slippage, and regime-based risk management.
          </p>
        </div>

        <div className="glass-card" style={{ overflow: "hidden" }}>
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ paddingLeft: 24 }}>Factor Signal</th>
                  <th>CAGR</th>
                  <th>Sharpe</th>
                  <th>Max Drawdown</th>
                  <th>Alpha (Ann.)</th>
                  <th>Beta</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { name: "Breakout Value", cagr: "102.18%", sharpe: "2.92", dd: "-31.47%", alpha: "82.66%", beta: "0.36" },
                  { name: "Small Cap Momentum", cagr: "94.47%", sharpe: "2.46", dd: "-26.92%", alpha: "82.26%", beta: "0.29" },
                  { name: "Trend Value", cagr: "88.24%", sharpe: "2.66", dd: "-35.96%", alpha: "70.00%", beta: "0.36" },
                  { name: "Five Factor Rotation", cagr: "87.43%", sharpe: "2.45", dd: "-29.09%", alpha: "63.31%", beta: "0.39" },
                  { name: "Donchian", cagr: "81.79%", sharpe: "2.49", dd: "-26.90%", alpha: "73.20%", beta: "0.30" },
                  { name: "Value", cagr: "81.10%", sharpe: "2.52", dd: "-35.21%", alpha: "63.44%", beta: "0.36" },
                  { name: "Quality Momentum", cagr: "78.00%", sharpe: "2.35", dd: "-33.48%", alpha: "61.29%", beta: "0.35" },
                  { name: "Trend Following", cagr: "81.55%", sharpe: "2.43", dd: "-28.58%", alpha: "72.49%", beta: "0.29" },
                  { name: "Macro Hedge", cagr: "57.65%", sharpe: "2.03", dd: "-30.19%", alpha: "43.59%", beta: "0.34" },
                  { name: "XU100 (Benchmark)", cagr: "36.40%", sharpe: "1.47", dd: "-34.01%", alpha: "23.36%", beta: "0.45" },
                ].map((row, i) => (
                  <tr key={i} style={row.name.includes("XU100") ? { opacity: 0.5, fontStyle: "italic" } : {}}>
                    <td
                      style={{
                        paddingLeft: 24,
                        fontWeight: 600,
                        color: row.name.includes("XU100") ? "var(--text-muted)" : "var(--text-primary)",
                      }}
                    >
                      {i + 1 <= 9 && !row.name.includes("XU100") && (
                        <span
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            justifyContent: "center",
                            width: 22,
                            height: 22,
                            borderRadius: 6,
                            background: "var(--accent-emerald-dim)",
                            color: "var(--accent-emerald)",
                            fontSize: "0.7rem",
                            fontWeight: 700,
                            marginRight: 10,
                          }}
                        >
                          {i + 1}
                        </span>
                      )}
                      {row.name}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-emerald)" }}>
                      {row.cagr}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{row.sharpe}</td>
                    <td style={{ fontFamily: "var(--font-mono)", color: "var(--accent-rose)" }}>
                      {row.dd}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>
                      {row.alpha}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{row.beta}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ padding: "16px 24px", borderTop: "1px solid var(--border-subtle)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Showing top 9 out of 34 factor models · Updated Feb 2026
            </span>
            <Link
              href="/dashboard"
              style={{
                fontSize: "0.85rem",
                color: "var(--accent-emerald)",
                textDecoration: "none",
                display: "flex",
                alignItems: "center",
                gap: 4,
                fontWeight: 600,
              }}
            >
              View all signals <ChevronRight size={16} />
            </Link>
          </div>
        </div>
      </section>

      {/* ===== AI AGENTS ===== */}
      <section
        style={{
          padding: "120px 24px",
          background: "var(--bg-secondary)",
        }}
      >
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 64 }}>
            <div className="badge badge-recovery" style={{ marginBottom: 16 }}>Multi-Agent Architecture</div>
            <h2
              style={{
                fontSize: "clamp(2rem, 4vw, 2.8rem)",
                fontWeight: 800,
                letterSpacing: "-0.02em",
                marginBottom: 16,
              }}
            >
              AI Agents That{" "}
              <span className="gradient-text-violet">Think Together</span>
            </h2>
            <p style={{ color: "var(--text-secondary)", maxWidth: 560, margin: "0 auto", fontSize: "1.05rem" }}>
              Three specialized agents collaborate to manage your portfolio.
              They explain decisions in plain language and operate autonomously.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 24 }}>
            {agents.map((a) => (
              <div
                key={a.name}
                className="glass-card"
                style={{ padding: 32 }}
              >
                <div
                  style={{
                    width: 52,
                    height: 52,
                    borderRadius: "var(--radius-md)",
                    background: a.bg,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    marginBottom: 20,
                  }}
                >
                  <a.icon size={24} color={a.color} />
                </div>
                <h3 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: 12 }}>
                  {a.name}
                </h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.6 }}>
                  {a.desc}
                </p>

                {/* Mock conversation */}
                <div
                  style={{
                    marginTop: 20,
                    padding: 16,
                    borderRadius: "var(--radius-md)",
                    background: "rgba(0,0,0,0.2)",
                    border: "1px solid var(--border-subtle)",
                    fontSize: "0.8rem",
                  }}
                >
                  <div style={{ color: "var(--text-muted)", marginBottom: 8 }}>
                    <span style={{ color: "var(--accent-emerald)" }}>You:</span>{" "}
                    {a.name === "Portfolio Manager"
                      ? "Why is momentum underperforming this month?"
                      : a.name === "Risk Manager"
                        ? "What's our current drawdown risk?"
                        : "What's driving BIST performance this week?"}
                  </div>
                  <div style={{ color: "var(--text-secondary)" }}>
                    <span style={{ color: a.color }}>Agent:</span>{" "}
                    {a.name === "Portfolio Manager"
                      ? "Momentum factor is down 1.3% MTD due to sector rotation away from banking. Our regime filter has shifted 15% allocation to quality factors..."
                      : a.name === "Risk Manager"
                        ? "Max drawdown is -4.2% from peak. Vol-targeting is active at 0.85x leverage. Current regime is Bull, so full equity exposure is maintained..."
                        : "BIST rallied +2.1% this week driven by banking sector (+3.8%). USD/TRY stability supporting valuations. Our trend value signal has 74% correlation..."}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== FEATURES ===== */}
      <section style={{ padding: "120px 24px", maxWidth: 1200, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: 64 }}>
          <h2
            style={{
              fontSize: "clamp(2rem, 4vw, 2.8rem)",
              fontWeight: 800,
              letterSpacing: "-0.02em",
              marginBottom: 16,
            }}
          >
            Everything You{" "}
            <span className="gradient-text">Need to Compete</span>
          </h2>
          <p style={{ color: "var(--text-secondary)", maxWidth: 520, margin: "0 auto", fontSize: "1.05rem" }}>
            A complete quantitative trading platform — from data ingestion to
            portfolio execution.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
            gap: 20,
          }}
        >
          {features.map((f, i) => (
            <div
              key={i}
              className="glass-card"
              style={{ padding: 28, display: "flex", gap: 16 }}
            >
              <div
                style={{
                  minWidth: 44,
                  height: 44,
                  borderRadius: "var(--radius-md)",
                  background: "var(--accent-emerald-dim)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <f.icon size={20} color="var(--accent-emerald)" />
              </div>
              <div>
                <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: 6 }}>
                  {f.title}
                </h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", lineHeight: 1.6 }}>
                  {f.desc}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ===== HOW IT WORKS ===== */}
      <section style={{ padding: "120px 24px", background: "var(--bg-secondary)" }}>
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 64 }}>
            <h2
              style={{
                fontSize: "clamp(2rem, 4vw, 2.8rem)",
                fontWeight: 800,
                letterSpacing: "-0.02em",
                marginBottom: 16,
              }}
            >
              How It <span className="gradient-text">Works</span>
            </h2>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 0, position: "relative" }}>
            {/* Vertical line */}
            <div
              style={{
                position: "absolute",
                left: 23,
                top: 0,
                bottom: 0,
                width: 2,
                background: "linear-gradient(180deg, var(--accent-emerald), var(--accent-cyan), var(--accent-violet))",
                opacity: 0.3,
              }}
            />

            {[
              {
                step: "01",
                title: "Connect & Analyze",
                desc: "Our engine ingests market data, fundamental reports, and macro indicators from Borsa Istanbul. 13 years of price data across 500+ securities.",
              },
              {
                step: "02",
                title: "Signal Generation",
                desc: "34+ factor models independently score every stock. Value, momentum, quality, breakout, size rotation — each with unique alpha sources.",
              },
              {
                step: "03",
                title: "Regime Detection",
                desc: "XGBoost + LSTM + HMM ensemble classifies market regimes in real-time. Automatically switches to gold in Bear/Stress periods.",
              },
              {
                step: "04",
                title: "AI Agent Orchestration",
                desc: "Portfolio Manager, Risk Manager, and Market Analyst agents collaborate to construct optimal portfolios and explain every decision.",
              },
              {
                step: "05",
                title: "Execute & Monitor",
                desc: "Monthly rebalancing with volatility targeting, inverse-vol sizing, and position stop-losses. 24/7 autonomous risk monitoring.",
              },
            ].map((item) => (
              <div
                key={item.step}
                style={{
                  display: "flex",
                  gap: 24,
                  padding: "24px 0",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    minWidth: 48,
                    height: 48,
                    borderRadius: "var(--radius-md)",
                    background: "var(--bg-primary)",
                    border: "2px solid var(--accent-emerald)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontFamily: "var(--font-mono)",
                    fontWeight: 700,
                    fontSize: "0.85rem",
                    color: "var(--accent-emerald)",
                    zIndex: 1,
                  }}
                >
                  {item.step}
                </div>
                <div>
                  <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 6 }}>
                    {item.title}
                  </h3>
                  <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.6 }}>
                    {item.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== CTA ===== */}
      <section
        style={{
          padding: "120px 24px",
          textAlign: "center",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "radial-gradient(circle at 50% 50%, rgba(16,185,129,0.06) 0%, transparent 60%)",
            pointerEvents: "none",
          }}
        />
        <div style={{ position: "relative", zIndex: 1, maxWidth: 600, margin: "0 auto" }}>
          <h2
            style={{
              fontSize: "clamp(2rem, 4vw, 2.5rem)",
              fontWeight: 800,
              letterSpacing: "-0.02em",
              marginBottom: 16,
            }}
          >
            Ready to Trade Like an{" "}
            <span className="gradient-text">Institution</span>?
          </h2>
          <p
            style={{
              color: "var(--text-secondary)",
              fontSize: "1.05rem",
              marginBottom: 40,
              lineHeight: 1.7,
            }}
          >
            Join the waitlist for early access. We&rsquo;re launching with BIST
            and expanding to global emerging markets.
          </p>
          <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
            <Link href="/dashboard" className="btn-primary" style={{ padding: "14px 36px", fontSize: "1rem" }}>
              Explore Dashboard
              <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer
        style={{
          borderTop: "1px solid var(--border-subtle)",
          padding: "40px 24px",
          textAlign: "center",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            marginBottom: 12,
          }}
        >
          <TrendingUp size={18} color="var(--accent-emerald)" />
          <span style={{ fontWeight: 700, fontSize: "0.95rem" }}>
            Quant<span className="gradient-text">AI</span>
          </span>
        </div>
        <p style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
          Safa Öksüz. © 2026 Quant AI Platform. Democratizing institutional finance with AI agents.
        </p>
        <p style={{ color: "var(--text-muted)", fontSize: "0.75rem", marginTop: 8 }}>
          Past performance does not guarantee future results. Backtested results include simulated transaction costs.
        </p>
      </footer>
    </>
  );
}
