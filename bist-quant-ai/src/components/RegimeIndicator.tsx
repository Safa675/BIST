"use client";

interface Props {
    regime: string;
}

const REGIME_CONFIG: Record<string, { color: string; bg: string; icon: string; desc: string }> = {
    Bull: {
        color: "var(--accent-emerald)",
        bg: "rgba(16,185,129,0.12)",
        icon: "🐂",
        desc: "Full equity exposure",
    },
    Bear: {
        color: "var(--accent-rose)",
        bg: "rgba(244,63,94,0.12)",
        icon: "🐻",
        desc: "Switched to gold",
    },
    Recovery: {
        color: "var(--accent-amber)",
        bg: "rgba(245,158,11,0.12)",
        icon: "🔄",
        desc: "Cautious exposure",
    },
    Stress: {
        color: "var(--accent-violet)",
        bg: "rgba(139,92,246,0.12)",
        icon: "⚡",
        desc: "Risk-off mode",
    },
};

export default function RegimeIndicator({ regime }: Props) {
    const config = REGIME_CONFIG[regime] || REGIME_CONFIG.Bull;

    return (
        <div
            style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 14px",
                borderRadius: 100,
                background: config.bg,
                border: `1px solid ${config.color}33`,
                fontSize: "0.8rem",
                fontWeight: 600,
                color: config.color,
            }}
        >
            <span style={{ fontSize: "1rem" }}>{config.icon}</span>
            <span>
                Regime: {regime}
            </span>
            <span style={{ fontSize: "0.7rem", opacity: 0.7 }}>· {config.desc}</span>
        </div>
    );
}
