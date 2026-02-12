"use client";

import { useState } from "react";
import { ArrowUpDown, ArrowDown, ArrowUp } from "lucide-react";

interface Signal {
    name: string;
    enabled: boolean;
    cagr: number;
    sharpe: number;
    beta?: number | null;
    max_dd: number;
    ytd: number;
    last_rebalance: string;
}

interface Props {
    signals: Signal[];
    selectedSignal: string;
    onSelectSignal: (name: string) => void;
}

type SortKey = "name" | "cagr" | "sharpe" | "beta" | "max_dd" | "ytd";
type SortDir = "asc" | "desc";

export default function SignalTable({ signals, selectedSignal, onSelectSignal }: Props) {
    const [sortKey, setSortKey] = useState<SortKey>("cagr");
    const [sortDir, setSortDir] = useState<SortDir>("desc");
    const [search, setSearch] = useState("");

    const handleSort = (key: SortKey) => {
        if (sortKey === key) {
            setSortDir(sortDir === "asc" ? "desc" : "asc");
        } else {
            setSortKey(key);
            setSortDir("desc");
        }
    };

    const sorted = [...signals]
        .filter((s) => s.name.toLowerCase().includes(search.toLowerCase()))
        .sort((a, b) => {
            const aVal = a[sortKey];
            const bVal = b[sortKey];
            if (typeof aVal === "string" && typeof bVal === "string") {
                return sortDir === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }
            const toSortableNumber = (value: unknown) => {
                if (typeof value === "number" && Number.isFinite(value)) return value;
                return sortDir === "asc" ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
            };
            const aNum = toSortableNumber(aVal);
            const bNum = toSortableNumber(bVal);
            return sortDir === "asc" ? aNum - bNum : bNum - aNum;
        });

    const renderSortIcon = (col: SortKey) => {
        if (sortKey !== col) return <ArrowUpDown size={12} style={{ opacity: 0.3 }} />;
        return sortDir === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />;
    };

    const formatName = (n: string) =>
        n.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

    return (
        <div className="glass-card" style={{ overflow: "hidden" }}>
            {/* Search bar */}
            <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", gap: 12 }}>
                <input
                    type="text"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Search signals..."
                    style={{
                        flex: 1,
                        padding: "8px 14px",
                        borderRadius: "var(--radius-sm)",
                        border: "1px solid var(--border-subtle)",
                        background: "rgba(0,0,0,0.3)",
                        color: "var(--text-primary)",
                        fontSize: "0.85rem",
                        outline: "none",
                    }}
                />
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", whiteSpace: "nowrap" }}>
                    {sorted.length} signals
                </span>
            </div>

            <div style={{ overflowX: "auto", maxHeight: 580, overflowY: "auto" }}>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th style={{ paddingLeft: 20, width: 32 }}>#</th>
                            <th onClick={() => handleSort("name")} style={{ cursor: "pointer" }}>
                                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                    Signal {renderSortIcon("name")}
                                </span>
                            </th>
                            <th onClick={() => handleSort("cagr")} style={{ cursor: "pointer" }}>
                                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                    CAGR {renderSortIcon("cagr")}
                                </span>
                            </th>
                            <th onClick={() => handleSort("sharpe")} style={{ cursor: "pointer" }}>
                                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                    Sharpe {renderSortIcon("sharpe")}
                                </span>
                            </th>
                            <th onClick={() => handleSort("beta")} style={{ cursor: "pointer" }}>
                                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                    Beta {renderSortIcon("beta")}
                                </span>
                            </th>
                            <th onClick={() => handleSort("max_dd")} style={{ cursor: "pointer" }}>
                                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                    Max DD {renderSortIcon("max_dd")}
                                </span>
                            </th>
                            <th onClick={() => handleSort("ytd")} style={{ cursor: "pointer" }}>
                                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                    YTD {renderSortIcon("ytd")}
                                </span>
                            </th>
                            <th>Last Rebal.</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sorted.map((signal, i) => {
                            const isSelected = signal.name === selectedSignal;
                            const isXu100 = signal.name === "xu100";
                            return (
                                <tr
                                    key={signal.name}
                                    onClick={() => onSelectSignal(signal.name)}
                                    style={{
                                        cursor: "pointer",
                                        background: isSelected ? "rgba(16,185,129,0.06)" : undefined,
                                        borderLeft: isSelected ? "3px solid var(--accent-emerald)" : "3px solid transparent",
                                        opacity: isXu100 ? 0.5 : 1,
                                    }}
                                >
                                    <td style={{ paddingLeft: 20, fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                        {i + 1}
                                    </td>
                                    <td style={{ fontWeight: 600, color: isSelected ? "var(--accent-emerald)" : "var(--text-primary)" }}>
                                        {formatName(signal.name)}
                                    </td>
                                    <td
                                        style={{
                                            fontFamily: "var(--font-mono)",
                                            fontWeight: 700,
                                            color: signal.cagr > 60 ? "var(--accent-emerald)" : signal.cagr > 40 ? "var(--accent-cyan)" : "var(--text-secondary)",
                                        }}
                                    >
                                        {signal.cagr.toFixed(2)}%
                                    </td>
                                    <td style={{ fontFamily: "var(--font-mono)", color: signal.sharpe >= 2 ? "var(--accent-emerald)" : "var(--text-secondary)" }}>
                                        {signal.sharpe.toFixed(2)}
                                    </td>
                                    <td style={{ fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>
                                        {typeof signal.beta === "number" ? signal.beta.toFixed(2) : "—"}
                                    </td>
                                    <td style={{ fontFamily: "var(--font-mono)", color: "var(--accent-rose)" }}>
                                        {signal.max_dd.toFixed(2)}%
                                    </td>
                                    <td
                                        style={{
                                            fontFamily: "var(--font-mono)",
                                            fontWeight: 600,
                                            color: signal.ytd > 0 ? "var(--accent-emerald)" : "var(--accent-rose)",
                                        }}
                                    >
                                        {signal.ytd > 0 ? "+" : ""}
                                        {signal.ytd.toFixed(2)}%
                                    </td>
                                    <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: "var(--text-muted)" }}>
                                        {signal.last_rebalance}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
