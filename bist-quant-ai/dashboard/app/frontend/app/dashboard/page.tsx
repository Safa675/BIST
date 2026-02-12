'use client'

import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  BarChart3,
  CalendarDays,
  ChevronDown,
  Layers,
  RefreshCw,
  TrendingUp,
} from 'lucide-react'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

interface Signal {
  name: string
  enabled: boolean
  cagr: number
  sharpe: number
  max_dd: number
  ytd: number
  excess_vs_xu030?: number | null
  excess_vs_xu100?: number | null
  last_rebalance: string
}

interface ManualHolding {
  ticker: string
  weight?: number | null
  quantity?: number | null
  avg_cost?: number | null
  last_price?: number | null
  market_value?: number | null
}

interface ManualHoldingsSummary {
  total_market_value?: number | null
  total_cost_value?: number | null
  market_value_date?: string | null
  currency?: string
  positions?: number
}

interface ManualHoldings {
  rows?: ManualHolding[]
  summary?: ManualHoldingsSummary
  notes?: string[]
}

interface ManualTrade {
  datetime?: string
  ticker?: string
  side?: string
  quantity?: number | null
  price?: number | null
  gross_amount?: number | null
}

interface ManualTrades {
  rows?: ManualTrade[]
  notes?: string[]
}

interface CustomDailyRow {
  date?: string
  portfolio_return?: number | null
  xu030_return?: number | null
  xu100_return?: number | null
  portfolio_cumulative?: number | null
  xu030_cumulative?: number | null
  xu100_cumulative?: number | null
}

interface CustomPortfolioSummary {
  obs_days?: number
  portfolio_cumulative?: number | null
  xu030_cumulative?: number | null
  xu100_cumulative?: number | null
  outperform_rate_vs_xu100?: number | null
}

interface CustomPortfolio {
  start_date?: string
  end_date?: string
  summary?: CustomPortfolioSummary
  daily?: CustomDailyRow[]
  notes?: string[]
}

interface DashboardData {
  last_update: string
  current_regime: string
  regime_distribution: Record<string, number>
  xu100_ytd: number
  trading_days: number
  active_signals: number
  signals: Signal[]
  holdings: Record<string, string[]>
  holdings_meta?: Record<string, { total_count?: number }>
  manual_holdings?: ManualHoldings
  manual_trades?: ManualTrades
  custom_portfolio?: CustomPortfolio
  payload_notes?: string[]
}

function asNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function valueClass(value: number | null | undefined) {
  const n = asNumber(value)
  if (n === null) return 'text-dark-400'
  return n >= 0 ? 'text-success' : 'text-danger'
}

function formatPercent(value: number | null | undefined, digits = 2) {
  const n = asNumber(value)
  if (n === null) return '--'
  return `${n >= 0 ? '+' : ''}${n.toLocaleString('tr-TR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}%`
}

function formatFractionPercent(value: number | null | undefined, digits = 2) {
  const n = asNumber(value)
  if (n === null) return '--'
  const pct = n * 100
  return `${pct >= 0 ? '+' : ''}${pct.toLocaleString('tr-TR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}%`
}

function formatCurrency(value: number | null | undefined, currency = 'TRY', digits = 2) {
  const n = asNumber(value)
  if (n === null) return '--'
  return `${n.toLocaleString('tr-TR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })} ${currency}`
}

function formatInteger(value: number | null | undefined) {
  const n = asNumber(value)
  if (n === null) return '--'
  return Math.round(n).toLocaleString('tr-TR')
}

function formatName(name: string) {
  return name
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function formatIsoDate(value: string | undefined | null) {
  if (!value) return '--'
  return value
}

function regimeClass(regime: string) {
  const normalized = regime.toLowerCase()
  if (normalized.includes('bull')) return 'bg-success/20 text-success border-success/30'
  if (normalized.includes('bear') || normalized.includes('stress')) return 'bg-danger/20 text-danger border-danger/30'
  if (normalized.includes('recovery')) return 'bg-primary-500/20 text-primary-300 border-primary-500/30'
  return 'bg-warning/20 text-warning border-warning/30'
}

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedSignal, setExpandedSignal] = useState<string | null>(null)

  const fetchData = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/dashboard`, { cache: 'no-store' })
      if (!response.ok) {
        throw new Error(`Dashboard request failed: ${response.status}`)
      }
      const payload = (await response.json()) as DashboardData
      setData(payload)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown dashboard error')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const signals = useMemo(() => {
    if (!data?.signals) return []
    return [...data.signals].sort((a, b) => b.cagr - a.cagr)
  }, [data])

  const manualHoldings = data?.manual_holdings?.rows ?? []
  const manualSummary = data?.manual_holdings?.summary ?? {}
  const manualTrades = data?.manual_trades?.rows ?? []
  const customPortfolio = data?.custom_portfolio
  const customSummary = customPortfolio?.summary ?? {}
  const dailyRows = [...(customPortfolio?.daily ?? [])].reverse()

  const summaryNotes = [
    ...(manualSummary.market_value_date ? [`Portfolio market-value date: ${manualSummary.market_value_date}`] : []),
    ...(customPortfolio?.notes ?? []),
    ...(data?.manual_holdings?.notes ?? []),
    ...(data?.manual_trades?.notes ?? []),
    ...(data?.payload_notes ?? []),
  ]

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-dark-800 bg-dark-950/90 backdrop-blur-lg">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-xl font-bold">BIST Daily Portfolio Dashboard</h1>
            <p className="text-sm text-dark-400">
              Last update: {data?.last_update || '--'}
            </p>
          </div>

          <button
            type="button"
            onClick={() => {
              setRefreshing(true)
              fetchData()
            }}
            className={`rounded-xl border border-dark-700 bg-dark-900 px-4 py-2 text-sm font-medium transition hover:border-primary-500/60 hover:text-primary-300 ${
              refreshing ? 'pointer-events-none' : ''
            }`}
          >
            <span className="inline-flex items-center gap-2">
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </span>
          </button>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl px-6 py-8">
        {error && (
          <div className="mb-6 rounded-2xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
            {error}
          </div>
        )}

        {loading && !data ? (
          <div className="rounded-2xl border border-dark-800 bg-dark-900/60 p-6 text-dark-300">Loading dashboard...</div>
        ) : (
          <>
            <section className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
              <TopCard
                icon={<TrendingUp className="h-4 w-4" />}
                label="XU100 YTD"
                value={formatPercent(data?.xu100_ytd)}
                valueClass={valueClass(data?.xu100_ytd)}
              />
              <TopCard
                icon={<Activity className="h-4 w-4" />}
                label="Current Regime"
                value={data?.current_regime || '--'}
                valueClass="text-white"
              />
              <TopCard
                icon={<Layers className="h-4 w-4" />}
                label="Active Signals"
                value={formatInteger(data?.active_signals)}
                valueClass="text-white"
              />
              <TopCard
                icon={<CalendarDays className="h-4 w-4" />}
                label="Trading Days"
                value={formatInteger(data?.trading_days)}
                valueClass="text-white"
              />
            </section>

            <motion.section
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="card mb-6"
            >
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold">Portfolio Summary</h2>
                  <p className="text-sm text-dark-400">
                    Score start date: {formatIsoDate(customPortfolio?.start_date || '2026-02-06')}
                  </p>
                </div>
                <span className={`rounded-xl border px-3 py-1 text-sm font-semibold ${regimeClass(data?.current_regime || '')}`}>
                  {data?.current_regime || 'Unknown'}
                </span>
              </div>

              <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
                <SummaryCard
                  label="Total Portfolio Value"
                  value={formatCurrency(
                    asNumber(manualSummary.total_market_value) ?? asNumber(manualSummary.total_cost_value),
                    manualSummary.currency || 'TRY'
                  )}
                />
                <SummaryCard
                  label="Portfolio Cumulative"
                  value={formatFractionPercent(customSummary.portfolio_cumulative)}
                  valueClass={valueClass(customSummary.portfolio_cumulative)}
                />
                <SummaryCard
                  label="XU030 Cumulative"
                  value={formatFractionPercent(customSummary.xu030_cumulative)}
                  valueClass={valueClass(customSummary.xu030_cumulative)}
                />
                <SummaryCard
                  label="XU100 Cumulative"
                  value={formatFractionPercent(customSummary.xu100_cumulative)}
                  valueClass={valueClass(customSummary.xu100_cumulative)}
                />
                <SummaryCard
                  label="Observed Days"
                  value={formatInteger(customSummary.obs_days)}
                />
                <SummaryCard
                  label="Outperform Rate vs XU100"
                  value={formatFractionPercent(customSummary.outperform_rate_vs_xu100)}
                  valueClass={valueClass(customSummary.outperform_rate_vs_xu100)}
                />
              </div>

              <div className="mt-5 space-y-1 text-sm text-dark-400">
                {summaryNotes.length === 0 ? (
                  <p>No portfolio warnings.</p>
                ) : (
                  summaryNotes.map((note, idx) => (
                    <p key={`${note}-${idx}`}>- {note}</p>
                  ))
                )}
              </div>
            </motion.section>

            <section className="mb-6 grid gap-6 xl:grid-cols-2">
              <DataCard title="Manual Holdings" subtitle={`${manualHoldings.length} rows`}>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[680px] text-sm">
                    <thead>
                      <tr className="border-b border-dark-800 text-left text-xs uppercase tracking-wide text-dark-400">
                        <th className="pb-3">Ticker</th>
                        <th className="pb-3">Weight</th>
                        <th className="pb-3">Qty</th>
                        <th className="pb-3">Avg Cost</th>
                        <th className="pb-3">Last Price</th>
                        <th className="pb-3">Market Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {manualHoldings.length === 0 ? (
                        <tr>
                          <td className="py-4 text-dark-400" colSpan={6}>
                            No manual holdings available.
                          </td>
                        </tr>
                      ) : (
                        manualHoldings.map((row) => (
                          <tr key={row.ticker} className="border-b border-dark-900 text-dark-100">
                            <td className="py-3 font-mono font-semibold">{row.ticker}</td>
                            <td className={`py-3 font-mono ${valueClass(row.weight)}`}>
                              {formatFractionPercent(row.weight)}
                            </td>
                            <td className="py-3 font-mono">{formatInteger(row.quantity)}</td>
                            <td className="py-3 font-mono">{formatCurrency(row.avg_cost, manualSummary.currency || 'TRY')}</td>
                            <td className="py-3 font-mono">{formatCurrency(row.last_price, manualSummary.currency || 'TRY')}</td>
                            <td className="py-3 font-mono">{formatCurrency(row.market_value, manualSummary.currency || 'TRY')}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </DataCard>

              <DataCard title="Manual Trades" subtitle={`${manualTrades.length} rows`}>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[620px] text-sm">
                    <thead>
                      <tr className="border-b border-dark-800 text-left text-xs uppercase tracking-wide text-dark-400">
                        <th className="pb-3">Datetime</th>
                        <th className="pb-3">Ticker</th>
                        <th className="pb-3">Side</th>
                        <th className="pb-3">Qty</th>
                        <th className="pb-3">Price</th>
                        <th className="pb-3">Gross</th>
                      </tr>
                    </thead>
                    <tbody>
                      {manualTrades.length === 0 ? (
                        <tr>
                          <td className="py-4 text-dark-400" colSpan={6}>
                            No manual trades available.
                          </td>
                        </tr>
                      ) : (
                        manualTrades.slice(0, 120).map((row, idx) => {
                          const side = (row.side || '').toUpperCase()
                          const sideColor = side === 'BUY' ? 'text-success' : side === 'SELL' ? 'text-danger' : 'text-dark-400'

                          return (
                            <tr key={`${row.datetime}-${row.ticker}-${idx}`} className="border-b border-dark-900">
                              <td className="py-3 font-mono">{row.datetime || '--'}</td>
                              <td className="py-3 font-mono font-semibold">{row.ticker || '--'}</td>
                              <td className={`py-3 font-semibold ${sideColor}`}>{side || '--'}</td>
                              <td className="py-3 font-mono">{formatInteger(row.quantity)}</td>
                              <td className="py-3 font-mono">{formatCurrency(row.price, manualSummary.currency || 'TRY')}</td>
                              <td className="py-3 font-mono">{formatCurrency(row.gross_amount, manualSummary.currency || 'TRY')}</td>
                            </tr>
                          )
                        })
                      )}
                    </tbody>
                  </table>
                </div>
              </DataCard>
            </section>

            <section className="mb-6">
              <DataCard
                title="Daily Performance"
                subtitle={`${dailyRows.length} rows | ${formatIsoDate(customPortfolio?.start_date)} to ${formatIsoDate(customPortfolio?.end_date)}`}
              >
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[880px] text-sm">
                    <thead>
                      <tr className="border-b border-dark-800 text-left text-xs uppercase tracking-wide text-dark-400">
                        <th className="pb-3">Date</th>
                        <th className="pb-3">Portfolio</th>
                        <th className="pb-3">XU030</th>
                        <th className="pb-3">XU100</th>
                        <th className="pb-3">Port Cum</th>
                        <th className="pb-3">XU030 Cum</th>
                        <th className="pb-3">XU100 Cum</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dailyRows.length === 0 ? (
                        <tr>
                          <td className="py-4 text-dark-400" colSpan={7}>
                            No daily portfolio rows available.
                          </td>
                        </tr>
                      ) : (
                        dailyRows.slice(0, 240).map((row, idx) => (
                          <tr key={`${row.date}-${idx}`} className="border-b border-dark-900">
                            <td className="py-3 font-mono">{row.date || '--'}</td>
                            <td className={`py-3 font-mono ${valueClass(row.portfolio_return)}`}>
                              {formatFractionPercent(row.portfolio_return)}
                            </td>
                            <td className={`py-3 font-mono ${valueClass(row.xu030_return)}`}>
                              {formatFractionPercent(row.xu030_return)}
                            </td>
                            <td className={`py-3 font-mono ${valueClass(row.xu100_return)}`}>
                              {formatFractionPercent(row.xu100_return)}
                            </td>
                            <td className={`py-3 font-mono ${valueClass(row.portfolio_cumulative)}`}>
                              {formatFractionPercent(row.portfolio_cumulative)}
                            </td>
                            <td className={`py-3 font-mono ${valueClass(row.xu030_cumulative)}`}>
                              {formatFractionPercent(row.xu030_cumulative)}
                            </td>
                            <td className={`py-3 font-mono ${valueClass(row.xu100_cumulative)}`}>
                              {formatFractionPercent(row.xu100_cumulative)}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </DataCard>
            </section>

            <section className="mb-4">
              <DataCard title="Signal Performance Metrics" subtitle={`${signals.length} strategies`}>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[1180px] text-sm">
                    <thead>
                      <tr className="border-b border-dark-800 text-left text-xs uppercase tracking-wide text-dark-400">
                        <th className="pb-3">Signal</th>
                        <th className="pb-3">Status</th>
                        <th className="pb-3">CAGR</th>
                        <th className="pb-3">Sharpe</th>
                        <th className="pb-3">Max DD</th>
                        <th className="pb-3">YTD</th>
                        <th className="pb-3">Excess vs XU030</th>
                        <th className="pb-3">Excess vs XU100</th>
                        <th className="pb-3">Assets</th>
                      </tr>
                    </thead>
                    <tbody>
                      {signals.length === 0 ? (
                        <tr>
                          <td className="py-4 text-dark-400" colSpan={9}>
                            No signals available.
                          </td>
                        </tr>
                      ) : (
                        signals.flatMap((signal) => {
                          const holdings = data?.holdings?.[signal.name] ?? []
                          const totalCount = data?.holdings_meta?.[signal.name]?.total_count ?? holdings.length
                          const isExpanded = expandedSignal === signal.name

                          const rows = [
                            <tr key={`${signal.name}-row`} className="border-b border-dark-900">
                                <td className="py-3 font-semibold">{formatName(signal.name)}</td>
                                <td className="py-3">
                                  <span
                                    className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                                      signal.enabled ? 'bg-success/20 text-success' : 'bg-dark-700 text-dark-300'
                                    }`}
                                  >
                                    {signal.enabled ? 'Active' : 'Off'}
                                  </span>
                                </td>
                                <td className={`py-3 font-mono ${valueClass(signal.cagr)}`}>{formatPercent(signal.cagr)}</td>
                                <td className="py-3 font-mono">{asNumber(signal.sharpe)?.toFixed(2) ?? '--'}</td>
                                <td className="py-3 font-mono text-danger">{formatPercent(signal.max_dd)}</td>
                                <td className={`py-3 font-mono ${valueClass(signal.ytd)}`}>{formatPercent(signal.ytd)}</td>
                                <td className={`py-3 font-mono ${valueClass(signal.excess_vs_xu030)}`}>
                                  {formatPercent(signal.excess_vs_xu030)}
                                </td>
                                <td className={`py-3 font-mono ${valueClass(signal.excess_vs_xu100)}`}>
                                  {formatPercent(signal.excess_vs_xu100)}
                                </td>
                                <td className="py-3">
                                  {totalCount > 0 ? (
                                    <button
                                      type="button"
                                      onClick={() => setExpandedSignal(isExpanded ? null : signal.name)}
                                      className="inline-flex items-center gap-1 rounded-lg border border-dark-700 px-2.5 py-1.5 text-xs text-dark-200 transition hover:border-primary-500/60 hover:text-primary-300"
                                    >
                                      {totalCount} assets
                                      <ChevronDown className={`h-3.5 w-3.5 transition ${isExpanded ? 'rotate-180' : ''}`} />
                                    </button>
                                  ) : (
                                    <span className="text-dark-500">--</span>
                                  )}
                                </td>
                              </tr>,
                          ]

                          if (isExpanded) {
                            rows.push(
                              <tr key={`${signal.name}-assets`} className="border-b border-dark-900 bg-dark-950/50">
                                <td colSpan={9} className="py-3">
                                  {holdings.length === 0 ? (
                                    <span className="text-sm text-dark-400">No assets available.</span>
                                  ) : (
                                    <div className="flex flex-wrap gap-2">
                                      {holdings.map((ticker) => (
                                        <span key={`${signal.name}-${ticker}`} className="rounded-lg border border-dark-700 bg-dark-900 px-2 py-1 font-mono text-xs">
                                          {ticker}
                                        </span>
                                      ))}
                                    </div>
                                  )}
                                </td>
                              </tr>
                            )
                          }

                          return rows
                        })
                      )}
                    </tbody>
                  </table>
                </div>
              </DataCard>
            </section>
          </>
        )}

        <footer className="pt-4 text-center text-xs text-dark-500">
          <p>Daily portfolio monitoring dashboard. Not investment advice.</p>
        </footer>
      </main>
    </div>
  )
}

function TopCard({
  icon,
  label,
  value,
  valueClass,
}: {
  icon: ReactNode
  label: string
  value: string
  valueClass?: string
}) {
  return (
    <div className="card">
      <div className="mb-2 flex items-center gap-2 text-sm text-dark-400">
        {icon}
        <span>{label}</span>
      </div>
      <div className={`text-xl font-bold ${valueClass || 'text-white'}`}>{value}</div>
    </div>
  )
}

function SummaryCard({
  label,
  value,
  valueClass,
}: {
  label: string
  value: string
  valueClass?: string
}) {
  return (
    <div className="rounded-xl border border-dark-800 bg-dark-950/60 px-4 py-3">
      <p className="text-xs uppercase tracking-wide text-dark-400">{label}</p>
      <p className={`mt-2 text-lg font-semibold ${valueClass || 'text-white'}`}>{value}</p>
    </div>
  )
}

function DataCard({
  title,
  subtitle,
  children,
}: {
  title: string
  subtitle?: string
  children: ReactNode
}) {
  return (
    <div className="card">
      <div className="mb-4 flex items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 text-base font-semibold">
          <BarChart3 className="h-4 w-4 text-primary-300" />
          {title}
        </h3>
        {subtitle ? <p className="text-xs text-dark-400">{subtitle}</p> : null}
      </div>
      {children}
    </div>
  )
}
