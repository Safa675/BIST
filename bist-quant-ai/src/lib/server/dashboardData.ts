import { execFile } from "child_process";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

const PROJECT_ROOT = process.cwd();
const GENERATOR_PATH = join(PROJECT_ROOT, "dashboard", "generate_dashboard_data.py");
const SNAPSHOT_PATH = join(PROJECT_ROOT, "public", "data", "dashboard_data.json");
const PYTHON_CANDIDATES = ["python3", "python"] as const;
const REFRESH_THROTTLE_MS = 10_000;

let lastRefreshAtMs = 0;

function readSnapshot(): Record<string, unknown> {
    if (!existsSync(SNAPSHOT_PATH)) {
        throw new Error(`Dashboard snapshot not found: ${SNAPSHOT_PATH}`);
    }

    const raw = readFileSync(SNAPSHOT_PATH, "utf-8");
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Dashboard snapshot root must be a JSON object.");
    }
    return parsed as Record<string, unknown>;
}

async function runGenerator(): Promise<void> {
    if (!existsSync(GENERATOR_PATH)) {
        throw new Error(`Dashboard generator not found: ${GENERATOR_PATH}`);
    }

    let lastError: unknown = null;
    for (const pythonCmd of PYTHON_CANDIDATES) {
        try {
            await execFileAsync(pythonCmd, [GENERATOR_PATH], {
                cwd: PROJECT_ROOT,
                timeout: 120_000,
                maxBuffer: 10 * 1024 * 1024,
            });
            return;
        } catch (error) {
            lastError = error;
        }
    }

    const detail = lastError instanceof Error ? lastError.message : String(lastError);
    throw new Error(`Failed to run dashboard generator with python3/python: ${detail}`);
}

export async function loadDashboardData(options?: {
    refresh?: boolean;
    force?: boolean;
}): Promise<{
    data: Record<string, unknown>;
    refreshed: boolean;
    refreshError?: string;
}> {
    const wantsRefresh = options?.refresh === true;
    const forceRefresh = options?.force === true;
    let refreshed = false;
    let refreshError: string | undefined;

    if (wantsRefresh) {
        const now = Date.now();
        const canRefresh = forceRefresh || now - lastRefreshAtMs >= REFRESH_THROTTLE_MS;
        if (canRefresh) {
            try {
                await runGenerator();
                refreshed = true;
                lastRefreshAtMs = now;
            } catch (error) {
                refreshError = error instanceof Error ? error.message : String(error);
            }
        }
    }

    return {
        data: readSnapshot(),
        refreshed,
        refreshError,
    };
}
