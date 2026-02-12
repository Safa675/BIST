import type { NextConfig } from "next";

const requiredAzureEnv = [
  "AZURE_OPENAI_ENDPOINT",
  "AZURE_OPENAI_API_KEY",
  "AZURE_OPENAI_DEPLOYMENT",
] as const;

const missingAzureEnv = requiredAzureEnv.filter((name) => !(process.env[name] || "").trim());
if (missingAzureEnv.length > 0) {
  throw new Error(
    `[Startup Check] Missing required Azure OpenAI env var(s): ${missingAzureEnv.join(", ")}. ` +
    `Set them before running Next.js to enable agent APIs.`
  );
}

console.info(
  `[Startup Check] Azure OpenAI configured for deployment "${process.env.AZURE_OPENAI_DEPLOYMENT}".`
);

const nextConfig: NextConfig = {
  /* config options here */
};

export default nextConfig;
