// Test Azure OpenAI deployments
// Loads configuration from environment variables (.env.local)

// Load environment variables from .env.local
require('dotenv').config({ path: '.env.local' });

const deployments = [
    'gpt-4',
    'gpt-4-turbo',
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-35-turbo',
    'gpt-3.5-turbo',
    'gpt-5.1-chat',
];

// Load from environment variables - no hardcoded values
const endpoint = process.env.AZURE_OPENAI_ENDPOINT;
const apiKey = process.env.AZURE_OPENAI_API_KEY;
const apiVersion = process.env.AZURE_OPENAI_API_VERSION || '2024-10-21';

// Validate required environment variables
if (!endpoint || !apiKey) {
    console.error('❌ Missing required environment variables.');
    console.error('Please ensure .env.local contains:');
    console.error('  - AZURE_OPENAI_ENDPOINT');
    console.error('  - AZURE_OPENAI_API_KEY');
    console.error('  - AZURE_OPENAI_API_VERSION (optional)');
    process.exit(1);
}

async function testDeployment(deployment) {
    const url = `${endpoint.replace(/\/$/, '')}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'api-key': apiKey,
            },
            body: JSON.stringify({
                messages: [{ role: 'user', content: 'test' }],
                max_tokens: 5,
            }),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (response.ok) {
            console.log(`✅ ${deployment}: SUCCESS`);
            return deployment;
        } else {
            const error = await response.text();
            console.log(`❌ ${deployment}: ${response.status} - ${error.substring(0, 100)}`);
            return null;
        }
    } catch (error) {
        console.log(`⚠️  ${deployment}: ${error.message}`);
        return null;
    }
}

async function main() {
    console.log(`Testing deployments at: ${endpoint}\n`);

    for (const deployment of deployments) {
        const result = await testDeployment(deployment);
        if (result) {
            console.log(`\n✨ Found working deployment: ${result}`);
            console.log(`\nUpdate your .env.local with:`);
            console.log(`AZURE_OPENAI_DEPLOYMENT=${result}`);
            break;
        }
    }
}

main().catch(console.error);
