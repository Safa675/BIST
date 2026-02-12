// Quick test for Azure OpenAI deployment
// Loads configuration from environment variables (.env.local)

// Load environment variables from .env.local
require('dotenv').config({ path: '.env.local' });

const deployments = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4',
    'gpt-4-turbo',
    'gpt-35-turbo',
    'gpt-4-32k',
    'gpt-35-turbo-16k',
    'gpt-5.1-chat',
    'my-gpt4',
    'my-gpt-4o',
    'deployment',
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
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'api-key': apiKey,
            },
            body: JSON.stringify({
                messages: [{ role: 'user', content: 'hi' }],
                max_tokens: 5,
            }),
            signal: AbortSignal.timeout(5000),
        });

        if (response.ok) {
            const data = await response.json();
            console.log(`✅ SUCCESS: ${deployment}`);
            console.log(`Response: ${JSON.stringify(data, null, 2)}`);
            return deployment;
        } else {
            const error = await response.text();
            let errorMsg = response.status;
            try {
                const errorObj = JSON.parse(error);
                errorMsg = errorObj.error?.code || errorObj.error?.message || response.status;
            } catch {}
            console.log(`❌ ${deployment}: ${errorMsg}`);
            return null;
        }
    } catch (error) {
        if (error.name === 'TimeoutError') {
            console.log(`⏱️  ${deployment}: Timeout (network issue?)`);
        } else {
            console.log(`⚠️  ${deployment}: ${error.message}`);
        }
        return null;
    }
}

async function main() {
    console.log(`Testing deployments at: ${endpoint}\n`);

    for (const deployment of deployments) {
        const result = await testDeployment(deployment);
        if (result) {
            console.log(`\n✨ FOUND WORKING DEPLOYMENT: ${result}`);
            console.log(`\nUpdate .env.local with: AZURE_OPENAI_DEPLOYMENT=${result}`);
            process.exit(0);
        }
        await new Promise(resolve => setTimeout(resolve, 500)); // Small delay between tests
    }

    console.log('\n❌ No working deployments found.');
    console.log('Please check your Azure Portal and create a deployment.');
}

main().catch(console.error);
