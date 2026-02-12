#!/bin/bash

# Common deployment names to test
DEPLOYMENTS=(
    "gpt-4"
    "gpt-4-turbo"
    "gpt-4o"
    "gpt-4o-mini"
    "gpt-35-turbo"
    "gpt-3.5-turbo"
    "gpt4"
    "gpt4o"
)

ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
API_KEY="${AZURE_OPENAI_API_KEY}"
API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-10-21}"

echo "Testing Azure OpenAI deployments at: $ENDPOINT"
echo "================================================"

for deployment in "${DEPLOYMENTS[@]}"; do
    echo -n "Testing deployment: $deployment ... "
    
    response=$(curl -s -w "\n%{http_code}" \
        "${ENDPOINT}openai/deployments/${deployment}/chat/completions?api-version=${API_VERSION}" \
        -H "Content-Type: application/json" \
        -H "api-key: ${API_KEY}" \
        -d '{
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
        }')
    
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" = "200" ]; then
        echo "✅ SUCCESS - This deployment exists!"
    elif [ "$http_code" = "404" ]; then
        echo "❌ Not found"
    else
        echo "⚠️  HTTP $http_code"
    fi
done
