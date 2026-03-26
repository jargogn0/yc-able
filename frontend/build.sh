#!/bin/bash
# Build script for Netlify: injects API_BASE_URL into the HTML
set -e

API_URL="${API_BASE_URL:-https://19labs-api.up.railway.app}"

# Copy the main HTML
cp ../19labs-app.html index.html

# Inject the api-base meta tag after the charset meta
sed -i.bak 's|<meta charset="UTF-8">|<meta charset="UTF-8">\n<meta name="api-base" content="'"$API_URL"'">|' index.html
rm -f index.html.bak

echo "Built frontend/index.html with API_BASE_URL=$API_URL"
