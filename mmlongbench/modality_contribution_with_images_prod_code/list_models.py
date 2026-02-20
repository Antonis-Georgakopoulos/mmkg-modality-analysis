#!/usr/bin/env python
"""
Simple script to list available models from the UVA API.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

# BASE_URL = "https://ai-research-proxy.azurewebsites.net"
BASE_URL = "https://llmproxy.uva.nl"
API_KEY = os.getenv("UVA_API_KEY")

if not API_KEY:
    print("ERROR: UVA_API_KEY not found in environment variables")
    exit(1)

print(f"Base URL: {BASE_URL}")
print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
print()

response = requests.get(
    f"{BASE_URL}/v1/models",
    headers={"Authorization": f"Bearer {API_KEY}"}
)

print(f"Status: {response.status_code}")
print()

if response.ok:
    data = response.json()
    print("Available Models:")
    print("-" * 40)
    for model in data.get("data", []):
        print(f"  - {model.get('id', 'unknown')}")
else:
    print(f"Error: {response.text}")
