#!/usr/bin/env python3
"""
Test script to verify cum_sigma split chart is receiving all data
"""

import requests
import json
import time

def test_cum_sigma_split():
    """Test the cum_sigma split chart"""
    base_url = 'http://localhost:5058'
    
    # First, let's load a session or create one
    print("Getting sessions...")
    resp = requests.get(f'{base_url}/sessions')
    print(f"Status: {resp.status_code}")
    print(f"Sessions: {resp.text[:200]}")
    
    # Try loading first session
    resp = requests.get(f'{base_url}/api/sessions')
    if resp.status_code == 200:
        sessions = resp.json()
        print(f"\nAvailable sessions: {list(sessions.keys())[:5]}")
        
        if sessions:
            first_key = list(sessions.keys())[0]
            print(f"\nLoading session: {first_key}")
            resp = requests.post(f'{base_url}/api/load-session', json={'name': first_key})
            print(f"Load status: {resp.status_code}")
            print(f"Response: {resp.text[:300]}")
    
    print("\nNote: Open http://localhost:5058 in browser, open DevTools (F12),")
    print("set chart to cum_sigma, enable split charts, and watch console output.")
    print("Look for lines like: '[_buildTileChart cum_sigma] tileData.length='")

if __name__ == '__main__':
    test_cum_sigma_split()
