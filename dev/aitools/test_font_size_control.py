#!/usr/bin/env python3
"""
Test script to verify that the font size control works correctly.
This script loads the HTML, checks the elements, and verifies the initialization.
"""

import re
import sys

def test_html_structure():
    """Verify HTML has the correct structure for font controls"""
    with open('fdv_chart_rev6/fdv_chart.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Check that font-item-select has "point" selected by default
    print("\n[TEST 1] Checking font-item-select default value...")
    if 'value="point" selected' in html:
        print("✅ PASS: font-item-select has 'point' selected by default")
        tests_passed += 1
    else:
        print("❌ FAIL: font-item-select does not have 'point' selected by default")
        tests_failed += 1
    
    # Test 2: Check that font-value-select is NOT disabled
    print("\n[TEST 2] Checking font-value-select is not disabled...")
    # Find the font-value-select element
    match = re.search(r'id="font-value-select"[^>]*disabled', html)
    if match:
        print("❌ FAIL: font-value-select still has 'disabled' attribute")
        tests_failed += 1
    else:
        print("✅ PASS: font-value-select is not disabled")
        tests_passed += 1
    
    # Test 3: Check that initialization code exists
    print("\n[TEST 3] Checking for initialization code...")
    if '_onFontItemSelect()' in html and 'INIT' in html:
        print("✅ PASS: Initialization code exists")
        tests_passed += 1
    else:
        print("❌ FAIL: Initialization code missing")
        tests_failed += 1
    
    # Test 4: Check that _applyFontSize has logging
    print("\n[TEST 4] Checking _applyFontSize function has debug logging...")
    if '[_applyFontSize]' in html:
        print("✅ PASS: _applyFontSize has debug logging")
        tests_passed += 1
    else:
        print("❌ FAIL: _applyFontSize missing debug logging")
        tests_failed += 1
    
    # Test 5: Check that split chart update logic exists
    print("\n[TEST 5] Checking split chart update logic...")
    if '_splitInsts' in html and 'pointRadius' in html:
        print("✅ PASS: Split chart update logic exists")
        tests_passed += 1
    else:
        print("❌ FAIL: Split chart update logic missing")
        tests_failed += 1
    
    # Test 6: Check that _selectedFontItem is initialized
    print("\n[TEST 6] Checking _selectedFontItem initialization...")
    if "var _selectedFontItem = ''" in html:
        print("✅ PASS: _selectedFontItem is initialized")
        tests_passed += 1
    else:
        print("❌ FAIL: _selectedFontItem not initialized")
        tests_failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("="*60)
    
    return tests_failed == 0

if __name__ == '__main__':
    success = test_html_structure()
    sys.exit(0 if success else 1)
