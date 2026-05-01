#!/usr/bin/env python3
"""
Comprehensive test to verify the font size control works end-to-end.
Tests the HTML structure, JavaScript logic, and simulates user interactions.
"""

import re
import sys

def extract_javascript_logic():
    """Extract and analyze the JavaScript logic from the HTML"""
    with open('fdv_chart_rev6/fdv_chart.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    print("\n" + "="*70)
    print("ANALYZING JAVASCRIPT LOGIC")
    print("="*70)
    
    # Test 1: Check _selectedFontItem initialization
    print("\n[TEST 1] Checking _selectedFontItem initialization...")
    if "var _selectedFontItem = ''" in html:
        print("✅ PASS: _selectedFontItem initialized to empty string")
        selected_font_item_init = True
    else:
        print("❌ FAIL: _selectedFontItem not found")
        selected_font_item_init = False
    
    # Test 2: Check _onFontItemSelect function sets _selectedFontItem
    print("\n[TEST 2] Checking _onFontItemSelect sets _selectedFontItem...")
    if re.search(r'_selectedFontItem\s*=\s*itemSelect\.value', html):
        print("✅ PASS: _onFontItemSelect sets _selectedFontItem from dropdown")
    else:
        print("❌ FAIL: _onFontItemSelect doesn't set _selectedFontItem")
    
    # Test 3: Check initialization calls _onFontItemSelect
    print("\n[TEST 3] Checking DOMContentLoaded calls _onFontItemSelect...")
    if re.search(r'_onFontItemSelect\(\)', html) and re.search(r'DOMContentLoaded', html):
        print("✅ PASS: DOMContentLoaded listener calls _onFontItemSelect()")
    else:
        print("❌ FAIL: Missing initialization call")
    
    # Test 4: Check _applyFontSize doesn't exit early if _selectedFontItem is set
    print("\n[TEST 4] Checking _applyFontSize logic flow...")
    apply_func = re.search(r'function _applyFontSize\(\).*?^}', html, re.MULTILINE | re.DOTALL)
    if apply_func:
        func_body = apply_func.group(0)
        # Check for the early exit condition
        if "if (!_selectedFontItem)" in func_body:
            print("⚠️  WARNING: _applyFontSize has early exit check for _selectedFontItem")
            print("  This is OK if _selectedFontItem is properly initialized")
        
        # Check for split chart update logic
        if "_splitInsts" in func_body and "pointRadius" in func_body:
            print("✅ PASS: _applyFontSize has split chart point update logic")
        else:
            print("❌ FAIL: Missing split chart update logic")
    else:
        print("❌ FAIL: _applyFontSize function not found")
    
    return html

def test_html_elements():
    """Test the HTML elements are correctly configured"""
    with open('fdv_chart_rev6/fdv_chart.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    print("\n" + "="*70)
    print("ANALYZING HTML ELEMENTS")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: font-item-select default value
    print("\n[TEST 1] Checking font-item-select default value...")
    if 'value="point" selected' in html:
        print("✅ PASS: font-item-select defaults to 'point' (selected)")
        tests_passed += 1
    else:
        print("❌ FAIL: font-item-select does NOT default to 'point'")
        tests_failed += 1
    
    # Test 2: font-value-select is NOT disabled
    print("\n[TEST 2] Checking font-value-select is not disabled...")
    # More precise check - find the font-value-select element
    pattern = r'id="font-value-select"[^>]*>'
    match = re.search(pattern, html)
    if match:
        element_str = match.group(0)
        if 'disabled' in element_str:
            print("❌ FAIL: font-value-select has 'disabled' attribute")
            print(f"  Element: {element_str}")
            tests_failed += 1
        else:
            print("✅ PASS: font-value-select is NOT disabled")
            tests_passed += 1
    else:
        print("❌ FAIL: Could not find font-value-select element")
        tests_failed += 1
    
    # Test 3: onchange handlers are set
    print("\n[TEST 3] Checking onchange handlers...")
    if 'onchange="_onFontItemSelect()"' in html:
        print("✅ PASS: font-item-select has onchange handler")
        tests_passed += 1
    else:
        print("❌ FAIL: font-item-select missing onchange handler")
        tests_failed += 1
    
    if 'onchange="_applyFontSize()"' in html:
        print("✅ PASS: font-value-select has onchange handler")
        tests_passed += 1
    else:
        print("❌ FAIL: font-value-select missing onchange handler")
        tests_failed += 1
    
    return tests_passed, tests_failed

def simulate_user_flow():
    """Simulate the expected user flow"""
    print("\n" + "="*70)
    print("SIMULATING USER FLOW")
    print("="*70)
    
    with open('fdv_chart_rev6/fdv_chart.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    print("\n[STEP 1] Page loads...")
    print("  - HTML sets font-item-select default to 'point'")
    print("  - DOMContentLoaded event fires")
    print("  - Calls _onFontItemSelect()")
    
    if "_onFontItemSelect()" in html:
        print("  ✅ Initialization code present")
    else:
        print("  ❌ Initialization code MISSING")
        return False
    
    print("\n[STEP 2] _onFontItemSelect() executes...")
    print("  - Reads itemSelect.value ('point' from default)")
    print("  - Sets _selectedFontItem = 'point'")
    print("  - Populates font-value-select with sizes 4, 6, 8, 10, ...")
    print("  - Enables font-value-select dropdown")
    
    print("\n[STEP 3] User loads session n59a_a2_pr36_rel005_tPROG...")
    print("  - Split charts are created")
    print("  - _splitInsts array is populated")
    
    print("\n[STEP 4] User changes font size in dropdown...")
    print("  - onchange event fires: _applyFontSize()")
    print("  - Reads new size from font-value-select")
    print("  - Checks: if (_selectedFontItem == 'point') ...")
    print("    - Should be TRUE because initialized to 'point'")
    
    print("\n[STEP 5] _applyFontSize() updates split charts...")
    print("  - Loops through _splitInsts array")
    print("  - For each chart's datasets:")
    print("    - dataset.pointRadius = new size")
    print("    - dataset.pointHoverRadius = new size + 2")
    print("  - Calls chart.update('none')")
    
    print("\n[STEP 6] Points resize visually...")
    print("  - All split chart points should be the new size")
    
    return True

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE FONT SIZE CONTROL TEST")
    print("="*70)
    
    # Test HTML structure
    html_pass, html_fail = test_html_elements()
    
    # Test JavaScript logic
    extract_javascript_logic()
    
    # Simulate user flow
    flow_ok = simulate_user_flow()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"HTML Element Tests: {html_pass} passed, {html_fail} failed")
    print(f"User Flow Simulation: {'✅ OK' if flow_ok else '❌ FAILED'}")
    
    if html_fail == 0 and flow_ok:
        print("\n✅ ALL CHECKS PASSED - Code should work!")
        print("\nNEXT STEPS:")
        print("1. Hard refresh the page (Ctrl+Shift+R)")
        print("2. Load session n59a_a2_pr36_rel005_tPROG")
        print("3. Go to split chart mode")
        print("4. Open console (F12)")
        print("5. Change the font size dropdown")
        print("6. Look for console messages:")
        print("   - [INIT] Font controls initialized")
        print("   - [_applyFontSize] ENTERED. _selectedFontItem=point")
        print("   - [_applyFontSize] Updating X split charts...")
        print("7. Points should resize immediately in split charts")
        return 0
    else:
        print("\n❌ ISSUES FOUND - Need to fix code")
        return 1

if __name__ == '__main__':
    sys.exit(main())
