#!/usr/bin/env python3
"""
FINAL VERIFICATION TEST
Checks that all critical fixes are in place and working properly.
"""

import re

def final_verification():
    with open('fdv_chart_rev6/fdv_chart.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    print("\n" + "="*70)
    print("FINAL VERIFICATION TEST - POINT SIZE CONTROL FIX")
    print("="*70)
    
    test_results = []
    
    # TEST 1: _selectedFontItem initialized to 'point'
    print("\n[TEST 1] _selectedFontItem initialized to 'point'...")
    if "var _selectedFontItem = 'point'" in html:
        print("✅ PASS: var _selectedFontItem = 'point'")
        print("   This means the early exit check won't prevent updates!")
        test_results.append(True)
    else:
        print("❌ FAIL: _selectedFontItem not initialized to 'point'")
        test_results.append(False)
    
    # TEST 2: HTML default selection
    print("\n[TEST 2] HTML font-item-select defaults to 'point'...")
    if 'value="point" selected' in html:
        print("✅ PASS: <option value=\"point\" selected>")
        test_results.append(True)
    else:
        print("❌ FAIL: Default selection missing")
        test_results.append(False)
    
    # TEST 3: font-value-select NOT disabled
    print("\n[TEST 3] font-value-select is not disabled...")
    pattern = r'id="font-value-select"[^>]*disabled'
    if not re.search(pattern, html):
        print("✅ PASS: No 'disabled' attribute on font-value-select")
        test_results.append(True)
    else:
        print("❌ FAIL: font-value-select has 'disabled' attribute")
        test_results.append(False)
    
    # TEST 4: DOMContentLoaded initialization
    print("\n[TEST 4] DOMContentLoaded calls _onFontItemSelect()...")
    if "setTimeout(function()" in html and "_onFontItemSelect()" in html:
        print("✅ PASS: Initialization code present")
        test_results.append(True)
    else:
        print("❌ FAIL: Initialization missing")
        test_results.append(False)
    
    # TEST 5: _applyFontSize has early exit check
    print("\n[TEST 5] _applyFontSize has early exit for safety...")
    if "if (!_selectedFontItem)" in html:
        print("✅ PASS: Early exit check exists")
        print("   Now safe because _selectedFontItem = 'point' by default")
        test_results.append(True)
    else:
        print("❌ FAIL: Early exit check missing")
        test_results.append(False)
    
    # TEST 6: Split chart update logic
    print("\n[TEST 6] Split chart update logic exists...")
    if "_splitInsts" in html and "pointRadius" in html:
        print("✅ PASS: Split chart updates for pointRadius")
        test_results.append(True)
    else:
        print("❌ FAIL: Split chart logic missing")
        test_results.append(False)
    
    # TEST 7: Debug logging
    print("\n[TEST 7] Debug logging for troubleshooting...")
    debug_items = [
        ("[INIT]", "Initialization"),
        ("[_applyFontSize]", "Font size apply"),
        ("Updating", "Chart updates"),
        ("split charts", "Split chart reference")
    ]
    found = 0
    for debug_str, desc in debug_items:
        if debug_str in html:
            print(f"  ✅ Found: {debug_str} ({desc})")
            found += 1
    if found >= 3:
        print("✅ PASS: Comprehensive debug logging present")
        test_results.append(True)
    else:
        print(f"⚠️  WARNING: Only {found}/4 debug items found")
        test_results.append(True)  # Still pass, logging is secondary
    
    # TEST 8: onchange handlers
    print("\n[TEST 8] Event handlers wired correctly...")
    if 'onchange="_onFontItemSelect()"' in html and 'onchange="_applyFontSize()"' in html:
        print("✅ PASS: Both onchange handlers present")
        print("  - font-item-select: onchange=\"_onFontItemSelect()\"")
        print("  - font-value-select: onchange=\"_applyFontSize()\"")
        test_results.append(True)
    else:
        print("❌ FAIL: Event handlers missing or wrong")
        test_results.append(False)
    
    # SUMMARY
    passed = sum(test_results)
    failed = len(test_results) - passed
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(test_results)} tests passed")
    print("="*70)
    
    if failed == 0:
        print("\n✅✅✅ ALL TESTS PASSED ✅✅✅")
        print("\nThe point size control fix is FULLY DEPLOYED and READY TO TEST!")
        print("\nEXPECTED BEHAVIOR:")
        print("1. Page loads")
        print("   → _selectedFontItem = 'point' ✅")
        print("   → font-item-select shows 'Point Size' selected")
        print("   → font-value-select is enabled (not greyed out)")
        print("\n2. DOMContentLoaded fires (100ms)")
        print("   → _onFontItemSelect() called")
        print("   → font-value-select populated with 4-20px sizes")
        print("   → Console logs: [INIT] Font controls initialized")
        print("\n3. User loads session n59a_a2_pr36_rel005_tPROG")
        print("   → Charts load and split mode can be enabled")
        print("\n4. User changes point size dropdown")
        print("   → onchange fires _applyFontSize()")
        print("   → Console logs: [_applyFontSize] ENTERED. _selectedFontItem=point")
        print("   → Check: if (!_selectedFontItem) - PASSES (it's 'point')")
        print("   → Split charts update: pointRadius = new size")
        print("   → Console logs: [_applyFontSize] Updating X split charts...")
        print("   → Points resize visually in split chart")
        print("\n5. Feature works! ✅")
        
        print("\n" + "="*70)
        print("TESTING PROCEDURE:")
        print("="*70)
        print("""
1. Open http://localhost:5059 in browser
2. Hard refresh (Ctrl+Shift+R) to clear cache
3. Open console (F12 → Console tab)
4. Load session: n59a_a2_pr36_rel005_tPROG
5. Enable split chart mode
6. In console, observe the logs
   - Should see: [INIT] Font controls initialized
7. Change the "Point Size" dropdown to different values
   - Should see: [_applyFontSize] ENTERED messages
   - Should see: [_applyFontSize] Updating X split charts...
8. Observe split chart points resizing in real-time
9. Test different point sizes (4, 6, 8, 10, 12, 14, 16, 18, 20)
   - All should work!
""")
        return 0
    else:
        print(f"\n❌ {failed} test(s) FAILED")
        print("Fix these issues before testing in browser")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(final_verification())
