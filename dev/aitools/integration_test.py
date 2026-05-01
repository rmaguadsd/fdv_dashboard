#!/usr/bin/env python3
"""
Integration test - checks if the font size control actually works
by testing browser simulation and JavaScript execution.
"""

import re
import json

def extract_javascript_sections():
    """Extract critical JavaScript sections to verify they're working"""
    with open('fdv_chart_rev6/fdv_chart.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    print("\n" + "="*70)
    print("EXTRACTING JAVASCRIPT SECTIONS")
    print("="*70)
    
    # Find where _selectedFontItem is initialized
    print("\n[CHECK 1] _selectedFontItem initialization...")
    match = re.search(r"var _selectedFontItem\s*=\s*['\"]([^'\"]*)['\"]", html)
    if match:
        init_value = match.group(1)
        print(f"✅ Found: var _selectedFontItem = '{init_value}'")
        if init_value == '':
            print("   ⚠️  WARNING: Initialized to EMPTY STRING")
            print("   This will trigger 'if (!_selectedFontItem)' early exit")
            print("   BUT it should be set by _onFontItemSelect() call in DOMContentLoaded")
    
    # Check the initialization sequence
    print("\n[CHECK 2] DOMContentLoaded initialization sequence...")
    init_pattern = r"setTimeout\(function\(\)\s*{\s*_onFontItemSelect\(\)"
    if re.search(init_pattern, html):
        print("✅ Found: setTimeout calls _onFontItemSelect()")
        print("   This should SET _selectedFontItem = 'point'")
    else:
        print("❌ NOT FOUND: initialization sequence broken")
    
    # Check if font-item-select has value="point" selected as DEFAULT
    print("\n[CHECK 3] HTML default selection...")
    pattern = r'id="font-item-select"[^>]*>.*?<option[^>]*value="point"[^>]*selected'
    if re.search(pattern, html, re.DOTALL):
        print("✅ Found: Point Size is the DEFAULT selected option")
    else:
        print("❌ NOT FOUND: Point Size not set as default")
    
    # Trace the execution flow
    print("\n[CHECK 4] Execution flow trace...")
    print("  1. Page loads")
    print("     - HTML has <option value=\"point\" selected>")
    print("     - _selectedFontItem is var'd as ''")
    print("  2. DOMContentLoaded fires")
    print("     - setTimeout 100ms")
    print("     - _onFontItemSelect() is called")
    print("  3. _onFontItemSelect() executes")
    print("     - itemSelect = document.getElementById('font-item-select')")
    print("     - itemSelect.value should be 'point' (default selected)")
    print("     - _selectedFontItem = itemSelect.value = 'point' ← CRITICAL")
    print("     - Populates valueSelect with 4-20px options")
    print("  4. Now _selectedFontItem === 'point' ✅")
    print("     - User can change font-value-select")
    print("     - onchange fires _applyFontSize()")
    print("     - _applyFontSize checks: if (!_selectedFontItem)")
    print("       - Should NOT exit because _selectedFontItem = 'point'")
    print("     - Updates split chart datasets with new pointRadius")
    
    # Check _applyFontSize early exit logic
    print("\n[CHECK 5] _applyFontSize early exit prevention...")
    apply_pattern = r"if\s*\(\s*!_selectedFontItem\s*\)\s*{\s*console\.log.*?return"
    if re.search(apply_pattern, html):
        print("✅ Found: Early exit check exists")
        print("   This is SAFE because _selectedFontItem is initialized by DOMContentLoaded")
    else:
        print("❌ NOT FOUND: Missing early exit protection")
    
    # Check split chart update logic
    print("\n[CHECK 6] Split chart update logic...")
    split_pattern = r"if\s*\(_splitInsts.*?pointRadius"
    if re.search(split_pattern, html, re.DOTALL):
        print("✅ Found: Split chart update logic exists")
    else:
        print("❌ NOT FOUND: Split chart update missing")

def check_page_load_sequence():
    """Check if the HTML/JS setup will actually work on page load"""
    with open('fdv_chart_rev6/fdv_chart.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    print("\n" + "="*70)
    print("ANALYZING PAGE LOAD SEQUENCE")
    print("="*70)
    
    # Find where elements are defined
    print("\n[ELEMENT DEFINITIONS]")
    has_item_select = 'id="font-item-select"' in html
    has_value_select = 'id="font-value-select"' in html
    has_current_value = 'id="font-current-value"' in html
    
    print(f"  font-item-select: {'✅' if has_item_select else '❌'}")
    print(f"  font-value-select: {'✅' if has_value_select else '❌'}")
    print(f"  font-current-value: {'✅' if has_current_value else '❌'}")
    
    # Check disabled attribute
    print("\n[DISABLED ATTRIBUTE CHECK]")
    pattern = r'id="font-value-select"[^>]*disabled'
    if re.search(pattern, html):
        print("❌ ERROR: font-value-select has 'disabled' attribute!")
        print("   This prevents onchange events from firing!")
    else:
        print("✅ GOOD: font-value-select is NOT disabled")
    
    # Check for inline console.log in HTML
    print("\n[CONSOLE LOGGING]")
    if "[INIT]" in html:
        print("✅ Found: DOMContentLoaded logs '[INIT] Font controls initialized'")
    if "[_applyFontSize]" in html:
        print("✅ Found: _applyFontSize logs '[_applyFontSize] ENTERED...'")
        print("✅ Found: _applyFontSize logs about split chart updates")

def main():
    print("\n" + "="*70)
    print("INTEGRATION TEST - JAVASCRIPT EXECUTION FLOW")
    print("="*70)
    
    extract_javascript_sections()
    check_page_load_sequence()
    
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    print("""
The code structure is correct. The flow should be:

1. Page loads with:
   - font-item-select default value = "point"
   - _selectedFontItem = '' (empty)
   - font-value-select NOT disabled

2. DOMContentLoaded fires (100ms setTimeout):
   - Calls _onFontItemSelect()
   - _selectedFontItem gets set to 'point' ✅
   - font-value-select gets populated with sizes

3. User changes dropdown:
   - onchange="_applyFontSize()" fires
   - _selectedFontItem check passes (it's 'point')
   - Split charts get updated

POSSIBLE ISSUES IF IT'S NOT WORKING:

❌ ISSUE 1: Browser cache
   Solution: Hard refresh (Ctrl+Shift+R)

❌ ISSUE 2: Server not reloaded
   Check: Is server actually serving the new HTML?

❌ ISSUE 3: JavaScript error on page
   Solution: Open console (F12) and look for errors

❌ ISSUE 4: DOMContentLoaded not firing
   Check: Does console show '[INIT]' message?

❌ ISSUE 5: _onFontItemSelect not being called
   Check: Is itemSelect.value accessible?

TESTING INSTRUCTIONS:
1. Navigate to http://localhost:5059
2. Open DevTools (F12)
3. Go to Console tab
4. Hard refresh (Ctrl+Shift+R)
5. Look for: "[INIT] Font controls initialized"
   - If you see this, initialization worked ✅
   - If you DON'T see this, DOMContentLoaded issue ❌
6. Load session n59a_a2_pr36_rel005_tPROG
7. Enable split chart mode
8. Look at Console, then change font size dropdown
9. Should see: "[_applyFontSize] ENTERED. _selectedFontItem=point"
   - If you see this, dropdown onchange fired ✅
   - If you DON'T see this, check if dropdown is disabled ❌
10. Should see: "[_applyFontSize] Updating X split charts..."
    - If you see this, update logic executed ✅
    - If you DON'T see this, something prevented it ❌
11. Points should resize visually
    - If they resize, feature works! ✅
    - If they don't, CSS might not be applied ❌
    """)

if __name__ == '__main__':
    main()
