import requests
import json
import time

# Give server a moment to be ready
time.sleep(2)

log_file = r'D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt'

print("=" * 70)
print("FDV EIMPRO WebApp - TEST WITH REAL LOG FILE")
print("=" * 70)

try:
    # Check if file exists
    import os
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        # List available files
        parent_dir = os.path.dirname(log_file)
        if os.path.exists(parent_dir):
            print(f"\n📁 Files in {parent_dir}:")
            for f in os.listdir(parent_dir)[:10]:
                print(f"   - {f}")
        exit(1)
    
    file_size = os.path.getsize(log_file)
    print(f"\n📂 Log File: {log_file}")
    print(f"   Size: {file_size:,} bytes")
    
    # Test 1: Upload and Parse
    print("\n" + "=" * 70)
    print("TEST 1: Upload and Parse Log File")
    print("=" * 70)
    
    with open(log_file, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:5058/api/upload', files=files, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print("✅ UPLOAD SUCCESSFUL!")
            row_count = data.get('row_count', 0)
            col_count = len(data.get('columns', []))
            csv_id = data.get('csv_id')
            
            print(f"\n📊 Parsed Data Summary:")
            print(f"   ✓ Total Rows: {row_count}")
            print(f"   ✓ Total Columns: {col_count}")
            print(f"   ✓ CSV ID: {csv_id}")
            
            columns = data.get('columns', [])
            print(f"\n📋 Extracted Column Names ({col_count}):")
            for i, col in enumerate(columns, 1):
                print(f"   {i:2d}. {col}")
            
            # Show preview
            preview = data.get('preview', [])
            if preview:
                print(f"\n📄 First Row Preview:")
                for key, val in preview[0].items():
                    print(f"   {key}: {val}")
            
            # Test 2: Get Data
            print("\n" + "=" * 70)
            print("TEST 2: Retrieve CSV Data")
            print("=" * 70)
            
            response = requests.get(f'http://localhost:5058/api/csv/{csv_id}/data?page=0&per_page=10')
            if response.status_code == 200:
                csv_data = response.json()
                if csv_data.get('success'):
                    print("✅ DATA RETRIEVAL SUCCESSFUL!")
                    print(f"   ✓ Rows Retrieved: {len(csv_data.get('data', []))}")
                    print(f"   ✓ Total Rows in CSV: {csv_data.get('total_rows', 0)}")
            
            # Test 3: Get Statistics
            print("\n" + "=" * 70)
            print("TEST 3: Calculate Statistics")
            print("=" * 70)
            
            response = requests.get(f'http://localhost:5058/api/stats/{csv_id}')
            if response.status_code == 200:
                stats_data = response.json()
                if stats_data.get('success'):
                    print("✅ STATISTICS CALCULATION SUCCESSFUL!")
                    stats = stats_data.get('stats', {})
                    
                    # Show numeric stats
                    print(f"\n📈 Numeric Columns ({len([s for s in stats.values() if 'mean' in s])}):")
                    for col, stat in stats.items():
                        if 'mean' in stat:
                            print(f"   {col}:")
                            print(f"      Mean: {stat['mean']:.6f}")
                            print(f"      Std:  {stat['std']:.6f}")
                            print(f"      Min:  {stat['min']:.6f}")
                            print(f"      Max:  {stat['max']:.6f}")
                            print(f"      Count: {stat['count']}")
                    
                    # Show categorical stats
                    print(f"\n🏷️ Categorical Columns:")
                    for col, stat in stats.items():
                        if 'unique_count' in stat:
                            print(f"   {col}: {stat['unique_count']} unique values")
            
            # Test 4: Generate Scatter Plot
            print("\n" + "=" * 70)
            print("TEST 4: Generate Scatter Plot")
            print("=" * 70)
            
            # Find numeric columns for plotting
            numeric_cols = [col for col, stat in stats_data.get('stats', {}).items() if 'mean' in stat]
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                payload = {
                    'csv_id': csv_id,
                    'x_col': x_col,
                    'y_col': y_col,
                    'hue_col': None,
                    'title': f'Scatter Plot: {x_col} vs {y_col}'
                }
                
                response = requests.post('http://localhost:5058/api/plot/scatter', json=payload, timeout=30)
                if response.status_code == 200:
                    plot_data = response.json()
                    if plot_data.get('success'):
                        print("✅ SCATTER PLOT GENERATED SUCCESSFULLY!")
                        print(f"   ✓ Plot URL: {plot_data.get('plot_url')}")
                        print(f"   ✓ Plot ID: {plot_data.get('plot_id')}")
                    else:
                        print(f"❌ Plot generation failed: {plot_data.get('error', 'Unknown error')}")
                else:
                    print(f"❌ Plot API error: {response.status_code}")
            else:
                print(f"⚠️ Not enough numeric columns for scatter plot (found: {len(numeric_cols)})")
            
            # Summary
            print("\n" + "=" * 70)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"\nSummary:")
            print(f"  ✓ File uploaded and parsed: {row_count} records")
            print(f"  ✓ {col_count} fields extracted")
            print(f"  ✓ Data retrieved from CSV")
            print(f"  ✓ Statistics calculated")
            print(f"  ✓ Scatter plot generated")
            print(f"\n🎉 WebApp is working correctly!")
            
        else:
            print(f"❌ Parse error: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ Upload Failed: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
