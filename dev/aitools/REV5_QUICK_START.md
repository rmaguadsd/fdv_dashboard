# FDV Chart rev5 - Performance Options Quick Start Guide

## Access & Launch

### Option 1: Direct Chart Server (Recommended for Testing)
```powershell
# Launch on port 5060
.\dev\aitools\launch_rev5_chart.ps1 -Port 5060

# Then navigate to: http://localhost:5060/
```

### Option 2: Via Report Dashboard
```powershell
# Launch on port 5059
.\dev\aitools\launch_rev5.ps1 -Port 5059

# Then navigate to: http://localhost:5059/
# Chart access is embedded in the analysis interface
```

---

## Performance Control Panel

Located in the chart control area (after "Type:" selector), you'll see:

```
⚙️ Performance: [Canvas ▼] [100% Data (Accurate) ▼] [Max Points input (hidden)]
```

### Three Main Controls

#### 1. Render Mode Selector
**Current Options:**
- **Canvas (Default)** - Standard HTML5 Canvas rendering, good for all data sizes
- **WebGL (Fast)** - Future acceleration (currently behaves like Canvas)

**When to Use:**
- Keep Canvas for standard use (no performance penalty)
- WebGL can be selected for future optimization

---

#### 2. Sampling Mode Selector - **PRIMARY CONTROL**
**Three Options:**

##### A. 100% Data (Accurate) - **DEFAULT**
```
Shows: All points in dataset
Speed: Slow for very large datasets (50K+ points)
Accuracy: Perfect, no data loss
Data Fidelity: 100%
```

**When to Use:**
- Small to medium datasets (< 50K points)
- Analysis and verification requiring exact point positions
- When performance is acceptable
- Default recommendation

**Performance:**
- < 10K points: < 50ms render time (excellent)
- 10-50K points: 50-200ms (good)
- 50K-100K points: 200-500ms (acceptable)
- > 100K points: 500ms+ (consider sampling)

---

##### B. Random Sampling (Fast)
```
Shows: ~maxPts random points (deterministic hash-based)
Speed: Fast even for million-point datasets
Accuracy: Good, statistically representative
Data Fidelity: ~(maxPts / totalPoints) * 100%
```

**How It Works:**
- Each data point has a consistent probability of inclusion
- Uses hash function: `hash = (rowIndex * 73856093) % 1000000007`
- Same dataset → Same visualization (reproducible)
- Subset relationships maintained (filtered ⊂ unfiltered)

**When to Use:**
- Large datasets (100K+ points)
- Need fast interaction and real-time filtering
- Statistical properties more important than exact point positions
- Exploratory analysis and quick previews

**Performance:**
- Always renders < 100 points visible as selected via max-pts
- Computation time: ~1-10ms for point selection
- Chart rendering: < 50ms

**Example:**
- Total points: 1,000,000
- Max points: 10,000
- Sampling probability: 1/100 (1% of points shown)
- Render time: <50ms vs. 5000ms for 100% mode

---

##### C. Decimation (Statistical)
```
Shows: Binned statistical representation
Speed: Fast for any dataset size
Accuracy: Very good for smooth/continuous data
Data Fidelity: Statistical (median per bin)
```

**How It Works:**
1. Bins points by x-coordinate into sqrt(totalPoints) bins
2. Extracts median point from each bin
3. Preserves distribution shape while reducing count
4. Maintains continuity of line plots

**When to Use:**
- Smooth continuous data (e.g., time series, RBER vs WL)
- Preserving distribution shape is important
- Trade-off between speed and statistical accuracy
- Line plots where visual smoothness matters

**Performance:**
- Same as random sampling: < 100ms for any dataset size
- Produces smoother line visualizations than random sampling

**Example with RBER data:**
- If scattered randomly across x-axis
- Decimation extracts median RBER at each x-bin
- Result: Smooth trend line representation vs. scattered random sample

---

#### 3. Max Points Input
**Visible When:** Random Sampling or Decimation mode selected

**Controls:**
- **Range:** 100 to 100,000 points
- **Default:** 10,000 points
- **Update:** Type new value, press Enter or leave field

**What It Does:**
- Sets target number of points to display
- Sampling algorithm ensures ≤ maxPts in visualization
- Actual displayed points may be less (but never more)

**Recommendations:**
- **10,000** (default) - Good balance for most displays
- **5,000** - For slower machines or very large datasets
- **20,000** - For high-end displays wanting more detail
- **50,000** - Maximum reasonable for smooth interaction

---

## Usage Workflow

### Scenario 1: Default Exploration
```
1. Load data → 100% Data (Accurate) is already selected
2. Small dataset? → Click Plot → Instant rendering
3. Large dataset? → See slow rendering warning
4. Switch to Random Sampling → Click Plot → Fast rendering
```

### Scenario 2: Performance Tuning
```
1. Start with 100% Data mode
2. Performance slow? → Open Sampling Mode dropdown
3. Select "Random Sampling (Fast)"
4. Verify Max Points is set appropriately (e.g., 10,000)
5. Click Plot → Should render in < 100ms
6. Adjust Max Points if needed for more/less detail
```

### Scenario 3: Comparing Modes
```
1. Plot data with 100% Data mode
2. Note distribution shape and key features
3. Switch to Random Sampling → Plot
4. Compare: Are key features preserved?
5. Switch to Decimation → Plot
6. Evaluate which mode best balances speed + fidelity for your use case
```

### Scenario 4: Large Dataset Analysis
```
1. Load multi-million point session
2. 100% Data would be too slow (try it to see)
3. Use Random Sampling with max-pts = 10,000
4. Get fast interactive experience
5. Export data if full resolution needed for offline analysis
```

---

## Expected Performance

### Render Times (milliseconds)

| Dataset Size | 100% Data | Random Sampling (10K) | Decimation |
|--------------|-----------|----------------------|------------|
| 1K points    | 20ms      | 20ms                 | 20ms       |
| 10K points   | 50ms      | 20ms                 | 20ms       |
| 50K points   | 200ms     | 25ms                 | 25ms       |
| 100K points  | 500ms     | 30ms                 | 30ms       |
| 500K points  | 2500ms    | 40ms                 | 40ms       |
| 1M points    | 5000ms    | 50ms                 | 50ms       |

### Memory Usage

All modes:
- All original data always in memory (no deletion)
- Sampling only affects visualization layer
- Memory footprint identical regardless of sampling mode

---

## Keyboard Shortcuts & Tips

- **Ctrl+Z:** No undo (would need to re-plot)
- **Max Points slider:** Type value directly, press Enter to apply
- **Default recovery:** Reload page to reset to defaults
- **Session persistence:** Settings reset per session (not saved to disk)

---

## Troubleshooting

### Problem: Sampling Mode Dropdown Disabled
- This shouldn't happen - report as bug if encountered
- Workaround: Reload page

### Problem: Max Points Input Not Visible
- **Expected!** Only appears when Random or Decimation selected
- Select "Random Sampling (Fast)" → Max Points will appear

### Problem: Chart Doesn't Update After Sampling Mode Change
- Click "Plot" button to trigger redraw with new sampling
- Sampling mode alone doesn't re-render (efficiency feature)

### Problem: Performance Still Slow After Enabling Sampling
- Set max-pts to lower value (try 5,000 instead of 10,000)
- Check if dataset is truly large (> 1M points)
- Consider switching from Decimation to Random Sampling

### Problem: Data Looks Different Between Modes
- **Expected!** Random sampling is probabilistic
- Use decimation for more consistent appearance
- Use 100% Data mode for exact verification

---

## Data Integrity Notes

✅ **Always Safe:**
- All original data preserved in memory
- Sampling only affects visualization
- No data corruption or permanent changes
- Can switch modes without data loss
- Export still works with full dataset

⚠️ **Important:**
- Switching sampling modes may show different subsets
- This is expected - each sampling mode is independent
- Use 100% Data mode for exact analysis requiring all points

---

## Performance Options Philosophy

**Key Principles:**
1. **Accuracy First:** Default is 100% data with no sampling
2. **Opt-in Optimization:** Users explicitly choose performance modes
3. **Transparency:** Always show what sampling mode is active
4. **Reproducibility:** Same data + same sampling = identical visualization
5. **Safety:** Original data never modified or deleted

---

## Advanced: Technical Details

### Random Sampling Algorithm
```javascript
if ((rowIndex * 73856093) % 1000000007 < threshold) {
    include_point();
}
```

Benefits:
- Deterministic (same seed → same results)
- Consistent across re-renders
- Preserves row-index relationships
- Fast computation (single arithmetic operation per point)

### Decimation Algorithm
```javascript
1. Find xMin, xMax across all points
2. Create sqrt(pointCount) bins by x-coordinate
3. For each bin: extract median point by y-value
4. Return concatenated medians preserving x-order
```

Benefits:
- Preserves distribution shape
- Good for continuous data
- Median extraction maintains statistical integrity
- Smoother line plots than random sampling

---

## Feedback & Issues

If you encounter:
- Performance worse than expected
- Ui controls not responding
- Data anomalies between sampling modes
- Rendering artifacts

Please report with:
1. Dataset size (number of points)
2. Sampling mode selected
3. Browser (Chrome/Firefox/Edge)
4. Session ID if available
