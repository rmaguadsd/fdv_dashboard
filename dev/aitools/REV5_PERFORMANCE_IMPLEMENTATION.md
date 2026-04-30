# FDV Chart rev5 - Performance Options Implementation

## Overview
Created rev5 with user-controlled performance options for the FDV Chart application. Rev5 is a frozen copy of rev4 (stable baseline) with new optional performance features layered on top.

## Key Features Added

### 1. Performance Control UI (Lines 747-766 in fdv_chart.html)
New control panel with two main dropdowns:

**Render Mode Selector:**
- Canvas (Default) - Standard HTML5 Canvas rendering
- WebGL (Fast) - Future WebGL acceleration (placeholder for now)

**Sampling Mode Selector:**
- 100% Data (Accurate) - Show all points, maximum fidelity
- Random Sampling (Fast) - Hash-based deterministic sampling
- Decimation (Statistical) - Bin-based statistical aggregation

**Max Points Input:**
- Configurable threshold (100-100000, default 10000)
- Only visible when sampling is enabled (random or decimation)

### 2. Global Performance Variables (Lines 1050-1053)
```javascript
var _renderMode = 'canvas';     // 'canvas' | 'webgl'
var _samplingMode = 'none';     // 'none' | 'random' | 'decimation'
var _maxPoints = 10000;         // configurable threshold for sampling
```

### 3. Event Handlers (Lines 5594-5644)
- `_onRenderModeChange()` - Switch rendering mode
- `_onSamplingModeChange()` - Switch sampling strategy and show/hide max-pts control
- `_updateMaxPointsValue()` - Update max points threshold

### 4. Sampling Implementation (Lines 4233-4305)
Three-mode sampling function: `_applyPointSampling(points, mode, maxPts)`

**Mode: 'none'** (Default)
- Returns all points unchanged
- No performance cost, 100% data fidelity

**Mode: 'random'** (Hash-based Deterministic)
- Implements reproducible sampling via hash function: `(rowIndex * 73856093) % 1000000007`
- Ensures identical results across renders for same data
- Maintains subset relationships (filtered ⊂ unfiltered)
- Fast computation, probabilistic coverage

**Mode: 'decimation'** (Statistical Aggregation)
- Bins points by x-coordinate
- Keeps median point from each bin
- Preserves distribution shape while reducing count
- Number of bins: `sqrt(pointCount)` capped at `maxPts`

### 5. Event Listeners (Lines 1495-1502)
Registered event listeners for max-pts-input:
- `change` event - When user finishes editing
- `input` event - Real-time updates during typing

## Data Flow

```
User Action (dropdown change)
    ↓
_onRenderModeChange() / _onSamplingModeChange()
    ↓
_applyPointSampling(points, mode, threshold)
    ↓
plotPts[] (visualization subset)
    ↓
Chart rendering with filtered data
```

## Performance Characteristics

| Mode | Data Points | Speed | Accuracy | Use Case |
|------|-------------|-------|----------|----------|
| None (100%) | All | Slow for huge datasets | Perfect | Analysis, verification |
| Random | < maxPts | Fast | Good | Exploration, quick previews |
| Decimation | < maxPts | Fast | Very Good | Long-tail distributions |

## Backward Compatibility

- **Default behavior unchanged**: Sampling is OFF by default (100% data mode)
- **No data loss**: All original data remains in memory; only visualization is affected
- **Rev4 stability frozen**: Rev4 remains untouched for production use
- **Opt-in performance**: Performance modes are optional user choices

## Files Modified

1. **fdv_chart_rev5/fdv_chart.html** (Main application)
   - Added performance control UI
   - Added global variables and event handlers
   - Added sampling logic function
   - Integrated sampling into main chart and split chart rendering

2. **launch_rev5_chart.ps1** (NEW - Launcher)
   - Starts FDV Chart rev5 on port 5060
   - Sets environment variables for rev5 mode
   - Kills existing listeners on target port

3. **launch_rev5.ps1** (NEW - Report launcher)
   - Alternative: Starts report on port 5059
   - Uses same flask app with rev5 chart assets

## Testing Checklist

- [ ] Launch chart on port 5060: `.\launch_rev5_chart.ps1 -Port 5060`
- [ ] Load test session: `n59a_a2_pr36_rel005_25c_rber_edtc2`
- [ ] Test default mode: Verify all data points visible
- [ ] Test random sampling: Enable, set max points to 1000, verify fast rendering
- [ ] Test decimation: Enable, verify distribution shape preserved
- [ ] Test render mode: Switch between Canvas/WebGL (visual consistency)
- [ ] Test max points adjustment: Real-time updates when changing threshold
- [ ] Test data consistency: Same data with same sampling mode = identical visualization

## Next Steps

1. Launch chart server on port 5060
2. Test with real FDV data session
3. Verify performance improvements with large datasets
4. Optionally implement WebGL rendering if needed
5. Consider deploying rev5 as default once stability confirmed

## Architecture Decisions

1. **Three-tier approach**: None (default) → Random → Decimation
   - Allows users to trade accuracy for speed
   - Random maintains deterministic behavior
   - Decimation preserves distribution characteristics

2. **Sampling in visualization layer only**
   - All data remains in memory
   - No data loss or corruption
   - Can switch modes without re-parsing

3. **User-controlled toggles**
   - Not forced performance optimization
   - Users make explicit choices
   - Transparent about tradeoffs

## Performance Expectations

- **100% Data mode**: Suitable for < 100K points
- **Random Sampling**: Good for 100K-1M points, target 10K-20K visualization points
- **Decimation**: Best for smooth continuous data, similar performance to random

When dataset exceeds ~50K points with 100% mode:
- Canvas rendering becomes slow (100-500ms per draw)
- User can enable sampling to achieve <50ms render times
- Trade-off is explicitly controlled via dropdown
