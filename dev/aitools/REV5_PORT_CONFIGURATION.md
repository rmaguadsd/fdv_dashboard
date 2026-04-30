# 🎯 FDV Chart Rev5 - Correct Port Configuration

## Port Issue & Resolution

**Problem:** Port 5060 is in the browser's restricted ports list
- Browsers block connections to certain ports for security reasons
- Port 5060 is one of them (ERR_UNSAFE_PORT)

**Solution:** Use **port 5059** instead
- Safe, unrestricted port
- Browser can access without issues
- Now the default in launch scripts

---

## How to Access

### ✅ Correct URL
```
http://localhost:5059/
```

### Launch Commands

**Chart Server (rev5):**
```powershell
.\dev\aitools\launch_rev5_chart.ps1 -Port 5059
# Or just:
.\dev\aitools\launch_rev5_chart.ps1
# Default is 5059
```

**Report Server (rev5):**
```powershell
.\dev\aitools\launch_rev5.ps1 -Port 5059
# Or just:
.\dev\aitools\launch_rev5.ps1
```

---

## Port Guidelines

| Port | Status | Browser Access |
|------|--------|----------------|
| 5058 | ✅ Safe | Works |
| 5059 | ✅ Safe | Works |
| 5060 | ❌ Restricted | Blocked |
| 5061 | ❌ Restricted | Blocked |
| > 5061 | ✅ Safe | Works |

**Rule:** Use ports 1024-5058 or 5062+ to avoid browser restrictions

---

## Current Configuration

- **Default Port:** 5059
- **Access URL:** http://localhost:5059/
- **Status:** ✅ Running and accessible

---

## Testing

1. Open: **http://localhost:5059/**
2. Load CSV file
3. See performance controls
4. Test sampling modes

You're all set! 🚀
