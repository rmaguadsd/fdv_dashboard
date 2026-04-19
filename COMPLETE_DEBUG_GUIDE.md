# Complete Streaming Debug Guide

## Current Situation
✅ Chat function works
✅ Message sent to server  
✅ Server responds with 200 OK
❌ Tokens not appearing (because old HTML is cached)

## The Fix: Hard Refresh

Your browser is using the **OLD HTML file** before my debugging code was added.

### Do This Now:
1. **Press Ctrl+F5** (Windows/Linux) or **Cmd+Shift+R** (Mac)
   - This clears the cache and downloads the updated HTML
2. **Wait for page to load completely**
3. **Open DevTools** (F12)
4. **Click Console tab**
5. **Send a chat message** (type "test" and press Enter)
6. **Look at the console**

### What You Should See (After Hard Refresh)

The console should show this sequence:
```
Chat send called, message: test
Adding user message to chat...
Setting busy state...
Adding thinking indicator...
Thinking indicator added, building context...
Starting chat send: test Model: llama3:latest
Chat response received: 200 OK

Chunk 1: done=false, size=2048
Processing 1 parts, buffer remaining: 0
Parsing JSON: {"token": " hello"}
Parsed object: {token: " hello"}
Got token:  hello

Chunk 2: done=false, size=1024
Processing 1 parts, buffer remaining: 0
Parsing JSON: {"token": " how"}
Parsed object: {token: " how"}
Got token:  how

Chunk 3: done=true, size=256
...
Stream done
```

### What Each Log Means

| Log | Meaning |
|-----|---------|
| `Chat send called` | User clicked send ✅ |
| `Adding user message` | Message visible in chat ✅ |
| `Setting busy state` | UI disabled (expected) ✅ |
| `Adding thinking indicator` | "...thinking" appears ✅ |
| `Chat response received: 200` | Server responded successfully ✅ |
| `Chunk N: done=false, size=X` | Received N bytes of data ✅ |
| `Parsing JSON: {...}` | JSON being extracted from stream ✅ |
| `Parsed object: {...}` | Successfully parsed ✅ |
| `Got token: ...` | Token extracted and displaying ✅ |

## If You Still Don't See "Chunk" Logs

If after hard refresh you're STILL not seeing "Chunk" logs, that means:
1. The streaming response is not being received properly
2. The fetch response isn't readable as a stream
3. There's a JavaScript error preventing the pump() function

**In that case:**
- Look for any **RED error messages** in console
- Copy those errors and report them
- They'll help us understand what's breaking

## Browser Cache Clearing Options

If Ctrl+F5 doesn't work:

**Option 1: DevTools method**
- F12 → Right-click the Refresh button → "Empty cache and hard refresh"

**Option 2: Browser settings**
- Close and reopen browser completely
- Go to Settings → Clear browsing data → Cache → Clear

**Option 3: Direct cache clear**
- Chrome: Ctrl+Shift+Delete
- Firefox: Ctrl+Shift+Delete
- Safari: Develop → Empty Caches

## Important: Server is Still Running

The FDV Chart server on port 5058 is still running with the updated code.
Just refresh your browser to get the new version!

## What Comes After Hard Refresh

Once you see the "Chunk" logs appearing:
1. **If tokens show in chat:** The issue is already fixed! ✅
2. **If tokens DON'T show but "Got token" appears:** UI/display issue
3. **If "Got token" doesn't appear:** Parsing issue with token extraction

Each scenario has a clear fix path once we see the debug logs.

---

**TL;DR: Press Ctrl+F5, send a message, show me the console output!**
