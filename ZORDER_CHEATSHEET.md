# Z-ORDER QUICK REFERENCE CARD

## 🎯 What Is It?
Control which color-by groups render **on top** when overlapping.

## 📍 Where?
Plot Panel → **Z-Order** field (right after "Color by:")

## 💡 How?
```
Z-Order: value1, value2, value3
↓
Result: value3 renders ON TOP
```

---

## 🚀 Quick Examples

### Put Group On Top
```
Z-Order: my_focus_group
```

### Control 3 Groups
```
Z-Order: bottom, middle, top
```
(last = top)

### Reverse Alphabetical
```
Groups: A, B, C, D
Z-Order: D, C, B, A
```
(A renders on top)

### Partial Control
```
Groups: A, B, C, D, E
Z-Order: E
```
(B,C,D alphabetical, then E on top)

---

## ✅ Rules

| Rule | Example | Result |
|------|---------|--------|
| Comma-separated | `a, b, c` | ✓ Works |
| Case-sensitive | `red` not `Red` | Must match legend |
| Whitespace trimmed | `a , b , c` | ✓ Works fine |
| Empty field | Leave blank | Returns to alphabetical |
| Partial list | `c, a` (b missing) | b first, then a, then c |

---

## 🔧 Common Tasks

| Task | Solution |
|------|----------|
| Put X on top | `Z-Order: X` |
| Put X on bottom | `Z-Order: Y, Z, X` (then move to end) |
| Reset to default | Clear field, click Plot |
| Control 5+ groups | List all: `a, b, c, d, e` |
| Multi-dim color | Use: `dim1~val1, dim2~val2` |

---

## ❌ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Typo in value | Copy from legend exactly |
| Wrong case | Check chart legend (case matters) |
| Forgot comma | Use: `a, b, c` NOT `a b c` |
| Semicolon | Use: comma NOT semicolon |
| Not clicking Plot | Enter values THEN click Plot |

---

## 💾 Saving

- **Recipe:** Preserves z-order + plot settings
- **Session:** Preserves z-order + all data
- **Browser:** Persists while tab open

---

## 📊 Visual Example

### Default (Alphabetical)
```
┌─────────────┐
│  C (top)    │  ← Rendered last (on top)
│  B (middle) │  ← Rendered middle
│  A (bottom) │  ← Rendered first (bottom)
└─────────────┘
```

### With Z-Order: C, A, B
```
┌─────────────┐
│  B (top)    │  ← Rendered last (on top) - NEW!
│  A (middle) │  ← Rendered middle
│  C (bottom) │  ← Rendered first (bottom) - MOVED!
└─────────────┘
```

---

## 🎯 Use Cases

- **Highlight:** Put important group on top
- **Compare:** Layer treatment over control
- **Hide:** Put background underneath
- **Find:** Keep rare events visible
- **Analyze:** Create visual hierarchy

---

## 🆘 Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| No effect | Wrong spelling | Copy from legend |
| Groups unordered | Incomplete list | List all groups |
| Still alphabetical | Field empty | Enter at least 1 value |

---

## 📈 Real Example

**Scenario:** Sales data by region (North, South, East, West)

**Problem:** West (low sales) hidden under North (high sales)

**Solution:**
```
Z-Order: North, South, East, West
```

**Result:** West line on top (now visible!)

---

## 🚀 30-Second Start

```
1. Open: http://localhost:5059
2. Parse: Upload CSV
3. Plot: Select X, Y, Color-by
4. Enter: Z-Order: group_name
5. Click: Plot
6. Done! ✓
```

---

## 📝 Format Reference

```
✓ Correct:   red, blue, green
✓ Correct:   red , blue , green   (spaces OK)
✓ Correct:   dim1~v1, dim2~v2     (multi-dim)
✗ Wrong:     red; blue; green      (semicolon)
✗ Wrong:     red blue green        (no comma)
✗ Wrong:     Red, blue             (case mismatch)
```

---

## 🔑 Key Takeaway

**Last value in Z-Order list renders ON TOP**

Everything else renders below it.

---

## 📚 Documentation

- **Quick:** ZORDER_QUICK_START.md
- **Complete:** ZORDER_COMPLETE_GUIDE.md
- **Technical:** ZORDER_IMPLEMENTATION.md

---

## ✅ Status

✅ Live and running  
✅ Server: localhost:5059  
✅ Ready to use  
✅ Production ready

---

**Ready? Open http://localhost:5059 and try it! 🚀**
