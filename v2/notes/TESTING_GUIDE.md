# 🧪 Testing Guide - Animation & Bug Fixes

## Quick Test Commands

### Test V1 (Simple Version)
```bash
cd v1
python play_pygame.py
```

### Test V2 (Extended Version)
```bash
cd v2
python play_pygame.py
```

---

## 🎯 Critical Test Cases

### 1. Attack Bug Fix - Human draws 2 cards
**Steps:**
1. Start game (you play first as human)
2. Wait for agent's turn
3. If agent plays Attack card, you should see:
   - ✅ Animation of Attack card flying from agent area to center
   - ✅ Message: "AGENT plays Attack!"
   - ✅ Message: "[AGENT] Played ATTACK - Human must draw 2!"
4. Click "Draw Card" button
5. **VERIFY:** You draw **2 cards** (not just 1)
   - Should see 2 separate draw animations
   - Log shows "[HUMAN] Drew: [card1]" and "[HUMAN] Drew: [card2]"

**Expected:** ✅ Human draws 2 cards
**Previous Bug:** ❌ Human only drew 1 card

---

### 2. Agent Card Play Animations
**Steps:**
1. Start game
2. Observe agent's actions
3. When agent plays Skip or Attack:
   - ✅ Card should fly from agent area (top left) to center
   - ✅ Card grows slightly (scale 1.0 → 1.3) during animation
   - ✅ Text appears: "AGENT plays Skip!" or "AGENT plays Attack!"
   - ✅ Animation lasts ~0.8 seconds
   - ✅ Color-coded: Yellow for Skip, Orange for Attack

**V2 Additional Cards:**
4. When agent plays new cards (if available):
   - SeeFuture: Purple color + "AGENT plays SeeFuture!"
   - DrawBottom: Cyan color + "AGENT plays DrawBottom!"
   - Shuffle: Coral color + "AGENT plays Shuffle!"

**Expected:** ✅ Clear visual indication of what card agent played
**Previous Bug:** ❌ Agent played cards invisibly, only log text

---

### 3. Bomb Explosion Animation
**Steps:**
1. Start game
2. Play until agent draws a bomb (may take several games)
3. When agent draws bomb:
   - ✅ Explosion animation appears (orange circles)
   - ✅ "BOOM!" text shows for ~2 seconds
   - ✅ Message: "💣 HIT A BOMB! Using Defuse..." (if has defuse)
   - ✅ Or: "💣 HIT A BOMB! NO DEFUSE - EXPLODED!" (if no defuse)
   - ✅ Animation visible for at least 2 seconds

4. When YOU draw a bomb:
   - Same explosion animation
   - If you have defuse: Defuse panel appears
   - If no defuse: Game over

**Expected:** ✅ Explosion clearly visible for 2+ seconds
**Previous Bug:** ❌ Explosion too fast or invisible

---

## 🎮 Interactive Testing Scenarios

### Scenario A: Full Attack Chain
```
Turn 1 (Human): Play Attack → Agent must draw 2
Turn 2 (Agent): Draws 2 cards (verify count)
Turn 3 (Agent): Plays Attack back → Human must draw 2
Turn 4 (Human): Draw 2 cards (verify!)
```

### Scenario B: Skip Chain
```
Turn 1 (Agent): Plays Skip → draws reduced
Turn 2 (Agent): Draws 0 cards, turn passes to human
Turn 3 (Human): Your turn (verify agent didn't draw)
```

### Scenario C: Bomb Defuse
```
1. Keep playing until someone draws bomb
2. If defuse available → explosion visible → defuse panel
3. Choose bomb position (Top/Middle/Bottom)
4. Game continues
```

---

## 📊 Visual Checklist

During gameplay, you should see:

**Card Play Animations (NEW):**
- [ ] Skip card flying from agent → center (yellow)
- [ ] Attack card flying from agent → center (orange)
- [ ] V2: SeeFuture card flying (purple)
- [ ] V2: DrawBottom card flying (cyan)
- [ ] V2: Shuffle card flying (coral)
- [ ] Text: "AGENT plays [CardName]!"
- [ ] Smooth ease-in-out movement
- [ ] Slight scale increase (card grows)

**Card Draw Animations (existing, should still work):**
- [ ] Cards fly from deck → player area
- [ ] Different target positions for agent vs human
- [ ] Multiple draws show multiple animations

**Bomb Animations (IMPROVED):**
- [ ] Orange explosion circles
- [ ] "BOOM!" text
- [ ] Visible for 2+ seconds
- [ ] Message with bomb emoji 💣

**Game Log (improved messages):**
- [ ] "[AGENT] Played ATTACK - Human must draw 2!"
- [ ] "[AGENT] 💣 HIT A BOMB! Using Defuse..."
- [ ] "[HUMAN] Drew 2 card(s)"

---

## 🐛 Known Issues to Verify Fixed

| Issue | Status | Test |
|-------|--------|------|
| Human only draws 1 card after Attack | ✅ FIXED | Test Case 1 |
| Agent card plays invisible | ✅ FIXED | Test Case 2 |
| Bomb explosion too fast/invisible | ✅ FIXED | Test Case 3 |

---

## 🔍 Debug Info

If you see issues, check the log panel (button at bottom):

**Good log output:**
```
[AGENT] Action: Play Attack
[AGENT] Played ATTACK - Human must draw 2!
[HUMAN] Drew: Skip
[HUMAN] Drew: Safe
```

**Bad log output (old bug):**
```
[AGENT] Action: Play Attack
[HUMAN] Drew: Skip
# Only 1 draw! Bug!
```

---

## 💡 Tips for Testing

1. **Use the log panel:** Click "Show Log" button to see detailed action history
2. **Multiple games:** Some bugs only appear after certain card combinations
3. **Watch the animations:** Don't click too fast, let animations complete
4. **Count cards:** Verify hand sizes match expected draws
5. **V2 testing:** Train a model first or use random actions

---

## 🚀 Performance Notes

- **Animation smoothness:** Should be 60 FPS
- **No lag:** Even with multiple animations overlapping
- **Memory:** No leaks during long sessions
- **Timing:** All animations should complete before next action

---

## 📝 Report Issues

If you find any problems:
1. Note which version (V1 or V2)
2. Describe what happened
3. Include log messages
4. Screenshot if possible
5. Steps to reproduce

---

**Happy Testing!** 🎮✨

All fixes have been applied to both V1 and V2.
Syntax verified on both files.
Ready to play!
