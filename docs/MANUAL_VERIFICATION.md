# MANUAL_VERIFICATION: checks that need real hardware

The scenarios below cannot run in an agent sandbox: they need a GPU, a
display, or both. They were written as each feature landed and they stay
here as a regression suite, so a change that touches an old surface can
be re-checked without reconstructing what "correct" looked like.

This file used to be the back half of `docs/HANDOFF.md`, where 132 numbered
items sat between two thousand lines of shipment narrative and a backlog.
Audit finding `META-01` moved them out so the handoff could go back to
being a page you read on the way in.

## What has been checked

State is recorded per range rather than per item, which is how it was
kept when these were written:

- **1 to 101**: validated on hardware by the maintainer.
- **102 to 126**: **not yet validated.** This is the outstanding debt,
  and it predates the audit campaign.
- **127 to 132**: confirmed as each one landed.
- **133 to 141**: confirmed on 2026-08-11, as each landed. Added by
  `DATA-05`, `DATA-04` and `RUNTIME-02`, all of which needed a display, a GPU,
  or two windows at once.
- **142 and 145**: confirmed on 2026-08-14. Between them they carry the
  user-facing claim of both findings: a refused switch costs nothing, and
  cancelling frees the worker.
- **143 and 144**: **not yet validated**, and deliberately awkward. Item 142
  is the reason: every cheap way of breaking a model is now caught before a
  worker spawns, so reaching a post-spawn load failure means staging one (see
  143). Worth doing once, not routinely.
- **146**: cannot be forced, and is a log-watch note rather than a scenario.
  See the item.
- **147 and 150**: confirmed on 2026-08-14.
- **148 and 149**: **not yet validated.** Both were attempted and both were
  hard to stage, for reasons now written into the items themselves: 148's
  setup is refused outright unless both models can genuinely load, and 149
  does nothing at all if the two windows pick the same model on the same
  device. Neither attempt showed the behaviour failing.
- **151 and 152**: **not yet validated.** 151 asks you to confirm the overlay
  is back to its pre-change feel after being removed and restored; 152 covers
  the one improvement kept from that detour.
- **153**: confirmed on 2026-08-15. A second launch from the desktop icon
  opens no second window.
- **154**: **not yet validated.**
- **155**: **not yet validated**, and it is a judgement call rather than a
  pass or fail. The change it covers is meant to be reverted if you do not
  like it.
- **156**: **not yet validated.** Three more reservations in the spirit of
  152, plus the one place the technique is deliberately not applied.

Update these ranges when you work through them. If an item turns out to
be wrong rather than failing, fix the item; a scenario that no longer
matches the app is worse than no scenario, because it costs a session to
discover that a correct result looks like a regression.

## How these relate to the audit campaign

Separate lists, on purpose. The hardware queue in
`docs/audit/IMPLEMENTATION_LEDGER.md` tracks findings whose automated half passes
and whose hardware confirmation is pending; it is short, current, and
empties as the campaign runs. This file is the standing regression
suite. A finding that lands a lasting behavior worth re-checking should
add an item here; a one-off confirmation belongs in the ledger.

## Forcing failures

Several items need an activation to fail. The runbook for that is inline
below, before item 58, because two items depend on it and an earlier
revision of them gave advice that was wrong twice over.

## The checklist

1. **Alternatives off** (turn the toggle off; the registry defaults it
   **on**): a SmolLM3 run still streams normally, the
   **Entropy** overlay appears and recolors tokens, the entropy profile draws
   under the scrubber and tracks the scrubber's frame, the metrics strip shows
   an `Entropy` value, hovering a token makes its profile column glow and swings
   the nats readout to it, and **no** What If button or hover popover appears.
2. **Alternatives on**: hovering a token opens the candidate popover with five
   rows, sane probabilities, the chosen token highlighted, and readable
   whitespace/control candidates. It should sit **above** the token (a
   preference kept for readability now that the native tooltip it originally
   dodged is gone) and flip **below** for a token near the top of the canvas,
   where sitting above would put it over the metrics strip. Reaching into the
   popover to click a candidate should keep it open and keep the column
   glowing.
3. **What If**: the button appears; clicking it underlines the captured positions;
   clicking a candidate truncates and regenerates from there; the run lands in
   the Confirm/Retry review; **Retry** re-enters substitution (not the diffusion
   edit session); **Confirm** saves.
4. **Diff vs Original** is offered for the confirmed AR branch and renders both
   layers (worth checking with a substitution near the end, where the branch and
   original differ in length).
5. **Analytics**: open the saved branch. Entropy overlay, the hover popover, the
   Edited column, and Diff vs Original should all work post-hoc. Confirm
   `alternatives.json` is present and position-indexed.
6. **Entropy chart**: the detail modal shows a third chart for AR runs (Timing,
   Confidence, Entropy by Position) whose bars match the shape of the generator's
   profile for the same run. Hover lights the column and names the token with its
   nats; zoom, pan, Reset, and the eye toggle all behave. Then open an AR run
   saved **before** this session: the section should stay hidden rather than
   drawing an empty chart. Switch between two runs without closing the modal to
   confirm no stale chart survives.
7. **Counterfactual layer** (open the confirmed What If branch): the substituted
   position carries an **orange** dashed marker and a faint orange column tint,
   matching the generator's remask color rather than the old green. Hovering at
   or right of that position gives two rows labeled Original and Edited;
   hovering left of it gives one unlabeled row. **The two rows at the marker
   should show the same nats and different tokens**, which is correct (see
   "Counterfactual entropy" above), so treat matching numbers there as a pass,
   not a bug. The Original/Edited slider appears; dragging it left fades the
   branch out and the pre-edit run in, and the change should be visible only
   right of the marker. Reopening a different run should reset the slider to
   Edited. An AR run edited **before** this session (snapshot without `e`) should
   show the marker but no slider and no second row.
8. **Tooltip placement**: on the Timing chart, hover anywhere along the rising
   line; the box should settle in the **top-left** rather than on the trendline,
   and should not flicker between corners as you sweep the pointer. On
   Confidence (a curve hugging the top) expect a bottom corner. Park the cursor
   in the box's chosen corner and confirm it moves out of the way. Burn-through
   should now be rare: it is the fallback for when all four corners are
   occupied, most likely on a heavily zoomed view.
9. **Error path**: substitute after switching models and back (which clears
   `last_run_state`). Expect a clean error message and the **original run
   restored**, not a truncated one.
10. **Regressions**: one LLaDA Edit Frames session still works end to end
    (`_stream_tokens` refactor and the `handleError` rollback are the shared
    surfaces), both models still save/load normally, and a diffusion run's
    detail modal still shows exactly its three charts with no Entropy section.
    The tooltip positioner is shared, so give the Convergence chart a hover too.

The rest cover the What If lifecycle and timing pass:

11. **Confirm locks What If**: SmolLM3 run, What If, pick a candidate, let it
    finish, click the green check. The button must grey out the instant you
    click (not when the save lands) and stay unclickable, and hovering it should
    still show the "already has a saved edit" tooltip. Try clicking it during
    the save specifically, since that was the open window.
12. **Retry then pick a later token**: same setup but click the blue Retry
    icon, then hover a token **after** the edit position and pick a candidate.
    It should run, with no "not among the captured candidates" error and no
    need to hit Generate first. Repeat picking a token *before* the edit
    position, which worked even with the bug, to confirm nothing regressed.
13. **Timing alignment on a fresh edited run**: save one, open Analytics. The
    Timing chart's last frame index should now match Confidence and Entropy
    (before, a 256-token run showed ~374 against 242), and the **Elapsed** row
    should agree with the chart's final y value rather than showing just the
    branch (2.08s against a ~7s chart was the symptom). The post-edit segment
    should still be light blue, now driven by `remask_edits` rather than an
    elapsed drop.
14. **Legacy runs still read correctly**: open an edited run saved *before* this
    pass. Its Time column and Elapsed row should now show the repaired total,
    its Timing chart should be unchanged (still misaligned, since the stored
    array cannot be re-cut), and its post-edit segment should still be light
    blue via the drop heuristic. Also confirm an unedited legacy run's elapsed
    is untouched.
15. **Diffusion parity and session round trip**: run a LLaDA Edit Frames resume
    and confirm the same alignment holds there. Then generate, edit, navigate to
    Analytics and back, and save: the restored session should still produce a
    complete edited run (the pre-edit signals ride in the `full` sessionStorage
    payload, which falls back to a lighter one when the quota is hit).

The rest cover the shared comparison layer. Every item has a degrade path,
because runs saved before the previous pass carry no `original_alternatives`.

16. **Diff mode is interactive again** (the regression this pass was blocked
    on). Open an edited run in Analytics, pick **Diff vs Original**, and hover a
    token: the tooltip, the candidate popover, and the entropy bar highlight
    should all now work, where before nothing happened. Drag the two opacity
    sliders past each other and confirm the popover starts reading the layer
    that is now more opaque. Then check the same in the generator's diff
    overlay. Nulls in a frame now draw as `░` rather than vanishing, so watch a
    substitution near the end where the two runs differ in length.
17. **The crossfade governs everything.** Open a What If branch saved **this
    session**. The Original/Edited slider should sit at the right of the
    **Token overlay** heading row, directly above the text it blends, and
    appear for any edited run with a snapshot (not only ones carrying
    entropy). Switch to **Commit Order** and confirm the legend and the
    slider sit together at the right rather than spreading across the row,
    and narrow the window until the header wraps. Drag it in **None**,
    then **Heatmap**, then **Entropy**, then **Commit Order**: the tokens should
    crossfade between the two runs in every mode, and the entropy chart's bars
    should follow the same slider. The key correctness check is Heatmap and
    Entropy: at full Original the colors must be the **pre-edit run's own**
    confidence and entropy, not the branch's colors under the original text.
    Commit Order has its own second steps array, so its original layer should
    show the pre-edit run's gradient. Switch runs and confirm it reopens on
    Edited. Then open an **unedited** run: no slider, one layer, everything
    exactly as before.
18. **Cross-highlighting, and that it lets go.** In Analytics, hover a bar and
    watch the matching token light up white in the overlay above; hover a token
    and watch its bar light. Then sweep off the chart three ways: down into the
    x-axis label strip, sideways off the canvas, and by closing the modal
    outright. The token must go dark every time (this was the sticky-highlight
    bug, and the axis-gutter exit is the one the old `onHover` wiring missed).
    On the generator, sweep the entropy profile and confirm the token lights and
    tracks correctly at both ends of the sequence (that is the math inversion;
    an off-by-one would show as a consistent one-token drift). With the
    crossfade mid-way, both layers should light at once.
18b. **One highlight look.** A direct token hover and an entropy-driven
    highlight should now be visually identical (white with a soft glow) on both
    pages, where the direct hover used to be a separate orange. Check it over a
    remasked token and at the Heatmap's warm end, which is where the old orange
    disappeared. Analytics tokens should highlight on direct hover at all,
    which they previously did not.
18c. **Entropy profile: no standing marker.** The strip should carry **no**
    drawn marker at rest, only the hover guide under the pointer. The 2px
    white stub that used to float at the top near the last column is gone.
    The scrubber's frame is still marked, but only by its bar sitting at full
    opacity against the others' 0.68, and the nats readout at the right
    should name that frame's own position: scrub to the **last** frame and
    confirm the readout reports the final position, not the one before it.
18d. **The highlight checkbox, and that nothing clobbers it.** The **Overlay**
    drawer on both pages should carry a **Highlight tokens** checkbox, ticked
    by default, taking effect the instant you toggle it. Untick it on the
    generator, open Analytics, and confirm its drawer opens unticked too, then
    toggle it there and confirm the generator agrees on return. Reload and
    confirm it holds. The load-bearing regression check: with it **unticked**,
    open the Settings page, change something (or just hit **Reset**), hit
    **Save**, then go back and confirm the checkbox is still unticked.
    Settings saves the whole blob and no longer shows this field, so a
    clobber here is the failure mode to watch. The status bar should no
    longer carry `Highlighted Tokens: On/Off`, and Settings' Appearance tab
    should be down to the diffusion-text rows.
18e. **Form state across a visit, and only a visit.** This is the reported bug:
    turn on **Alternatives**, change a number or two, tick **Experimental**,
    type into the prompt, then go to Analytics and come back. Everything
    should be exactly as you left it. Repeat the round trip **before** running
    anything and again **after** a completed run; both paths were broken, and
    the second is where a completed run's prompt should override the draft
    rather than the other way round. Then hit **Generate** and confirm the
    params are still there afterward (they used to be cleared at the *start*
    of a run). Switch models and back, confirming each model keeps its own
    values across the reload. Finally close the app and relaunch: everything
    should be back to the recommended defaults, which is the intended
    boundary. Worth one look at a half-typed value (leave a trailing decimal
    point), which should come back as typed rather than rounded.
18f. **Reset to Defaults.** The **Reset** button at the right of the
    Experimental row should start greyed out, light up the moment anything
    differs, and grey out again once you undo the change by hand. Click it
    with Experimental on and several params changed: everything returns to
    the defaults, Experimental clears, and the range tooltips follow the
    narrower bounds. It should leave the prompt alone, and it should be
    unavailable while a generation is running.
19. **Popover pagination.** On a branch saved this session, hover a token
    **past** the substitution: the heading should read "Position N: Edited" with
    a `‹ ›` pager, the current side disabled. Click `‹` for the pre-edit
    candidates, and confirm the marked "chosen" row changes to the token the
    original run actually drew. The box should stay put rather than jumping when
    the two pages differ in row count, and reaching into it must not close it.
    Hover a token **before** the substitution: no pager, since both runs
    recorded the same set there. In Analytics, drag the crossfade below halfway
    and confirm a freshly hovered token opens on **Original**. On the generator
    with What If armed, the Edited page should be clickable and the Original
    page read-only (no hint line, no substitution on click).
20. **Degrade paths.** Open an edited run saved **before** the previous pass:
    the crossfade should be absent (no snapshot at all) or present with a
    single-dataset chart (snapshot without `e`), and the popover should show no
    pager (no `original_alternatives`) while otherwise working. Nothing here
    should throw; the failure mode to watch for is a blank overlay.
21. **Both runs draw on the line charts.** Open a What If branch (or a LLaDA
    Edit Frames resume) saved with its pre-edit snapshot. Timing and Confidence
    should each show two lines: grey solid for the original, colored dashed for
    the branch, overlapping exactly until the edit and separating after it.
    Neither chart should be filled. An **unedited** run should still show one
    filled line and **no pins** at all.
22. **The pins, all three states.** Both are lit green on open. Click 2: only
    the grey original remains and 1 becomes unclickable (cursor does not change
    to a pointer, clicking it does nothing). Click 2 again for both. Click 1:
    only the dashed branch remains. Confirm the two charts are independent of
    each other, and that tooltips drop the row for whichever run is hidden.
23. **The drag preview and the ease back.** With both pins lit, drag the
    Original / Edited crossfade above the token view. Timing and Confidence
    should crossfade along with the tokens and the entropy bars, and the pins
    should dim while you hold. Release: the two line charts should settle back
    to showing both over roughly a fifth of a second, and the pins should
    brighten. Release with the pointer dragged well outside the slider (and
    outside the modal) and confirm it still hands back.
24. **Keyboard is intentionally not previewing.** Click the slider thumb, then
    use the arrow keys. The tokens and entropy bars should move; Timing and
    Confidence should not. This is by design, not a bug.
25. **Zoom dock.** All four charts should carry the `+` / `-` / reset pill in
    their bottom-left corner, faint until you hover the chart. Check it does
    not overlap the first x-axis tick label or the axis title on any of them,
    and that zoom, pan, and reset still work. Scroll-wheel zoom over the plot
    should be unaffected.
26. **Tooltip swatches.** Hover any of Convergence, Timing, Confidence, and a
    compare-panel chart. Each swatch should be a solid chip of that series'
    line color, not a white box with a colored rim. On an edited run the two
    chips should be visibly different (grey for the original).
27. **Processor row.** The run summary above the charts should carry a
    `GPU: <name>` line (or `CPU: <name>` for a SmolLM3 CPU run, which is the
    case the old timing header got wrong), and the Timing header should now
    read just "Timing" plus its controls. An older run saved without the field
    should fall back to the detected GPU rather than showing an empty row.

The generator crossfade pass (all on the **generation** page, not Analytics):

28. **Nothing changed before a branch exists.** Generate a plain run of each
    kind (LLaDA, SmolLM3) and scrub through it in every overlay. No
    Original / Edited row should appear below the scrubber, and the tokens
    should look exactly as they did: masks fading up with confidence as you
    scrub a mid-run frame, Heatmap and Entropy and Commit Order colors
    unchanged, tooltips carrying the same lines. This is the regression check
    that matters most, since every span on the page now comes from the shared
    builder rather than the old inline one.
29. **The crossfade appears and blends.** Run SmolLM3 with Alternatives, use
    **What If?**, and confirm. The Original / Edited row should appear below
    the scrubber. Drag it: the text should fade between the two runs, and the
    entropy strip's columns should fade with it. Switch to **Heatmap** and
    drag to full Original: the colors should be the *original* run's
    confidence, not the branch's colors sitting under the original words.
    Repeat with a LLaDA Edit Frames resume.
30. **Pointer follows the opaque side.** With the slider left of center, hover
    a token past the divergence. The tooltip should describe the original
    run's token, and the candidate popover should open on its **Original**
    page. Drag right of center and hover the same position: both should flip
    to the branch. Then scrub to a mid-run frame and confirm the masks still
    grade by confidence in *both* layers.
31. **Edit modes still render one layer.** With a branch in place, click
    **Edit Frames** (LLaDA). The crossfade row must disappear, the tokens must
    go back to a single layer, and clicking one must still select it for
    remasking (orange, fully opaque). Exit and confirm the crossfade returns.
    Do the same with **What If?** armed on SmolLM3: the dotted underlines must
    be present and clickable, with no ghost layer behind them.
32. **Diff still owns its own controls.** Pick **Diff vs Original**: the
    crossfade row must hide and the two opacity sliders plus Difference blend
    must take its place, working as before. Switch back to any other overlay
    and confirm the crossfade returns at the value you left it.
33. **Commit Order colors each run by its own schedule.** On an edited LLaDA
    run, pick **Commit Order** and drag the crossfade fully left. The ghost
    layer's tints and its "Resolved at step" tooltips should describe the
    original run, which for positions past the edit means different values
    than the branch shows at full right. If the two look identical past the
    edit, `originalCommitSteps` is not being reached.
34. **The generator's entropy strip marks the edit.** Run SmolLM3 with
    **Alternatives** on, use **What If?** on a mid-sequence token, and let the
    branch finish. A dashed orange line over a faint tint must appear on the
    substituted position in the profile below the scrubber, matching the
    Analytics entropy chart's marker. Scrub the frame slider and drag the
    Original / Edited crossfade end to end: the marker must stay put through
    both. Hover the marked column and confirm the white hover glow draws *over*
    the orange tint rather than under it. Then run a second What If on a
    different position and confirm **both** markers show. On an unedited run
    the strip must have no marker at all.
35. **The band shows the cost of the edit.** Open an edited SmolLM3 or LLaDA
    run in Analytics. The timing and confidence charts should show a wash
    filling the gap between the grey solid line and the dashed branch, empty
    to the left of the edit and opening up to the right of it. Where the
    branch is higher the wash takes the chart's color (blue on timing, amber
    on confidence); where the original is higher it should be grey. Check both
    directions if you can produce them: a branch that is slower than the
    original gives blue, one that finishes faster gives grey.
36. **The band respects the pins and the scrub.** Unpin **1**, then **2** (one
    at a time): the band must vanish entirely with either, since it needs both
    edges. Re-pin both and drag the token view's Original / Edited crossfade:
    the band should fade with whichever line is fading out and come back as
    the charts ease home on release. It must not double-dim, that is, it
    should never look noticeably more transparent than the fainter of the two
    lines bounding it.
37. **Tooltip swatches are solid chips.** Hover any point on the timing,
    confidence, or convergence chart, and on the compare panel's charts. Each
    tooltip row's swatch must be a flat square of the series color with no
    colored ring around it and no white edge inside it. Check the entropy
    chart too: its swatches show the hovered bar's own ramp color and should
    be unchanged apart from losing the same white edge.
38. **The original bug is gone.** Generate a run and do **not** save it. Enter
    **What If?**, which auto-saves the original in the background, and pick a
    candidate immediately. Both *Saving original run* and *Running edit from
    frame N to end* must be up at the same time, side by side on one row
    separated by a faint dot, the resume nearest the resting message, each
    animating its own dots independently. Before this pass the save's message
    vanished. Let both finish: each should drift left and fade as its work
    ends, taking its separator with it and leaving *Done.* alone at the
    right.
39. **The single-message case looks unchanged.** Generate a plain run with
    nothing else happening. The *Running* text must sit on the same baseline
    as *Step* and *Elapsed*, at the far right where the status message has
    always sat, with nothing above or below it. This is the detail most
    likely to be a pixel off, since the row's alignment is no longer pinned
    by hand the way the old absolute column was.
40. **The footer keeps the record, and only the footer.** Save a completed
    run. The chip reads *Saving original run* and then disappears as the
    footer settles on *Saved original run to results/...*; at no point should
    two lines say the same thing. Navigate to Analytics and back: the footer
    line must still be there (it is what `saveSessionState` persists) and
    there should be no leftover chip. Then force a failure if you can (stop
    the server mid-save): the chip just goes, and the footer carries the full
    error text in red. Save an edited run too and confirm both the chip and
    the footer say *edited* rather than *original*, **and that both saves
    report the same shape of path**: `results/<run>`, never an absolute
    `/home/...` one. That asymmetry was the bug this pass fixed, and the
    edited save is the branch that used to show the long form.
41. **Nothing lingers or leaks.** Run several generate/save cycles in a row
    and confirm chips always drain, never pile up permanently, and never
    leave a half-faded ghost or an orphaned separator dot. Retry a run while
    one is going (Generate again after an error) and confirm the old chip
    goes rather than sitting there animating forever.
42. **The ellipsis animates in all three text modes, at a fixed width.** In
    Settings, cycle through the diffusion text effect (off, default, cycle)
    and start a run each time. The dots must tick in all three. In *Cycle*
    watch the word specifically: it re-diffuses every second or so, and the
    dots must keep ticking straight through that, with the message's left
    edge dead still. Any horizontal twitch as the word re-diffuses, or as the
    dot count passes through zero, means the fixed slot has been lost.
43. **Long messages clamp instead of colliding.** Save a run whose results
    path is long (a deep output directory, or just narrow the window). The
    footer line must truncate with an ellipsis at a small gap to the right of
    *Elapsed:*, never overlapping or sliding under it. Widen the window again
    and it should return to the full path. Note the boundary sits further
    right on diffusion runs, where the commit legend occupies that space.
44. **The row gives way at the left, not the right.** Narrow the window with
    two messages up (easiest during a What If auto-save) until they no longer
    fit. The oldest must fade out against the left edge rather than being cut
    with a hard vertical edge, and the resting message on the right must keep
    its full width throughout: it should never be the thing squeezed to make
    room for a chip.
45. **Chips rise in and step aside going out.** Start a run and watch the
    message appear: it should rise from below the row, out of the window's
    bottom edge, fading as it comes, not slide in from the right. Then watch
    one end. It must drift *left*, away from the resting line, and be gone
    quickly. The specific failure to look for is the fading text and the
    footer line printing over each other for a moment, which is what the
    old shared entrance/exit rule did. Check the rise clears the footer's
    padding cleanly rather than appearing to start mid-air; if it looks
    wrong, `--status-rise` in `style.css` is the single knob, and it must
    stay equal to the footer's bottom padding plus `#app`'s.
46. **Neighbours slide, they do not jump.** With one chip up, start a second
    operation (the What If auto-save overlap is easiest). The first chip must
    *glide* left as the new one arrives, not teleport. Then let a chip finish
    and watch the survivors glide back right. Hardest case, and the one worth
    doing deliberately: save a run with a long path while a chip is still
    up, so the resting line grows from empty to its full width in one go;
    the chip should ease across rather than being flung. Then turn on the
    OS "reduce motion" setting and confirm all of this degrades to plain
    fades with no sliding at all.
47. **The row does not fight the other floating surfaces.** Trigger a model
    download so the draggable download toast appears: at its default
    bottom-left it should not touch the row at bottom-right. Dragging the
    toast onto the row is expected to overlap and is not a bug.

**Resident navigation, picker flip, and load-bar corrections (needs a GPU
and a display).** The polish pass before this one was verified clean apart
from the three items it left open, which are 50, 51, and 54 below; item 55
is carried over untouched because it was never reported on.

48. **Re-selecting the resident model is navigation.** Generate with a model,
    go to the Main Menu, and pick that same model again. The confirm must read
    *Go back to the Generation page?* rather than offering to load it, there
    should be no loading animation or bar, and **the run must still be on the
    canvas when you arrive**, hyperparameters included. Do the same from the
    header selector, which should be a no-op there too. Then check the case
    that must still clear: with SmolLM3 resident on GPU, pick its **CPU** row
    from the menu. That is a real switch, so it must load properly and land on
    an empty canvas.
49. **A genuine switch still clears, from both paths.** Generate with SmolLM3,
    switch to a diffusion model and back, from the header selector and then
    from the menu. Only the run output should go; hyperparameters are keyed per
    model on purpose and must survive. Then generate, visit Analytics, and come
    back: that run must still be there, which is the case an over-eager clear
    would break.
50. **The overlay picker opens upward near the bottom.** Drag the collapsed
    drawer handle to the bottom of the output area, open it, and open the
    Overlay picker. The choices must appear **above** the drawer, fully inside
    the output area, rather than being cut off by its border. Drag the handle
    back to the top and confirm the picker opens downward again. Repeat on the
    Analytics page and in its run-detail modal. Then check the drawer drag
    itself still behaves: a drag must not open it, a plain click must, the
    position must survive a reload, and the two pages must remember separate
    positions.
51. **DiffusionGemma reserves a tail for the copy.** Activate DiffusionGemma
    and watch the sub-line. *Loading weights* should climb to roughly 90% and
    stop there, then *Moving to GPU* should carry the last stretch to 100%.
    The specific failure this replaced was the bar reaching 100% while still
    reading *Loading weights*, with *Moving to GPU* flashing past at full.
    Neither phase should jump: a leap from 90 to 100 in one step means the copy
    is not being tracked.
52. **LLaDA and SmolLM3 bars are unchanged.** These two take a different code
    path from DiffusionGemma and are meant to be untouched by that fix, so this
    is the regression check. Activate each on GPU: the bar should track the
    wait the way it did before, with no stall at 90% and no jump.
53. **"Moving to GPU" now arrives earlier.** This is an intended change. On
    SmolLM3 the sub-line used to stay on *Loading weights* until VRAM overtook
    RAM; it should now flip to *Moving to GPU* as soon as the copy starts, and
    stay there. LLaDA streams straight to the GPU and should read *Moving to
    GPU* almost immediately, as before.
54. **The bar climbs and finishes rather than cutting off.** Activate a model
    from the Main Menu, which is the fast path where this was worst, and watch
    the inline bar: it should step up several times and be seen at **100%**
    before the page moves to the generator, not vanish somewhere partway. Do
    the same for a header switch (which reloads) and for the very first load
    after a full restart (which comes up through the boot path). All three
    should complete visibly.
55. **A CPU load never claims a GPU.** Activate SmolLM3 with the device toggle
    on CPU. The sub-line must stay on *Loading weights* for the whole load
    **including the final frame**, and never say *Moving to GPU*. The bar
    should still be roughly linear: this is the fp32-vs-BF16 target case, so a
    bar that finishes at half is the scale factor being wrong.
56. **The download bar still works.** Delete a model's cache (or use a model
    you have never loaded) and activate it. The download phase must still read
    "Downloading NN%" both in the menu's inline bar and in the generator
    overlay, and must hand over cleanly to the load phase afterwards rather
    than resetting or double-counting.
57. **A failed load does not leave a bar behind.** Force an activation failure
    (see the note below this list for how). The overlay must close, the error
    must surface, and a subsequent successful switch must start from an empty
    track rather than the failed run's fill.

**How to force an activation failure.** Earlier revisions of items 57 and 73
said to request DiffusionGemma on CPU, "which is refused". That is wrong twice
over and cost the maintainer a session's worth of hunting, so it is written out
here once. There is no CPU option for a diffusion model in the UI at all
(`isAutoregressive` gates the device toggle in
[menu.js](../src/web/static/menu.js); diffusion rows get a static GPU tag), and
reaching past the UI would not be refused either: `_resolve_device` accepts
`cpu` for any model and `activate()` **skips** the VRAM pre-flight for it
([server.py](../src/web/server.py)), so the app would earnestly try to load LLaDA
into host RAM. Three things do work, and the first two fail through different
paths, which matters depending on what you are testing:

- **Rename a venv** (fails *inside* `activate()`, so the POST returns non-ok
  and the poll never runs). `mv .venv-dgemma .venv-dgemma.bak`, then activate
  DiffusionGemma: it raises "venv python not found" before spawning anything.
  Instant, no GPU, no weights touched, and a second `mv` puts it back. This is
  the one for item 57.
- **Kill the worker mid-load** (fails through the *poll*, which is the path
  that matters for anything about the progress track, since the bar is on
  screen when the error lands). `pkill -f "run_worker.*llada"` while it loads;
  `_monitor_startup` sees the dead process and reports "worker exited during
  startup". This is the one for item 73.
- **Occupy the VRAM** from another process and activate LLaDA, which trips
  `_preflight_vram`. The most realistic and the slowest to set up.

**Reveal signal, birth glow, and Tokens per Second (needs a GPU and a
display).** The case to press hardest is 58: it is the peak concurrent-glow
scenario on the weakest renderer, and it exercises the rendering rewrite and the
animation at the same time.

58. **LLaDA at `steps=8`, `gen_length=160`, in the `desktop.py` window.** This
    is the stress case: eight steps over 160 positions means roughly twenty
    tokens are born at once, each drawing a blurred halo, on WebKitGTK. Watch
    for stutter during the run and for glows that linger past half a second.
    Then run the same prompt at `steps=64`, where reveals are sparse and each
    flash should be individually legible.
59. **The streaming view looks the way it always did.** This is the regression
    check for the character-to-token rewrite, so compare against memory
    carefully. Masked positions must still carry their soft glow (that is a
    `live-tokens` rule restoring what `.token-mask` deliberately drops for the
    scrubber), mask opacity must still track confidence, and line wrapping must
    be unchanged. One difference is expected and is arguably a fix: the live
    text is now the concatenation of per-token decodes rather than one decode of
    the whole sequence, which is exactly what the scrubber has always shown, so
    the text should no longer shift subtly the moment a run finishes.
60. **Scrubbing does not retrigger the glow.** Finish a run, then drag the
    scrubber back and forth. Nothing should flash: the glow fires only from the
    live path. Then confirm the scrubber's own masked tokens do **not** glow,
    which is the check that `live-tokens` was taken off the container when the
    run ended.
61. **The glow toggle.** Turn **Token birth glow** off in Settings, save, and
    run: no flashes, and no stutter either. Turn it back on. Then set the OS to
    prefer reduced motion and run again with the setting on; there should be no
    animation at all.
62. **DiffusionGemma does not strobe.** Its draft tokens churn before settling,
    which is the case the monotone reveal set exists for. Each position should
    flash **once**. Watch a multi-canvas run in particular: a new canvas is
    fresh noise, so its positions are entitled to flash again, but positions
    within one canvas are not.
63. **Elapsed no longer jumps backwards.** Generate, note the footer Elapsed,
    then run an Edit Frames resume (or a What If substitution on SmolLM3). The
    number must keep climbing from where it was, not restart near zero. This is
    the bug the fix targets, so it is worth doing before anything else on the
    footer.
64. **Tokens per Second in the footer.** It should climb and settle during a
    run. Click it: the label swaps to the last step and the number gets noisier,
    which is expected. Reload the page and confirm the mode stuck. Then check
    that a Settings **Reset to defaults** does *not* flip it back, since its
    control is the footer and not that page. Generate, go to Analytics, come
    back: the readout should still be there and should honor the current mode.
65. **The Timing pager in Analytics.** Open a run: the section reads **Elapsed
    Time** with two small arrows beside the heading. The right arrow swaps to
    **Tokens per Second**, which should be a smooth curve that settles rather
    than a sawtooth, and the chart must be **correctly sized** rather than
    squashed, since it was built while its section was briefly visible. Zoom,
    pan, Reset, and the eye toggle should all work on it. Switch between two
    runs without closing the modal to confirm no stale chart survives. On an
    edited **SmolLM3** run the compare pins appear and the pre-edit line draws;
    on an edited **diffusion** run they must stay hidden, which is deliberate.
66. **Both elapsed totals.** Open an edited run's detail: the summary lists
    *Elapsed (original)* and *Elapsed (edited)* rather than one figure. An
    unedited run, and an edited run saved before the pre-edit signal existed,
    should both still show the single *Elapsed* row.

**Per-class glow tuning, sub-setting grouping, and the load sweep (needs a GPU
and a display).** Press 68 hardest: maximum brightness against the longest fade
is the worst case for paint cost, and it is the only combination that reaches
the raised concurrency cap.

67. **Defaults are unchanged.** Before touching a slider, run LLaDA and
    SmolLM3 and confirm the glow looks exactly as it did last session. Both
    classes default to 100% and 500ms, and the cap arithmetic is chosen to land
    on the old fixed 48 at that fade, so any visible difference here means the
    scaling or the fallbacks are wrong rather than merely mistuned.
68. **The worst case: SmolLM3 on GPU at 200% and 2000ms.** This is the stress
    test. Set **Tune for** to Autoregressive, push both sliders to maximum,
    save, and run a long generation in the `desktop.py` window. Expect a long
    bright meteor trail; watch for stutter, since doubled radii quadruple the
    blurred area per token and up to 192 tokens can be glowing at once. Then
    confirm the tail **fades** rather than being cut off partway, which is the
    check that the cap followed the fade instead of staying at 48.
69. **The two classes are independent.** With the autoregressive pair still at
    maximum, switch **Tune for** to Diffusion: its sliders must snap back to
    100% / 500ms, not inherit what you just set. Run LLaDA to confirm it still
    uses its own pair, then run SmolLM3 again to confirm it kept the loud one.
    Reload the page between the two to check both survived the save.
70. **The preview matches the real thing.** Drag each slider and watch the copy
    block replay a second or so after you let go. What it shows should be what
    a run then does, since both go through the same function and the same
    keyframes. Check the shape against the class: **Autoregressive** sweeps
    left to right, **Diffusion** scatters with no marching or directional
    pattern to it, and in bursts of visibly differing size. Push **Fade time**
    to 2000ms and confirm the trail still has a **dark head ahead of it and a
    faded tail behind it** rather than filling solid; this is the tightest
    case, peaking at 28 of 38 lit, so it is the one to actually look at. Drop
    it to 200ms and confirm only about six words are lit at a time. Replay
    twice at one setting and confirm the scatter lights the **same words in the
    same order** both times, which is what makes two settings comparable. Check
    the block's **right edge lines up with the slider readouts** above it.
    Click the block to replay
    without moving a slider. Turn **Token birth glow** off: all four sub-rows
    dim, become unclickable, and any sequence in flight stops immediately
    rather than finishing into a dimmed row.
71. **Sub-setting grouping.** In Settings, confirm there is no hairline between
    a preference and its sub-settings, that the sub-rows are indented under it,
    and that a single line closes each group off from the next preference (with
    no line at all after the last group, since it ends the panel). Toggle
    **Render diffusion-style text** off: **Mode** must now dim in place rather
    than disappear, and must not be clickable or keyboard-reachable while dim.
    Check the dim level reads as inactive rather than broken, which is the
    override cancelling the custom select's own opacity.
72. **The load sweep, on both surfaces.** Activate a model from the Main Menu
    and watch the inline bar: a sweeping track labeled *Starting worker* should
    appear immediately, where previously the row showed nothing for several
    seconds. It must hand over to *Loading weights* with a real percentage in
    one eased move rather than a jump, then to *Moving to GPU*, then hold a
    brief full bar before the page moves on. Repeat as a **switch** from the
    generator header, which uses the other renderer. DiffusionGemma is the one
    to check for the handoff, since its reserved tail makes the fill behave
    differently from the other two.
73. **The sweep's edge cases.** Force a failed activation, using the **kill the
    worker mid-load** recipe from the note under item 57 rather than either of
    the others: it is the only one that fails while the sweep is on screen,
    which is the transition being tested. The overlay must still close and
    surface the error, and a later successful switch must start from a clean
    track rather than the failed run's state. Then set the OS to prefer
    reduced motion and activate
    again: the track should show a dim, still bar rather than a moving one, and
    the labels must still change phase. Finally confirm the pager arrows on the
    Analytics charts and in the SmolLM3 candidate popover now read bright when
    they act and dim when they do not, and that the thin bar above the
    Analytics **Original / Edited** slider is gone while the generator's own
    crossfade **keeps** its separator.

*Items 74 to 80 cover the token metrics strip. None could be exercised
in-sandbox (no display).*

74. **Both pages idle correctly.** Open the generator before generating
    anything: the strip sits directly above the output canvas, holds its
    labels, shows a dash for each value, and the canvas below it is not
    clipped or scrolled. Open an Analytics run detail: the same row sits
    between the **Token overlay** heading and the bordered canvas, and the
    modal is still `90vh` with no scrollbar it did not have before. Confirm
    the reserved height does not shift when you hover in and out.
75. **The two hover sources agree.** On an AR run, hover a token and note the
    values, then move to the same column in the entropy profile below the
    scrubber: the strip should read the identical position and numbers.
    Leaving either surface returns it to idle, except that reaching *into* the
    candidate popover must keep the reading, since that popover is about the
    position you are still reading. Repeat both directions in the Analytics
    modal against its Entropy chart.
76. **It follows the frame, not just the pointer.** Park the pointer on one
    token and drive the scrubber with the arrow buttons: position stays put
    while the token, confidence and entropy change under it, and the strip
    goes idle on a frame where that position does not exist (the target
    placeholder during guided editing is the clearest case). Then hover a
    token during a **live** run: the values update as the run streams, which
    is new (streaming tokens never had a tooltip).
77. **The crossfade names its run.** On a confirmed What If branch, drag the
    **Original / Edited** slider: past the midpoint the tag at the right end
    of the strip flips, and the confidence and entropy change to the other
    run's values for positions right of the substitution. Do the same with the
    Diff overlay's two opacity sliders. With no branch, the tag is absent
    entirely rather than reading "Edited".
78. **The overlay extras.** Under **Commit Order** on a diffusion run, hovering
    a resolved token adds `Resolved at step: N`, and a position that never
    settled adds nothing. Under **Diff vs Original**, a changed position reads
    `was: X` and a remask origin reads `(remasked here)`. Switching overlays
    with the pointer held still should swap the extra without moving anything
    else in the row.
79. **Dashes, whitespace, and no tooltips anywhere.** On a LLaDA or
    DiffusionGemma run, entropy reads as a dash rather than 0, since diffusion
    tokens carry no `e`. Hover a token that is a plain space: it shows as a
    middle dot, not an empty box. Then rest the pointer on any token, on both
    pages, for a couple of seconds: **no native tooltip should appear** over
    any token, in any overlay, in either layer. (Buttons and table cells still
    have their own titles; those are meant to stay.)
80. **The popover clears the strip.** On an AR run with **Alternatives**,
    hover a token in the **first line or two** of the output: the candidate
    popover should open *below* the token rather than above it, leaving the
    metrics strip and the hyperparameter row uncovered. Hover a token further
    down and it should go back to opening above. Scroll the canvas so a
    mid-run token sits at the very top and confirm it flips there too, since
    the test is the token's position on screen and not its position in the
    run. Repeat both in the Analytics detail modal, where the strip is the
    thing being protected. Reaching down into a below-placed popover to click
    a candidate should still keep it open.
81. **The tokenizer row.** Do a fresh SmolLM3 run and save it, then open it in
    Analytics: the detail panel should carry a **Tokenizer** row naming the
    class with its vocabulary size in parentheses. Open any run saved before
    this session and confirm the row is simply **absent** rather than blank or
    "unknown". Repeat the save on LLaDA and on DiffusionGemma, since the field
    is reported by the shared worker base and each of the three loads its
    tokenizer differently. `results/<run>/metadata.json` should carry
    `reproducibility.tokenizer` with `class`, `name_or_path`, `is_fast` and
    `vocab_size`.
82. **The pin survives pointer drift.** On an AR run with **Alternatives**,
    arm **What If?**, hover a token, and click into **Enter your own**. Now
    move the pointer well off the popover, across other tokens, and out of the
    output area entirely. The popover must stay open, keep your text, and stop
    retargeting to whatever you pass over. Type a few more characters from
    where the pointer now sits to confirm the field still has focus.
83. **The pin survives a scroll and a resize.** With a draft live, scroll the
    page and resize the window. The box should stay put with the draft intact
    rather than closing or re-anchoring. (Staying put is intended: re-anchoring
    would slide the field out from under the pointer.)
84. **The three cancels.** With a draft live, press **Escape**: the popover
    closes and the draft is gone. Do it again and click the red cross instead:
    the popover stays open on the candidates. Do it a third time and click
    somewhere else on the page: the popover closes. In all three cases hovering
    a token afterwards should open a fresh, empty entry.
85. **The leading space.** Hover a **mid-sentence** token (one whose text
    starts with a space) and click into the field: it should already contain a
    single space. Hover a token at the very start of a sentence and it should
    be empty. Backspace the seeded space away, then move the pointer around
    inside the popover to force a redraw: the space must **not** come back.
86. **Confirm gates at exactly one token.** Type something that resolves to
    several pieces (`unfortunatelyy` is a good bet): the pieces render in
    alternating tints with their ids, the count reads orange, and the green
    check is disabled. Delete a character until it resolves to one: the count
    goes green and the check enables, live, without needing to blur the field.
    Confirm and check the solidified row appears with a **yours** tag.
87. **A typed token reports an honest probability.** Confirm a deliberately
    unlikely word, click the solidified row to run it, and read the metrics
    strip at that position: the confidence should be genuinely low (well under
    the captured candidates' probabilities) rather than a placeholder. Then do
    the same with a word that *was* one of the five captured candidates: the
    measured value should match what the popover showed for it. Both cases:
    the position's **entropy** must be unchanged from the original run.
88. **Retry on a solidified row.** After confirming, click the small
    counter-clockwise icon at the row's right. It must return you to an empty,
    re-seeded field **without** running the substitution it is sitting inside.
89. **The typed path still lands as a real edit.** Run a typed substitution to
    completion, Confirm, and check the branch behaves like any other: the edit
    marker at that position, **Diff vs Original** available, the entropy
    profile carrying both runs, and the saved run opening cleanly in Analytics.
90. **Top-k does something, and nothing by default.** Generate with Top-k at
    its default -1 and a fixed seed, then repeat: the output should be
    identical to what the same seed produced before this session. Now set
    Top-k to 1 with a temperature above 0: the run should become effectively
    greedy. Set it to 5 with a high temperature and confirm the output is more
    conservative than the same temperature at Top-k -1. Also try 0 explicitly:
    it should behave exactly like -1, which is what keeps runs saved under the
    old default replaying unchanged.
91. **Top-k does not disturb What If.** With Top-k set to something small, run
    a substitution: the continuation is greedy by design, so the branch should
    look exactly as it would have with Top-k off.
92. **The preview does not stall behind a run.** Start a long generation, and
    while it is streaming, open the entry on a *previous* completed run's
    popover if one is available. Typing should still preview without waiting
    for the generation to finish. (If the UI does not allow reaching a popover
    mid-run, this is satisfied by construction and can be skipped.)
93. **The typed row carries a figure.** Confirm a token and watch the right end
    of the solidified row, left of the retry icon: a dim `…` while the probe is
    out, then a percentage. On GPU that may be too fast to see, so check it on
    the CPU build where the pass is slow enough to catch. A deliberately wild
    word should read `<0.1%` rather than `0.0%`.
94. **The figure and the run agree.** Confirm a word that *was* one of the five
    captured candidates: the typed row's percentage must match the percentage
    that candidate's own row shows. Then click it to run, and the metrics strip
    at that position afterwards must report the same number again. Three
    surfaces, one distribution.
95. **The candidate readout fills the strip's right half.** Hover any row of
    the popover: a green-tinted chip with that candidate's text should appear
    to the right of the Entropy field, followed by its probability to three
    significant figures. The left group must keep reporting the token you are
    hovering rather than going blank. Move off the row and the green group
    should disappear entirely, not blank out to dashes.
96. **The typed row's readout carries a rank.** Hover the solidified row: the
    strip should show the probability *and* something like `#41,203 of
    128,256`. The captured five deliberately show no rank, since their rank is
    their order in the list you are already looking at. Confirm the denominator
    is the model's output width (128,256 for SmolLM3), not the tokenizer's
    128,000.
97. **Hovering across one row does not flicker.** Sweep the pointer slowly
    across a single candidate row, over its bar and its percentage: the strip's
    right half must hold steady rather than blinking as you cross the children.
98. **A retry does not leave a stale readout.** Hover the solidified row so the
    strip fills, then click retry. The green group must clear rather than
    keeping the token you just abandoned. Same check after flipping the
    popover's **Original** / **Edited** pager while a row is hovered.
99. **A probe cannot arrive for the wrong token.** On the CPU build, where the
    pass is slow: confirm a token, and while the `…` is still showing, click
    retry and confirm a *different* token. The row must not briefly show the
    first token's odds.
100. **The probe waits for a generation rather than colliding with it.** If you
    can reach a popover while a generation is streaming, confirming there
    should report that a generation is already running rather than starting a
    second forward pass. (Satisfied by construction if the UI blocks it.)
101. **Nothing regressed in the worker's dispatch.** The load gate and the
    streaming handlers were refactored, so exercise all three plainly: a cold
    model load (progress bar, then ready), a normal generation, a diffusion
    resume, and a What If substitution. Then cancel a run mid-flight, and try
    to start a second generation while one is running: it should still refuse
    with "A generation is already running."
102. **Item 94 is closed.** Repeat it exactly: confirm a word that *was* one of
    the five captured candidates. The typed row's percentage must now equal
    that candidate's own row to the digit, not merely approach it, because the
    figure is read from the record rather than measured again. Then run it and
    check the strip afterwards. This is the check the whole pass exists for.
103. **A genuine measurement also agrees.** Type something the position did
    *not* capture, note the percentage, then click it to run and read the
    strip at that position. Those must match, which is what says the probe and
    the substitution are using the same cache rather than two reconstructions
    of it. Do this on a run of at least a few dozen tokens, where the prefix is
    long enough for the cache to matter.
104. **The cache actually speeds up a substitution.** On the CPU build, where
    the difference is legible: run a long generation, then substitute at a late
    position. It should begin producing tokens noticeably sooner than the same
    substitution did before this pass, since the prefix is no longer prefilled
    from scratch. GPU may be too fast to feel.
105. **A stale cache cannot be used.** Generate, then generate *again* with a
    different prompt, then substitute in the new run. The fallback is a fresh
    prefill, so the only visible failure would be a wrong probability or a
    branch that reads as though it continued the old prompt. Also substitute
    twice in a row in the same run: the second must behave exactly like the
    first, since slicing must not have consumed the cache.
106. **Position 0 still works.** Substitute at the very first generated
    position, which has no cached token to decode against and takes the prefill
    path. Both the probe figure and the branch should be normal.
107. **A sixth row appears when the run reached outside the five.** Set
    Temperature high (1.2 or so), generate, then hover tokens: positions where
    sampling picked outside the top five should show a sixth row below a dashed
    rule, marked as the chosen one, with a rank in the hundreds or thousands.
    The five above it must be unchanged, still five. At Temperature 0 or a
    greedy run, no position should show a sixth row.
108. **The sixth row is not clickable.** Arm What If and hover a position that
    has one: the sixth row should not highlight as a target or respond to a
    click, while the five above it still do.
109. **Every row shows a rank on hover.** Hover each of the five in turn: the
    strip's right half should read `#1 of 128,256` through `#5 of 128,256`,
    matching their order in the list. Check an *old* run in Analytics too, saved
    before `model_vocab_size` existed: the rank should still show its number,
    with the denominator omitted rather than the whole reading suppressed.
110. **Edited positions stay tinted.** Run a What If substitution, confirm it,
    and scan the output: the position you forced should keep a soft orange wash
    for as long as the run is on screen, distinguishable from the brighter mark
    a position wears while being edited. Switch to the Heatmap and the Entropy
    overlays and confirm both still read normally on top of it. Repeat on a
    LLaDA run with several remask edits, and check the same run in Analytics.
111. **Entropy bars dim behind the scrubber.** Drag the frame scrubber back to
    the middle of an AR run: columns past that frame should fade while the
    current one stays highlighted. Then do the same in the Analytics detail
    modal, and while there, drag the **Original** / **Edited** crossfade: the
    dimming must survive the crossfade rather than being overwritten by it, and
    the bars should not flash or fully redraw as you scrub.
112. **Nothing regressed in a plain run.** The sampler's inner loop changed, so
    generate normally on both the GPU and CPU builds and confirm the output,
    the per-token confidences, and the entropy profile all look as they did.
    Cancel a run mid-flight, and start a fresh run immediately after a
    substitution.
113. **Memory does not creep.** The cache is now held across a run. Generate
    several long runs back to back on GPU, watching the VRAM headroom pill: it
    should return to the same figure after each, since a new generation
    releases the previous run's cache before building its own.
114. **The context readout appears and counts.** Load each model in turn. Once
    it is ready, the line under the prompt box should read `N / W`. Check `W`
    against the checkpoint's own config: LLaDA and SmolLM3 should report
    `max_position_embeddings`, and DiffusionGemma's should be its window, not
    256. Then type: `N` should settle a moment after you stop, and an empty
    prompt should report the template's own overhead rather than 0, because the
    role markers are always there.
115. **The count is the templated count, not a character estimate.** With
    SmolLM3 or DiffusionGemma loaded, note `N`, then flip **Thinking** and
    watch it change without the prompt changing. Flip it back and it should
    return to the original figure. If it does not move at all, the re-request
    on the flag change is broken.
116. **The saved figure matches what was on screen.** Generate, note the
    readout, save, then open the run in Analytics: the **Prompt tokens** and
    **Context window** rows should carry the same two numbers. Open a run saved
    before this session and confirm both rows are simply absent rather than
    showing zero or a dash.
117. **The overflow warning fires on the right key.** With LLaDA, raise **Gen
    Length** until prompt plus budget passes the window: the readout should
    turn amber. Do the same with **Max Tokens** on SmolLM3. Then on
    DiffusionGemma set **Max Tokens** above 256 but well inside the window and
    confirm it stays neutral, since chaining canvases is legitimate.
118. **Import by button.** Click the import control, pick a `.txt` file of a
    few KB, and confirm the text lands intact (including newlines) and the
    readout immediately reports its cost. Repeat with a `.md` file and confirm
    the markdown is inserted raw rather than stripped.
119. **Import by drag.** Drag the same file onto the textarea: it should show a
    drop target as the file crosses it, and insert on release. Then drag
    something that is not text (a PDF or an image) and confirm it is refused
    rather than inserted as bytes, and that the drop styling clears either way.
120. **The caps hold, and in the right order.** Drop a file over 1 MiB and
    confirm it is refused with a message naming the limit, and that nothing was
    inserted. Then take one between 200,000 characters and 1 MiB (300 KB of
    ASCII will do): it should be accepted, truncated to 200,000, and say so.
    The two bounds are deliberately different, since refusing a file you have
    already decoded is the wrong order.
121. **Replacing a non-empty prompt asks first.** With text in the box, import:
    the confirm modal should name the file. Cancel and the prompt must be
    untouched. Confirm and it is replaced. Then browse **prompt history** to an
    entry and import from there: the history UI should exit cleanly rather than
    leaving its controls active over imported text.
122. **Import is unavailable mid-run.** Start a generation and confirm the
    import button greys out for the duration and comes back afterwards.
123. **The star files and reads back.** In Analytics, click a row's star: it
    should fill and stay filled, a **Favorites** tab should appear with a count
    of 1, and the run should be listed under it. Reload the page and confirm
    both the filled star and the tab survive. Then close the app entirely,
    relaunch, and check again, since the point of the server-side store is that
    it survives more than a reload. Do that last part in the **desktop app**
    too, which is where the old localStorage approach failed.
124. **The caret files into several at once.** Hover a row, open its caret, and
    tick two collections: the run should appear under both tabs and the star
    should be filled. Untick one and it stays filled, because the star reports
    membership anywhere. Create a collection from inside the chooser and
    confirm the run is filed into it straight away.
125. **Tabs rename, delete, and scope.** Rename a collection from its pencil
    and confirm the tab and the chooser both show the new name. Delete one and
    confirm the modal says the runs survive, then confirm they do: they should
    still be listed under **All**. With a collection tab active, tick **Select
    all** and confirm the count matches only the visible rows, then switch tabs
    and confirm the selection cleared. Try a bulk delete from inside a
    collection and confirm exactly the visible runs went.
126. **A deleted run leaves no trace in a collection.** File a run, then delete
    it from the table: the tab count should drop immediately, without a
    reload. Then file another run, delete its folder from `results/` by hand,
    reload Analytics, and confirm the id was pruned server-side (check
    `results/ui_state.json` if you want to see it), and that the tab count
    never claimed the missing run.
127. **The star column reads down the table.** The star and its caret now sit
    in their own column between the checkbox and Date. Confirm the header
    lines up over them, that hovering a row reveals the caret without the
    column widening or the Date text shifting (it uses `visibility`, so it
    should not), and that clicking either one still does what it did from the
    right-hand side. Scroll the table with the sticky header up: the new
    header cell should stay in line with the rest.
128. **The group headers still span the whole table.** Set **Group by** to
    Model or Date: each group's heading row should reach the full width, with
    no empty cell left at the right edge. That row's `colSpan` had to grow by
    one for the new column, and an off-by-one shows up here and nowhere else.
129. **The trashcan sits on the row's centerline.** Compare a row's trashcan
    against its star, its Edited check and its text. Then check the header's
    bulk-delete button by ticking a few rows: its count should still fit
    beside the icon, since the actions column kept its width even though it
    now holds only one icon.
130. **The import dialog lists three entries.** Click import and open the file
    type dropdown. It should read **Accepted types (\*.txt \*.md)**, **Plain
    text document (\*.txt)** and **Markdown document (\*.md)**, with no
    repeats. "Accepted types" is Qt's own wording for the combined filter, not
    ours, so it cannot be renamed from the page. Confirm the combined entry
    shows both file kinds at once.
131. **A `.markdown` file still imports.** It is deliberately out of the accept
    list to keep that dropdown to two named types, but `isImportableTextFile`
    still allows it. Drag one onto the textarea and confirm it lands. In the
    browser as well as the desktop app, since the two use different file
    choosers and only the desktop one was the reason for this change.
132. **The bulk trashcan sits on the column labels' centerline.** Tick two rows
    and sight along the header: the can should straddle the caps of EDITED and
    TIME rather than hang below them, and the count beside it should not have
    drifted off the can. The correction is a single pixel against a 10px font,
    so judge it at the window size you actually use; say so if it now reads
    high and the lift can come back out.
133. **A damaged run is a visible row, not a gap.** Copy a saved run's folder
    aside, then corrupt the copy: overwrite its `metadata.json` with `{oops`.
    Reload Analytics. The corrupt run should appear as its own row, in amber
    italics, reading `<folder>: Unreadable metadata: ...`, and **every other
    run should still be listed**. Before this change it vanished silently,
    which reads as a deleted run. Sight along the table: the amber row's
    single wide cell has to end where the other rows' Edited column ends, so
    the trashcans stay in one line down the right edge. It should have no
    checkbox and no star. Click it: the detail panel should say the run could
    not be opened, give the reason, and note the folder is still on disk,
    with no spinner and no charts left over from the run you had open before.
    Then delete it from its row and confirm the confirmation names the right
    folder and that it goes.
134. **A run from a future version says so.** In another copied folder, edit
    `metadata.json` and set `"schema_version": 99`. Reload. That row should
    read `Saved by a newer version of this app (format 99). Update to open
    it.`, and specifically must **not** say unreadable or corrupt: the run is
    probably fine and this build is the old one. Wording matters here because
    the wrong wording invites deleting a good run.
135. **A new run carries `frames.jsonl` and reads back from it.** Save any run
    and open its folder. Alongside `history.txt` there should be
    `frames.jsonl`, one JSON object per line, and `metadata.json` should
    carry `schema_version` and a `capture` block naming which signals the run
    captured. Now rename `history.txt` aside and reload Analytics: the run's
    charts and overlays should be unaffected, because nothing reads the
    transcript any more. Put it back. Then do the same to an **older** run
    (one saved before this session): that one *should* break, because the
    transcript is all it has, and it should show as an invalid row rather
    than failing the page.
136. **A run keeps its own provenance across a model switch.** This is the
    scenario the fix exists for and it needs two browser windows on the same
    server. In window A, run LLaDA and let it finish, but do **not** save. In
    window B, switch to SmolLM3 and let it load. Go back to A and save. Open
    the saved `metadata.json`: `backend` must be LLaDA, the `reproducibility`
    block's `tokenizer` must be LLaDA's, `versions` must be the LLaDA venv's
    (`transformers` 4.38.2, not 4.53+), `context.context_length` must be
    LLaDA's window, and `reproducibility.attested` must be `true`. Before this
    change every one of those would have described SmolLM3. Check the
    Analytics detail panel for the same run agrees.
137. **A CPU run is not recorded as a GPU run.** Launch with SmolLM3 forced
    onto CPU (the device selector, or a host with no GPU), run and save.
    `metadata.json` should say `"processor": "CPU"` with the CPU's name, and
    the Analytics Processor column should agree. The supervisor only ever knew
    what it asked for, so a CUDA request that fell back used to be saved as
    GPU, which quietly makes the run's timings incomparable with real GPU
    runs.
138. **A guided edit's terminal frame carries provenance too.** Run LLaDA, do
    an Edit Frames resume with "run to here" so the worker ends the run at the
    frame budget rather than the sampler ending it, then Confirm and save.
    `reproducibility.attested` must still be `true`. That terminal frame is
    built by the worker on a separate code path from an ordinary finish, and
    it is the one most likely to be missed. Repeat on DiffusionGemma, whose
    resume path is different again.
139. **An older completed run still saves.** If you have a browser tab with a
    run finished before this session (a restored session snapshot), save it.
    It should save normally, with `reproducibility.attested` set to `false`,
    describing the resident model exactly as it always did. The fallback is
    what stops this change from stranding a run that is already on screen.
140. **The GIF names the model that made it.** Save a SmolLM3 run and a
    DiffusionGemma run, then open each `diffusion.gif`. The heading should
    read `SmolLM3-3B RESPONSE (Autoregressive):` and `DiffusionGemma-26B-A4B
    RESPONSE (Diffusion):`. Every GIF used to say `LLaDA RESPONSE
    (Diffusion):` whichever model ran.
141. **A long run's GIF is sampled and says so.** Run something past 300
    frames (raise steps, or use a long DiffusionGemma run) and save it. Its
    GIF heading should carry `[300 of N frames]`, the animation should still
    begin fully masked and end on the finished text, and the progression in
    between should look even rather than jumping at the end. Watch the
    supervisor's memory while it saves: it should stay flat rather than
    climbing by gigabytes, which is the actual point of the change.
142. **A refused switch costs you nothing.** With LLaDA loaded and a finished
    run on screen, use the header selector to switch to DiffusionGemma on
    **CPU** (or temporarily rename `.venv-dgemma` to force a missing
    interpreter). The switch should be refused with a readable reason, and
    critically: LLaDA should still be loaded, still generating, and the run
    should still be on screen. Before this change the resident model was
    unloaded first and the run discarded, both for an error that needed no
    VRAM to discover. Try the same from the Main Menu: the row should show the
    error and the previously loaded model should still be resident.
143. **A worker that fails to load is really gone.** Harder to stage than it
    looks, because item 142 now catches every cheap way of breaking a model
    before a worker is ever spawned. This one needs a failure that happens
    *after* the process starts, and the lever is that validation checks only
    that the checkpoint *directory* exists while the worker reads one file
    inside it. From `~/models/diffusiongemma-26B-A4B-it-nf4`, run
    `mv model_nf4.pt model_nf4.pt.bak` (same filesystem, instant, reversible),
    then activate DiffusionGemma. It passes validation, spawns, and fails
    inside `load()`. Watch `nvidia-smi` and `ps aux | rg run_worker`: the
    process must disappear and its VRAM must come back. Before this change it
    stayed alive holding memory while the supervisor reported an error. Move
    the file back afterwards.
144. **A failed load lands you on the menu, with the reason.** After 143,
    navigate to `/generate` directly. You should be redirected to the Main
    Menu and the menu should show the load error. Previously the generator
    page opened and appeared usable, because the gate asked only whether a
    process existed. Then activate any model successfully and return to the
    menu: the old error must be gone.
145. **Cancel still frees the worker.** Start a slow load and hit Cancel
    mid-load. VRAM should come back, the menu should be usable, and no error
    should be shown, since cancelling is not a failure. Then check the
    supervisor log: a cancel should not produce a stack trace.
146. **A stubborn worker is killed and waited for.** Not a scenario you can
    run: forcing it needs a process wedged in the kernel, which is not
    something to arrange on purpose. Kept as a numbered item only so the
    escalation has somewhere to be recorded if it ever fires. Two lines to
    watch for in the supervisor log: `ignored SIGTERM; killing` is the
    escalation working, and `survived SIGKILL` means a process stuck in
    uninterruptible I/O or a wedged driver call, which is worth reporting
    rather than ignoring.
**Items 147 to 150 are robustness probes, not descriptions of normal use.**
Nobody keeps two generator windows open on purpose; the setup exists to prove
a property that a stale background tab or a double-clicked launcher can reach
by accident. Two things make them easy to run wrongly. **Both windows must be
on the same supervisor**, so two browser tabs on the same address, because
`desktop.py` and `main.py` bind different ports and a second desktop launch
now stands down rather than starting a rival (see `LIFE-05` under Deviations).
And **both windows must pick different models**, since the server treats
re-selecting the resident model on the same device as a no-op and correctly
does nothing at all.

147. **Two windows cannot navigate for each other.** Open the Main Menu in two
    windows. In A, select LLaDA and watch it load. Before it finishes, in B,
    select SmolLM3. A should stop where it is rather than jumping to the
    generator when SmolLM3 becomes ready; B should navigate normally. Before
    this change A polled a global "is it ready" and navigated for whichever
    load finished, so it could land on the generator configured for a model it
    never asked for.
148. **Cancel reaches only your own load.** Same setup: A is loading LLaDA, B
    replaces it with SmolLM3. Press **Cancel** in A. It should refuse with a
    message naming SmolLM3 and saying it was started elsewhere, and B's load
    should keep going. Then press Cancel in B: that one should work and free
    the VRAM. Before this change A's Cancel killed B's load silently.

    **Setting this up is the hard part**, and the first attempt failed on it.
    Both windows have to reach an in-flight load, so both models must actually
    be loadable *at that moment*: item 142's pre-eviction check will refuse the
    second one outright if the GPU cannot hold it, and you get that refusal
    instead of the scenario. Start from an idle GPU with nothing resident.
    Beware the Main Menu's numbers while doing this; they are a snapshot from
    when the page was drawn and do not refresh, so a row can still claim a
    model fits after that stopped being true (recorded under Deviations in
    `docs/audit/IMPLEMENTATION_LEDGER.md`). Reload the menu to get a current
    reading before judging what will fit.
149. **A generator reloads when its model is taken, and keeps the run.** Load
    a model, generate something, and leave the run on screen **without
    saving**. In a second window, go to the Main Menu and switch to a
    *different* model, or the same model on the other device. Then open
    Analytics: the run must be there, saved, and attributed to the model that
    produced it. This is the item worth doing carefully, because the rescue
    save is the part that can lose work. Repeat with a run you *have* already
    saved: it should reload without filing a duplicate.

    Three things about the timing, all of which made the first attempt hard to
    read. **The two models must differ.** Re-selecting the resident model on
    the same device is a no-op the server does nothing for, so no worker is
    replaced, the socket never drops, and correctly nothing happens.
    **The reload waits for a reconnect.** The page learns about the change
    when its WebSocket reopens, which takes about two seconds after the new
    model finishes loading, and until then it shows a red "No model is active"
    while it retries. That is expected, not the failure.
    **The messages are brief.** "The model was changed to X" and the saving
    line are up only until the reload fires, which on a fast save is a
    fraction of a second. Judge this item by where the page lands and what is
    in Analytics, not by catching the message.
150. **The ordinary case is untouched.** The resident frame is sent on every
    socket open, so the common path has to be invisible. Load a model, use it
    normally, navigate to Analytics and back, let the socket drop and
    reconnect (stop and restart nothing, just leave it idle a while). At no
    point should the page reload itself or mention a model change.
151. **The generator still opens behind its curtain.** This item briefly said
    the opposite. The overlay was removed on the reasoning that the
    `/generate` gate makes a model load impossible to be waiting for, which
    was true and beside the point: the curtain also covers the page building
    itself, and without it the parameter column and the restored run were
    visibly assembling. It is back. Navigate from Analytics to Generation and
    expect the brief black "Loading model" flash, then a page that is already
    correct when it appears. Confirm it also still covers a real load: switch
    models from the header and it should stay up for the whole thing.
    The flash is a placeholder, not the destination. Removing the need for it
    means rendering the page's opening state into the HTML at serve time,
    which is `ORG-02`'s in stage 5 and is recorded there. Do not delete the
    overlay again without doing that first.
152. **The scrubber holds its place.** Kept from the attempt above, because
    it removes a source of movement rather than hiding one. Behind the
    curtain, a restored run must not resize the canvas as it appears. Easiest
    to judge on a fresh launch with no run: there is a reserved gap below the
    canvas where the scrubber will eventually be, and it should stay exactly
    that size once a run fills it. Say if the empty gap reads worse than the
    jump did.
153. **A second launch joins the first instead of fighting it.** With the
    desktop app already open, launch it again from the icon. No second window
    should appear and no second supervisor should start; the terminal prints
    `desktop: already running (pid N)`. Whether the existing window actually
    comes to the front depends on your window manager and is best-effort, so
    a message with no raise is a pass, not a failure. Confirm the important
    half with `nvidia-smi` and `ps aux | rg run_worker`: exactly one
    supervisor, and loading a model must still work normally in the original
    window.
154. **An unrelated process on 8760 does not lock the app out.** Occupy the
    port with something that is not this app, for instance
    `python3 -m http.server 8760`, then launch the desktop app. It should
    start normally on a fallback port rather than standing down, because the
    thing holding 8760 is not one of ours. The terminal says it is falling
    back to an ephemeral port and that web storage will not persist for that
    launch, which is the pre-existing behaviour and still correct.
155. **The page cross-fade, which is an experiment with an exit.** Move between
    the Main Menu, Generation and Analytics in the desktop app. The document
    swap should cross-fade rather than hard-cut. Judge it *with* the loading
    overlay in place, since that is the shipped combination: the transition
    covers the swap and the overlay covers the page assembling afterwards.
    If it does not clearly improve on the hard cut, say so and the rule comes
    out. It is one commit and one CSS block precisely so that is cheap.
    Two things not to read as bugs. In a browser that does not support it
    (Firefox on the `main.py` entry point) there is simply no transition,
    which is why the two entry points may feel different. And the fade is
    generic, with nothing morphing between pages, because naming elements is
    the follow-on work rather than part of the experiment.
156. **Three more things hold their places, and one deliberately does not.**
    Same idea as 152, applied to the shifts you spotted once the cross-fade
    stopped hiding them. Judge each on a fresh launch, since first paint is
    where they show.
    **Below the prompt box.** The token count under the prompt box needs a
    ready worker, so it is empty for a moment on every load. Nothing below
    the prompt box should move when it fills in.
    **The header links.** The Analytics link carries a count of unopened
    runs, which also arrives after a fetch. The links around it should not
    slide when it appears, and should not move again when the count crosses
    from nine to ten. Three digits will still nudge them, which is a hundred
    unopened runs and accepted.
    **The entropy row, only on SmolLM3.** Run SmolLM3 and the entropy strip
    below the canvas should occupy its space before the run produces it, so
    the canvas does not shrink when the run ends. Then load a diffusion
    model and confirm the opposite: no empty strip and no gap at all, because
    a row held for a model that never fills it is worse than the shift it
    would prevent. This is the case where reserving is the wrong answer, so
    a stray gap under a LLaDA or DiffusionGemma run is the failure to report.
