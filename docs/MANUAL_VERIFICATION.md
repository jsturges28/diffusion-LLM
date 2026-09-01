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
- **148**: **not reachable on this hardware**, which is a different answer
  from failing. The scenario needs two models loading at once, and item 142's
  pre-eviction check refuses the second on a card that cannot hold both;
  LLaDA and SmolLM3 together come to roughly the whole 23.49 GiB. The
  automated half covers the same fence in
  `tests/web/test_activation_identity.py`, where a cancel naming another
  operation is refused. Leave it here rather than deleting it: a larger card
  makes it stageable again.
- **149**: confirmed on 2026-08-17, after a first attempt that was hard to
  read. The trap is in the item: two windows picking the same model on the
  same device is a no-op the server correctly does nothing for.
- **151 and 152**: confirmed on 2026-08-15. The restored overlay reads as it
  did before the detour, and the reserved scrubber was the better of the two
  changes.
- **153**: confirmed on 2026-08-15. A second launch from the desktop icon
  opens no second window.
- **154**: confirmed on 2026-08-17. An unrelated process on 8760 sends the
  app to a fallback port instead of locking it out.
- **155**: **kept**, judged on 2026-08-15. The cross-fade itself was worth
  having. What it exposed was that several elements still moved just after
  the fade settled, which is 156 and is now fixed, so the experiment stays
  rather than being reverted.
- **156**: confirmed on 2026-08-17. The three reservations hold and the
  entropy row stays absent on a diffusion model, which is the half that
  would have been easy to get backwards.
- **162 to 166**: confirmed on 2026-08-18. The `ORG-02` state core and the
  save work that came out of testing it. The maintainer went past the
  scripts looking for a way to break the phase table and did not find one.
  166 took two attempts, and the first one is written into the item now: a
  saved run opened in Analytics is read from disk, which has no quota, so
  it looks perfect at any length and reads as a pass. Only returning to the
  Generation page exercises the session snapshot this item is about.
- **157 to 161**: confirmed on 2026-08-17, stage 4 pass three's whole queue.
  Two of them took three attempts each, and not because anything was broken.
  Both failures were the items' own fault and both are now fixed in place, so
  read 159 and 160 as they stand rather than assuming they are hard:

  The first was **how to get a second window**. Item 153 had just established
  that a second desktop launch stands down, which makes "two windows" read as
  impossible. A browser pointed at the running app's address is the answer,
  and neither item said so.

  The second was **which token to type**, in 160. Typing one of the five
  candidates on screen is answered from the run's own record without a probe
  ever being sent, so nothing could be refused and the item looked like it
  had failed when it had simply not been exercised.

- **167 to 170**: confirmed on 2026-08-18, `LIFE-04`'s whole queue. Stop
  halts each model within about a step, the GPU returns to idle, the run
  stays scrubbable and savable, and the worker takes fresh work straight
  after. Leaving the page stops a run the same way, and a guided edit
  finishing at its budget still reads as completed rather than stopped.

  Going past 168's script turned up one gap, which is recorded under
  `LIFE-04` in the ledger rather than here because nothing about it fails.
  Stop a run, then edit it and resume to the end: the saved run is
  correctly *not* marked stopped, because the branch really did run the
  schedule out. What is unrecorded is that the pre-edit baseline bundled
  inside it was the truncated one, so the Original/Edited comparison reads
  as though the intervention lengthened the run.

- **171 to 175**: confirmed on 2026-08-18, the analytics read path's whole
  queue. The table survives its catalog being cut to a twentieth of the
  size, the detail panel still knows everything now that it fetches its own
  record, convergence stops depending on word length, multi-canvas
  throughput carries committed canvases forward, and compare names what it
  leaves out.

  174 was the one worth doing slowly. The generator footer read `T/s: 14.8`
  and the Analytics curve landed at about 15 on the same run, which is the
  agreement that did not exist before. Looking at the same run's convergence
  chart then turned up something the item did not ask for and the ledger now
  records: for DiffusionGemma the curve counts stability rather than
  settlement, and a canvas reading 90% resolved can have a mean model
  confidence of 0.165.

- **176 to 206**: confirmed by the maintainer as each landed, across the
  sessions that shipped them, but never written into this list at the time.
  The dates are recoverable from the commits that added each item and are
  not reconstructed here rather than guessed at. Treat them as checked; if
  one of them matters enough to want a date, the commit that added the item
  is the record.
- **207 to 213**: confirmed on 2026-08-31, the collections polish queue.
  211 took two passes: the target rows rendered and counted correctly and
  could not be clicked, because the list listened for `change` and a target
  row is a button. The fix and what it says about source-inspection tests
  are in `docs/ROADMAP.md`.
- **214 and 215**: confirmed on 2026-08-31. Four kills, three reloads, then
  the cap, and a readable line per kill in the log. Note that 214 was
  written against a first attempt that aborted the app at launch, so its
  first instruction is to check the window opens at all.
- **216**: **outstanding**, and deliberately not scheduled. It needs the
  machine left idle until the screen blanks, so it costs twenty minutes of
  waiting to force and nothing to catch in passing. The maintainer will run
  it the next time the app happens to sit idle. It blocks nothing.

  When it fires, the question is not only whether the window recovered. If
  the window is white and
  `~/.local/share/llm-xai-visualizer/renderer-crashes.log` has **no new
  line**, the renderer did not die, the diagnosis behind that whole change
  was wrong, and the investigation restarts elsewhere. That negative is the
  more informative result and is the reason the log exists.

Update these ranges when you work through them. If an item turns out to
be wrong rather than failing, fix the item; a scenario that no longer
matches the app is worse than no scenario, because it costs a session to
discover that a correct result looks like a regression.

The gap at 176 to 206 is the failure mode this paragraph warns about,
arriving from the other direction: the items were run and the list was
not updated, so a later session cannot tell a checked item from an
unchecked one without reading the transcripts. Update the range in the
same pass as the confirmation.

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
157. **The ordinary single-window path is untouched.** Everything in stage 4
    pass three fences requests that should not be answered, so the first
    thing to confirm is that it fences nothing it should not. In one window,
    on any model: generate, then Edit Frames and resume from a frame, then
    Retry. On SmolLM3 also open What If, type a token to get a measurement,
    and confirm the substitution. None of it should mention a stale run, and
    nothing should behave differently from before. If this item fails,
    everything below it is moot.
158. **A reload does not make the run uneditable.** The page carries the run
    it is holding across a reload, and without that the restored output
    would look editable and refuse. Generate, reload the page with the run
    still on screen, then Edit Frames and resume. It should work exactly as
    if you had not reloaded. The same applies to going to Analytics and
    coming back.
159. **The second window's run locks the first out, and says so.**
    **Getting two windows is the part to read first**, because item 153 makes
    it look impossible: a second launch from the icon deliberately stands
    down. Launch once, then point an ordinary browser at the address the app
    is serving. That is a second client on the same supervisor and the same
    worker, which is exactly what this needs. Only **one** model is required,
    so a single GPU is no obstacle.
    Launch from a terminal rather than the icon while doing any of these:
    `.venv/bin/python desktop.py` prints the port it chose, and it does not
    always choose 8760. Anything else holding that port sends it to a
    fallback (item 154), and a browser tab on 8760 then reaches the wrong
    thing, or nothing. Check with
    `python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8760/api/app').read().decode())"`,
    which answers with a pid when it is ours.
    **The scenario.** Generate in window one and leave its output on screen.
    Generate in window two. Back in window one, Edit Frames and resume: it
    should refuse with a message about the run having been replaced, and roll
    back to the output it had rather than stranding a half-truncated run.
    Worth doing a second time with the *same prompt in both windows*, because
    equal shapes are what used to let the request through and produce a
    plausible wrong answer instead of an error.
    Then confirm window two still works: its own Edit Frames and resume
    should succeed. Refusing both windows would look like a fix and is not.
160. **A refused probe no longer closes What If.** SmolLM3 only, two windows
    as set up in 159.
    **Type a token that is not one of the five candidates on screen.** This
    is the whole trick, and the first two attempts at this item failed on it.
    A typed token that matches a captured candidate is answered from the
    run's own record without any probe being sent, deliberately, because the
    recorded figure is the run's real arithmetic. So nothing reaches the
    worker, nothing is refused, and the item looks broken when it has simply
    not run. Gibberish, backspaced until the row says it resolves to one
    token, is the quickest way there.
    **Do not race the generation.** Let window two's run finish completely.
    Window one's probe then names a run the worker no longer holds and is
    refused as stale, which needs no timing at all. Racing it works too and
    gives a busy refusal instead, but SmolLM3 finishes faster than you can
    type.
    **The scenario.** Generate in window one with Alternatives on. Generate
    in window two and let it finish. Back in window one, open What If, pick a
    position, type a token from outside the five, and confirm.
    What should happen: no measurement appears, an error says the run has
    been replaced, and What If stays open with your edit intact. The error
    goes to the resting status line on the right of the status bar, in red,
    not into the popover you are working in. What used to happen, and what to
    report if you see it: the panel closes and the edit is discarded, because
    a refused measurement was treated as though the run itself had died.
161. **Collections survive a fast navigation.** In Analytics, file a run into
    a collection and immediately click through to another page, faster than
    a quarter second if you can. Come back. The run should still be in the
    collection. Before this it could silently revert, one window, no race.
    Also worth one destructive check if you are willing: stop the server,
    then file a run into a collection in a still-open Analytics tab. A toast
    should say the collection could not be saved and that the change is
    local to that window. Previously it looked saved and then was not.
162. **The whole edit workflow still behaves.** `ORG-02` moved the run's
    frame arrays and its editing phases behind modules, and the guided edit
    flow needs a GPU, so none of it could be exercised where it was written.
    This item is the exercise. It should be entirely boring; report anything
    that is not.
    On a diffusion model: generate, Edit Frames, pick a frame, remask some
    tokens, lock them in, then **Edit another** and pick a target frame, run
    to it, remask again, and finally run to the end. Confirm at the review
    step. Then generate again, enter Edit Frames, and this time **Retry**
    from the review step rather than confirming. Then do one more where you
    simply leave the edit session partway through, which should put the
    original run back rather than leaving a truncated one.
    On SmolLM3: What If, substitute a captured candidate, and Confirm; then
    another where you Retry instead.
    **What a failure looks like now is different, and louder.** An illegal
    move between phases throws rather than proceeding, so a wrong table
    shows up as the workflow stopping dead, not as a subtly wrong screen. If
    that happens, the browser console will name the move it refused, for
    instance `illegal run phase move: choice -> review`, and that line is
    the whole bug report.
163. **A restored run is still complete.** The frame arrays are serialised
    and read back through one place now, and the keys did not change, so a
    snapshot written before this still loads. Generate, go to Analytics and
    come back, and confirm the scrubber, the token view, the overlays and
    the timing footer all still work on the restored run. Then save it and
    check in Analytics that the timing chart's x axis lines up with the
    other charts, which is the specific thing that broke the last time one
    of these arrays went missing.
    Long runs are worth one pass of their own. A run long enough to exceed
    the session storage quota falls back to a lighter snapshot that carries
    only three of the six arrays, and the run should come back readable with
    the character renderer rather than not at all. Editing it afterwards
    should still work.
164. **A save cannot become two runs.** The one you found. Generate on any
    model, click the save icon, and immediately click through to Analytics,
    faster than the save can answer. Come back to Generation, then open
    **Edit Frames** (or **What If?** on SmolLM3) and confirm an edit.
    Analytics should hold **one** row for that generation, carrying the
    edit. Before this it held two.
    Worth a second pass without the interruption, since that is the common
    path: generate, save, edit, confirm. Still one row, and the edit is in
    it. The bottom-right message naming where the run landed is still brief
    and still disappears on navigation; that part is unchanged and is not
    what this item is about.
165. **Opening an editor writes nothing.** The behaviour change to notice.
    Generate a run and do **not** save it. Open **Edit Frames**, look
    around, select a frame, mark some tokens, then leave the session
    without confirming. Nothing should have been written: Analytics has no
    new row, and the run is back on screen with the save icon still
    offered. Same again with **What If?** on SmolLM3.
    Then confirm one for real, and check that the single row it produces
    carries both the original and the branch: **Diff vs Original** should be
    available on it, and the Original/Edited crossfade should work.
    Two things that follow from this and are worth confirming rather than
    assuming. Opening What If on a long run should now be instant, because
    it no longer posts the whole run first. And if you abandon an edit and
    then close the tab, that run is gone, which is the protection this
    deliberately gave up: saving is explicit now.
166. **A run that came back thin says so instead of saving thin.** Needs the
    long-run setup from 163: SmolLM3, Experimental on, a prompt long enough
    to exceed the session storage quota, around a thousand tokens.
    **Read this before running it, because there is a false pass that is
    very convincing.** Saving the run and then opening it in Analytics is
    *not* this test, and it will look perfect however long the run is: a
    saved run is read from disk, which has no quota, so hover, candidates,
    the entropy profile and the crossfade all work at any length. This
    item is only about the session snapshot, and the only step that
    exercises it is coming **back to the Generation page**.
    So: generate, go to Analytics, come back to Generation. The run reads
    and scrubs but has no hover, no candidates and no entropy profile,
    which is expected and is the condition under test.
    Now press the save icon. It should refuse with a line saying the run
    came back without its per-token detail and cannot be saved in full.
    Before this it saved the hollowed-out version silently, which is the
    run with no output and one chart family.
    This is a stopgap over the storage shape, not a fix; `RUNTIME-01` is
    where the refusal stops being necessary.
167. **Stop actually stops the model.** The half of `LIFE-04` no sandbox
    can prove. Do this once per model, because all three reach the stop
    by a different route: LLaDA and SmolLM3 check between steps, and
    DiffusionGemma is unwound from inside its streamer.
    Start a long run, at least a few hundred tokens so there is time to
    act. **Generate** should have become **Stop**. Watch `nvidia-smi` (or
    the Settings VRAM readout) alongside it, press Stop partway, and
    confirm three things: the frames halt within about a step rather than
    running on to the token budget, GPU utilisation drops back to idle,
    and the status line reads **Stopped.** rather than **Done.**
    The run should still be there: scrub it, hover tokens, and confirm
    the save icon is offered. Then generate again immediately without
    reloading, which is the part that catches a worker left thinking it
    is still busy.
168. **A stopped run says so in Analytics.** Follows on from 167. Save the
    stopped run, open Analytics, and check its duration column reads
    something like `12.3s (stopped)`. Open it and confirm the detail panel
    carries **Completion: Stopped before the model finished**.
    Then check the negative half on any completed run: no `(stopped)` on
    the duration and no Completion row at all. A run saved before today
    should look the same as a completed one, not like an unknown.
169. **Leaving the page stops the run too.** The behaviour change most
    likely to surprise. Start a long run and, while it is going, click
    through to **Analytics**. Confirm with `nvidia-smi` that the GPU goes
    idle rather than continuing to churn for a page that is no longer
    watching. Come back to Generation: the frames it had produced are
    still there, and the run reads as stopped rather than as finished.
    Worth doing once by closing the tab outright as well, since that is
    the same path and the one a user hits by accident.
170. **A guided edit is not mistaken for a stopped run.** The distinction
    that shares a code path with the one above, on a diffusion model.
    Run a generation, open **Edit Frames**, pick a frame partway through
    and resume **to a chosen frame** rather than to the end. It stops at
    the frame you asked for, exactly as a cancelled run stops early, and
    it must still read **Done.** rather than **Stopped.**: it is a
    completed request. Saving it should produce a run with no Completion
    row and no `(stopped)` marker.
171. **The Analytics list against the real archive.** The change with the
    widest blast radius and no GPU requirement, so do this one first.
    Open Analytics on your 222 runs and confirm the page still behaves
    exactly as it did: sort by every column header, group by each of
    Date, Model, Processor, Prompt and Edited, switch between collection
    tabs, star and unstar a run, select several rows and check the count,
    and delete one. The catalog now carries about a twentieth of what it
    did, so the thing to watch for is a column that has gone blank or a
    count that has gone wrong, not a crash.
    Worth timing the load, since that is the point: it was fetching
    roughly 1.3 MB of metadata for a table that draws six fields.
172. **A run's detail panel still knows everything.** Follows from 171,
    and is where a field dropped too eagerly would show up. Open a run
    and check the panel lists its full prompt, every hyperparameter, the
    processor, the tokenizer and vocabulary rows, the context block and
    the elapsed total. Those used to come from the row; they arrive in
    their own request now.
    Open one with a very long prompt too. The row's tooltip shows the
    first 240 characters with an ellipsis, and the panel shows the whole
    thing.
173. **Convergence stops depending on word length.** Two SmolLM3 runs are
    the wrong tool here (they have no convergence chart); use LLaDA or
    DiffusionGemma. Generate two runs at the same **Steps** and
    **Gen Length**, one on a prompt that produces long words and one on
    a prompt that produces short ones, and compare their convergence
    curves in Analytics. They should have close to the same shape,
    because they resolved the same number of positions per step. Before
    this the long-word run climbed faster for no reason but its spelling.
    Then open an older run, from before token records were saved, and
    confirm the chart carries an italic line above it saying the curve
    counts characters. A recent run must not carry that line.
174. **Multi-canvas throughput no longer falls off a cliff.** Needs
    DiffusionGemma and a prompt long enough to chain a second canvas
    (over 256 tokens). Save the run, open it in Analytics, and page the
    timing chart across to **Tokens per Second**. The curve should carry
    on across the canvas boundary marker rather than dropping toward
    zero when the second canvas starts, which is what it used to do.
    Cross-check the figure against what the generator's own **T/s**
    footer read during the run. The two used to disagree on multi-canvas
    runs; they should now agree closely.
175. **Compare explains what it left out.** Select a diffusion run and a
    SmolLM3 run together and press Compare. The chart draws the
    diffusion run, and a note above it names the SmolLM3 one and says
    autoregressive runs have no convergence curve. Before this it simply
    vanished.
    Then check the legend on a DiffusionGemma run: it should name the
    model and real parameters, not `undefined`. Try selecting more than
    twelve runs, which should be refused with a message rather than
    quietly working. And open a comparison, close it immediately, and
    open a different one: the second must not be overwritten by the
    first arriving late.
176. **The DiffusionGemma convergence curve stops lying.** Open the
    multi-canvas run from item 174 in Analytics, or generate a fresh
    one. The curve should now start near zero in each canvas and climb,
    where it used to leap to about 90% within one frame and then
    sawtooth. Beside the **Convergence** heading there should be a
    small **?** icon; hovering it explains that this model has no mask
    token and that the curve counts positions holding what their canvas
    committed. A LLaDA run's heading carries no icon at all.
    If you have a run old enough to predate token records, its icon
    should be tinted amber and say the reading is approximate. That
    tint is the point: it means a reader who never hovers can still
    see that the curve is the weak one.
    The interesting part is comparing it against what you remember. The
    old curve said the canvas was nearly done almost immediately; the
    new one says it takes most of the canvas's steps to settle, which
    is what the text on screen was doing all along.
177. **The LLaDA curve is untouched.** The half that would be easy to
    break. Open any LLaDA run and confirm its convergence chart looks
    exactly as it did, climbing smoothly from zero, with **no** caption
    above it. A caption there, or a curve that now starts partway up,
    means the settlement measure has been applied where the mask flag
    was already ground truth.
    A run edited and resumed is the sharpest case, since that is where
    the wrong measure would open near 18% instead of 0%.
178. **Throughput still agrees with the generator.** Convergence moved
    and throughput deliberately did not, so this is the check that they
    were separated properly. Generate a multi-canvas DiffusionGemma run,
    note the **T/s** figure in the generator footer when it finishes,
    then open the run in Analytics and page the timing chart across to
    **Tokens per Second**. The two should still land close together, as
    they did in item 174.
179. **The step readout follows the scrubber.** Once per model, since
    the two report differently. Finish a run, then drag the scrubber
    from end to start and watch the left of the status bar. On LLaDA
    and SmolLM3 it should count down as `Step N/M`; on DiffusionGemma
    it should read `Step N, Canvas C` with the canvas number changing
    as you cross a boundary. Before this it sat frozen on whatever the
    run ended at.
    Then go to Analytics and come back, and scrub again. The readout
    has to keep working, which is the part that needs the step total to
    have survived the trip.
    **The denominator has to describe the run, not the last branch.**
    Note what a fresh LLaDA run reads at some frame, say `Step 64/128`.
    Then edit at a frame, resume to the end, and scrub back: the total
    must still be 128, not the number of steps the branch ran. Retry
    out of an edit and check the same frame again, and edit at a
    second frame and retry before rechecking the first. All of them
    should read 128. Found on 2026-08-28 during item 182, where a
    branch from frame 28 left the run reading `Step 64/100`. The live
    line during a resume is different on purpose and should still
    count the branch: `Resuming 12/64` is right while it runs.

## Intervention checkpoints (XAI-01)

The sampler work is tested without a model, so what is left here is
everything a user would actually see. Items 180 to 182 confirm fixes.
Item 183 is different: it tests a prediction, and a null result is
information rather than a bug report.

180. **An edited LLaDA branch reports real confidence.** Generate a
    LLaDA run, enter **Edit Frames**, remask a few positions somewhere
    in the middle, and resume. Then turn on the **Heatmap** overlay and
    scrub to the first frame of the resumed branch.
    Before this, every token the edit did not touch was drawn at full
    confidence, so the prefix was a flat wall of bright green and the
    mean confidence under it read near 1.00. It should now be varied,
    the same mixture of tones the original run showed at those
    positions, and the mean should be a plausible number well under 1.
    The positions you remasked are the control: they should show as
    masks with no confidence at all until a step reveals them again.
181. **A resumed DiffusionGemma canvas does not flash empty.** Load
    DiffusionGemma, generate a single-canvas run (256 tokens or fewer,
    since resume is single-canvas), edit a few positions and resume,
    watching the first resumed frame closely.
    Previously it rendered as an entirely masked canvas for one frame
    before snapping back, and the tokens then climbed from zero
    confidence even though they had been settled for many steps. It
    should now show the canvas you branched from, with the positions
    you remasked as the only unsettled ones.
182. **One edit repeats.** The audit's own verification, and the reason
    the random state is retained at all. Generate a LLaDA run, enter
    **Edit Frames**, remask a few positions at a chosen frame and
    **Resume to End**. Note the output. Click the blue **Retry**, go
    back to the *same* frame, remask the *same* positions, and resume
    again.
    The second branch should match the first, character for character.
    An earlier attempt at this item asked you to generate a second run
    in between; that cannot work, because starting a generation
    retires the previous run's retained state and the first run then
    cannot be edited at all.
    Two things had to be true for this to pass, and each fails
    differently. The random state is one: without it the resume drew
    from wherever the process happened to be, so the branch moved
    depending on what you had done since. The rewind is the other:
    a completed resume replaces the worker's retained frames with the
    branch it produced, and until Retry told the worker to undo that,
    the second attempt re-entered a frame from the *first branch*
    while you were looking at the original run. That one is worth
    provoking deliberately, so if you have the patience: edit at a
    late frame, Retry, then edit at an *earlier* frame and Retry
    again, then repeat the late edit. It should still match.
    Worth knowing what this does not claim: a resumed branch is not
    expected to match the *original* run frame for frame, because the
    resume re-enters the whole generation region as one block rather
    than the original block schedule. Two resumes agreeing is the
    property under test.
    The same repeat works on single-canvas DiffusionGemma, where the
    stability state comes back with the canvas.
183. **Does the DiffusionGemma canvas brighten toward a boundary?**
    Prediction, not a fix. Generate a multi-canvas DiffusionGemma run
    and watch the masked positions rather than the text. (This item
    predates the removal of the **Entropy signal** toggle and used to
    ask for it to be switched on; there is nothing to switch now.)
    Each mask is faded by the model's certainty in the guess behind
    it. The predicted behaviour is that
    a canvas brightens roughly together as it approaches its adaptive
    stop, then resets dim when the next canvas begins.
    If it does not brighten, say so rather than treating it as a
    failure. It would mean the confidence means something other than
    the candidate-reveal and adaptive-stopping items in the ROADMAP
    assume, and those want rethinking before they are built.
184. **Masks grade themselves while the run is being written.** Found
    on 2026-08-28 while looking at 183: the grading only ever ran on
    the scrubber, so a live canvas was flat and the brightness
    appeared on rewind. Watch a **LLaDA** run stream and look at the
    masks rather than the text. They should start solid and brighten
    individually as each nears its reveal, which is the "heating up"
    the README describes, instead of sitting at one shade until the
    run ends. Scrub back afterwards and it should look the same as
    it did live.
    Then a **DiffusionGemma** run, where the same grading should be
    visible during generation. This is what item 183 was trying to
    look at, so the two are best done together: with the live view
    graded, the question of whether a canvas brightens together
    toward its adaptive stop can be watched as it happens rather
    than reconstructed by scrubbing. (This item also predates the
    toggle's removal; the "with the signal off it should stay flat"
    contrast it used to ask for is no longer reachable.)
    **Watch the frame rate.** This grades every position on every
    frame, up to 256 on a DiffusionGemma canvas. The renderer only
    writes to the page where a value actually moved, so it should be
    unnoticeable, but the sandbox cannot measure that. If streaming
    feels choppier than you remember, say so: the change is one
    property on one object and is cheap to take back out.

## Owning a download (TRUST-04)

A download is now a child process rather than threads inside the
supervisor. The sandbox can prove the supervisor asks it to stop; it
cannot prove the operating system obliges, or that the app closes
inside its 35 seconds while a real transfer is running. That is what
these are for. All of them need a model whose weights are **not**
cached, and a slow enough connection that you have time to act.

185. **Cancel stops a fetch, and keeps what arrived.** From the Main
    Menu, click an uncached model to start its download. Let the bar
    reach somewhere clearly partway, note the percentage, then press
    **Cancel** beside it. The row should go back to "Click to
    Download".
    Now click it again. It should resume from roughly where you
    stopped rather than from zero, which is the whole reason nothing
    deletes the partial parts. If it restarts at 0%, the cache was
    cleaned when it should not have been.
    Worth confirming with `ls` if you want to be certain: the blobs
    directory under your Hugging Face cache should still hold
    `*.incomplete` files after the cancel.
    **The row should say how to resume.** Confirmed on 2026-08-28
    that the functional half works and this half did not: the row
    kept the frozen percentage under a Cancel button that no longer
    had anything to cancel, and the only way to continue was to
    click the veneer somewhere the button was not. It should now
    read **Click to Resume Download**, with no bar and no Cancel.
    Then reload the menu, or close and reopen the app. The prompt
    must still say resume, since the server reports the partial
    cache rather than the page remembering it, and the row should
    look exactly as it did before the reload. That sameness is the
    check: a frozen bar was tried first and dropped, because it
    existed only in the window that cancelled and only until a
    reload, so one state rendered two ways depending on where you
    were standing.
186. **A cancelled download really stops.** The half a test cannot
    reach, because this sandbox refuses to signal a process in its
    own session. Start a download, and while it runs check that a
    child exists: `pgrep -af src.inference.download_main`. Press
    **Cancel**, then run it again. The process should be gone.
    If it is still there, the supervisor asked and the child did not
    listen, which is exactly what the escalation to SIGKILL is for
    and worth reporting with the log.
187. **Closing the app takes the download with it.** The finding's
    actual motivation. Start a download from the desktop app, let it
    run, then close the window while it is still going.
    The app should exit promptly rather than hanging near its
    35-second shutdown bound, and `pgrep -af src.inference.download_main`
    afterwards should find nothing. Before this, the fetch kept
    running with nothing able to reach it.
    Also worth trying the harsher version once: start a download and
    `kill -9` the supervisor. The child should still go, because it
    is spawned with the same parent-death signal a model worker gets.
188. **Two clients, one download.** Note that a second `desktop.py`
    will not open: `LIFE-05` single-instances it, and running the
    command again prints "already running... focusing that window
    instead", which is correct rather than a failure. An earlier
    version of this item asked for two desktop windows and could
    not be performed at all.
    So: leave the desktop window open and point a browser at the
    address it printed, `http://127.0.0.1:<port>`. Two clients, one
    supervisor, which is the case the operation number exists for.
    Start a download from one and confirm the other shows the same
    progress, through the toast or its own row.
    Now cancel from the client that did **not** start it. It should
    either stop the download cleanly or refuse with a sentence
    naming the other window; what it must not do is silently
    nothing. Then confirm the client that did start it notices,
    rather than sitting on a bar that has stopped moving.
    **Both rows should then read the same.** Whichever client
    cancelled, the other must also offer **Click to Resume
    Download**, without being reloaded. Found on 2026-08-28 saying
    "Click to Download" instead, because a page's model flags came
    from its own load and nothing told it the cache had changed
    since. A download ending now re-reads them in every window.
    Worth trying the success case for the same reason: let a
    download finish in one client and confirm the other stops
    offering to download that model, rather than waiting for a
    reload to notice it is already there.
189. **The progress bar still moves.** Cheap to overlook, and easy to
    break: progress is now measured by the supervisor watching the
    cache directory while a different process does the writing. Just
    confirm the percentage climbs smoothly rather than sitting at 0%
    and jumping to 100% at the end.
    If it sits at zero, the child is fetching without leaving
    measurable `*.incomplete` parts, which would mean the Xet
    downloader was not disabled in the child the way it is in the
    supervisor.

## Revealing the mask candidate

A Settings toggle draws the token a diffusion model is currently
holding at each unsettled position, instead of `░`. The sandbox can
prove the guess is recorded and that the renderer substitutes it. It
cannot answer the question the feature actually asks, which is
whether a canvas full of plausible-looking words is informative or
just noise. That is what these are for, and a negative answer is a
real result: the setting is off by default and can stay that way, or
the default can move, on the strength of what you see.

190. **A LLaDA canvas names what it is holding.** Turn on
    **Settings > Appearance > Reveal the mask candidate**, Save, and
    generate a LLaDA run. The canvas should fill with dim tinted
    words from the first step rather than with blocks, each word
    firming up as its position grows confident and then turning the
    resolved color when it commits.
    Three specifics worth confirming, because each is a separate
    decision in the code. The first frame should still be all
    blocks, since the model has not looked at the canvas yet. A
    revealed word should stay visibly *dimmer and differently
    colored* than a settled one at all times, which is what stops a
    guess from reading as an answer. And scrubbing back over the
    finished run should look the same as it did live.
    Then the reading question: does watching the draft rewrite
    itself tell you more than watching blocks resolve, or does it
    read as noise? Both answers are useful. Say which.
191. **A DiffusionGemma canvas, and the placeholder finding.** Same
    setting, a multi-canvas DiffusionGemma run. The words are faded
    by confidence as well, which this item used to ask you to enable
    and which is now unconditional.
    The specific thing to look for is the one the ROADMAP and the
    ledger describe in a paragraph each: early in a canvas the
    display should fill with `" the"` and other high-frequency
    filler, which is then eaten by real content as the canvas
    settles. That is exactly why the mask-flag convergence curve
    overstates progress for this model, and the reveal should make
    it self-evident rather than something to take on faith. If the
    canvas does *not* fill with filler, that is worth reporting,
    because the explanation currently in two documents would be
    wrong.
192. **Saved runs answer to the setting.** Open a DiffusionGemma run
    in Analytics, scrub to a mid-run frame, and toggle the setting
    (Save, then reload Analytics, since both pages read preferences
    at load). The same saved frame should switch between blocks and
    words.
    Then a LLaDA run saved **before** today: it should show blocks
    either way, and that is correct rather than a bug. LLaDA wrote
    the glyph into the record, so there is nothing else stored to
    show; only runs generated from now on carry the guess. A LLaDA
    run generated today should behave like the DiffusionGemma one.
193. **The reveal does not leak into the places it should not.**
    Three views, quickly, with the setting on and a diffusion run
    that has been edited.
    In **Edit Frames**, click a resolved token to select it for
    remasking: it should turn into a block, not stay a word. The
    selection is a statement about what the next run will redraw,
    and drawing the old word there would undo the point.
    In **Edit Another Frame**, the faded preview of the frame you
    are about to regenerate should honor the setting too, since it
    is the thing you compare a branch against.
    Hovering a masked position should still report `░` and the
    position's confidence in the metrics strip. The strip says what
    the canvas *is*; the reveal is a reading of it. If that reads as
    a contradiction rather than as a distinction, say so, because
    the strip could follow the setting instead.

## A mask that shows its confidence (retuned ramp)

The grading was already running and spending almost none of the
alpha channel: measured on a saved LLaDA run, a whole frame's masks
sat between 0.48 and 0.65, which is not a gradient anyone can see.
The curve is now a square root over a 0.05 floor, chosen against the
measured confidence distribution rather than by eye, and it is shared
by both pages for the first time. The sandbox can prove the arithmetic
and the wiring; only a screen can say whether the result is legible.

194. **A LLaDA canvas grades visibly now.** Generate a LLaDA run and
    watch the masks rather than the text, first with **Reveal the
    mask candidate** off so the blocks are the only thing moving.
    Early frames should look genuinely faint and uneven, with
    individual positions firming up as they approach their reveal,
    instead of the flat wash of green this replaced. Scrub back
    afterwards: it should look the same as it did live.
    Then the judgement the numbers cannot make. Is the low end too
    faint to be useful, or is a barely-there mask the honest drawing
    of a position the model has no opinion about? If a typical frame
    now reads as empty rather than as uncertain, say so: the floor
    and the curve are two constants in one function and easy to move.
195. **Frame 0 is solid, and frame 1 is not.** The one visible step
    this introduces, called out so it is not mistaken for a flicker.
    An unmeasured position now draws solid rather than dim, and
    LLaDA's opening frame carries no confidence at all, so a run
    should open on a full field of solid blocks and drop into the
    ramp on the first step.
    That change is what keeps a run saved before the measurement
    existed from rendering as a near-blank canvas. Open an older
    diffusion run in Analytics: its masks should be solid and
    uniform. A canvas that looks empty there is a real bug, not a
    taste question.
    This item used to name a live second case, a DiffusionGemma run
    with the Entropy Signal off. That toggle is gone, so the only
    unmeasured positions left are LLaDA's opening frame and runs
    already on disk.
196. **Analytics agrees with the generator on the same run.** The
    asymmetry that prompted this: Analytics never graded its masks
    at all, so a run looked different depending on which page you
    opened it from, even though the confidence was in the saved file
    the whole time.
    Take one saved diffusion run. Scrub it on the generator, note a
    frame, then open the same run and frame in Analytics. The masks
    should be identically faded. Check both comparison views on an
    edited run too, the **Original / Edited** crossfade and **Diff
    vs Original**, on both pages: the diff overlay was flat
    everywhere before this and is the likeliest place for a gap to
    survive.
197. **Confirm and Retry survive a scrub.** Run a guided edit through
    to the end. On the last frame the green check and the blue retry
    appear as before. Now scrub back several frames: both should
    stay, and the status line should name the frame you are on
    rather than telling you to go back to the end.
    Then use them from there, which is the half that matters.
    Confirming from a mid-run frame should save the whole run and
    leave the scrubber on the last frame; retrying should discard
    the branch and drop you at the first editable frame. Neither
    reads the scrubber, so a save that came out short would mean
    something deeper than this change.
198. **A large edit no longer widens the tooltip.** In Analytics,
    open an edited run with many remasked positions (the 44-token
    edit that surfaced this is ideal) and hover the resume point on
    the **Convergence** chart. It should read over two lines,
    `User remasked 44 tokens:` then
    `[2, 3, 4, 5, 6, ... and 39 others]`, and the box should sit
    inside the chart rather than running off its side.
    Truncating alone was not enough and the split is why: the
    tooltip is drawn onto the chart canvas, so it cannot exceed a
    380px column, which is about 57 characters at this font. Five
    positions on one line came to 60 and clipped by three. Two lines
    put the longest at 34.
    The placement is still open. A list whose length depends on the
    data does not really belong in a fixed-width box drawn inside a
    chart, and the other two charts say only
    `Resume point (44 tokens remasked)`. Moving the list to a
    reserved row under the chart is scoped in the ROADMAP; this item
    covers only that the current box no longer clips.

## Confidence without a switch (DiffusionGemma)

The Entropy Signal toggle is gone and DiffusionGemma measures every
position on every step. The sandbox proved the reduction returns the
same numbers as the softmax it replaced, and that the parameter and
its plumbing are gone. What it cannot touch is the GPU: whether the
chunked reduction is actually cheaper in practice, and whether a
model that now always pays for it still runs at the speed you
remember.

199. **The parameter is gone and nothing misses it.** Load
    DiffusionGemma and look at the hyperparameter panel. There
    should be no **Entropy signal** control, and the panel should
    lay out cleanly rather than leaving a gap where it was.
    Generate a short run. Every masked position should be faded by
    confidence with nothing switched on, which on an older build
    required the toggle.
    Then open an older DiffusionGemma run in Analytics, one saved
    with the signal off. It should still open, still show its
    parameters including the recorded `entropy_signal: false`, and
    still draw solid masks. The parameter row survives because the
    detail panel renders whatever a run recorded; only the control
    went away.
200. **The run is not slower, and preferably faster.** This is the
    claim the whole change rests on and the one the sandbox cannot
    make. The old path built a float32 softmax over the full
    vocabulary for every step, roughly a quarter of a gigabyte held
    at once; the new one reduces 32 positions at a time.
    Generate a DiffusionGemma run and compare the elapsed time and
    T/s against a recent run of the same prompt and step count from
    before this change. It was previously only paid when the toggle
    was on, so compare against a **signal-on** run rather than a
    signal-off one, or the comparison is against a run that was not
    measuring anything.
    Expect it to be the same or slightly better. Noticeably slower
    would mean the chunk size is wrong for this hardware, which is
    one constant.
    Worth a glance at VRAM headroom in the header while it runs, if
    the difference is visible there at all.
201. **A resumed edit still re-enters its canvas.** The resume
    checkpoint lost a field, so this is the path most likely to have
    been broken by an invisible mistake.
    Run DiffusionGemma, **Edit Frames**, remask a few tokens on a
    mid-run frame, and resume to the end. The first resumed frame
    should show the inherited canvas rather than a wall of blocks,
    and the tokens you did not touch should not flash as newly born.
    Confirm and reopen the run in Analytics to check the branch
    saved intact.

## Two windows filing at once (DATA-02)

Collections are now the server's: the page sends one operation per
gesture and takes back the list, instead of writing its own copy of
the array. The sandbox races sixteen threads and eight forked
processes through that path and proves nothing is lost, and it proves
the old read-then-write shape fails the same test. What it cannot do
is drive two real browsers, which is the case the finding was written
about, so these are the ones only you can run.

202. **Two windows, two different runs, both survive.** Open
    Analytics in two browser windows side by side. In the first,
    star run A. In the second, which has not been reloaded and so
    still shows its older list, star a different run B.
    Reload either window. **Both** stars should be filled. Under the
    old write path the second window would have written a list
    computed before A existed, and A would be gone with no error
    anywhere.
    Then the sharper version: with both windows still open and
    neither reloaded, star two more runs in quick succession, one in
    each. Both should survive too, since each request carries only
    the gesture.
203. **The browser and the desktop app, against one directory.** The
    case a thread lock cannot cover, and the reason there is a file
    lock: these are two supervisor processes.
    Launch `desktop.py` and also open Analytics in a browser at the
    same address. File a run into a collection in each. Reload both.
    Neither filing should be missing.
204. **Refresh catches a window up.** Two windows again. File a run
    in the first. In the second, without reloading, press
    **Refresh**: the new membership should appear.
    This is the whole of the staleness fix, so it is worth knowing
    what it does not do. A window that just sits there stays behind
    until it acts; nothing polls or pushes.
205. **A refusal says so, and changes nothing.** Make collections up
    to the cap of 24, then try to make one more. The refusal should
    name the limit rather than failing silently, and no empty
    collection should appear.
    Then the useful negative: stop the server, and with the page
    still open click a star. It should report that the change could
    not be saved, and the star should **not** appear filled. The old
    path wrote the browser's copy first, so it looked saved and then
    vanished the next time anything hydrated.
206. **Filing from the caret still works in one step.** Hover a row,
    open the caret, and type a name into the new-collection field.
    The collection should be created with that run already in it,
    and the checkbox list should show it ticked. Those are one
    request now, so a half-made collection with nothing in it would
    be the failure to look for.

## Collections polish: overflow, copy, and filing in bulk

Three fixes that fell out of verifying `DATA-02`, and the two
conveniences they exposed. The fixes are cheap to check and the
filing paths are the ones worth spending time on, since the whole
point of doing them as one request is a failure mode that only shows
up when something goes wrong partway.

207. **The strip scrolls instead of taking the page with it.** Make
    collections up to the cap of 24, or enough to overflow the
    toolbar. The tab strip should scroll sideways inside the
    toolbar, keeping **Group by** on its left and leaving the runs
    table where it was. Before this it laid out to roughly 1,800px
    and dragged the whole page sideways, so the table scrolled off
    to the left. Check the toolbar's height does not change either:
    wrapping to a second row was the alternative and was rejected
    for exactly that.
208. **A refusal reads like the app, not like the API.** At the cap,
    try to make one more collection. The message should say the
    limit is reached, not `at most 24 collections`, which is the
    server's own wording and was leaking into the toast.
    Try the other reachable one too: open a row's caret, leave the
    name field blank, and press Enter. It should ask for a name.
209. **An empty collection is deleted without asking.** Make a
    collection and delete it straight away with the cross on its
    tab. It should just go. Then file a run into another one and
    delete that: it should still ask first, and still leave the run
    in `results/` and under All. The confirmation was never the
    point for a collection holding nothing, and one that always
    fires is one people stop reading.
210. **Filing several runs into Favorites.** Under **All**, tick
    three or four rows. A star and a caret should appear beside the
    bulk trashcan, with the selection count on the star. Click the
    star: all of them should be filed into Favorites, a toast should
    say how many, and the Favorites tab count should jump by that
    many in one step, not tick up one at a time.
    Then the case worth the trouble: with some of a selection
    already in Favorites, star it again. It should add only the
    rest, and say so.
211. **Choosing a target for a selection.** Tick several rows and
    click the caret beside the star. The dialog should list
    collections as **targets** rather than checkboxes, each saying
    how many of the selection it already holds, with one that holds
    all of them greyed out. Click a target: the runs are filed and
    the dialog closes. Naming a new collection there should create
    it with the whole selection already in it.
    The failure to look for is a partial file. It is one request, so
    a collection that gained four of six runs would mean the
    batching is not doing what it exists to do.
212. **Filing from inside a collection.** Open a collection, then
    turn on **Show all runs** beside the tab strip. Every run should
    appear with the tab still selected, and the ones already in this
    collection dimmed. Tick a few of the undimmed ones and click the
    bulk star: they should be filed into **this** collection, not
    into Favorites, and the star's tooltip should have said so
    before you clicked.
    Then switch to another tab and back. Show all should be off
    again, and the collection should show only its members.
213. **An empty collection points at the toggle.** Make a fresh
    collection and open it. The message should tell you to turn on
    Show all runs and file from there, rather than sending you to
    the All tab, which used to be the only way in and meant leaving
    the collection you were trying to fill.

## The window that went white (desktop renderer recovery)

Chromium runs the page in a process of its own, and when that process
dies QtWebEngine leaves the view blank: no error, no event the page
can see, nothing in any log. pywebview does not connect the signal
that reports it, which is why the window sat white until the app was
restarted.

Reported after the machine idles and the screen blanks or locks,
which fits a GPU context lost to suspend that the renderer does not
survive. That cannot be staged in a test and the agent sandbox has no
display and no GPU, so everything above the signal is covered by
`tests/test_desktop_renderer_watch.py` and everything below it is
here.

The first attempt at this **aborted the app at launch**, which is why
214 starts by checking that it opens at all. The watch was being
installed from pywebview's post-start worker thread, and Qt objects
belong to the thread that made them, so reaching for the view's page
from another one killed the process with "Cannot create children for
a parent that is in a different thread". It is now installed inside a
wrapper around the backend's constructor, which runs on the GUI
thread. If anything like that recurs, launching with
`LLM_VISUALIZER_NO_RENDERER_WATCH=1` skips the watch entirely and
should restore a plain working window.

214. **A crashed renderer brings itself back.** Launch
    `.venv/bin/python desktop.py` from a terminal, so stderr is
    visible. **The window should open normally**, which is the first
    thing to confirm: the previous attempt aborted here with a
    `Trace/breakpoint trap (core dumped)` before showing anything.
    Then find the renderer with `pgrep -af QtWebEngineProcess` and
    kill the one whose command line carries `--type=renderer`:

    ```bash
    pkill -f "QtWebEngineProcess.*--type=renderer"
    ```

    The window should blank for an instant and come back on its own,
    and the terminal should print a `[desktop] renderer ...` line
    naming the status and exit code. Before this it stayed white.
    Do it four times in a row: the fourth should **not** reload, and
    should say it is giving up. That cap is deliberate, so that a
    renderer dying in a loop cannot spin forever unnoticed.
215. **The crash is written where a desktop launch can be read.**
    After the above, look at
    `~/.local/share/llm-xai-visualizer/renderer-crashes.log`. There
    should be one timestamped line per kill. This file is the point
    of the whole exercise: launched from the app icon there is no
    terminal, so without it the only evidence is a white window and
    a memory of roughly when.
216. **The original scenario, which is the real test.** Leave the
    app open and let the machine idle until the screen blanks or
    locks, at least twenty minutes. Come back and wake it. The
    window should either be intact or have reloaded itself; if it is
    white, the log will say whether the renderer died and how, which
    is the evidence that was missing before.
    If the log is **empty** and the window is still white, the
    renderer did not die and the cause is elsewhere. That is a
    genuinely useful negative: say so, because it rules out the
    hypothesis this work was built on.

## Autoregressive frames that append (RUNTIME-01, stage one)

SmolLM3 frames now carry the one position they added rather than the
whole sequence so far, the browser keeps one flat list, and the
server rebuilds the per-frame arrays on save. The stored files are
byte-identical either way, pinned by
`tests/web/test_append_expansion.py`, so **nothing about a saved run
should look different**. That is what these items check: the change
is supposed to be invisible except in how much it costs.

The automated half covers reconstruction, the desync guard and the
byte identity. What it cannot cover is a real model at a real length
on real hardware, which is where the quota failure showed up in the
first place.

217. **A short run reads as it always did.** Generate 128 tokens with
    SmolLM3. Scrub across the whole run: the text should grow by one
    token per frame with no gaps or repeats, hover should give
    candidates, and the entropy profile should be populated. This is
    the boring one and it is the one most likely to catch a
    reconstruction that is off by one.
218. **The failure that started this.** Generate 2,048 tokens with
    SmolLM3, let it finish, then navigate to Analytics and back.
    Before this change the run came back with no hover, no
    candidates and no entropy profile, because the session snapshot
    exceeded the storage quota and fell back to a light payload. It
    should now come back whole.
    Worth timing while you are there: saving used to take 30 to 45
    seconds with visible stutter and Analytics took about 10 seconds
    to paint. Both should be markedly faster, though the stored file
    is still the same size, so Analytics will not be *fast* until
    stage two.
219. **An edited run still compares.** Take a finished SmolLM3 run,
    use What If to substitute a token partway through, let the
    branch run out, and save. The Original/Edited crossfade, the
    diff overlay and both elapsed series should read as before. The
    baseline is stored flat now too, so this is where a mistake in
    the pre-edit copy would show.
220. **The substitution splice lands where it should.** During that
    edit, watch the canvas at the moment the branch starts. The
    forced token should appear at the position you chose, with the
    text before it unchanged. The worker now sends only the forced
    position rather than restating the kept prefix, so an off-by-one
    here would show as the branch starting one token early or late.
221. **A saved run opens unchanged in Analytics.** Open a run saved
    before this change and one saved after, side by side. Both
    should render identically: same token overlay, same charts, same
    commit order. The point of expanding server-side is that
    Analytics cannot tell which is which, and this is the only place
    that claim is checked against a run that predates the change.
