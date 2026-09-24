# Guide

How to use the LLM Visualizer, feature by feature, and the mechanics
behind what you are looking at.

This is the long form. [README.md](../README.md) is the front page and
gets you running; this file is what each control does and why it reads
the way it does. For where the project is going and why lines were
drawn, see [ROADMAP.md](ROADMAP.md); for a cold start on the
architecture, [HANDOFF.md](HANDOFF.md).

The in-app **Help** modal covers the same ground more briefly, and is
the right thing to reach for while you are actually using the app.

## Contents

- [Building the DiffusionGemma checkpoint](#building-the-diffusiongemma-checkpoint)
- [Launching and choosing a model](#launching-and-choosing-a-model)
- [Running a generation](#running-a-generation)
- [Reading a run](#reading-a-run)
- [Intervening in a run](#intervening-in-a-run)
- [Analytics and saving](#analytics-and-saving)
- [How the models generate](#how-the-models-generate)
- [Sampling parameters](#sampling-parameters)


## Building the DiffusionGemma checkpoint

The one model with a setup step beyond `pip install`, which is why
[README.md](../README.md) points here rather than carrying it.

DiffusionGemma is gated on Hugging Face. Accept its license, then
download the bf16 base at a specific commit so the artifact you build
can name what it was built from:

```bash
.venv-dgemma/bin/huggingface-cli download \
    google/diffusiongemma-26B-A4B-it \
    --revision <commit> \
    --local-dir ~/models/diffusiongemma-26B-A4B-it-bf16
```

Then produce the local 4-bit checkpoint. Only the MoE experts are
quantized to NF4, which is what makes it fit in 24 GB:

```bash
.venv-dgemma/bin/python scripts/quantize_diffusiongemma_nf4.py \
    --base-revision <commit>
```

This writes the NF4 checkpoint to the path the registry references
(`~/models/diffusiongemma-26B-A4B-it-nf4`). The build happens in a
`.incomplete` sibling directory and is moved into place with a single
rename, so an interrupted run never leaves something that looks
installed. The finished directory carries an `artifact_manifest.json`
naming the base checkpoint and its revision, this repository's commit,
and the state dict's size and SHA-256; the app requires that manifest
before it treats the checkpoint as present.

If you built this checkpoint before manifests existed, attest it in
place rather than rebuilding it. This needs no GPU and no base
checkpoint:

```bash
.venv/bin/python scripts/quantize_diffusiongemma_nf4.py --adopt \
    --out ~/models/diffusiongemma-26B-A4B-it-nf4
```

## Launching and choosing a model

Everything between starting the server and having a model resident.

```bash
python3 main.py            # or: python3 main.py --port 8000
```

The app binds to `127.0.0.1`, so it is reachable only from this machine. You can serve it to your network with `--host 0.0.0.0`, and it will warn you when you do: there is no authentication, so anyone who can reach the port can load and unload models, save runs, and permanently delete them. Treat that as a trusted-network convenience rather than a supported deployment.

Saved runs and durable UI state go to this repository's `results/` no matter which directory you start from, so launching by absolute path, from a service, or from a desktop entry all reach the same data. Point somewhere else with `--results-dir /path/to/runs` (or the `DIFFUSION_LLM_RESULTS_DIR` environment variable, which the flag overrides); the resolved path is logged at startup and shown in full wherever the UI names it.

Open [http://localhost:8000](http://localhost:8000). You land on the **Main Menu** (titled **LLM Visualizer**): a looping title screen over a model picker that shows the detected GPU (with free VRAM) and CPU (with free RAM). Each row carries a family glyph (diffusion vs autoregressive) beside its name and a device tag on the right: a static **GPU** for the diffusion models, and a **GPU / CPU** toggle for the autoregressive model. A small pill extending left of each tag shows the signed **VRAM headroom** (green `+X.X GiB` if it fits, red `-X.X GiB` if it is short); hovering it details required vs available VRAM. Available counts memory reclaimed from unloading the current model, so switching accounts for the resident one freeing up.

Selecting a model contracts the menu to that row and asks you to **confirm** (green check to load, red X to go back); picking the model that is already loaded asks *Go back to the Generation page?* instead, since nothing needs loading and the run you left is still there. **On the first activation its weights download from Hugging Face** (LLaDA ~16 GB, SmolLM3 ~6 GB, cached under `~/.cache/huggingface`); a model whose weights are not cached yet is gated behind a **Click to Download** veneer with a progress bar so you can pre-fetch it (its description stays hidden until the download finishes and is then revealed, and a success/error message with **Ok** confirms the result). A row reading **Resident** is the model currently loaded in memory. Type a prompt, adjust parameters, and click **Generate** to watch generation stream live. To switch models (or the autoregressive model's device) later, use the **Model** selector in the generator header, which confirms the switch the same way; its collapsed tag tickers between the device and VRAM headroom (a "Device tag ticker" Setting toggles this off), and hovering a listed model shows its full VRAM readout (only one model is resident at a time).

## Running a generation

The prompt, the run itself, and everything the status bar tells you while it happens.

After a run completes, a **Save** button appears and a **frame scrubber** slides into view below the output area. While a save is in progress the status bar shows "Saving original run…" and the scrubber is dimmed and frozen until it finishes.

**Stopping a run.** While a run is in flight, **Generate** becomes **Stop**. The samplers check between decode steps, so a run ends within roughly one step of the click rather than having to reach its token budget. A stopped run keeps everything it produced: the frames stay on the scrubber, and Save, Edit Frames and What If? all still apply. What it does not do is claim it finished. The status line reads "Stopped.", and a saved stopped run is recorded as partial, showing its duration in Analytics as `12.3s (stopped)` with a **Completion** row in its detail panel, so a truncated answer cannot later be mistaken for one the model chose to end. Leaving the page mid-run stops it the same way, rather than leaving the model computing for a page that is no longer watching.

**Run readouts.** The status bar's left side carries the current **Step**, the **Elapsed** wall-clock time, and **T/s** (tokens per second). Both totals count the run as a whole, so an edit continues from where the run had reached rather than restarting the clock. T/s is the one interactive readout: it shows the run average by default, and clicking it (or focusing it and pressing Enter) switches to the last step alone, a noisier reading that tracks throughput as it changes. The choice is persisted and is not affected by **Reset** on the Settings page, since its control is the readout itself.

**Resource meter.** Beside them, a small bordered **sparkline** holding the last minute of what the machine is doing, sampled by the worker twice a second rather than per frame, so it keeps moving between runs as well as during them. It shows one resource, chosen by where the model was placed: on a GPU it is **VRAM**, the whole card's memory in use against its total, so it answers whether you are near the limit rather than what this run alone holds; on a model running on **CPU** it is instead how hard the worker is working as a share of every core, since there is no VRAM of its own to report and the useful question on a slow CPU run is whether it is progressing. Full height always means all of the resource, so the two read on one scale. It is absent rather than flat on a machine that can report neither figure, since an empty meter would read as an idle one, and it disappears while the connection to a model is down (which is what switching models looks like) rather than leaving a stale reading up, starting a fresh line when the new model is serving. It does not cover a model *loading*, because the page is not connected to a model until that model can answer. Nothing here is saved; what a finished run cost is recorded separately as **Peak VRAM** (see Analytics).

**Status row.** The status bar's right side separates what is *happening* from where the run *stands*. The rightmost slot holds the resting line, "Done.", "Stopped.", where a save landed, or the full error text, which is also the one persisted across a trip to Analytics and back. Work in flight appears to its left as short dot-separated messages that animate while their operation runs and then leave, newest nearest the resting line, so concurrent operations stay readable instead of overwriting each other. That is what makes entering **What If?** on an unsaved run legible: it saves the original in the background, and picking a candidate immediately leaves "Saving original run" and "Running edit from frame 81 to end" up side by side. Results are spelled out only in the resting line, so a message just disappears when its work succeeds rather than restating what is next to it. Messages name their subject as well as their verb: a save says whether it is writing the *original* or the *edited* run, and a resume names the stretch it regenerates ("from frame 12 to 40", or "to end"). Saved runs always report a path relative to the repository, however the save reached disk. Each message rises in from the bottom of the window and slips left on its way out, and the ones already up glide aside to make room rather than jumping, all of which reduces to plain fades if your system asks for reduced motion. The row stops short of the readouts on the left, fading its oldest message out against that edge, and the resting line never yields its own space to make room.

**Prompt history.** A small history icon at the top-right of the prompt box (shown once you have run at least one prompt) recalls earlier prompts, persisted per-browser. Click it to browse: `‹` steps to older prompts and `›` back to newer ones, the green check keeps the shown prompt for editing, and the red cross restores what you had typed. The counter numbers them in the order you typed them, oldest as 1, so browsing opens on your most recent prompt at `N / N` and the count rises as you move right, the way the arrows do. Prompts are recorded automatically each time you Generate.

**New Run.** Once you have finalized a run by saving an edit (see Interactive remasking below), the **Generate** button becomes **New Run** in the same spot. Clicking it clears the canvas and prompt for a fresh start (the prompt box shows its "Enter a prompt" placeholder) and restores the **Generate** button.

**Desktop window.** To run the UI in a native window instead of a browser (after the optional pywebview setup in Setup):

```bash
.venv/bin/python desktop.py
```

This owns the server lifecycle: it starts the supervisor on a private localhost port, opens the window, and gracefully stops the active model worker (freeing its VRAM) when you close it. The browser path (`main.py`) still works and serves the same app.

**Freeing stuck GPU memory.** A worker is normally stopped when you switch models or close the app, and the supervisor sweeps stray workers on startup (plus a `PR_SET_PDEATHSIG` guard). If a hard crash ever leaves one behind holding VRAM, list GPU processes with `nvidia-smi` and clear them with `pkill -f "src.backends.run_worker"`, then relaunch.

## Reading a run

The overlays, the metrics strip, and the per-token signals that make a finished run worth looking at.

### Visual overlays and settings

A collapsible **Overlay** drawer in the top-right of the output area recolors the frame you are viewing. Collapsed, its handle can be dragged up and down the right edge to move it clear of whatever the run has drawn there; the position is remembered per page. Dragged low, its picker opens upward so the choices stay inside the output area rather than being clipped by its border. It defaults to **None** and offers:

- **Heatmap:** recolor resolved tokens by confidence (dim, desaturated tones for low, bright green for high).
- **Entropy:** recolor resolved tokens by the entropy of the distribution they were sampled from, on a cool blue (decisive) to hot amber (torn) ramp. This answers a different question than the Heatmap: confidence is how likely the chosen token was, entropy is how spread the model's whole distribution was. Listed for any run that carries the signal (autoregressive runs today, where it is always captured).
- **Commit Order:** tint resolved tokens by the step at which they settled, from light green (early) to red-orange (late), with a matching gradient legend in the status bar (diffusion runs only).
- **Diff vs Original:** compare an edited run against the original. Diffusion runs list it up front (disabled until you have edited and resumed a run); autoregressive runs list it once a **What If?** substitution has produced a branch. When active, a slim control row below the scrubber provides independent **Original** / **Edited** opacity sliders and a **Difference blend** toggle, alongside a `Diverged N/total` readout.

Once a run has a branch, an **Original** / **Edited** crossfade appears below the scrubber, the same control the Analytics detail modal carries. It stacks the retained pre-edit run under the branch and mixes between them in *any* overlay, so the two can be faded against each other while reading a Heatmap or an Entropy view rather than only inside Diff. Each layer is colored from its own run's values (at full Original the Heatmap shows the original run's confidence, not the branch's colors under the original text), and the more opaque side takes the pointer, so hovering, the tooltip, and the candidate popover read the run on screen, with the popover opening on that run's page. The entropy profile follows the same slider. **Diff vs Original** keeps its own two sliders instead, because a difference blend needs both layers up at once rather than traded off. The crossfade hides while a run is being edited, where the tokens are a click target rather than something to read, and resets to **Edited** whenever a run completes.

Runs that captured **Alternatives** also get two XAI affordances. Hovering any token opens a **candidate popover** listing the top five tokens the model weighed at that position, with a proportional bar and probability each, and the one it actually chose marked. After a **What If?** branch, positions at or past the substitution get a small **Original** / **Edited** pager in the popover's heading, so you can flip between what the two runs were weighing at the same position; each page marks the token its own run drew, and only the Edited page is clickable while substitution is armed. Below the scrubber, an **entropy profile** draws one column per position, tall and hot where the model was torn, with the column for the frame under the scrubber highlighted and its value read out in nats. Columns past the scrubber dim to a fraction of their weight, so the chart agrees with the canvas above it about which tokens exist at the frame you are looking at rather than showing the whole run's shape at every step. Once a run has been edited, each edited position is marked with a dashed line and a faint tint, the same marker the Analytics entropy chart uses, so the place the branch was forced stays visible while scrubbing and crossfading, and the edited positions themselves keep a soft orange wash in the output for as long as the run is on screen. Each marker is colored by the **frame its edit was made at**, on the same green-to-warm scale as Commit Order, so a run remasked in several rounds shows the order of its interventions rather than one flat color and a marker's hue can be read against a token's; a position remasked twice takes the color of its most recent edit, and a run saved before this keeps the flat orange it always had. The markers sit below full strength, since they annotate the entropy bars rather than compete with them. That tint is deliberately quieter than the one an in-progress edit uses and means a different thing: the bright mark says "selected, about to be redrawn", the wash says "this run was intervened here", which stays true afterwards. It is a background, so it composes under the Heatmap and Entropy overlays instead of competing with them for the token's color. Because an autoregressive model samples each position exactly once, this is a profile across the sequence, not a trajectory of one position over time.

The profile and the tokens **cross-highlight** in both directions: hovering a token lights its column, and hovering a column lights the token it belongs to. A token lit from the profile looks identical to one under the cursor, since both mean "this token". A token that renders to nothing, a line break, gets a thin upright bar standing where it sits instead, because a highlight with no width to fill would leave a column plainly pointing at an empty space. Sweeping the profile lights tokens regardless of the **Highlight tokens** setting, because reading a column back to its word is an analysis affordance rather than a comfort preference.

After a **What If?** branch the profile carries both runs, the pre-edit columns underneath and the branch's on top, mixed by the run crossfade. It spans whichever run ran longer so the columns stay aligned by position even when the branch came out a different length, and the nats readout reports whichever run the crossfade favors.

**Highlight tokens** itself is a checkbox in the **Overlay** drawer, on the generator and in the Analytics detail modal, rather than a Settings row: it acts on the tokens the drawer sits over, applies immediately without a Save step, and is on by default. The value is still server-persisted and shared between the two pages.

The remaining persistent preferences live on a shared **Settings page** (`/settings.html`), reached from a **gear icon** in the header of the generator, the Main Menu, and Analytics. It has a left tab rail and stages changes behind **Save** / **Reset**; all settings are server-persisted and shared across pages:

- **Appearance** tab: **Render diffusion-style text** (dynamic status messages resolve from scrambled block-glyph noise, like a denoising pass, in the green palette, skipped automatically under `prefers-reduced-motion`; a **Mode** sub-setting picks **Default** to resolve once or **Cycle** to keep re-diffusing while the status is active, and the same effect drives small button interactions) and **Token birth glow** (each token flashes a soft white halo at full strength the instant it is generated and then fades, so a run leaves a visible trail; live generation only, never on the scrubber, and skipped under `prefers-reduced-motion`). The glow has three sub-settings: **Tune for** selects the model class, then **Brightness** (50-200%) and **Fade time** (200-2000ms) are stored per class, since visible trail length is roughly generation rate times fade and an autoregressive run outpaces a diffusion step by an order of magnitude. A **Preview** token replays the flash as you drag, because nothing generates on the Settings page. Last on the tab is **Reveal the mask candidate** (off by default), which draws the token a diffusion model is currently holding at each unsettled position instead of the `░` glyph, so the canvas reads as a draft firming up rather than as blocks dissolving. The position still reads as unsettled: it keeps the mask tint and the confidence fade, so a dim word is a guess the model is not committed to. It applies live, on the scrubber, in both comparison overlays, and retroactively to saved runs in Analytics.
- Sub-settings are indented under the preference they depend on, with no separator between them, and are dimmed rather than hidden when that preference is off, so what exists and what it belongs to stay visible.
- **Interface** tab: **Device tag ticker** (the scrolling GPU/device readout).

### Metrics strip

A single always-present row directly above the token canvas, on both the generator and the Analytics detail modal, reads out the hovered position: the token (with visible stand-ins for whitespace), `position / total`, confidence and entropy with a small bar each on the overlays' own ramps, the overlay-specific extra (`Resolved at step N` under Commit Order, `was: X` under Diff), and an `Original` / `Edited` tag while both runs are stacked. It replaced the native `title` tooltip, which the browser delayed, would not let us style or place, and could only ever be bound to one element, so it fed nothing from the entropy chart.

Two sources drive it: a token hover, and an entropy hover (the generator's profile, the Analytics chart), so a tall bar can be read back to a word without moving the pointer to the text. It also follows the frame, so a held position updates while scrubbing and during live generation. Absent is distinguished from zero: a dash means the run does not carry the value, rather than that the model measured nothing. Every model records entropy now; on the diffusion models it is the current step's, so it moves as you scrub. Height is reserved permanently rather than shown on hover, which would push the canvas down every time the pointer crossed into it.

### Confidence and the heatmap

Every resolved token carries a **confidence** value in [0, 1], and every frame carries the mean confidence of its resolved tokens. The source differs per model, cheap by default:

- **LLaDA:** the softmax probability of the token at the moment it was unmasked (fixed thereafter, since resolved tokens are never revisited).
- **DiffusionGemma:** the max-softmax probability from the model's logits, at every position on every step, settled or not. That is what lets each mask be faded by the model's certainty in the guess behind it. It used to be optional, behind an **Entropy signal** toggle, with a stability proxy (consecutive steps holding the same prediction) standing in when it was off; the toggle is gone, because the measurement now costs a bounded reduction rather than a canvas-sized softmax and a run without it was not a cheaper run, just one with a hole where the number goes. Runs recorded before that change keep whatever they recorded, and their unsettled positions draw solid rather than faded.

Hovering any token reads its position and confidence for that frame into the **metrics strip** above the canvas, and the **Heatmap** overlay recolors resolved tokens by confidence. The strip's left half holds that reading; its right half stays empty until you hover a row of the candidate popover, which fills it with that candidate's probability at full precision, headed by a green chip where the committed token's chip is grey. Per-frame mean confidence and canvas indices are also persisted for the analytics charts.

### Commit order and counterfactual diff

The frame history also drives two explainability overlays:

- **Commit order** colors each resolved token by *when* it settled into its final value, on a gradient from light green (early) to red-orange (late). This exposes the model's resolution trajectory across a run.
- **Diff vs Original** becomes available after you edit and resume a run. It compares the edited output against a snapshot of the original run, stacking the two with independent opacity sliders and an optional *difference blend* (matching tokens cancel to black, divergences glow), so you can see exactly how an intervention propagated.

Both overlays are derived from the recorded per-token frames. In the live generator they render across every frame; the underlying per-token data (display text, mask flag, vocab id, and confidence) is also persisted with each saved run, so both overlays are reviewable after the fact in the Analytics Suite (see below) rather than being lost on reload. The mask fade is part of that: a saved diffusion run replays with its masks graded exactly as they were live, in every view including both comparison overlays, since the confidence a mask fades by is recorded per position rather than computed for the moment.

## Intervening in a run

Remasking, resuming, and counterfactuals: the tools that let you change what the model did and compare the result.

### Interactive remasking and resume

The scrubber steps through every intermediate frame. Navigate with the slider, the arrow buttons, or the keyboard (Left / Right arrows, Home / End). The remasking and resume tools work for LLaDA and for single-canvas DiffusionGemma runs; on multi-canvas DiffusionGemma runs the **Edit Frames** button is disabled (multi-canvas resume is on the roadmap).

**Guided multi-frame editing.** Click **Edit Frames** to chain edits across one or more frames:

1. **Select a frame:** the scrubber starts at frame 0 and only allows forward navigation. Navigate to the frame you want to edit and click **Select Frame**.
2. **Remask tokens:** click resolved tokens to remask them (they turn orange). Click again to deselect. When satisfied, click **Lock In**. For LLaDA the tokens are set back to `[MASK]`; for DiffusionGemma they are *renoised*, so committed neighbours may also shift on resume.
3. **Choose next action:**
   - **Edit Another Frame:** enters target selection mode. A faded preview of the original run is shown at each frame as a reference, noting that output will diverge from your edits. Navigate to the target frame and click **Run to Here**; the model resumes up to that frame and places you into edit mode on it.
   - **Resume to End:** resumes the model through all remaining steps to produce the final output.

A single edit followed by **Resume to End** is the simple case; you can also chain as many edits as you like. Each partial resume generates only the frames between your last edit and the next target, so earlier edits propagate forward. The scrubber enforces forward-only navigation; later edits cannot precede earlier ones. Clicking **Exit** discards the in-progress edits and restores the original run. All remask edits (frame indices and token positions) are recorded and saved with the run.

**Confirming an edit.** Saving is yours to trigger: entering **Edit Frames** no longer writes a run behind your back, and the pre-edit snapshot travels with the edited run rather than needing a separate entry. After **Resume to End** completes, the editor stays open on the final frame and offers two choices in place of **Select Frame**: a green **Confirm** (checkmark), which saves the edited run, and a blue **Retry** (counter-clockwise arrow), which discards the edits and restarts editing from frame 0. Once an edited run has been saved, **Edit Frames** is disabled for that run (with a "this run already has a saved edit" tooltip) until you **Generate** again, so a single run cannot accrue two conflicting saved edits.

**An edited branch reports what the model actually said.** Resuming used to hand every token the edit did not touch a confidence of 1.0, so the Heatmap over an edited run showed a wall of certainty the model never expressed, and its mean confidence averaged those invented numbers. Each surviving position now keeps the probability it was really revealed at, and the positions you remasked carry none until a step reveals them again. The worker also retains the random state of the frame you branch from, so repeating the same edit on the same frame gives the same branch even if you generated something else in between. That is a claim about repeating an *edit*: a resumed branch is not expected to match the original run frame for frame, because it re-enters the generation region as a single block rather than the original block schedule.

**What If? (autoregressive counterfactuals).** Left-to-right models get a different intervention in place of Edit Frames. After a SmolLM3 run generated with **Alternatives** on, a **What If?** button appears beside the scrubber. Clicking it arms substitution: every position that captured candidates gets a dotted underline, hovering one opens the candidate popover, and clicking a candidate replaces the token there and regenerates the rest of the run from that point. There is no frame-selection step, because for a left-to-right model the frame and the position are the same choice. The continuation is decoded greedily so the divergence you see is the intervention's effect rather than fresh sampling noise. The result lands in the same **Confirm** / **Retry** review as a diffusion edit, after which **Diff vs Original** becomes available for the branch.

**Typing your own token.** Below the five candidates, an **Enter your own** field forces a token the model never weighed. Clicking into it slides a green check and a red cross out from behind its right edge; confirming turns the text into a row you click to run exactly like a candidate, with a small retry icon beside it. Escape, the cross, or a click outside all cancel, and while a draft is live the popover pins itself: it stops following the pointer and ignores scroll, so reaching for the buttons cannot destroy what you have written.

As you type, the field shows what the text actually resolves to: its **token pieces**, each with its vocabulary id, in alternating tints so a split inside a word is visible. This is where tokenization stops being an abstraction, since `unfortunately` is likely one token while `unfortunatelyy` is likely three. Confirm is enabled only at exactly one piece, and the count turns orange otherwise, because a replacement of any other length would shift every position after it and the diff, the entropy profile, and the edit marker all read positionally. The preview is a standalone encode against the loaded tokenizer, answered by the worker outside the generation lock; it is honest about context because substitution keeps the prefix ids verbatim rather than re-encoding the sequence, so no boundary effects can arise.

The field is pre-seeded with a leading space when the token being replaced carries one, read off that token rather than guessed from sentence position, since most tokenizers keep the space inside the word token and dropping it would weld the replacement onto the previous word. A single backspace removes it. Empty or holding only that seeded space, the field shows a placeholder that spells the space out as a `·` dot, so the automatic seeding is visible rather than mysterious.

A typed token reports its **true** probability, read from the model's distribution at that position (at no extra compute during a substitution: the prefill stops just short of the forced token, so the last position's logits are that distribution), which lets the readout honestly show that you forced something the model gave 0.003 to. Its entropy is unchanged, because entropy describes the distribution at that position and not the token pushed into it. A candidate picked from the five keeps the probability its own run recorded.

Confirming also **probes** that position: one forward pass measures the token's probability and its **rank** among everything the model could have said, and the solidified row shows the percentage (floored at `<0.1%`, since a typed token is most interesting where it is improbable and a bare `0.0%` would be a lie). The probe reads the same distribution the substitution will, so the odds shown before you run cannot contradict what the run then reports. Rank is what stays legible when the percentage has collapsed: `#41,203 of 128,256` says what a rounded zero cannot, and it costs a comparison and a sum on a distribution already in hand.

Typing a token the position already listed skips the probe entirely and reports the stored figure, which is better information and not merely cheaper. A run samples position *n* from a single decode step against an incrementally built cache, and a probe that rebuilds the same prefix as one fresh prefill lands about an ulp away in bf16, which is enough to show a candidate as 39.8% in one place and 38.3% in another. Where the run's own KV cache has been retained the probe reproduces the sampled figure exactly, because it makes the same call the run made rather than a reconstruction of it: a one-token decode against the cache sliced to everything before that position. The retained cache also removes the prefix prefill from a **What If?** substitution, which was that operation's dominant cost. It is sliced through fresh views rather than `crop`, which transformers implements in place and would consume the cache a later probe needs, and every disagreement between the cache and the prefix being asked about falls back to a fresh prefill rather than answering confidently from the wrong sequence.

**Every row shows its rank on hover**, not only the typed one. The captured five come out of `torch.topk` in descending order, so a row's position in the list *is* its rank, and the denominator is the model's output width rather than the tokenizer's vocabulary: those differ wherever a checkpoint pads its embedding (128,256 against 128,000 for SmolLM3), and a rank is a place among the tokens that could have been ranked. And when a position committed a token from *outside* the five, which a warm temperature does routinely, that token is appended as a sixth row carrying its own unrounded probability and explicit rank, ruled off from the list above it. It never displaces the fifth, because the five are a statement about what the model preferred and dropping one to make room would break it; and it is not a substitution target, since forcing the token already sitting there would spend a full regeneration arriving back where it started.

## Analytics and saving

What gets written to disk, and how the Analytics Suite reads it back.

### Analytics Suite

Click **Analytics** in the header (or navigate to `/analytics.html`) to open the Analytics Suite. It reads saved runs from `results/` and provides interactive charts for comparing behavior across configurations and models.

Every page works with no outbound network. The chart libraries (Chart.js, Hammer.js, the zoom plugin) and the JetBrains Mono webfont are vendored under `src/web/static/vendor/` with their licenses and a manifest of source URLs and SHA-256 hashes, rather than pulled from a CDN, so a blocked or absent network cannot take the run browser down with it and no third-party origin runs code alongside the model and deletion APIs. Refresh or bump them with `.venv/bin/python scripts/vendor_assets.py`.

- **Run browser:** group runs by date, model, prompt, or whether a run was edited. Columns are shared across models and ordered Date, Model, Prompt, Time, and a sortable **Edited** column (a checkmark, textured from the diffusion mask glyph, marks runs that carry a pre-edit snapshot for a Diff vs Original; blank otherwise). The leading Date column carries the pulsing green "new run" dot. Clicking a row opens a wide **detail modal** (fades in like About/Help; close with the X or by clicking outside) laid out with the token overlay canvas as the centerpiece on the left and the run's info plus the convergence, timing, confidence, and entropy charts stacked on the right. For an edited run saved with its pre-edit snapshot, the **Token overlay** heading row carries a run-level **Original** / **Edited** crossfade, sitting directly above the text it blends, that governs the token view, which stacks the two runs and blends between them in whichever overlay is active, and the entropy chart's bar layers follow the same slider. The timing and confidence charts follow it only while it is being dragged, since those two carry their own pins. The more opaque side takes the pointer, so hovering, the candidate popover, and cross-highlighting all read the run you are actually looking at.
- **Manage runs:** delete a saved run with the row's red trashcan action. Select rows with the checkboxes to enable **bulk delete** (a trashcan with the selected count appears in the actions header) and highlight the selected rows. Either path opens a confirmation modal ("Delete this run?" / "Delete N runs?") showing the folder path or count, and a toast confirms the deletion.
- **Convergence chart:** percentage of resolved tokens per frame, counted from the per-token records the run saved, so it measures positions rather than text. That distinction matters: an earlier version divided mask glyphs by decoded characters, which made a position resolving into a long token look like ten times the progress of one resolving into a short token, and two runs with the same schedule could disagree purely on word length. A run saved without those records (the older ones, and a few that stored bare token ids) still gets a curve from the character count, captioned above the chart to say so. User remask edits are highlighted as blue segments with hover details.

  Which record it counts depends on the model, because "resolved" does not mean the same thing for both. LLaDA masks a position with a real vocabulary token, so its mask flag is ground truth and the curve counts it directly. DiffusionGemma has no mask token: it renoises unsettled positions to fresh real tokens, and the sampler infers resolution from a position holding still. That is stability rather than settlement, and the two part company badly early in a canvas, where the model fills the whole thing with its highest-frequency token and that filler is perfectly stable. Measured on a real run, a canvas reading 90.2% resolved held only 8.6% of what it eventually committed, at a mean model confidence of 0.165. So for DiffusionGemma the curve counts positions already holding what their canvas committed, per canvas, which is exact and needs nothing the run did not already save. A **?** icon beside the heading explains which measure a run got, and appears only where there is something to explain: a LLaDA run's heading stays clean. For a run old enough to have no token records at all the curve falls back to counting mask characters, and that icon is tinted amber, so an approximate reading is visible as one without hovering for it.
- **Timing charts (two pages, one slot):** pager arrows beside the heading flip between **Elapsed Time**, cumulative elapsed per frame (accumulating across resumes, with remask transitions highlighted in green), and **Tokens per Second**, the same run read as a rate: tokens produced by each frame over the seconds taken to get there. The rate is a running average rather than a per-step reading, which on a diffusion run would mostly trace the sampler's reveal schedule; it needs no new stored data, so it works on runs saved long before the metric existed. The numerator counts each canvas against its own size and adds it to what came before, which is what makes a multi-canvas DiffusionGemma run read correctly: it previously subtracted from a single first-frame baseline, so committing one canvas and starting the next sent the curve back toward zero and lost a whole canvas of production. It also means the chart and the generator's live **T/s** footer now count the same thing, which they did not. The pre-edit comparison is offered for autoregressive runs only, since a saved run keeps the original's timings but not its canvas and a rate needs both. The run summary above the charts lists the processor and the elapsed total; an edited run lists two, **Elapsed (original)** and **Elapsed (edited)**, so the cost of the intervention is visible.
- **Compare runs:** tick two or more rows and press **Compare** to overlay their convergence curves. Every selection is accounted for: a run that cannot contribute a curve is named above the chart with the reason, whether it was deleted, unreadable, saved by a newer build, or autoregressive and so has no masked canvas to converge. It used to simply not appear, so three ticked runs could draw one line with nothing saying where the others went. Legend labels are built from each model's own parameters, so a DiffusionGemma or SmolLM3 run reads properly instead of showing LLaDA field names it does not have. A comparison is capped at twelve runs, and opening a new one supersedes the last, so a slow response cannot repaint a panel you have already closed.
- **Confidence chart:** mean per-token confidence per frame, which climbs as a canvas converges. Shown for runs saved with confidence data.
- **Two-run comparison on the line charts:** an edited run saved with its pre-edit snapshot draws both runs on the timing and confidence charts at once, the original solid in grey and the branch dashed in the chart's own color, so the point where the dashes leave the solid line is the cost or the confidence the intervention actually changed. The area **between** the two curves is washed in, colored by whichever run bounds it from above: the branch's own hue where the branch leads, the original's grey where it does not. Because the runs share their prefix exactly, the band is empty until the edit and opens up only where the intervention reached, and the rule reads the same on both charts without calling either direction good or bad (higher means slower on timing but better on confidence). Two **pins** in each chart header, **1** for the original and **2** for the branch, choose which are drawn and light green when showing; both are on when a run opens, and the last lit pin cannot be turned off since a chart drawing neither run has nothing to read. The band fades with whichever run is closer to invisible, since a band bounded by a line that is not drawn has no reading in it. Dragging the token view's Original / Edited crossfade borrows these two charts for the length of the drag so the whole modal moves together, then eases them back to their pins on release.
- **Entropy chart:** per-token entropy, indexed by **position** rather than by frame (and drawn as bars for that reason: an autoregressive model decides each position once, so entropy is a property of the position, not a point in a time series). One bar per generated token on the Entropy overlay's cool-blue to hot-amber ramp, hover lighting the column and naming the token alongside its value in nats. Bars and tokens **cross-highlight**: hovering a bar lights the matching token in the overlay above, and hovering a token lights its bar. Edited runs get a dashed marker and tint on each edited position, colored by the frame that edit was made at on the Commit Order scale, exactly as the generator's entropy profile marks them; from there rightward the tooltip splits into labeled **Original** and **Edited** rows (at the marked position itself the nats match and only the token differs, since forcing a token changes what was drawn, not the distribution it was drawn from), and the token view's crossfade blends the pre-edit run's bars against the branch's. Shown for runs saved with the entropy signal.
- **Canvas boundaries:** for multi-canvas DiffusionGemma runs, dashed amber markers on the charts mark where one canvas commits and the next begins. Single-canvas runs show none.
- **Token overlay + per-frame scrubber:** a scrubbable view of the run's tokens inside the detail modal, with a corner **Overlay** drawer mirroring the generator's. A frame scrubber (prev / slider / next, `Frame i / N`) replays every saved frame through the active overlay, opening on the final frame. The drawer offers **None** and **Heatmap** for every run with token records (Heatmap recolors resolved tokens by their persisted confidence), plus **Commit Order** and **Diff vs Original** for diffusion runs. Commit Order tints each token by when it settled (early-to-late gradient legend); Diff vs Original (available only for edited runs with a saved snapshot) stacks the original and edited runs with independent **Original** / **Edited** opacity sliders and a **Difference blend** toggle, plus a `Diverged N/total` readout, matching the generator's layered diff (the original layer clamps to its final frame past its end). Runs saved with entropy add the **Entropy** overlay, and runs saved with captured candidates get the same hover popover as the generator, so a What If branch and the decision behind it are both replayable post-hoc. Autoregressive runs, which have no masked canvas, omit Commit Order. Hovering a token shows its position, persisted confidence, and entropy where saved. This makes the generator's explainability overlays durable and scrubbable post-hoc; runs saved before durable overlays (or without token data) show a short unavailable note.
- **Chart controls:** scroll-wheel zoom and +/-/Reset on every chart. Tooltip boxes park in whichever corner of the plot area is free of both the data and the pointer (preferring top-left, then top-right, bottom-left, bottom-right) and stay fully inside the plot area rather than spilling onto the axes; each chart has a toggle to hide/show its box. When no corner is free, the covered segment and the hovered point glow through the box.

### Saving and reproducibility

Clicking **Save** writes a timestamped folder under `results/` containing `metadata.json`, `final.txt`, `frames.jsonl` (frame text, one JSON object per frame), `history.txt` (the same frames as a plain-text transcript, for reading by eye), `tokens.json` (per-frame, per-token records: display text, mask flag, vocab id, confidence, and entropy where captured), and `diffusion.gif`. Edited runs also write `original_tokens.json`, the pre-edit snapshot that powers the durable Diff vs Original overlay. Runs that captured competing candidates write `alternatives.json`, indexed by token position rather than by frame, since a position's candidate set is fixed the moment it is sampled.

A run is published whole or not at all: everything is written to a staging directory, checked against the run's own manifest, and moved into place with `metadata.json` last, since that file is what makes a directory a run. Each run carries a `schema_version` and a `capture` block naming the signals it recorded, so a reader is told what a run holds instead of inferring it from which files happen to be present. Runs saved before versioning still load, through an adapter for their era. A run this build cannot read is listed in Analytics as its own row explaining why, rather than disappearing or taking the catalog down with it.

The metadata captures the model, prompt, hyperparameters, any remask edits, per-frame timing, canvas indices, mean confidence, what the run cost the card, and reproducibility info: seed, GPU name, git commit, the worker's torch/transformers versions, and the tokenizer that produced the run's ids (class, checkpoint path, vocabulary size, and whether it is a fast tokenizer). The cost block holds the peak VRAM the run's generation held together with the allocation it started from, which Analytics reports on one **Peak VRAM** row as the peak and, in brackets, how far above the baseline it reached. The pair is given because the peak is mostly the weights the model had already loaded, so the bracketed figure is the part that moves when generation settings or the sampler change. A run on CPU has no such cost and saves no block at all, rather than a block of zeros that would read later as a measurement. All of that is attested by the worker at the moment the run finishes and travels with the run, including the device the model actually loaded onto, which is not always the one requested. That matters because two browser windows share one supervisor: a run finished in one window and saved after the other switched models used to be described by the model that replaced it. Analytics shows the tokenizer on the run's detail panel; runs saved before a field existed simply omit its row.

## How the models generate

The mechanics behind the overlays: what a diffusion step actually does, and how each model differs.

### Autoregressive vs diffusion

Autoregressive LLMs generate one token at a time, left to right:

$p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_{<t})$

Diffusion LLMs instead start from a corrupted sequence and refine the whole thing in parallel over *N* steps, using **bidirectional** attention (no causal mask), re-corrupting the least certain positions between steps until the sequence converges.

### LLaDA: masked discrete diffusion

- **Forward process (corruption):** independently replace each token with `[MASK]` with probability *t* in [0, 1]. At *t* = 0 the text is clean; at *t* = 1 everything is masked.
- **Reverse process (generation):** starting from a fully masked canvas, the Transformer predicts all masked positions at once, then **re-masks** the least confident predictions. Repeat for *N* steps until nothing is masked.

The training loss is cross-entropy on masked positions only, weighted by 1/*t*, which provides a variational upper bound on negative log-likelihood. This makes LLaDA a principled generative model, not a fill-in-the-blank system like BERT. In the UI, unresolved positions render as `░`, faded by the model's live predicted confidence for that position: a mask the model has no opinion about is barely visible, and it firms up as the model grows sure of what it will become, then resolves into the token. That confidence is measured on a specific prediction, and **Reveal the mask candidate** in Settings draws it, so the same position shows the word the model is holding, fading in as it commits to it. The fade follows a square-root curve rather than a straight line, because a masked position's confidence is skewed low (median 0.11 to 0.21 across a measured run) and a linear ramp would crowd an entire canvas into one indistinguishable shade.

### DiffusionGemma: block-autoregressive text diffusion

DiffusionGemma denoises a fixed **256-token canvas** on an encoder-decoder MoE backbone. Rather than a `[MASK]` placeholder, unresolved positions carry noisy tokens that the sampler renoises between steps under an entropy bound. Two properties make it distinct from LLaDA:

- **Adaptive stopping:** a canvas can finish in fewer than the configured maximum steps once its predictions stabilize, so simpler prompts run faster.
- **Block-autoregressive chaining:** for outputs longer than one canvas, it commits a canvas and then seeds the next, chaining multiple canvases. The status readout therefore reads `Step X, Canvas Y` rather than a fixed step total.

An optional **thinking** channel exposes a step-by-step reasoning pass, which the UI separates into a collapsible panel above the answer.

## Sampling parameters

**LLaDA**

| Parameter | Description |
|---|---|
| Steps | Number of denoising steps. More steps generally mean higher quality and slower generation. |
| Gen Length | Length of the masked canvas (output token count). Must be divisible by Block Length. |
| Block Length | Block size for semi-autoregressive sampling. When smaller than Gen Length, blocks resolve left-to-right with diffusion inside each block; set equal to Gen Length for pure diffusion. |
| Temperature | Gumbel noise temperature for categorical sampling. 0 is greedy (argmax). |
| CFG Scale | Classifier-free guidance strength. 0 disables it; higher values increase prompt adherence. |
| Seed | Random seed for reproducibility; -1 is nondeterministic. |
| Remasking | Strategy: `low_confidence` (default, re-mask least confident tokens) or `random`. |

**DiffusionGemma**

| Parameter | Description |
|---|---|
| Max Tokens | Output budget. Generation happens in 256-token canvases; larger budgets chain multiple canvases. |
| Denoising Steps | Upper bound on steps per canvas. Adaptive stopping may use fewer. |
| Temp Start / Temp End | Endpoints of a linear temperature schedule across the denoising steps (hotter early, cooler late). |
| Seed | Random seed for reproducibility; -1 is nondeterministic. |
| Thinking | Enables the step-by-step reasoning channel, shown in a separate panel. |

**SmolLM3**

| Parameter | Description |
|---|---|
| Max Tokens | Number of tokens to generate. The recommended ceiling is lower on CPU, where decoding is slower. |
| Temperature | Sampling temperature. 0 is greedy (argmax). |
| Top-p | Nucleus sampling probability mass. |
| Seed | Random seed for reproducibility; -1 is nondeterministic. |
| Thinking | Enables the extended reasoning channel, shown in a separate panel. |
| Top-k | Keep only the k likeliest tokens before Top-p applies. A hard truncation where Top-p's is adaptive, so it caps how far into the tail sampling can reach when the model is torn. Applied before Top-p (matching Hugging Face); -1, the default, disables it, spelled that way because a k of 0 would read as "no candidates at all". Distinct from the five candidates **Alternatives** records. |
| Alternatives | Captures the top five competing tokens at each position. Powers the hover popover and is required for **What If?** substitution (slightly slower, on by default). |

All parameters are configurable in the web UI with recommended bounds enforced by default. An **Experimental** toggle lifts the bounds for exploratory use.

Hyperparameters, the Experimental toggle, and the prompt draft persist for the life of the app, per model, so navigating to Analytics and back leaves the setup intact; values are stored as typed, so a half-finished number is not rounded off. Closing the app clears them, and a fresh launch starts from the recommended defaults. A **Reset** button on the Experimental row restores every hyperparameter and the toggle for the current model and device, and is disabled while nothing differs from the defaults.
