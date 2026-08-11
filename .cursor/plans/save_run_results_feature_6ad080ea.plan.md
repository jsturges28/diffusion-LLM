---
name: Save Run Results Feature
overview: Add a "Save" button to the web UI that becomes active after each generation run completes, allowing the user to save a GIF, metadata, frame history, and final output to a `Results/` directory.
todos:
  - id: backend-save-endpoint
    content: Add POST /api/save endpoint to server.py with file-saving logic (metadata, final, history, GIF) writing to Results/
    status: completed
  - id: frontend-frame-accumulation
    content: Add frame history accumulation in app.js (push in handleFrame, clear on new generation, store final text on done)
    status: completed
  - id: frontend-save-button-html
    content: Add Save button element to index.html controls area
    status: completed
  - id: frontend-save-button-js
    content: "Implement Save button click handler in app.js: POST to /api/save, disable/enable logic, success/error feedback"
    status: completed
  - id: frontend-save-button-css
    content: Style the Save button and any toast/feedback elements in style.css
    status: completed
  - id: gitignore-results
    content: Add Results/ to .gitignore
    status: completed
isProject: false
---

# Save Run Results Feature

## Architecture

The client already receives every frame's text via WebSocket. The approach is:

1. **Client accumulates frame history** in a JS array during generation
2. After run completes, a **Save button becomes clickable**
3. On click, client POSTs accumulated data to a **new REST endpoint**
4. Server writes files to `Results/<timestamp>/` and generates the GIF using the existing PIL renderer

```mermaid
sequenceDiagram
    participant User
    participant Client as app.js
    participant Server as server.py
    participant GIF as render_gif.py

    Note over Client: During generation
    Server->>Client: frame (text, index, total_steps)
    Client->>Client: Push text to frameHistory[]

    Server->>Client: done (final_text)
    Client->>Client: Enable Save button

    Note over User: User clicks Save
    User->>Client: Click Save
    Client->>Server: POST /api/save {prompt, params, frames, final_text}
    Server->>Server: Create Results/YYYY-MM-DD_HH-MM-SS/
    Server->>Server: Write metadata.json, final.txt, history.txt
    Server->>GIF: history_to_gif(frames, path)
    Server->>Client: 200 {path: "Results/..."}
    Client->>User: Show success toast
```



## Backend: New REST endpoint in `server.py`

Add a `POST /api/save` endpoint (registered **before** the catch-all static mount at the bottom of [server.py](src/web/server.py)):

- Accepts JSON body: `{prompt, params, frames: string[], final_text}`
- Creates a timestamped directory under `Results/` using the same `make_run_dir` pattern from [main.py](main.py) (lines 44-48)
- Saves four files:
  - `metadata.json` — prompt, params, final_text, timestamp, model name
  - `final.txt` — the final clean text
  - `history.txt` — frame-by-frame dump with `===== FRAME N =====` headers (same format as existing [artifacts](artifacts/2026-02-26_01-33-13_llada/))
  - `diffusion.gif` — generated via the existing `history_to_gif()` from [render_gif.py](src/inference/render_gif.py)
- Returns JSON `{success: true, path: "Results/..."}` or an error

The endpoint must be mounted **above** the static files catch-all on line 280 of `server.py`, otherwise FastAPI's `StaticFiles` will intercept the route.

## Frontend: Save button and frame accumulation in `app.js`

### State additions

- `var frameHistory = []` — accumulates frame text strings during a run
- `var lastRunParams = null` — stores the params/prompt used for the completed run
- `var lastFinalText = null` — stores the final text from the `done` message

### Frame accumulation

- In `handleFrame(data)`: push `data.text` onto `frameHistory`
- In `startGeneration()`: clear `frameHistory`, `lastRunParams`, `lastFinalText`
- In `handleDone(data)`: store `data.final_text` in `lastFinalText`, store prompt+params in `lastRunParams`, enable the Save button

### Save button behavior

- Add a **Save** button next to Generate/Cancel in the controls area of [index.html](src/web/static/index.html)
- Starts **hidden** (or disabled+hidden). Becomes visible and enabled in `handleDone()`
- Hides again when a new generation starts (`startGeneration()`)
- On click: sends `POST /api/save` with `{prompt, params, frames: frameHistory, final_text: lastFinalText}`
- While saving: button shows "Saving..." and is disabled
- On success: brief toast/status message "Saved to Results/..."
- On error: status bar shows the error

### Save button styling in `style.css`

- Style consistent with existing Generate/Cancel buttons
- Use a distinct but cohesive color (e.g., a blue/cyan accent) to differentiate it from the green Generate button

## Files to modify

- **[src/web/server.py](src/web/server.py)** — add `POST /api/save` endpoint with save logic
- **[src/web/static/app.js](src/web/static/app.js)** — frame accumulation, save button logic, POST request
- **[src/web/static/index.html](src/web/static/index.html)** — add Save button element
- **[src/web/static/style.css](src/web/static/style.css)** — Save button styling, toast/feedback styling

## Files NOT modified

- `src/inference/render_gif.py` — reused as-is
- `src/inference/streaming_sampler.py` — no changes (per project constraint)

## Edge cases to handle

- Save button disabled if the run was cancelled (no valid `done` message)
- Save button disabled during an active generation
- Prevent double-saves (disable button after click until response)
- Large frame histories: gen_length=1024 with steps=1024 could produce ~1024 frame strings. The POST payload could be large but manageable (each frame is at most a few KB of text)

