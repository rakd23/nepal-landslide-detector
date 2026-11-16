"""
Step 6: Nepal Landslide Detector web app.

FastAPI backend + Leaflet map. Shows the Step 5 notebook results, runs new detections in the
background for any location and dates (pipeline.py), and lets you review each detection as a
real landslide or a false alarm to measure real-world precision.

Run from the ~/geoai/webapp folder:
    uvicorn app:app
then open http://127.0.0.1:8000 in your browser.

Environment variables (optional, used when deploying online):
    GEOAI_ROOT   folder with the model files and runs (default ~/geoai)
    MAX_QUEUE    how many runs may wait in line at once (default 5)
    DEMO_MODE    set to 1 on a public deployment: runs that exist when the server starts become
                 read-only "example" runs (visitors can't delete them or change their reviews),
                 and the developer-only notebook run is hidden
"""
import json
import os
import secrets
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.base import BaseHTTPMiddleware

import pipeline

ROOT = pipeline.ROOT
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"
RUN_FILES = {"landslides.geojson", "removed.geojson", "prob.png", "before.png", "after.png"}
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "5"))
DEMO_MODE = os.environ.get("DEMO_MODE", "") == "1"

app = FastAPI(title="Nepal Landslide Detector", version="3.0")


class NoCacheHTML(BaseHTTPMiddleware):
    """Make browsers always fetch the latest page instead of a stale cached copy."""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response


app.add_middleware(NoCacheHTML)

runs: dict[str, dict] = {}
runs_lock = threading.Lock()
worker = ThreadPoolExecutor(max_workers=1)   # one detection at a time: the Mac has one GPU


# ---------------------------------------------------------------- run bookkeeping

def save_run(run):
    (RUNS_DIR / run["id"]).mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / run["id"] / "run.json").write_text(json.dumps(run, indent=2))


def update_run(run_id, **changes):
    with runs_lock:
        run = runs[run_id]
        run.update(changes)
        save_run(run)


RESTART_MESSAGE = ("The server restarted while this run was in progress (the free server goes to sleep "
                   "when nobody is using it), so the run stopped. Start it again from New detection.")


def friendly_error(e):
    """Turn a pipeline exception into a message a visitor can act on."""
    text = str(e) or e.__class__.__name__
    if isinstance(e, ValueError):          # the pipeline's own checks (clouds, no imagery) already explain themselves
        return text
    low = (text + " " + e.__class__.__name__).lower()
    if isinstance(e, MemoryError) or "out of memory" in low or "cannot allocate" in low:
        return ("This area was too large for the free server's memory. Choose a smaller area size "
                "(5 × 5 km) and run it again.")
    network = ("timeout", "timed out", "connection", "http", "curl", "ssl", "403", "404", "409", "429",
               "500", "502", "503", "504", "rasterioioerror", "apierror", "temporary failure")
    if any(k in low for k in network):
        return ("Couldn't download the satellite images: Microsoft Planetary Computer, which supplies them, "
                "didn't respond properly. This is usually temporary, so wait a minute and run it again. "
                f"(Details: {text[:200]})")
    return ("Something went wrong while processing the images. Try a different date range or a smaller area. "
            f"(Details: {e.__class__.__name__}: {text[:200]})")


def load_runs():
    for path in sorted(RUNS_DIR.glob("*/run.json")):
        run = json.loads(path.read_text())
        if run["status"] in ("queued", "running"):
            run.update(status="failed", message=RESTART_MESSAGE)
            save_run(run)
        runs[run["id"]] = run


def import_notebook_run():
    """Show the Step 5 notebook results as a run, refreshing it if the notebook ran again."""
    src = ROOT / "outputs" / "run_summary.json"
    if not src.exists():
        return
    run_id = "notebook"
    existing = runs.get(run_id)
    mtime = src.stat().st_mtime
    if existing and existing.get("source_mtime") == mtime and existing.get("version") == 3:
        return
    try:
        summary = pipeline.import_notebook_outputs(RUNS_DIR / run_id)
    except Exception as e:  # noqa: BLE001
        print("Could not import notebook outputs:", e)
        return
    lat, lon = summary["center"]
    run = {
        "id": run_id, "name": "Notebook run (Step 5)", "status": "done", "message": "",
        "created": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
        "params": {"lat": lat, "lon": lon}, "summary": summary, "source_mtime": mtime, "version": 3,
    }
    with runs_lock:
        runs[run_id] = run
        save_run(run)


load_runs()
import_notebook_run()

# On a public demo, the runs shipped with the app are examples: visible to everyone, never editable.
EXAMPLE_IDS = {rid for rid, r in runs.items() if r["status"] == "done"} if DEMO_MODE else set()
HIDDEN_IDS = {"notebook"} if DEMO_MODE else set()


def featured_run_id():
    """The example a first-time visitor should see: the default model's run with the most reviews."""
    if not EXAMPLE_IDS:
        return None
    best_model = pipeline.default_model()
    def score(rid):
        r = runs[rid]
        n_reviews = len(json.loads((RUNS_DIR / rid / "reviews.json").read_text())) if (RUNS_DIR / rid / "reviews.json").exists() else 0
        return ((r.get("params") or {}).get("model") == best_model, n_reviews, r.get("created", ""))
    candidates = [rid for rid in EXAMPLE_IDS if rid not in HIDDEN_IDS]
    return max(candidates, key=score) if candidates else None


def check_editable(run_id):
    if run_id in EXAMPLE_IDS:
        raise HTTPException(403, "This is an example run, so it can't be changed. "
                                 "Start your own run from New detection to review or delete detections.")


def execute(run_id):
    p = runs[run_id]["params"]
    update_run(run_id, status="running", message="starting")
    try:
        summary = pipeline.run_detection(
            lat=p["lat"], lon=p["lon"],
            before_dates=f"{p['before_start']}/{p['before_end']}",
            after_dates=f"{p['after_start']}/{p['after_end']}",
            half_size_km=p["half_size_km"], out_dir=RUNS_DIR / run_id,
            model_name=p.get("model"), normalization=p.get("normalization"), tta=p.get("tta", True),
            filters=p.get("filters"),
            log=lambda msg: update_run(run_id, message=msg),
        )
        update_run(run_id, status="done", message="", summary=summary)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        update_run(run_id, status="failed", message=friendly_error(e))


# ---------------------------------------------------------------- API models

class Filters(BaseModel):
    slope: bool = True
    vegetation: bool = True
    water: bool = True
    min_slope_deg: float = Field(pipeline.DEFAULT_FILTERS["min_slope_deg"], ge=0, le=45)
    min_ndvi_drop: float = Field(pipeline.DEFAULT_FILTERS["min_ndvi_drop"], ge=0, le=1)
    max_water_fraction: float = Field(pipeline.DEFAULT_FILTERS["max_water_fraction"], ge=0, le=1)


class RunRequest(BaseModel):
    name: str = Field("", max_length=80)
    lat: float = Field(ge=-56, le=72)
    lon: float = Field(ge=-180, le=180)
    before_start: date
    before_end: date
    after_start: date
    after_end: date
    half_size_km: float = Field(5.0, ge=1.0, le=10.0)
    model: Optional[str] = None
    normalization: Optional[Literal["quantile", "zscore"]] = None
    tta: bool = True
    filters: Filters = Filters()

    @model_validator(mode="after")
    def check(self):
        if self.before_start >= self.before_end or self.after_start >= self.after_end:
            raise ValueError("Each date range must start before it ends.")
        if self.before_end > self.after_start:
            raise ValueError("The before range must end before the after range starts.")
        if self.before_start < date(2015, 7, 1):
            raise ValueError("Sentinel-2 imagery starts in mid-2015.")
        for start, end in ((self.before_start, self.before_end), (self.after_start, self.after_end)):
            if (end - start).days > 400:
                raise ValueError("Keep each date range under about a year.")
        if self.model and self.model not in pipeline.MODELS:
            raise ValueError(f"Unknown model {self.model}.")
        return self


class Review(BaseModel):
    label: Optional[Literal["real", "false"]] = None


def public(run):
    out = {k: v for k, v in run.items() if k not in ("source_mtime", "version")}
    out["example"] = run["id"] in EXAMPLE_IDS
    return out


def get_existing(run_id):
    run = runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


# ---------------------------------------------------------------- routes

@app.get("/api/options")
def options():
    """Models, normalization modes and filter defaults the New detection form can offer."""
    return {
        "models": pipeline.available_models(),
        "default_model": pipeline.default_model(),
        "normalizations": pipeline.normalization_modes(),
        "filters": pipeline.DEFAULT_FILTERS,
        "demo": DEMO_MODE,
        "featured_run": featured_run_id(),
        "max_queue": MAX_QUEUE,
    }


@app.get("/api/runs")
def list_runs():
    import_notebook_run()
    with runs_lock:
        visible = (public(r) for rid, r in runs.items() if rid not in HIDDEN_IDS)
        # visitors' own runs first (newest first), then the examples
        return sorted(visible, key=lambda r: (not r["example"], r["created"]), reverse=True)


@app.post("/api/runs", status_code=201)
def create_run(req: RunRequest):
    waiting = sum(r["status"] in ("queued", "running") for r in runs.values())
    if waiting >= MAX_QUEUE:
        raise HTTPException(429, f"{waiting} runs are already waiting in line (the free server runs one at a time). "
                                 "Try again in a few minutes, or explore the example runs meanwhile.")
    if req.model and req.model not in [m["name"] for m in pipeline.available_models()]:
        raise HTTPException(422, f"Model {req.model} isn't trained yet.")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    name = req.name.strip() or f"{req.lat:.4f}, {req.lon:.4f}"
    run = {
        "id": run_id, "name": name, "status": "queued", "message": "waiting for the previous run",
        "created": datetime.now().isoformat(timespec="seconds"),
        "params": json.loads(req.model_dump_json()), "summary": None,
    }
    with runs_lock:
        runs[run_id] = run
        save_run(run)
    worker.submit(execute, run_id)
    return public(run)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return public(get_existing(run_id))


def read_reviews(run_id):
    path = RUNS_DIR / run_id / "reviews.json"
    return json.loads(path.read_text()) if path.exists() else {}


@app.get("/api/runs/{run_id}/reviews")
def get_reviews(run_id: str):
    get_existing(run_id)
    return read_reviews(run_id)


@app.put("/api/runs/{run_id}/reviews/{fid}")
def set_review(run_id: str, fid: int, review: Review):
    get_existing(run_id)
    check_editable(run_id)
    with runs_lock:
        reviews = read_reviews(run_id)
        if review.label is None:
            reviews.pop(str(fid), None)
        else:
            reviews[str(fid)] = review.label
        (RUNS_DIR / run_id / "reviews.json").write_text(json.dumps(reviews, indent=2))
    return reviews


@app.get("/api/runs/{run_id}/export.geojson")
def export_run(run_id: str):
    """Detections with your review labels, ready for QGIS, Google Earth or the evaluation notebook."""
    run = get_existing(run_id)
    path = RUNS_DIR / run_id / "landslides.geojson"
    if run["status"] != "done" or not path.exists():
        raise HTTPException(404, "This run has no results yet.")
    geo = json.loads(path.read_text())
    reviews = read_reviews(run_id)
    for f in geo["features"]:
        f["properties"]["review"] = reviews.get(str(f["properties"].get("fid")))
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run["name"])[:40] or run_id
    return JSONResponse(geo, media_type="application/geo+json",
                        headers={"Content-Disposition": f'attachment; filename="{safe}_landslides.geojson"'})


@app.get("/api/runs/{run_id}/{filename}")
def run_file(run_id: str, filename: str):
    get_existing(run_id)
    if filename not in RUN_FILES:
        raise HTTPException(404, "Not found")
    path = RUNS_DIR / run_id / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} isn't available for this run")
    media = "application/geo+json" if filename.endswith(".geojson") else "image/png"
    return FileResponse(path, media_type=media, headers={"Cache-Control": "no-cache"})


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str):
    run = get_existing(run_id)
    check_editable(run_id)
    if run["status"] in ("queued", "running"):
        raise HTTPException(409, "Wait for this run to finish before deleting it.")
    if run_id == "notebook":
        raise HTTPException(409, "The notebook run comes from ~/geoai/outputs. Rerun Step 5 to change it.")
    with runs_lock:
        runs.pop(run_id)
    shutil.rmtree(RUNS_DIR / run_id, ignore_errors=True)


# Serve the map page last so the /api routes above take priority.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
