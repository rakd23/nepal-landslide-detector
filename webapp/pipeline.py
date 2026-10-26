"""
Landslide detection pipeline: the same steps as 03_detect_nepal.ipynb, packaged so the
web app can run them for any location and dates.

run_detection() downloads before/after Sentinel-2 imagery + elevation, runs the U-Net,
keeps landslides that are new in the after image, removes likely false alarms
(flat ground, no vegetation loss, water), and writes map-ready files to out_dir.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import io
import json
import math
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(os.environ.get("GEOAI_ROOT", Path.home() / "geoai"))

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12"]
MODEL_BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", None, "B11", "B12"]  # None = B10
BAD_SCL = [0, 1, 3, 8, 9, 10, 11]   # no data, saturated, cloud shadow, clouds, cirrus, snow
WATER_SCL = 6
MAX_BAD_FRACTION = 0.6              # give up if even the clearest image is mostly cloud

DEFAULT_FILTERS = {
    "min_slope_deg": 10.0,          # landslides need a slope; flat ground is usually fields, riverbeds, towns
    "min_ndvi_drop": 0.10,          # a fresh scar removes vegetation, so NDVI should fall
    "max_water_fraction": 0.3,      # rivers and sandbars change with every flood
}

MODELS = {   # name -> (config file, label)
    "v1": ("model_config.json", "v1: U-Net from scratch"),
    "v2": ("model_config_v2.json", "v2: U-Net + pretrained ResNet34"),
}


# ---------------------------------------------------------------- models

def double_conv(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """The U-Net trained from scratch in Step 4."""
    def __init__(self, in_ch=14, base=32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.pool = nn.MaxPool2d(2)
        self.enc1, self.enc2 = double_conv(in_ch, c1), double_conv(c1, c2)
        self.enc3, self.enc4 = double_conv(c2, c3), double_conv(c3, c4)
        self.bottleneck = double_conv(c4, c5)
        self.up4, self.dec4 = nn.ConvTranspose2d(c5, c4, 2, stride=2), double_conv(c5, c4)
        self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, stride=2), double_conv(c4, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), double_conv(c3, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), double_conv(c2, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_model(config):
    arch = config.get("arch", "unet")
    if arch == "unet":
        return UNet(in_ch=config["in_channels"], base=config["base"])
    if arch == "smp_unet":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name=config["encoder"], encoder_weights=None,
                        in_channels=config["in_channels"], classes=1)
    raise ValueError(f"Unknown model architecture: {arch}")


def available_models():
    out = []
    for name, (cfg_file, label) in MODELS.items():
        path = ROOT / cfg_file
        if path.exists():
            cfg = json.loads(path.read_text())
            if (ROOT / cfg.get("weights", "landslide_unet.pt")).exists():
                out.append({"name": name, "label": label, "val_f1": cfg.get("val_f1"),
                            "val_f1_tta": cfg.get("val_f1_tta", cfg.get("val_f1") if cfg.get("tta") else None),
                            "threshold": cfg.get("threshold")})
    return out


def default_model():
    """The available model with the best validation F1 (with flip averaging when measured)."""
    models = available_models()
    if not models:
        return "v1"
    score = lambda m: m["val_f1_tta"] if m["val_f1_tta"] is not None else (m["val_f1"] or 0)
    return max(models, key=score)["name"]


@lru_cache(maxsize=2)
def get_model(name="v1"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    config = json.loads((ROOT / MODELS[name][0]).read_text())
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(ROOT / config.get("weights", "landslide_unet.pt"), map_location=device))
    model.eval()
    return model, config, device


@lru_cache(maxsize=1)
def get_catalog():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


# ---------------------------------------------------------------- imagery

def make_bbox(lat, lon, half_size_km):
    dlat = half_size_km / 111.0
    dlon = half_size_km / (111.0 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def utm_crs(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def find_clearest_day(catalog, bbox, crs, dates, log, max_tile_cloud=20, max_candidates=6):
    import odc.stac
    items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=dates,
                                query={"eo:cloud_cover": {"lt": max_tile_cloud}}).items())
    if not items:
        raise ValueError(f"No Sentinel-2 images under {max_tile_cloud}% cloud between {dates.replace('/', ' and ')}. "
                         "Try a wider date range or dry-season months (Oct–Feb).")
    by_day = {}
    for it in items:
        by_day.setdefault(it.datetime.date(), []).append(it)
    candidates = sorted(by_day, key=lambda d: np.mean([it.properties["eo:cloud_cover"] for it in by_day[d]]))
    best_day, best_bad = None, 2.0
    for i, day in enumerate(candidates[:max_candidates], 1):
        log(f"checking clouds on {day} ({i}/{min(len(candidates), max_candidates)})")
        scl = odc.stac.load(by_day[day], bands=["SCL"], bbox=bbox, crs=crs, resolution=60,
                            groupby="solar_day", dtype="uint16", nodata=0)["SCL"].values[0]
        bad = float(np.isin(scl, BAD_SCL).mean())
        if bad < best_bad:
            best_day, best_bad = day, bad
        if bad < 0.02:
            break
    if best_bad > MAX_BAD_FRACTION:
        raise ValueError(f"The clearest image between {dates.replace('/', ' and ')} is still "
                         f"{best_bad:.0%} cloud or missing over this area. Try different dates.")
    return best_day, by_day[best_day], best_bad


def load_scene(items, bbox, crs):
    import odc.stac
    ds = odc.stac.load(items, bands=S2_BANDS + ["SCL"], bbox=bbox, crs=crs, resolution=10,
                       groupby="solar_day", dtype="uint16", nodata=0,
                       resampling={"*": "bilinear", "SCL": "nearest"})
    return ds.isel(time=0)


def load_dem(catalog, bbox, geobox):
    import odc.stac
    items = list(catalog.search(collections=["cop-dem-glo-30"], bbox=bbox).items())
    if not items:
        raise ValueError("No elevation data found for this area.")
    ds = odc.stac.load(items, bands=["data"], geobox=geobox, resampling="bilinear", dtype="float32")
    return ds["data"].max(dim="time").values


def slope_degrees(dem, pixel_size=10.0):
    dy, dx = np.gradient(dem, pixel_size)
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def ndvi(ds):
    nir = ds["B08"].values.astype(np.float32)
    red = ds["B04"].values.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (nir - red) / (nir + red)
    v[(nir <= 0) | (red <= 0)] = np.nan
    return v


# ---------------------------------------------------------------- normalization

def load_normalizer():
    """Training statistics from Step 3 (mean/std) and, if the v2 notebook saved them, per-band quantiles."""
    stats = ROOT / "band_stats.npy"
    quant = ROOT / "band_quantiles.npy"
    mean, std = np.load(stats) if stats.exists() else (None, None)
    quantiles = np.load(quant) if quant.exists() else None
    return mean, std, quantiles


def normalization_modes():
    mean, _, quantiles = load_normalizer()
    modes = ["zscore"]
    if quantiles is not None and mean is not None:
        modes.insert(0, "quantile")
    return modes


def standardize(a, valid):
    m, s = a[valid].mean(), a[valid].std() + 1e-6
    return np.where(valid, (a - m) / s, 0).astype(np.float32)


def quantile_match(a, valid, train_q, mean, std):
    """Map this image's value distribution onto the training data's distribution for the same band,
    then apply the training normalization. Handles skewed bands better than a plain z-score."""
    scene_q = np.percentile(a[valid], np.linspace(0, 100, len(train_q)))
    scene_q = np.maximum.accumulate(scene_q + np.arange(len(scene_q)) * 1e-6)   # strictly increasing for interp
    matched = np.interp(a, scene_q, train_q)
    return np.where(valid, (matched - mean) / (std + 1e-6), 0).astype(np.float32)


def build_input(ds, dem, slope, mode="zscore"):
    valid = (ds["B02"].values > 0) & ~np.isin(ds["SCL"].values, BAD_SCL) & np.isfinite(dem)
    if valid.sum() < 1000:
        raise ValueError("Almost no clear pixels in this image.")
    mean, std, quantiles = load_normalizer()
    use_q = mode == "quantile" and quantiles is not None and mean is not None
    raw = [None if b is None else ds[b].values.astype(np.float32) for b in MODEL_BAND_ORDER]
    raw += [np.nan_to_num(slope), np.nan_to_num(dem)]
    layers = []
    for i, a in enumerate(raw):
        if a is None:
            layers.append(np.zeros(valid.shape, np.float32))       # B10 not in L2A: use the average value
        elif use_q:
            layers.append(quantile_match(a, valid, quantiles[i], mean[i], std[i]))
        else:
            layers.append(standardize(a, valid))
    return np.stack(layers), valid


# ---------------------------------------------------------------- inference

@torch.no_grad()
def predict_large(x, model, device, tile=128, stride=64, batch_size=16, tta=True):
    """Slide a 128x128 window over the image and average overlapping predictions.
    With tta=True each tile is also predicted flipped three ways and the results averaged
    (test-time augmentation), which smooths out noisy predictions."""
    C, H, W = x.shape
    H2 = max(tile, int(np.ceil((H - tile) / stride)) * stride + tile)
    W2 = max(tile, int(np.ceil((W - tile) / stride)) * stride + tile)
    xp = np.zeros((C, H2, W2), np.float32)
    xp[:, :H, :W] = x
    probs = np.zeros((H2, W2), np.float32)
    counts = np.zeros((H2, W2), np.float32)
    flips = [(), (3,), (2,), (2, 3)] if tta else [()]
    coords = [(r, c) for r in range(0, H2 - tile + 1, stride) for c in range(0, W2 - tile + 1, stride)]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        xb = torch.from_numpy(np.stack([xp[:, r:r + tile, c:c + tile] for r, c in chunk])).to(device)
        pb = torch.zeros(len(chunk), 1, tile, tile, device=device)
        for dims in flips:
            xi = torch.flip(xb, dims) if dims else xb
            pi = torch.sigmoid(model(xi))
            pb += torch.flip(pi, dims) if dims else pi
        pb = (pb / len(flips)).cpu().numpy()[:, 0]
        for (r, c), p in zip(chunk, pb):
            probs[r:r + tile, c:c + tile] += p
            counts[r:r + tile, c:c + tile] += 1
    return (probs / counts)[:H, :W]


# ---------------------------------------------------------------- map overlays

def to_latlon(arrays, transform, crs):
    """Reproject 2-D arrays from the UTM grid to lat/lon.
    Returns (reprojected arrays, Leaflet bounds [[south, west], [north, east]])."""
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    H, W = arrays[0].shape
    left, bottom, right, top = array_bounds(H, W, transform)
    dst_t, dw, dh = calculate_default_transform(crs, "EPSG:4326", W, H, left, bottom, right, top)
    out = []
    for a in arrays:
        dst = np.full((dh, dw), np.nan, np.float32)
        reproject(a.astype(np.float32), dst, src_transform=transform, src_crs=crs,
                  dst_transform=dst_t, dst_crs="EPSG:4326", resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
        out.append(dst)
    west, north = dst_t * (0, 0)
    east, south = dst_t * (dw, dh)
    return out, [[south, west], [north, east]]


def save_png(rgba, path):
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(np.nan_to_num(rgba), 0, 1), format="png")
    Path(path).write_bytes(buf.getvalue())


def probability_png(prob_ll, path):
    p = np.nan_to_num(prob_ll)
    rgba = plt.get_cmap("inferno")(p)
    rgba[..., 3] = np.clip((p - 0.2) / 0.6, 0, 1)   # transparent below 0.2
    save_png(rgba, path)


def true_color_png(ds, transform, crs, path):
    bands = [ds[b].values.astype(np.float32) for b in ("B04", "B03", "B02")]
    valid = bands[0] > 0
    for i, b in enumerate(bands):
        lo, hi = np.percentile(b[valid], (2, 98)) if valid.any() else (0, 1)
        bands[i] = np.where(valid, np.clip((b - lo) / (hi - lo + 1e-6), 0, 1), np.nan)
    ll, bounds = to_latlon(bands, transform, crs)
    alpha = np.isfinite(ll[0]).astype(np.float32)
    save_png(np.dstack(ll + [alpha]), path)
    return bounds


# ---------------------------------------------------------------- detections + filters

def outline_and_filter(new_mask, prob, slope, ndvi_change, water, transform, crs, min_area_m2, filters):
    """Turn flagged pixels into polygons, measure each one, and split them into kept vs removed."""
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape

    shapes = [shape(g) for g, _ in features.shapes(new_mask.astype(np.uint8), mask=new_mask, transform=transform)]
    gdf = gpd.GeoDataFrame(geometry=shapes, crs=crs)
    gdf["area_m2"] = gdf.geometry.area.round(0)
    gdf = gdf[gdf["area_m2"] >= min_area_m2].sort_values("area_m2", ascending=False).reset_index(drop=True)
    if gdf.empty:
        empty = gdf.assign(max_prob=[], mean_slope=[], ndvi_change=[], water_fraction=[])
        return empty, empty.assign(reason=[])

    # Label every pixel with its polygon number, then summarise all polygons at once.
    labels = features.rasterize(((g, i + 1) for i, g in enumerate(gdf.geometry)),
                                out_shape=new_mask.shape, transform=transform, dtype="int32")
    n = len(gdf) + 1
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n).astype(np.float64)
    counts[counts == 0] = 1

    def mean_of(arr):
        a = np.nan_to_num(arr.ravel().astype(np.float64))
        ok = np.isfinite(arr.ravel())
        s = np.bincount(flat, weights=a * ok, minlength=n)
        c = np.bincount(flat, weights=ok.astype(np.float64), minlength=n)
        return (s / np.maximum(c, 1))[1:]

    max_prob = np.zeros(n)
    np.maximum.at(max_prob, flat, prob.ravel())
    gdf["max_prob"] = np.round(max_prob[1:], 3)
    gdf["mean_slope"] = np.round(mean_of(slope), 1)
    gdf["ndvi_change"] = np.round(mean_of(ndvi_change), 3)
    gdf["water_fraction"] = np.round(np.bincount(flat, weights=water.ravel().astype(np.float64), minlength=n)[1:] / counts[1:], 3)

    reasons = []
    for _, r in gdf.iterrows():
        why = []
        if filters.get("slope") and r["mean_slope"] < filters["min_slope_deg"]:
            why.append(f"flat ground (avg slope {r['mean_slope']:.0f}°)")
        if filters.get("vegetation") and not (r["ndvi_change"] <= -filters["min_ndvi_drop"]):
            why.append(f"no vegetation loss (NDVI change {r['ndvi_change']:+.2f})")
        if filters.get("water") and r["water_fraction"] > filters["max_water_fraction"]:
            why.append(f"water or river ({r['water_fraction']:.0%} water)")
        reasons.append("; ".join(why))
    gdf["reason"] = reasons
    kept = gdf[gdf["reason"] == ""].drop(columns="reason").reset_index(drop=True)
    removed = gdf[gdf["reason"] != ""].reset_index(drop=True)
    return kept, removed


def write_geojson(gdf, path, image_date):
    out = gdf.to_crs("EPSG:4326").copy()
    out.insert(0, "fid", range(1, len(out) + 1))
    out["image_date"] = str(image_date)
    if out.empty:
        Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    else:
        out.to_file(path, driver="GeoJSON")


# ---------------------------------------------------------------- full run

def run_detection(lat, lon, before_dates, after_dates, out_dir, half_size_km=5.0,
                  model_name=None, normalization=None, tta=True, filters=None,
                  threshold=None, min_area_m2=1000, log=print):
    import rasterio
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_name or default_model()
    modes = normalization_modes()
    normalization = normalization if normalization in modes else modes[0]
    flt = {"slope": True, "vegetation": True, "water": True, **DEFAULT_FILTERS, **(filters or {})}

    log(f"loading model {model_name}")
    model, config, device = get_model(model_name)
    threshold = float(config["threshold"] if threshold is None else threshold)
    catalog = get_catalog()

    bbox = make_bbox(lat, lon, half_size_km)
    crs = utm_crs(lat, lon)

    log("searching for a clear before image")
    before_day, before_items, before_bad = find_clearest_day(catalog, bbox, crs, before_dates, log)
    log(f"downloading before image ({before_day})")
    before = load_scene(before_items, bbox, crs)

    log("searching for a clear after image")
    after_day, after_items, after_bad = find_clearest_day(catalog, bbox, crs, after_dates, log)
    log(f"downloading after image ({after_day})")
    after = load_scene(after_items, bbox, crs)

    geobox = before.odc.geobox
    if after.odc.geobox.shape != geobox.shape:
        raise ValueError("Before and after images don't line up.")
    transform = geobox.affine

    log("downloading elevation")
    dem = load_dem(catalog, bbox, geobox)
    slope = slope_degrees(dem)

    log("running the model" + (" (with flips)" if tta else ""))
    x_b, valid_b = build_input(before, dem, slope, normalization)
    x_a, valid_a = build_input(after, dem, slope, normalization)
    prob_b = predict_large(x_b, model, device, tta=tta)
    prob_a = predict_large(x_a, model, device, tta=tta)
    prob_b[~valid_b] = 0
    prob_a[~valid_a] = 0

    log("outlining and filtering new landslides")
    new_mask = (prob_a > threshold) & (prob_b <= threshold) & valid_b & valid_a
    ndvi_change = ndvi(after) - ndvi(before)
    water = (before["SCL"].values == WATER_SCL) | (after["SCL"].values == WATER_SCL)
    kept, removed = outline_and_filter(new_mask, prob_a, slope, ndvi_change, water,
                                       transform, crs, min_area_m2, flt)
    write_geojson(kept, out_dir / "landslides.geojson", after_day)
    write_geojson(removed, out_dir / "removed.geojson", after_day)

    log("saving map layers")
    with rasterio.open(out_dir / "prob_after.tif", "w", driver="GTiff", height=prob_a.shape[0],
                       width=prob_a.shape[1], count=1, dtype="float32", crs=crs,
                       transform=transform, compress="deflate") as dst:
        dst.write(prob_a.astype(np.float32), 1)
    (prob_ll,), bounds = to_latlon([prob_a], transform, crs)
    probability_png(prob_ll, out_dir / "prob.png")
    true_color_png(before, transform, crs, out_dir / "before.png")
    true_color_png(after, transform, crs, out_dir / "after.png")

    removed_by = {}
    for reason in removed.get("reason", []):
        for part in reason.split("; "):
            key = part.split(" (")[0]
            removed_by[key] = removed_by.get(key, 0) + 1

    summary = {
        "center": [lat, lon], "bbox": list(bbox), "half_size_km": half_size_km,
        "before_date": str(before_day), "after_date": str(after_day),
        "before_cloud": round(before_bad, 3), "after_cloud": round(after_bad, 3),
        "model": model_name, "model_label": MODELS[model_name][1], "model_val_f1": config.get("val_f1"),
        "normalization": normalization, "tta": bool(tta),
        "threshold": threshold, "min_area_m2": min_area_m2,
        "filters": {k: flt[k] for k in ("slope", "vegetation", "water", *DEFAULT_FILTERS)},
        "num_new_landslides": int(len(kept)),
        "num_removed": int(len(removed)), "removed_by": removed_by,
        "total_area_km2": round(float(kept["area_m2"].sum()) / 1e6, 3) if len(kept) else 0.0,
        "overlay_bounds": bounds, "has_imagery": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def import_notebook_outputs(out_dir):
    """Turn the files 03_detect_nepal.ipynb saved in ~/geoai/outputs into a web-app run."""
    import geopandas as gpd
    import rasterio
    src = ROOT / "outputs"
    nb_summary = json.loads((src / "run_summary.json").read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(src / "new_landslides.geojson")
    if "fid" in gdf.columns:
        gdf = gdf.drop(columns="fid")
    gdf = gdf.sort_values("area_m2", ascending=False).reset_index(drop=True) if len(gdf) else gdf
    gdf.insert(0, "fid", range(1, len(gdf) + 1))
    if gdf.empty:
        (out_dir / "landslides.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    else:
        gdf.to_file(out_dir / "landslides.geojson", driver="GeoJSON")

    with rasterio.open(src / "landslide_prob_after.tif") as f:
        prob, transform, crs = f.read(1), f.transform, f.crs.to_string()
    (prob_ll,), bounds = to_latlon([prob], transform, crs)
    probability_png(prob_ll, out_dir / "prob.png")
    config_path = ROOT / "model_config.json"
    summary = {
        **nb_summary, "overlay_bounds": bounds, "has_imagery": False,
        "model": "v1", "model_label": MODELS["v1"][1], "normalization": "zscore", "tta": False,
        "filters": None, "num_removed": 0, "removed_by": {},
        "model_val_f1": json.loads(config_path.read_text()).get("val_f1") if config_path.exists() else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
