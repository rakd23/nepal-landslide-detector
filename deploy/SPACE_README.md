---
title: Nepal Landslide Detector
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Find new landslides in Sentinel-2 imagery with a U-Net
---

# Nepal Landslide Detector

Pick a place and two date ranges. The app downloads before and after Sentinel-2 images from
Microsoft's Planetary Computer, runs a U-Net trained on the Landslide4Sense dataset, and outlines
landslides that appeared in between. Likely false alarms (flat ground, no vegetation loss, rivers)
are filtered out.

- Model training data: Landslide4Sense (Ghorbanzadeh et al., 2022), CC BY 4.0
- Imagery: Copernicus Sentinel-2 via Microsoft Planetary Computer
- Elevation: Copernicus DEM GLO-30

This demo runs on a shared CPU, so a detection takes a few minutes, and results are cleared
whenever the Space restarts. Detections are unverified.
