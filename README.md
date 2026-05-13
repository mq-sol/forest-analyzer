# forest-analyzer

Drone / Aerial Forest Analysis Platform

## Overview

`forest-analyzer` is a personal R&D project for analyzing drone and aerial imagery focused on:

- Forest analysis
- Pine wilt detection
- Time series monitoring
- GeoTIFF processing
- Web GIS visualization

The project is designed as a modular microservice-oriented architecture and is intended to evolve into a web service / SaaS platform.

---

# Goals

## Short Term (MVP)

- Forest mask generation
- GeoTIFF analysis
- White/dead pine candidate detection
- GeoJSON export
- Web GIS visualization

## Mid Term

- CHM (Canopy Height Model) support
- DSM / DTM analysis
- Time series comparison
- Annotation tools
- Semi-automatic training data generation

## Long Term

- AI-based pine wilt detection
- Continuous monitoring
- Notification system
- SaaS platform
- Distributed processing
- GPU worker support

---

# Architecture

```text
Drone Images
    ↓
OpenDroneMap / WebODM
    ↓
Orthophoto / DSM / DTM
    ↓
Forest Mask
    ↓
Pine Detection
    ↓
GeoJSON / Vector Tiles
    ↓
Web GIS
```

---

# Project Structure

```text
forest-analyzer/
├── data/
├── services/
├── api/
├── web/
├── notebooks/
├── tests/
├── docker/
└── scripts/
```

---

# Development Environment

## OS

- WSL2
- Ubuntu 24.x

## Editor

- VSCode

## Language

- Python 3.12

## Main Libraries

- OpenCV
- rasterio
- geopandas
- shapely
- scikit-image
- click
- FastAPI

---

# Current Status

## Implemented

- [ ] Project structure
- [ ] Forest mask CLI
- [ ] ExG processing
- [ ] Texture analysis
- [ ] GeoJSON export

## Planned

- [ ] CHM processing
- [ ] Pine wilt detection
- [ ] Web GIS viewer
- [ ] Time series analysis
- [ ] Annotation UI
- [ ] AI model training

---

# Data Sources

## Input

- Drone JPG images
- Orthophoto GeoTIFF
- DSM / DTM
- OpenAerialMap datasets

## Output

- Forest masks
- Pine candidate polygons
- GeoJSON
- PNG visualization
- Analysis reports

---

# Core Concepts

## Forest Mask

Extract forest areas from orthophotos using:

- Excess Green Index (ExG)
- Texture analysis
- Morphology filtering
- Height filtering (future)

## Pine Wilt Detection

Detect abnormal pine trees using:

- Color differences
- Texture differences
- Time series comparison
- CHM analysis (future)

---

# Roadmap

## Phase 1

- Forest mask generation
- CLI-based processing
- GeoJSON export

## Phase 2

- Web API
- Web GIS viewer
- Tile generation

## Phase 3

- CHM support
- Time series comparison
- Annotation support

## Phase 4

- AI segmentation
- SaaS architecture
- Distributed processing

---

# License

TBD

---

# Author

Personal R&D project.
