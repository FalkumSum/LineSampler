# -*- coding: utf-8 -*-
"""
Line Sampler (Streamlit + Leaflet DualMap) — stable HTML rendering
- Input two points (Lat/Lon or UTM)
- EPSG selector only if UTM mode
- Sample points every N meters
- Two synchronized maps: left basemap (selectable), right blank
- Red line + blue points on both
- Exports: CSV (lat/lon+UTM), KML, Shapefile (ZIP)
"""

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString, Point
from pyproj import CRS, Transformer
import geopandas as gpd
import simplekml

# Folium maps (rendered as raw HTML to avoid temp-file issues)
import folium
from folium.plugins import DualMap
import streamlit.components.v1 as components


# =========================
# Data structures
# =========================
@dataclass
class LatLon:
    lat: float
    lon: float

@dataclass
class UTM:
    easting: float
    northing: float
    epsg: int  # e.g., 32632


# =========================
# CRS / Projection helpers
# =========================
def utm_epsg_from_latlon(lat: float, lon: float) -> int:
    """Infer UTM EPSG from a lat/lon (WGS84)."""
    zone = int(math.floor((lon + 180) / 6) + 1)
    is_northern = lat >= 0
    return (32600 if is_northern else 32700) + zone

def get_transformer(src_epsg: int, dst_epsg: int) -> Transformer:
    return Transformer.from_crs(CRS.from_epsg(src_epsg), CRS.from_epsg(dst_epsg), always_xy=True)

def wgs84_to_utm(lat: float, lon: float, utm_epsg: int) -> Tuple[float, float]:
    tr = get_transformer(4326, utm_epsg)
    e, n = tr.transform(lon, lat)
    return e, n

def utm_to_wgs84(easting: float, northing: float, utm_epsg: int) -> Tuple[float, float]:
    tr = get_transformer(utm_epsg, 4326)
    lon, lat = tr.transform(easting, northing)
    return lat, lon


# =========================
# Geometry helpers
# =========================
def interpolate_points_along_line(line: LineString, spacing_m: float) -> List[Point]:
    """Return Points every `spacing_m` along LineString (meters CRS). Includes start & end."""
    length = max(0.0, float(line.length))
    if spacing_m <= 0 or length == 0.0:
        return [line.interpolate(0.0), line.interpolate(length)]
    dists = list(np.arange(0.0, length + 1e-9, spacing_m))
    if dists[-1] < length:
        dists.append(length)
    return [line.interpolate(d) for d in dists]


# =========================
# Export helpers
# =========================
def build_points_dataframe(points_ll: List[Tuple[float, float]],
                           points_utm: List[Tuple[float, float]],
                           utm_epsg: int) -> pd.DataFrame:
    rows = []
    for idx, ((lat, lon), (e, n)) in enumerate(zip(points_ll, points_utm)):
        rows.append({
            "id": idx,
            "lat": lat,
            "lon": lon,
            "utm_easting": e,
            "utm_northing": n,
            "utm_epsg": utm_epsg
        })
    return pd.DataFrame(rows)

def export_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def export_kml(points_ll: List[Tuple[float, float]], line_ll: List[Tuple[float, float]]) -> bytes:
    """Plain KML."""
    kml = simplekml.Kml()
    ls = kml.newlinestring(name="Line")
    ls.coords = [(lon, lat) for lat, lon in line_ll]
    ls.altitudemode = simplekml.AltitudeMode.clamptoground
    ls.style.linestyle.width = 3
    for i, (lat, lon) in enumerate(points_ll):
        p = kml.newpoint(name=f"P{i}", coords=[(lon, lat)])
        p.altitudemode = simplekml.AltitudeMode.clamptoground
    return kml.kml().encode("utf-8")

def export_shapefile_zip(points_utm: List[Tuple[float, float]],
                         line_utm: List[Tuple[float, float]],
                         utm_epsg: int) -> bytes:
    """ZIP with points and line Shapefiles in given UTM CRS."""
    tmpdir = tempfile.mkdtemp()
    try:
        gdf_points = gpd.GeoDataFrame(
            {"id": list(range(len(points_utm)))},
            geometry=[Point(e, n) for e, n in points_utm],
            crs=f"EPSG:{utm_epsg}",
        )
        points_dir = os.path.join(tmpdir, "points")
        os.makedirs(points_dir, exist_ok=True)
        gdf_points.to_file(os.path.join(points_dir, "points.shp"), driver="ESRI Shapefile")

        gdf_line = gpd.GeoDataFrame(
            {"name": ["line"]},
            geometry=[LineString([(e, n) for e, n in line_utm])],
            crs=f"EPSG:{utm_epsg}",
        )
        line_dir = os.path.join(tmpdir, "line")
        os.makedirs(line_dir, exist_ok=True)
        gdf_line.to_file(os.path.join(line_dir, "line.shp"), driver="ESRI Shapefile")

        staging_root = os.path.join(tmpdir, "shapefiles")
        os.makedirs(staging_root, exist_ok=True)
        shutil.copytree(points_dir, os.path.join(staging_root, "points"))
        shutil.copytree(line_dir, os.path.join(staging_root, "line"))
        zip_base = os.path.join(tmpdir, "export_shapefiles")
        shutil.make_archive(zip_base, "zip", tmpdir, "shapefiles")

        with open(zip_base + ".zip", "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# =========================
# Streamlit UI (with persistence)
# =========================
st.set_page_config(page_title="Line Sampler (DualMap)", layout="wide")
st.title("Line Sampler")
st.caption("Enter two points, sample every N meters, view two synchronized maps (left basemap, right blank), export CSV/KML/Shapefile.")

# Keep generated output visible across reruns
if "generated" not in st.session_state:
    st.session_state.generated = False

with st.sidebar:
    st.header("Inputs")
    mode = st.radio("Input coordinate system", ["Lat/Lon (WGS84)", "UTM"], index=0)

    spacing = st.number_input(
        "Point spacing (meters)",
        min_value=0.0,
        value=20.0,
        step=1.0,
        help="0 = only endpoints"
    )

    st.markdown("---")
    st.subheader("Left map basemap")
    basemap = st.selectbox(
        "Background (left map)",
        ["OpenStreetMap", "OpenTopoMap", "ESRI World Imagery (satellite)", "ESRI World Street Map"],
        index=0,
    )

# Point inputs
st.subheader("Point A")
colA1, colA2, colA3 = st.columns(3)
if mode == "Lat/Lon (WGS84)":
    with colA1:
        A_lat = st.number_input("A Latitude", value=55.6761, format="%.8f")
    with colA2:
        A_lon = st.number_input("A Longitude", value=12.5683, format="%.8f")
    with colA3:
        st.write("")
    A_ll = LatLon(A_lat, A_lon)
else:
    with colA1:
        A_e = st.number_input("A Easting", value=720000.0, format="%.3f")
    with colA2:
        A_n = st.number_input("A Northing", value=6170000.0, format="%.3f")
    with colA3:
        A_epsg = st.number_input("UTM EPSG (for A & B)", value=32632, step=1)
    A_utm = UTM(A_e, A_n, A_epsg)

st.subheader("Point B")
colB1, colB2, colB3 = st.columns(3)
if mode == "Lat/Lon (WGS84)":
    with colB1:
        B_lat = st.number_input("B Latitude", value=55.60, format="%.8f")
    with colB2:
        B_lon = st.number_input("B Longitude", value=12.50, format="%.8f")
    with colB3:
        st.write("")
    B_ll = LatLon(B_lat, B_lon)
else:
    with colB1:
        B_e = st.number_input("B Easting", value=725000.0, format="%.3f")
    with colB2:
        B_n = st.number_input("B Northing", value=6175000.0, format="%.3f")
    with colB3:
        st.write("")
    B_utm = UTM(B_e, B_n, A_epsg)

st.markdown("---")
# Buttons: Generate / Reset
c_gen, c_reset = st.columns([1, 1])
with c_gen:
    if st.button("Generate", type="primary"):
        st.session_state.generated = True
with c_reset:
    if st.button("Reset"):
        st.session_state.generated = False

# =========================
# Main processing
# =========================
if st.session_state.generated:
    try:
        # Normalize inputs to UTM working CRS & also produce lat/lon for view
        if mode == "Lat/Lon (WGS84)":
            mid_lat = (A_ll.lat + B_ll.lat) / 2.0
            mid_lon = (A_ll.lon + B_ll.lon) / 2.0
            working_epsg = utm_epsg_from_latlon(mid_lat, mid_lon)

            A_e, A_n = wgs84_to_utm(A_ll.lat, A_ll.lon, working_epsg)
            B_e, B_n = wgs84_to_utm(B_ll.lat, B_ll.lon, working_epsg)

            line_ll = [(A_ll.lat, A_ll.lon), (B_ll.lat, B_ll.lon)]
            line_utm = [(A_e, A_n), (B_e, B_n)]
        else:
            # Single EPSG for both points (user picks in sidebar when UTM mode)
            working_epsg = int(A_epsg)
            line_utm = [(A_utm.easting, A_utm.northing), (B_utm.easting, B_utm.northing)]
            # Convert for view/exports
            A_lat, A_lon = utm_to_wgs84(A_utm.easting, A_utm.northing, working_epsg)
            B_lat, B_lon = utm_to_wgs84(B_utm.easting, B_utm.northing, working_epsg)
            line_ll = [(A_lat, A_lon), (B_lat, B_lon)]

        # Sample points in meters CRS
        line_geom_utm = LineString(line_utm)
        pts_utm_geom = interpolate_points_along_line(line_geom_utm, spacing)
        pts_utm = [(p.x, p.y) for p in pts_utm_geom]
        # Back to lat/lon
        pts_ll = [utm_to_wgs84(e, n, working_epsg) for (e, n) in pts_utm]

        # Build DataFrame and exports
        df_export = build_points_dataframe(pts_ll, pts_utm, working_epsg)
        csv_bytes = export_csv_bytes(df_export)
        kml_bytes = export_kml(points_ll=pts_ll, line_ll=line_ll)
        shp_zip_bytes = export_shapefile_zip(points_utm=pts_utm, line_utm=line_utm, utm_epsg=working_epsg)

        # Compute map center/zoom
        all_lats = [lat for (lat, lon) in line_ll] + [lat for (lat, lon) in pts_ll]
        all_lons = [lon for (lat, lon) in line_ll] + [lon for (lat, lon) in pts_ll]
        if all_lats:
            lat_min, lat_max = min(all_lats), max(all_lats)
            lon_min, lon_max = min(all_lons), max(all_lons)
            center = [(lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0]
            # crude zoom from span
            span_deg = max(1e-9, max(abs(lat_max - lat_min), abs(lon_max - lon_min)))
            zoom = int(max(2, min(18, math.log2(360.0 / span_deg))))
        else:
            center, zoom = [0.0, 0.0], 2

        # --- Dual synchronized maps ---
        dual = DualMap(location=center, tiles=None, zoom_start=zoom, control_scale=True)

        # Left: basemap
        if basemap == "OpenStreetMap":
            folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=False).add_to(dual.m1)
        elif basemap == "OpenTopoMap":
            folium.TileLayer(
                tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                attr="© OpenTopoMap, © OpenStreetMap contributors",
                name="OpenTopoMap", control=False
            ).add_to(dual.m1)
        elif basemap == "ESRI World Imagery (satellite)":
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
                name="ESRI Imagery", control=False
            ).add_to(dual.m1)
        elif basemap == "ESRI World Street Map":
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                attr="Source: Esri, HERE, Garmin, © OpenStreetMap contributors",
                name="ESRI Street", control=False
            ).add_to(dual.m1)

        # Right: blank (keep tiles=None on dual.m2)

        # Add red line + blue points to BOTH maps
        folium.PolyLine([(lat, lon) for (lat, lon) in line_ll], color="#ff0000", weight=4).add_to(dual.m1)
        folium.PolyLine([(lat, lon) for (lat, lon) in line_ll], color="#ff0000", weight=4).add_to(dual.m2)

        for i, (lat, lon) in enumerate(pts_ll):
            folium.CircleMarker(location=[lat, lon], radius=4, color="#0000ff",
                                fill=True, fillOpacity=1.0,
                                tooltip=f"P{i}: {lat:.6f}, {lon:.6f}").add_to(dual.m1)
            folium.CircleMarker(location=[lat, lon], radius=4, color="#0000ff",
                                fill=True, fillOpacity=1.0,
                                tooltip=f"P{i}: {lat:.6f}, {lon:.6f}").add_to(dual.m2)

        # Render maps as stable HTML (prevents temp-file errors & rerun flicker)
        map_html = dual.get_root().render()
        components.html(map_html, height=560, scrolling=False)

        # Downloads + summary
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("⬇️ CSV (lat/lon + UTM)", data=csv_bytes,
                               file_name="points_along_line.csv", mime="text/csv")
        with c2:
            st.download_button("⬇️ KML", data=kml_bytes,
                               file_name="line_and_points.kml", mime="application/vnd.google-earth.kml+xml")
        with c3:
            st.download_button("⬇️ Shapefile (ZIP)", data=shp_zip_bytes,
                               file_name="shapefiles.zip", mime="application/zip")

        st.success(
            f"Generated {len(pts_utm)} points — Working CRS: EPSG:{working_epsg} — "
            f"Line length ≈ {line_geom_utm.length:.2f} m"
        )
        st.dataframe(df_export.head(20), use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.info("Set inputs and click **Generate** to show the synchronized maps and downloads.")
