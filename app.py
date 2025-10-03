# -*- coding: utf-8 -*-
"""
Line Sampler (Streamlit + pydeck)
- Input two points (Lat/Lon or UTM)
- Sample points every N meters along the line
- Export CSV (lat/lon + UTM), KML, and Shapefile (ZIP)
- Preview map with a single basemap selector (or blank)
- Line = red, Points = blue (fixed styles)
"""

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from shapely.geometry import LineString, Point
from pyproj import CRS, Transformer
import geopandas as gpd
import simplekml


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
# Map preview (pydeck)
# =========================
def _autoscale_zoom(lat_min, lon_min, lat_max, lon_max) -> float:
    """Crude zoom estimate from bounds."""
    span_deg = max(1e-9, max(abs(lat_max - lat_min), abs(lon_max - lon_min)))
    return float(max(2, min(18, math.log2(360.0 / span_deg))))

def make_preview_deck(points_ll: List[Tuple[float, float]],
                      line_ll: List[Tuple[float, float]],
                      basemap: str = "OpenStreetMap") -> pdk.Deck:
    """pydeck preview with optional basemap. Line red, points blue, fixed sizes."""
    point_df = pd.DataFrame(points_ll, columns=["lat", "lon"]) if points_ll else pd.DataFrame(columns=["lat","lon"])
    path_df = pd.DataFrame(
        [{"path": [{"lon": lon, "lat": lat} for lat, lon in line_ll]}]
    ) if line_ll else pd.DataFrame([{"path": []}])

    layers = []

    # Basemap layer (bottom) — use single-host URLs to avoid subdomain quirks.
    basemap_urls = {
        "None (blank)": None,
        "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "OpenTopoMap": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        "ESRI World Imagery (satellite)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "ESRI World Street Map": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    }
    url = basemap_urls.get(basemap, basemap_urls["OpenStreetMap"])
    if url is not None:
        layers.append(pdk.Layer(
            "TileLayer",
            data=url,
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
            opacity=1.0,
        ))

    # Points (BLUE)
    if not point_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=point_df,
            get_position=["lon", "lat"],
            get_radius=8,
            get_fill_color=[0, 0, 255, 255],  # RGBA
            pickable=True,
        ))

    # Line (RED)
    if not path_df.empty:
        layers.append(pdk.Layer(
            "PathLayer",
            data=path_df,
            get_path="path",
            width_min_pixels=5,
            get_color=[255, 0, 0, 255],       # RGBA
            pickable=False,
        ))

    # View (fit to data)
    all_lats, all_lons = [], []
    if not point_df.empty:
        all_lats += point_df["lat"].tolist()
        all_lons += point_df["lon"].tolist()
    for lat, lon in line_ll:
        all_lats.append(lat)
        all_lons.append(lon)

    if all_lats:
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_center = (lat_min + lat_max) / 2.0
        lon_center = (lon_min + lon_max) / 2.0
        zoom = _autoscale_zoom(lat_min, lon_min, lat_max, lon_max)
    else:
        lat_center, lon_center, zoom = 0.0, 0.0, 2

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=zoom),
        tooltip={"text": "{lat}, {lon}"}
    )


# =========================
# Exporters
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
    """Plain KML (colors handled only in preview)."""
    kml = simplekml.Kml()
    # Line
    ls = kml.newlinestring(name="Line")
    ls.coords = [(lon, lat) for lat, lon in line_ll]
    ls.altitudemode = simplekml.AltitudeMode.clamptoground
    ls.style.linestyle.width = 3
    # Points
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
        # Points
        gdf_points = gpd.GeoDataFrame(
            {"id": list(range(len(points_utm)))},
            geometry=[Point(e, n) for e, n in points_utm],
            crs=f"EPSG:{utm_epsg}",
        )
        points_dir = os.path.join(tmpdir, "points")
        os.makedirs(points_dir, exist_ok=True)
        gdf_points.to_file(os.path.join(points_dir, "points.shp"), driver="ESRI Shapefile")

        # Line
        gdf_line = gpd.GeoDataFrame(
            {"name": ["line"]},
            geometry=[LineString([(e, n) for e, n in line_utm])],
            crs=f"EPSG:{utm_epsg}",
        )
        line_dir = os.path.join(tmpdir, "line")
        os.makedirs(line_dir, exist_ok=True)
        gdf_line.to_file(os.path.join(line_dir, "line.shp"), driver="ESRI Shapefile")

        # Zip
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
# Streamlit UI
# =========================
st.set_page_config(page_title="Line Sampler (Lat/Lon & UTM)", layout="centered")
st.title("Line Sampler")
st.caption("Enter two points, sample every N meters, and export Shapefile, KML, and CSV.")

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
    st.subheader("Basemap (preview only)")
    basemap = st.selectbox(
        "Background map",
        ["OpenStreetMap", "OpenTopoMap", "ESRI World Imagery (satellite)", "ESRI World Street Map", "None (blank)"],
        index=0,
    )

    st.markdown("---")
    st.subheader("Output UTM CRS")
    if mode == "Lat/Lon (WGS84)":
        utm_choice = st.selectbox("UTM EPSG", ["Auto from midpoint", "Manually specify"], index=0)
        if utm_choice == "Manually specify":
            output_utm_epsg = st.number_input("Output UTM EPSG (e.g., 32632)", value=32632, step=1)
        else:
            output_utm_epsg = None
    else:
        st.caption("When UTM is the input, outputs use the same EPSG.")
        output_utm_epsg = None

# Inputs for Point A
st.subheader("Point A")
colA1, colA2, colA3 = st.columns(3)
if mode == "Lat/Lon (WGS84)":
    with colA1:
        A_lat = st.number_input("A Latitude", value=55.6761, format="%.8f")
    with colA2:
        A_lon = st.number_input("A Longitude", value=12.5683, format="%.8f")
    with colA3:
        st.write("")  # spacer
    A = LatLon(A_lat, A_lon)
else:
    with colA1:
        A_e = st.number_input("A Easting", value=720000.0, format="%.3f")
    with colA2:
        A_n = st.number_input("A Northing", value=6170000.0, format="%.3f")
    with colA3:
        A_epsg = st.number_input("A UTM EPSG (e.g., 32632)", value=32632, step=1)
    A = UTM(A_e, A_n, A_epsg)

# Inputs for Point B
st.subheader("Point B")
colB1, colB2, colB3 = st.columns(3)
if mode == "Lat/Lon (WGS84)":
    with colB1:
        B_lat = st.number_input("B Latitude", value=55.60, format="%.8f")
    with colB2:
        B_lon = st.number_input("B Longitude", value=12.50, format="%.8f")
    with colB3:
        st.write("")  # spacer
    B = LatLon(B_lat, B_lon)
else:
    with colB1:
        B_e = st.number_input("B Easting", value=725000.0, format="%.3f")
    with colB2:
        B_n = st.number_input("B Northing", value=6175000.0, format="%.3f")
    with colB3:
        B_epsg = st.number_input("B UTM EPSG (must match A)", value=A_epsg if 'A_epsg' in locals() else 32632, step=1)
    B = UTM(B_e, B_n, B_epsg)

st.markdown("---")
go = st.button("Generate")

if go:
    try:
        # Normalize to UTM working CRS
        if mode == "Lat/Lon (WGS84)":
            # Choose UTM EPSG
            if output_utm_epsg is None:
                mid_lat = (A.lat + B.lat) / 2.0
                mid_lon = (A.lon + B.lon) / 2.0
                working_epsg = utm_epsg_from_latlon(mid_lat, mid_lon)
            else:
                working_epsg = int(output_utm_epsg)

            A_e, A_n = wgs84_to_utm(A.lat, A.lon, working_epsg)
            B_e, B_n = wgs84_to_utm(B.lat, B.lon, working_epsg)

            line_utm = [(A_e, A_n), (B_e, B_n)]
            line_ll = [(A.lat, A.lon), (B.lat, B.lon)]

        else:  # UTM input
            if A.epsg != B.epsg:
                st.error("UTM EPSG for Point A and Point B must match.")
                st.stop()
            working_epsg = int(A.epsg)
            line_utm = [(A.easting, A.northing), (B.easting, B.northing)]

            # Convert to Lat/Lon for preview/CSV/KML
            A_lat, A_lon = utm_to_wgs84(A.easting, A.northing, working_epsg)
            B_lat, B_lon = utm_to_wgs84(B.easting, B.northing, working_epsg)
            line_ll = [(A_lat, A_lon), (B_lat, B_lon)]

        # Build geometry in meters CRS and sample points
        line_geom_utm = LineString(line_utm)
        pts_utm_geom = interpolate_points_along_line(line_geom_utm, spacing)

        # Prepare outputs
        pts_utm = [(p.x, p.y) for p in pts_utm_geom]
        pts_ll = [utm_to_wgs84(e, n, working_epsg) for (e, n) in pts_utm]  # (lat, lon)

        # DataFrame for export
        df_export = build_points_dataframe(pts_ll, pts_utm, working_epsg)

        # Preview map
        deck = make_preview_deck(
            points_ll=pts_ll,
            line_ll=line_ll,
            basemap=basemap,
        )
        st.pydeck_chart(deck)

        # Downloads
        csv_bytes = export_csv_bytes(df_export)
        st.download_button(
            "Download CSV (lat/lon + UTM)",
            data=csv_bytes,
            file_name="points_along_line.csv",
            mime="text/csv"
        )

        kml_bytes = export_kml(points_ll=pts_ll, line_ll=line_ll)
        st.download_button(
            "Download KML",
            data=kml_bytes,
            file_name="line_and_points.kml",
            mime="application/vnd.google-earth.kml+xml"
        )

        shp_zip_bytes = export_shapefile_zip(points_utm=pts_utm, line_utm=line_utm, utm_epsg=working_epsg)
        st.download_button(
            "Download Shapefile (ZIP)",
            data=shp_zip_bytes,
            file_name="shapefiles.zip",
            mime="application/zip"
        )

        # Summary + preview table
        st.success(
            f"Generated {len(pts_utm)} points. Working CRS: EPSG:{working_epsg}. "
            f"Line length ≈ {line_geom_utm.length:.2f} m."
        )
        st.dataframe(df_export.head(20))

    except Exception as e:
        st.error(f"Something went wrong: {e}")
