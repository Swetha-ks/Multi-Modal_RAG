"""
maps.py — Free map-based address selection (Leaflet/OpenStreetMap via
folium + streamlit-folium) and a delivery-route visualization for the
Orders page. No paid API key required anywhere in this module.
"""

import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

WAREHOUSE_LOCATION = (12.9716, 77.5946)
WAREHOUSE_LABEL = "ShopAI Fulfillment Center"

_geolocator = Nominatim(user_agent="shopai_ecommerce_app")


def reverse_geocode(lat: float, lon: float) -> str:
    try:
        location = _geolocator.reverse((lat, lon), timeout=5, language="en")
        if location:
            return location.address
    except (GeocoderTimedOut, GeocoderServiceError, Exception):
        pass
    return f"Lat {lat:.5f}, Lon {lon:.5f} (address lookup unavailable — check your connection)"


def build_picker_map(center=None, marker=None):
    center = center or WAREHOUSE_LOCATION
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")
    if marker:
        folium.Marker(
            marker, tooltip="Delivery location", icon=folium.Icon(color="red", icon="home")
        ).add_to(m)
    return m


def _interpolate(p1, p2, fraction):
    return (p1[0] + (p2[0] - p1[0]) * fraction, p1[1] + (p2[1] - p1[1]) * fraction)


def build_tracking_map(delivery_location, progress_fraction: float):
    if not delivery_location:
        return None

    midpoint = _interpolate(WAREHOUSE_LOCATION, delivery_location, 0.5)
    m = folium.Map(location=midpoint, zoom_start=11, tiles="OpenStreetMap")

    folium.Marker(
        WAREHOUSE_LOCATION, tooltip=WAREHOUSE_LABEL,
        icon=folium.Icon(color="blue", icon="industry", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        delivery_location, tooltip="Delivery address",
        icon=folium.Icon(color="green", icon="home"),
    ).add_to(m)
    folium.PolyLine(
        [WAREHOUSE_LOCATION, delivery_location], color="#D4AF37", weight=3, opacity=0.8, dash_array="8"
    ).add_to(m)

    current_pos = _interpolate(WAREHOUSE_LOCATION, delivery_location, progress_fraction)
    folium.Marker(
        current_pos, tooltip="Current order position (estimated)",
        icon=folium.Icon(color="orange", icon="truck", prefix="fa"),
    ).add_to(m)

    m.fit_bounds([WAREHOUSE_LOCATION, delivery_location], padding=(30, 30))
    return m
