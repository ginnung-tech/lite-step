"""WGS84 -> projected CRS for ``IfcMapConversion``.

The declared EPSG code and the eastings/northings must agree: a surveyor loads
the IFC against the CRS it names. The helper uses Redfearn transverse-Mercator
math (not a spherical approximation) and picks EPSG:25832 for Denmark.
"""
from __future__ import annotations

import pytest

from lite_step.ifc.generator import _utm_crs_for, _wgs84_to_utm

# Reference values from the Redfearn implementation the site pipeline uses for
# its EPSG:25832 raster bboxes (``_wgs84_to_utm32n``), pinned as constants.
_REFERENCE_25832 = [
    # (lat, lng, easting, northing)
    (55.6761, 12.5683, 724351.9286492015, 6175804.022863195),   # Copenhagen
    (55.67, 12.57, 724493.7164610737, 6175131.129083958),
    (56.1629, 10.2039, 574766.3925123963, 6224862.6488411315),  # Aarhus
]


@pytest.mark.parametrize("lat,lng,e_ref,n_ref", _REFERENCE_25832)
def test_danish_coordinates_match_the_25832_reference_within_centimetres(lat, lng, e_ref, n_ref):
    e, n, epsg = _wgs84_to_utm(lat, lng)
    assert epsg == 25832
    assert e == pytest.approx(e_ref, abs=0.01)
    assert n == pytest.approx(n_ref, abs=0.01)


def test_copenhagen_is_not_spherical_error_away():
    """The spherical helper put Copenhagen ~10^4 m from the truth."""
    _e, n, _ = _wgs84_to_utm(55.6761, 12.5683)
    assert abs(n - 6175804.02) < 1.0


def test_danish_box_is_one_crs_even_east_of_twelve_degrees():
    assert _utm_crs_for(55.68, 12.57)[1] == 25832   # Copenhagen: zone 33 by longitude
    assert _utm_crs_for(55.12, 14.90)[1] == 25832   # Bornholm
    assert _utm_crs_for(56.16, 9.0)[1] == 25832


def test_outside_denmark_picks_the_zone_by_longitude():
    assert _utm_crs_for(52.52, 13.405)[1] == 32633    # Berlin
    assert _utm_crs_for(45.0, 3.0)[1] == 32631
    assert _utm_crs_for(-33.87, 151.21)[1] == 32756  # southern hemisphere: 327zz


def test_central_meridian_point_outside_denmark_is_exact():
    """On the zone-31 central meridian easting is exactly the 500 km false
    easting; northing at 45 N is 0.9996 x the meridian arc (4 984 944.378 m)."""
    e, n, epsg = _wgs84_to_utm(45.0, 3.0)
    assert epsg == 32631
    assert e == pytest.approx(500_000.0, abs=1e-6)
    assert n == pytest.approx(0.9996 * 4_984_944.378, abs=0.01)


def test_southern_hemisphere_adds_the_false_northing():
    e_n, n_n, _ = _wgs84_to_utm(33.87, 151.21)
    e_s, n_s, epsg = _wgs84_to_utm(-33.87, 151.21)
    assert epsg == 32756
    assert e_s == pytest.approx(e_n, abs=1e-6)
    assert n_s == pytest.approx(10_000_000 - n_n, abs=1e-6)
