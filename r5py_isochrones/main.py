from __future__ import annotations

import datetime as dt

import r5py as r5
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from loguru import logger


WGS84 = "epsg:4326"


def isochrone_g(
    transport_network: r5.TransportNetwork,
    transport_modes: list[r5.TransportMode],
    origins: gpd.GeoDataFrame,
    time_bounds: list[float],
    grid: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame | None = None,
    departure: dt.datetime | None = None,
    snap_to_network: bool | int = False,
    **kwargs: dict,
) -> gpd.GeoDataFrame:
    """
    Compute grid-style isochrones from the given origin points.

    More specifically, given a GeoDataFrame of WGS84 origin points with a unique
    identifier column 'id' and a grid of WGS84 polygons with unique identifier column
    'id', whose polygons ideally cover the study area without overlaps
    and are clipped to coastlines, do the following.
    Choose one representative point from each grid cell to form a set of destination points.
    Alternatively, use ``destinations``, if given.
    For each origin and for each duration t (minutes) in the given list of time bounds,
    find all destinations that are at most t minutes away when departing from the
    origin at the given departure time (which defaults to the current datetime),
    using the given transport modes, and travelling through the given transport nework.
    Collect all grid cells containing those reachable destinations, dissolve them,
    and set that as the isochrone for that origin and duration.
    Return a GeoDataFrame with the following columns.

    - ``'from_id'``: origin point ID
    - ``'travel_time_percentile'``: string; defaluts to 'p50' if no percentiles are given
    - ``'time_bound'``: float; one of the given time bounds
    - ``'geometry'``: (Multi)Polygon isochrone.

    You can customise the isochrone calculation as follows.

    - Snap the origin points to the street network before routing if and only if
      ``snap_to_network``
      If ``True``, then the default search radius
      defined in com.conveyal.r5.streets.StreetLayer.LINK_RADIUS_METERS is used;
      if int, then use that many meters as the search radius for snapping.
    - Pass in any keyword arguments accepted by :class:`r5py.RegionalTask`,
      e.g. `departure_time_window`, `percentiles`, `max_time_walking`.

    NOTES:

    - All given GeoDataFrame will be converted to coordinate reference system (CRS) WGS84 before routing,
      then converted back to the CRS of the grid.
    - Uses r5py to do the routing.

    """
    if "id" not in set(origins.columns) & set(grid.columns):
        raise ValueError("Origins and grid must both contain an 'id' column")

    # Prepare inputs
    logger.info("Prepare inputs")
    time_bounds = sorted(set(time_bounds))
    origins = origins.to_crs(WGS84)
    final_crs = grid.crs
    grid = grid.to_crs(WGS84)
    if destinations is None:
        destinations = grid.assign(geometry=lambda x: x.representative_point())
    else:
        if "id" not in destinations.columns:
            raise ValueError("Destinations must contain an 'id' column")
        destinations = destinations.to_crs(WGS84)

    # Compute travel times
    logger.info("Compute travel times")
    ttm = r5.TravelTimeMatrixComputer(
        transport_network,
        origins=origins,
        destinations=destinations,
        departure=departure,
        transport_modes=transport_modes,
        snap_to_network=snap_to_network,
        **kwargs,
    )
    f = (
        ttm.compute_travel_times()
        .dropna()
        .rename(columns={"travel_time": "travel_time_p50"})
        # Melt in case of multiple travel time percentiles
        .melt(id_vars=["from_id", "to_id"], var_name="pctile", value_name="travel_time")
        .assign(pctile=lambda x: x["pctile"].str.split("_").str[-1])
    )
    if f.empty:
        return gpd.GeoDataFrame()

    # Build isochrones from the grid cells of the reachable points
    logger.info("Build isochrones")
    records = []
    for (from_id, pctile), group in f.groupby(["from_id", "pctile"]):
        for time_bound in time_bounds:
            iso = grid.merge(
                group.loc[lambda x: x["travel_time"] <= time_bound].rename(
                    columns={"to_id": "id"}
                )
            ).dissolve()
            records.append(
                {
                    "from_id": from_id,
                    "travel_time_percentile": pctile,
                    "time_bound": time_bound,
                    "geometry": iso["geometry"].iat[0] if not iso.empty else np.nan,
                }
            )

    return gpd.GeoDataFrame(pd.DataFrame.from_records(records), crs=WGS84).to_crs(final_crs)

def get_osm_nodes(transport_network: r5.TransportNetwork) -> gpd.GeoDataFrame:
    """
    Return the OSM nodes underlying the given transport network.
    Include in the GeoDataFrame an ID column 'id' that is simply the index of the
    GeoDataFrame.
    """
    import com.conveyal.r5

    k = com.conveyal.r5.streets.VertexStore.FIXED_FACTOR
    v = transport_network._transport_network.streetLayer.vertexStore
    lonlats = zip(list(v.fixedLons.toArray()), list(v.fixedLats.toArray()))
    nodes = gpd.GeoDataFrame(
        geometry=[shapely.Point(lon / k, lat / k) for lon, lat in lonlats],
        crs=WGS84,
    )
    nodes["id"] = nodes.index
    return nodes

def isochrone_ch(
    transport_network: r5.TransportNetwork,
    transport_modes: list[r5.TransportMode],
    origins: gpd.GeoDataFrame,
    time_bounds: list[float],
    destinations: gpd.GeoDataFrame | None = None,
    departure: dt.datetime|None=None,
    snap_to_network: bool|int=False,
    sample_frac: float=0.8,
    concave_hull_ratio=0.15,
    **kwargs: dict,
) -> gpd.GeoDataFrame:
    """
    Compute concave-hull-style isochrones from the given origin points.

    More specifically, given a GeoDataFrame of WGS84 origin points with a unique
    identifier column 'id' do the following.
    Choose a set of destination points by sampling ``sample_frac`` of all OSM nodes in
    the underlying network.
    Alternatively, use ``destinations``, if given.
    For each origin and for each duration t (minutes) in the given list of time bounds,
    find all destinations that are at most t minutes away when departing from the
    origin at the given departure time (which defaults to the current datetime),
    using the given transport modes, and travelling through the given transport nework.
    Collect all such destinations, compute their concave hull using the given
    concave hull ratio, and set that as the isochrone for that origin and duration.
    Return a GeoDataFrame with the following columns.

    - ``'from_id'``: origin point ID
    - ``'travel_time_percentile'``: string; defaluts to 'p50' if no percentiles are given
    - ``'time_bound'``: float; one of the given time bounds
    - ``'geometry'``: (Multi)Polygon isochrone.

    You can customise the isochrone calculation as follows.

    - Snap the origin points to the street network before routing if and only if
      ``snap_to_network``
      If ``True``, then the default search radius
      defined in com.conveyal.r5.streets.StreetLayer.LINK_RADIUS_METERS is used;
      if int, then use that many meters as the search radius for snapping.
    - Pass in any keyword arguments accepted by :class:`r5py.RegionalTask`,
      e.g. `departure_time_window`, `percentiles`, `max_time_walking`.

    NOTES:

    - All given GeoDataFrames will be converted to coordinate reference system (CRS) WGS84 before routing,
      then converted back to the CRS of the origin points.
    - Uses r5py to do the routing.
    - Takes longer than :func:`isochrone_g` and often excludes disconnected components.

    """
    if "id" not in origins.columns:
        raise ValueError("Origins must contain an 'id' column")

    logger.info("Prepare inputs")
    final_crs = origins.crs
    time_bounds = sorted(set(time_bounds))
    if destinations is None:
        destinations = get_osm_nodes(transport_network).sample(frac=sample_frac, random_state=1)
    else:
        if "id" not in destinations.columns:
            raise ValueError("Destinations must contain an 'id' column")
        destinations = destinations.to_crs(WGS84)

    # Compute travel times
    logger.info("Compute travel times")
    ttm = r5.TravelTimeMatrixComputer(
        transport_network,
        origins=origins.to_crs(WGS84),
        destinations=destinations,
        departure=departure,
        transport_modes=transport_modes,
        snap_to_network=snap_to_network,
        **kwargs,
    )
    f = (
        ttm.compute_travel_times()
        .dropna()
        .rename(columns={"travel_time": "travel_time_p50"})
        # Melt in case of multiple travel time percentiles
        .melt(id_vars=["from_id", "to_id"], var_name="pctile", value_name="travel_time")
        .assign(pctile=lambda x: x["pctile"].str.split("_").str[-1])
    )
    if f.empty:
        return gpd.GeoDataFrame()

    # Build isochrones as concave hulls of reachable points
    logger.info("Build isochrones")
    records = []
    for (from_id, pctile), group in f.groupby(["from_id", "pctile"]):
        for time_bound in time_bounds:
            reachable_nodes = destinations.merge(
                group
                .loc[lambda x: x["travel_time"] <= time_bound]
                .rename(columns={"to_id": "id"})
            )
            iso = shapely.concave_hull(reachable_nodes.unary_union, ratio=concave_hull_ratio)
            records.append(
                {
                    "from_id": from_id,
                    "travel_time_percentile": pctile,
                    "time_bound": time_bound,
                    "geometry": iso if not iso.is_empty else np.nan,
                }
            )

    logger.info("Build GeoDataFrame")
    return gpd.GeoDataFrame(pd.DataFrame.from_records(records), crs=WGS84).to_crs(final_crs)

