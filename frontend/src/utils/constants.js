export const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
export const WS_URL = import.meta.env.VITE_WS_URL || "ws://127.0.0.1:8000/ws/live";
export const PUBLIC_MAP_STYLE = {
  version: 8,
  sources: {
    voyager: {
      type: "raster",
      tiles: [
        "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://d.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
      ],
      tileSize: 256,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
    },
  },
  layers: [
    {
      id: "ocean-background",
      type: "background",
      paint: {
        "background-color": "#07233b",
      },
    },
    {
      id: "voyager-raster",
      type: "raster",
      source: "voyager",
      minzoom: 0,
      maxzoom: 20,
      paint: {
        "raster-opacity": 0.96,
        "raster-saturation": -0.1,
        "raster-contrast": 0.06,
      },
    },
  ],
};
