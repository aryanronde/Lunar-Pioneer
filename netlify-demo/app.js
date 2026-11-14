/* Static frontend fixture. No scientific or network calculation runs here. */
(() => {
  "use strict";

  const demo = {
    date: "2026-10-01",
    start: "04:05",
    lastSample: "22:05",
    cadenceMinutes: 60,
    earthVisibleSamples: 19,
    site: {
      name: "Site06 — Nobile Rim 1",
      latitudeDeg: -85.4381,
      longitudeDeg: 37.36667,
      frame: "MOON_ME_DE421",
      projectedXMetres: 84000.01002578877,
      projectedYMetres: 110000.01079172149,
    },
    knownGeometry: [
      { timestamp: "2026-10-01T04:05:00Z", target: "Sun", azimuth: null, elevation: -10.2788, horizon: 0.2343, clearance: -10.5131, status: "Blocked" },
      { timestamp: "2026-10-01T04:05:00Z", target: "Earth", azimuth: null, elevation: 13.1260, horizon: 9.5764, clearance: 3.5496, status: "Visible" },
      { timestamp: "2026-10-01T18:05:00Z", target: "Earth", azimuth: null, elevation: 13.1440, horizon: 9.9517, clearance: 3.1923, status: "Visible" },
    ],
  };

  const mapSvg = document.getElementById("mission-map");
  const plot = { left: 174, top: 32, size: 420 };
  const baseSpanKm = 24;
  const rangeKm = 10;
  const center = {
    x: demo.site.projectedXMetres / 1000,
    y: demo.site.projectedYMetres / 1000,
  };
  let view = { x: center.x, y: center.y, span: baseSpanKm };

  function svgElement(tag, attributes = {}, text = "") {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    if (text) node.textContent = text;
    return node;
  }

  function mapX(xKm) {
    return plot.left + ((xKm - (view.x - view.span / 2)) / view.span) * plot.size;
  }

  function mapY(yKm) {
    return plot.top + plot.size - ((yKm - (view.y - view.span / 2)) / view.span) * plot.size;
  }

  function drawMap() {
    mapSvg.replaceChildren();
    mapSvg.append(svgElement("rect", { x: 0, y: 0, width: 800, height: 530, fill: "#edf3fa" }));
    mapSvg.append(svgElement("rect", { x: plot.left, y: plot.top, width: plot.size, height: plot.size, rx: 2, fill: "#f5f8fc", stroke: "#cbd8e6" }));

    const xStart = view.x - view.span / 2;
    const yStart = view.y - view.span / 2;
    const step = view.span <= 30 ? 5 : view.span <= 60 ? 10 : 20;
    const xTickStart = Math.ceil(xStart / step) * step;
    const yTickStart = Math.ceil(yStart / step) * step;
    for (let x = xTickStart; x <= view.x + view.span / 2; x += step) {
      const px = mapX(x);
      mapSvg.append(svgElement("line", { x1: px, y1: plot.top, x2: px, y2: plot.top + plot.size, stroke: "#dce5ef", "stroke-width": 1 }));
      mapSvg.append(svgElement("text", { x: px, y: plot.top + plot.size + 22, "text-anchor": "middle", class: "map-tick" }, x.toFixed(0)));
    }
    for (let y = yTickStart; y <= view.y + view.span / 2; y += step) {
      const py = mapY(y);
      mapSvg.append(svgElement("line", { x1: plot.left, y1: py, x2: plot.left + plot.size, y2: py, stroke: "#dce5ef", "stroke-width": 1 }));
      mapSvg.append(svgElement("text", { x: plot.left - 12, y: py + 4, "text-anchor": "end", class: "map-tick" }, y.toFixed(0)));
    }

    const points = [];
    for (let i = 0; i <= 360; i += 1) {
      const angle = (i / 360) * Math.PI * 2;
      const x = center.x + rangeKm * Math.cos(angle);
      const y = center.y + rangeKm * Math.sin(angle);
      points.push(`${i === 0 ? "M" : "L"}${mapX(x).toFixed(2)},${mapY(y).toFixed(2)}`);
    }
    points.push("Z");
    mapSvg.append(svgElement("path", { d: points.join(" "), fill: "#e6533b0c", stroke: "#e6533b", "stroke-width": 2.5, "stroke-dasharray": "8 6" }));

    const siteX = mapX(center.x);
    const siteY = mapY(center.y);
    mapSvg.append(svgElement("circle", { cx: siteX, cy: siteY, r: 10, fill: "#e6533b", stroke: "#ffffff", "stroke-width": 3 }));
    mapSvg.append(svgElement("circle", { cx: siteX, cy: siteY, r: 17, fill: "none", stroke: "#e6533b", "stroke-width": 1, opacity: 0.35 }));
    mapSvg.append(svgElement("text", { x: siteX + 16, y: siteY - 15, class: "map-site-label" }, "SITE06"));

    mapSvg.append(svgElement("text", { x: plot.left + plot.size / 2, y: 510, "text-anchor": "middle", class: "map-axis-label" }, "South-polar stereographic X (km) · +X ≈ 90°E"));
    const yLabel = svgElement("text", { x: 26, y: plot.top + plot.size / 2, "text-anchor": "middle", class: "map-axis-label", transform: `rotate(-90 26 ${plot.top + plot.size / 2})` }, "South-polar stereographic Y (km) · +Y toward 0° longitude");
    mapSvg.append(yLabel);
    mapSvg.append(svgElement("text", { x: 755, y: 30, "text-anchor": "end", class: "map-orientation" }, "↑ +Y · 0° lon"));
    mapSvg.append(svgElement("text", { x: 755, y: 47, "text-anchor": "end", class: "map-orientation" }, "→ +X · 90°E"));
    mapSvg.append(svgElement("text", { x: plot.left, y: 20, class: "map-coordinate-caption" }, `Projected center · X ${center.x.toFixed(3)} km · Y ${center.y.toFixed(3)} km`));
  }

  function drawTimeline() {
    const svg = document.getElementById("visibility-timeline");
    svg.replaceChildren();
    const left = 145;
    const right = 960;
    const xFor = (index) => left + (index / 18) * (right - left);
    const sunY = 64;
    const earthY = 128;
    svg.append(svgElement("line", { x1: left, y1: sunY, x2: right, y2: sunY, stroke: "#e0e6ed", "stroke-width": 2 }));
    svg.append(svgElement("line", { x1: left, y1: earthY, x2: right, y2: earthY, stroke: "#e0e6ed", "stroke-width": 2 }));
    svg.append(svgElement("text", { x: 18, y: sunY + 4, class: "timeline-label" }, "SUN"));
    svg.append(svgElement("text", { x: 18, y: earthY + 4, class: "timeline-label" }, "EARTH"));

    // The sample includes one confirmed Sun state, so later Sun points are omitted.
    svg.append(svgElement("circle", { cx: xFor(0), cy: sunY, r: 7, fill: "#bd3434", stroke: "white", "stroke-width": 2 }));
    svg.append(svgElement("text", { x: xFor(0) + 10, y: sunY - 13, class: "timeline-small" }, "Blocked sample"));

    for (let index = 0; index < demo.earthVisibleSamples; index += 1) {
      svg.append(svgElement("circle", { cx: xFor(index), cy: earthY, r: 5.5, fill: "#16834a", stroke: "white", "stroke-width": 1.5 }));
    }
    svg.append(svgElement("line", { x1: left, y1: 170, x2: right, y2: 170, stroke: "#99a9bb", "stroke-width": 1 }));
    [0, 6, 12, 18].forEach((index) => {
      const x = xFor(index);
      const hour = 4 + index;
      const label = `${String(hour).padStart(2, "0")}:05`;
      svg.append(svgElement("line", { x1: x, y1: 166, x2: x, y2: 175, stroke: "#8999ac", "stroke-width": 1 }));
      svg.append(svgElement("text", { x, y: 191, "text-anchor": index === 0 ? "start" : index === 18 ? "end" : "middle", class: "timeline-time" }, label));
    });
  }

  function renderGeometry() {
    const body = document.getElementById("geometry-body");
    body.replaceChildren();
    for (const row of demo.knownGeometry) {
      const tr = document.createElement("tr");
      const time = row.timestamp.replace("T", " ").replace("Z", " UTC");
      const targetClass = row.target.toLowerCase();
      const cells = [
        time,
        `<span class="target-tag ${targetClass}">${row.target}</span>`,
        row.azimuth === null ? "—" : `${row.azimuth.toFixed(4)}°`,
        `${row.elevation >= 0 ? "+" : ""}${row.elevation.toFixed(4)}°`,
        `${row.horizon >= 0 ? "+" : ""}${row.horizon.toFixed(4)}°`,
        `${row.clearance >= 0 ? "+" : ""}${row.clearance.toFixed(4)}°`,
        `<span class="status-pill ${row.status.toLowerCase()}">${row.status}</span>`,
      ];
      cells.forEach((value, index) => {
        const td = document.createElement("td");
        if (index === 1 || index === 6) td.innerHTML = value;
        else td.textContent = value;
        tr.append(td);
      });
      body.append(tr);
    }
  }

  function announceDemoRun(event) {
    event.preventDefault();
    const date = document.getElementById("analysis-date").value;
    const start = document.getElementById("start-time").value;
    const end = document.getElementById("end-time").value;
    const cadence = document.getElementById("cadence").value;
    const status = document.getElementById("run-status");
    status.textContent = `Demo controls set to ${date}, ${start}–${end} UTC, ${cadence}. The dashboard continues to show the saved 2026-10-01 sample; no scientific calculation is running.`;
  }

  function bindMapNavigation() {
    let drag = null;
    mapSvg.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      drag = { x: event.clientX, y: event.clientY, viewX: view.x, viewY: view.y };
      mapSvg.classList.add("dragging");
      mapSvg.setPointerCapture(event.pointerId);
    });
    mapSvg.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const bounds = mapSvg.getBoundingClientRect();
      const pixelsPerCss = bounds.width / 800;
      const dxKm = ((event.clientX - drag.x) / pixelsPerCss / plot.size) * view.span;
      const dyKm = ((event.clientY - drag.y) / pixelsPerCss / plot.size) * view.span;
      view.x = drag.viewX - dxKm;
      view.y = drag.viewY + dyKm;
      drawMap();
    });
    const stop = () => { drag = null; mapSvg.classList.remove("dragging"); };
    mapSvg.addEventListener("pointerup", stop);
    mapSvg.addEventListener("pointercancel", stop);
    mapSvg.addEventListener("wheel", (event) => {
      event.preventDefault();
      const bounds = mapSvg.getBoundingClientRect();
      const cssScale = bounds.width / 800;
      const px = (event.clientX - bounds.left) / cssScale;
      const py = (event.clientY - bounds.top) / cssScale;
      const relativeX = (px - plot.left) / plot.size;
      const relativeY = 1 - (py - plot.top) / plot.size;
      const anchorX = view.x + (relativeX - 0.5) * view.span;
      const anchorY = view.y + (relativeY - 0.5) * view.span;
      const newSpan = Math.min(100, Math.max(4, view.span * (event.deltaY < 0 ? 0.85 : 1.18)));
      view.x = anchorX - (relativeX - 0.5) * newSpan;
      view.y = anchorY - (relativeY - 0.5) * newSpan;
      view.span = newSpan;
      drawMap();
    }, { passive: false });
    document.getElementById("map-reset").addEventListener("click", () => {
      view = { x: center.x, y: center.y, span: baseSpanKm };
      drawMap();
    });
  }

  document.getElementById("demo-controls").addEventListener("submit", announceDemoRun);
  drawMap();
  drawTimeline();
  renderGeometry();
  bindMapNavigation();
})();
