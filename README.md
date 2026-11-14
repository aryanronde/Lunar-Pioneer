# CLPS Lunar Mission Browser and LUMEN

This repository contains two related but distinct pieces of software:

The **CLPS Lunar Mission Browser is the primary application**. LUMEN is an
experimental research extension, not an operational mission design. **Site06
(Nobile Rim 1) is the supported candidate** for the current CLPS analysis; it is
not proven suitable. Site07 material is experimental and review-only, not a
production candidate.

- **CLPS Lunar Mission Browser**: a Streamlit dashboard for exploring sampled
  Sun illumination and Earth communication visibility at Site06, with optional
  NASA LOLA terrain context. Site06 — Nobile Rim 1 is currently the only
  supported analysis site; other landing sites are not implemented.
- **LUMEN (Lunar Utility through Mirror-Enhanced Networks)**: a separate
  research model for an elevated lunar charging station, rover energy
  management, and a steerable orbital reflector that redirects sunlight to a
  station or rover.

LUMEN investigates a specific south-polar operations problem: terrain can block
direct sunlight even when sunlight is available elsewhere. The model asks
whether a reflector and receiver could route some solar energy to a lunar
surface asset, how much of that energy may reach a receiver, and whether the
resulting electrical energy could support station operations or rover safe
return. It is a research and accounting pipeline, not an operational mission
design or a claim of demonstrated hardware performance.

## Project layout

```text
app.py                         Streamlit entry point; CLPS dashboard and separate LUMEN section
core/                          CLPS geometry, terrain, analysis, and all LUMEN calculations
core/lumen/                    LUMEN orbit, visibility, optics, power, energy, simulation, audit, site, UI
data/spice/                    Locally provisioned SPICE kernels (not included in GitHub upload)
reports/                       Saved site evidence and generated LUMEN demo reports
scripts/lumen_stage12_demo.py  Deterministic synthetic demo/report generator
tests/                         CLPS and LUMEN unittest suites
requirements.txt               Project runtime dependencies
```

The CLPS mission pipeline and its saved horizon reports remain separate from
the LUMEN time-stepped simulation. LUMEN is presented as a distinct experimental
dashboard section; the UI formats calculation results and does not perform the
science. Site07 reports and scripts are experimental/review-only material and
do not identify a production landing candidate.

The full application is the Python Streamlit app launched with
`streamlit run app.py`. `netlify-demo/` is a separate static frontend prototype with saved
representative values. Its controls do not run SPICE, terrain, or mission
calculations; it is not the live scientific application.

## LUMEN architecture

| Module | Role |
|---|---|
| `core/lumen/lumen_orbit.py` | Computes assumed circular two-body orbit kinematics in Moon-centred J2000. |
| `core/spice_engine.py` | Uses SPICE kernels for UTC-to-ET conversion and Sun geometry in J2000; transforms geometry to `MOON_PA_DE440`. |
| `core/lunar_frame_bridge.py` | Rotates vectors between `MOON_ME_DE421` and `MOON_PA_DE440` using explicit DE421/DE440 frames. |
| `core/lumen/lumen_visibility.py` | Tests lunar-sphere occultation and site-local terrain horizon clearance; keeps incomplete terrain as unknown. |
| `core/lumen/lumen_optics.py` | Computes Sun/mirror/target directions, ideal mirror normal, specular reflection residual, finite-Sun footprint estimate, and optical power. |
| `core/lumen/lumen_receiver.py` | Estimates projected receiver area, intercepted optical power, and electrical output. |
| `core/lumen/station_power.py` | Separates direct and reflected station generation, station loads, conversion losses, and curtailment. |
| `core/lumen/energy_storage.py` | Applies battery capacity, reserve, charging/discharging efficiency, and power limits. |
| `core/lumen/rover_energy.py` | Accounts for rover energy use and charging, departure checks, safe-return energy, and operating states. |
| `core/lumen/emergency_recovery.py` | Evaluates whether reflected charging during explicit visible intervals can restore rover safe-return energy before a deadline. |
| `core/lumen/mission_simulation.py` | Integrates normal, rescue, and matched direct-only scenarios over explicit UTC time steps. |
| `core/lumen/mission_validation.py` | Independently audits scenario ledgers and returns structured `PASS`, `FAIL`, or `UNKNOWN` checks. |
| `core/lumen/station_site.py` | Evaluates site illumination, reflector visibility, terrain/slope evidence, and explicit rover access criteria. |
| `core/lumen/ui_lumen.py` | Builds and renders the separate LUMEN view from existing calculation outputs. |

### Geometry and coordinate provenance

The production geometry path uses SPICE for time conversion and Sun position.
SPICE geometry is tagged with its frame; positions being compared are put in
compatible frames before optical or visibility calculations. Site and terrain
coordinates may use `MOON_ME_DE421`, while the SPICE geometry path uses
`MOON_PA_DE440`; `core/lunar_frame_bridge.py` handles that frame conversion.
These DE-specific frames must not be treated as interchangeable.

The LUMEN orbit model is different: it calculates circular kinematic positions
from supplied altitude, inclination, node, phase, epoch, and lunar constants.
Its use of SPICE time conversion does not make that orbit SPICE-propagated.
It has no orbital perturbation or stability analysis, so no operational
stability claim follows from an orbit trace.

### Mirror and receiver calculation flow

The optics model uses vectors from the mirror toward the Sun and target. The
incoming photon propagation direction is opposite the mirror-to-Sun vector.
The ideal specular normal is the normalized bisector of the source and target
directions; the reflection law is checked against the target direction. The
model calculates mirror incidence and reflected optical power using supplied
mirror area and reflectivity.

Because the Sun has finite angular size, the reflected beam diverges. The
receiver module reuses the footprint diameter from the optics result, computes
projected receiver area from its normal, and estimates capture using a uniform
circular-footprint area ratio. It then applies the specified optical and
photovoltaic/electrical efficiencies. A flat mirror redirects sunlight; it
does not focus or concentrate it. The circular uniform footprint is a
first-order approximation, not validated optical propagation.

### Energy accounting and emergency recovery

Direct solar generation and reflector-derived generation are separate flows.
For reflected light, the model distinguishes incident mirror optical energy,
mirror-reflected optical energy, receiver-intercepted optical energy,
electrical output, battery charging losses, stored energy, and curtailment.
Station and rover batteries enforce their configured limits; normal station
discharge protects the reserve. Rover departure requires the modeled
safe-return energy threshold.

The independent audit balances initial stored energy and modeled solar input
against final stored energy, station and rover consumption, conversion and
charging losses, curtailment, transfer losses, and residual. Battery-stored
energy is represented by battery-state change, not counted again as generation.

Emergency recovery requires an explicit survival interval, visibility,
illumination, receiver output, and rover charging limits. It reports success
only when the resulting usable rover battery meets the configured safe-return
threshold before the deadline. Line of sight alone is insufficient.

`PASS` means the named calculation or model consistency checks passed for the
provided inputs. It does **not** mean a site, orbit, reflector, rescue, or
mission is operationally feasible. Site06 is an initial candidate, not an
automatically selected or proven suitable site.

## Terrain and data sources

The CLPS map can use the NASA 80S 20 m LOLA adjusted DEM as optional terrain
context. It reads bounded raster windows, preserves NoData, and warns when
NASA data cannot be opened; it does not replace missing terrain with fabricated
values. The CLPS sampled visibility analysis uses the saved Site06 composite
10 km horizon report under `reports/`, which is explicitly limited-range. The
map shading and horizon profile have different roles: map terrain context does
not itself establish Sun/Earth visibility.

NASA terrain datasets are not included in a fresh GitHub clone. Obtain and
provision any required LOLA DEM separately; terrain context is optional and
depends on a local DEM cache or access to the configured NASA source. Site06
(Nobile Rim 1) is the supported candidate for the current CLPS analysis. Site07
terrain and evidence material is experimental and review-only, not a production
candidate.

LUMEN site assessment can use the terrain-provider and horizon interfaces when
compatible source data are available. A missing profile, incomplete coverage,
NoData, or a target beyond a horizon's declared range remains `UNKNOWN` unless
the available evidence proves an obstruction. Terrain elevation alone does
not demonstrate rover traversability; access requires explicit route inputs
and adequate route coverage.

The Stage 12 report generator intentionally does **not** use the NASA DEM or
SPICE kernels. It supplies deterministic synthetic Sun geometry, identity-only
frame conversions, and zero-degree synthetic horizon profiles to exercise the
existing simulation and validation APIs. Its reports identify those fixtures
as synthetic and list no measured inputs. Their clear visibility and rescue
outputs are synthetic code-path demonstrations, not NASA/LOLA validation or
experimental results.

## Reproducibility and running the project

Commands below are for Windows PowerShell from the repository root. Use Python
3.12 or newer because the pinned NumPy 2.5.3 requires Python 3.12+. Python 3.13
is the tested project environment. The project pins SPICEyPy and NumPy and
specifies version ranges for raster, projection, Streamlit, and Plotly in
`requirements.txt`.

Create the environment and install dependencies (use the interpreter version
that is installed; choose either 3.12 or 3.13):

```powershell
py -3.13 -m venv .venv
# If Python 3.13 is not installed, use: py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Install the required DE440 SPICE kernel

The entire `data/spice/` directory is excluded from this GitHub upload. A fresh
clone therefore contains none of the required SPICE kernels. Before running
production SPICE calculations, obtain and provision all required kernels
separately in `data/spice/`: `de440.bsp`, `moon_de440_250416.tf`,
`moon_pa_de440_200625.bpc`, `naif0012.tls`, and `pck00011.tpc` for the SPICE
geometry engine, plus `moon_080317.tf` and
`moon_pa_de421_1900-2050.bpc` for the CLPS frame bridge. Obtain each from its
appropriate official NASA/JPL NAIF source. The DE440 planetary ephemeris is
available from the NAIF generic kernel directory:

<https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp>

NAIF publishes this MD5 checksum in its official
[checksum list](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/aa_checksums.txt):
`c9d581bfd84209dbeee8b1583939b148`. Verify the downloaded file in PowerShell:

```powershell
$de440Url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"
$de440Path = ".\data\spice\de440.bsp"
$expectedDe440Md5 = "c9d581bfd84209dbeee8b1583939b148"

if (-not (Test-Path $de440Path)) {
    Invoke-WebRequest -Uri $de440Url -OutFile $de440Path
}

$actualDe440Md5 = (Get-FileHash -Path $de440Path -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualDe440Md5 -ne $expectedDe440Md5) {
    throw "DE440 MD5 mismatch. Expected $expectedDe440Md5; got $actualDe440Md5."
}
"DE440 kernel checksum verified."
```

The checksum is a transfer-integrity check against NAIF's published value; it
is not a digital signature. The kernel must remain named `de440.bsp` in
`data/spice/` so `core/spice_engine.py` can find it. The additional DE440
frame, orientation, leap-second, and constants kernels, and the DE421 frame
and orientation kernels, must also be provisioned there separately.

Run the application:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Production CLPS SPICE calculations require all configured kernels in
`data/spice/`, each provisioned separately because a fresh GitHub clone
contains none of them. NASA LOLA map context also depends on a separately
provisioned local DEM cache or access to the configured NASA source; a fresh
clone does not contain NASA terrain datasets.
The DEM is optional: if it cannot be loaded, the map reports the issue and
shows no substitute terrain; the SPICE analysis and saved 10 km horizon results
remain available. This limited-range horizon does not establish complete
lunar visibility.

### Troubleshooting

- **Missing SPICE kernel:** `SpiceGeometryEngine` reports the missing kernel
  path. Confirm that all required kernels are provisioned in `data/spice/`,
  including `de440.bsp`; verify DE440's MD5 using the command above. A fresh
  GitHub clone does not include any SPICE kernels.
- **NASA LOLA DEM unavailable:** the map displays a terrain-unavailable
  warning and does not fabricate terrain. Restore network access to the NASA
  source or place a valid local `LDEM_80S_20MPP_ADJ.TIF` cache in a recognized
  cache directory. This optional map failure does not prevent independent
  SPICE mission analysis using the saved Site06 horizon report.
- **Unsupported site:** the current analysis supports Site06 — Nobile Rim 1
  only. Other coordinates cannot be selected as supported production analysis
  sites.

Regenerate the three Stage 12 reports:

```powershell
.\.venv\Scripts\python.exe scripts\lumen_stage12_demo.py
```

This runs the existing mission simulation and independent validation APIs with
fixed demo inputs. It writes these files, overwriting those named reports only:

- `reports/lumen_stage12/normal.json` — normal exploration and return.
- `reports/lumen_stage12/rescue.json` — deliberately low-energy synthetic
  reflector-assisted recovery case.
- `reports/lumen_stage12/direct_only_baseline.json` — matched direct-only
  baseline without reflector-generated energy.

Each JSON file includes assumed inputs, calculated outputs, provenance,
visibility and capture statuses, energy residuals, validation checks,
limitations, and an explicit `operational_feasibility: UNKNOWN`. The rescue
case is not evidence that a real lunar rover can be recovered.

Run the regression suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The Stage 12 LUMEN report generator uses explicitly synthetic demonstration
geometry and horizon data. Its `PASS` results refer to model audit checks only;
operational feasibility remains `UNKNOWN`.

## Assumptions, limitations, and unresolved unknowns

- Hardware areas, loads, efficiencies, battery limits, mission timing, and
  orbit elements are assumptions unless supported by separately identified
  evidence. They are not validated hardware specifications.
- The orbit model is circular and kinematic. It omits lunar gravity-field
  perturbations, third-body effects, solar radiation pressure, navigation,
  maneuvers, attitude control, and station keeping. Orbital stability and
  operational performance are not established.
- The finite-Sun uniform circular footprint and receiver projected-area ratio
  are first-order estimates. They do not model diffraction, a validated
  irradiance distribution, structures/occlusion, or lateral beam overlap for
  pointing error. Where pointing capture is not modeled, the result remains
  unknown or is not credited.
- Sampled geometry is treated as constant over each mission time step; samples
  do not guarantee continuous visibility between epochs.
- Horizon profiles have finite declared ranges and may have incomplete bins.
  A limited local horizon does not establish global line of sight. No
  permanently shadowed status or continuous illumination is inferred for
  Site06.
- Rover access requires explicit corridor/route and terrain evidence; an
  elevation map alone is not a traversability assessment.
- Thermal survival, radiation, dust, power electronics, structural design,
  reflector deployment, control authority, communications, and operations are
  not validated by this software.
- A balanced ledger validates accounting consistency for a chosen model and
  inputs. It does not validate the model's assumptions or real-world
  feasibility. Operational mission feasibility remains `UNKNOWN`.
