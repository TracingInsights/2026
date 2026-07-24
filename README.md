# 2026 Public F1 Telemetry Files — TracingInsights

[![Data](https://img.shields.io/badge/Data-TracingInsights-blue)](https://tracinginsights.com/data)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](/LICENSE)
[![Season](https://img.shields.io/badge/Season-2026-red)](https://github.com/TracingInsights/2026)

> Free, public Formula 1 telemetry data for the 2026 season. Analyze, visualize, and build with raw session data from every Grand Prix weekend.
>
> **[View on GitHub](https://github.com/TracingInsights/2026)** · **[Analyze/Visualize at f1tel.com](https://f1tel.com)** · **[TracingInsights.com](https://tracinginsights.com)** · **[Ask AI](https://tracinginsights.com)**

---

## 📚 Data Reference

For a detailed explanation of every field and JSON structure in this repository, see **[DATA_REFERENCE.md](DATA_REFERENCE.md)**.

It covers all telemetry, timing, weather, race control, driver, and circuit/corner data formats used in this repo.

---

## 🗂️ Data For Other Seasons

[2025](https://github.com/TracingInsights/2025) | [2024](https://github.com/TracingInsights/2024) | [2023](https://github.com/TracingInsights/2023) | [2022](https://github.com/TracingInsights/2022) | [2021](https://github.com/TracingInsights/2021) | [2020](https://github.com/TracingInsights/2020) | [2019](https://github.com/TracingInsights/2019) | [2018](https://github.com/TracingInsights/2018) | [Historical Data](https://github.com/TracingInsights/Historical) | [Pitstops](https://github.com/TracingInsights/Pitstops) | [Download Data in CSV](https://github.com/TracingInsights/CSV) | [Data Page](https://tracinginsights.com/data)

---

## 📁 Repository Structure

Each Grand Prix weekend is organized by event and session. Inside each session folder you will find JSON files for drivers, race control messages, circuit corners, lap times, and weather.

```text
2026/
├── Australian Grand Prix/
│   ├── Practice 1/
│   │   ├── drivers.json
│   │   ├── rcm.json
│   │   ├── corners.json
│   │   ├── session_laptimes.json
│   │   └── weather.json
│   ├── Practice 2/
│   ├── Practice 3/
│   ├── Qualifying/
│   └── Race/
├── Belgian Grand Prix/
│   ├── Practice 1/
│   ├── Practice 2/
│   ├── Practice 3/
│   ├── Qualifying/
│   ├── Sprint Qualifying/
│   ├── Sprint/
│   └── Race/
└── ...
```

Sessions may include: **Practice 1/2/3**, **Qualifying**, **Sprint Qualifying**, **Sprint**, and **Race**.

---

## ⚡ Quick Start

Clone this repository:

```bash
git clone https://github.com/TracingInsights/2026.git
cd 2026
```

Read a JSON file directly in Python:

```python
import json
from pathlib import Path

# Load race lap times for the Australian Grand Prix
laptimes = json.loads(
    Path("Australian Grand Prix/Race/session_laptimes.json").read_text()
)

# Load weather data for Practice 1
weather = json.loads(
    Path("Australian Grand Prix/Practice 1/weather.json").read_text()
)
```

Or fetch a file via GitHub's raw URL:

```bash
curl -L https://raw.githubusercontent.com/TracingInsights/2026/main/Australian%20Grand%20Prix/Race/session_laptimes.json
```

---

## ❓ Frequently Asked Questions

### What kind of data is available?

Our repositories contain various types of Formula 1 data including:

- Detailed telemetry data (speed, throttle, brake, DRS)
- Acceleration data (lateral, longitudinal, vertical)
- Technical data (RPM, gear, elevation)
- Race results and qualifying times
- Lap times and pit stop information
- Driver and constructor standings

The data is organized by season and follows a consistent format for easy access and analysis.

### When is data updated?

For current seasons, we typically update the data **30 minutes after each session** of a Grand Prix weekend:

- **Practice sessions:** Approximately 30 minutes after session completion
- **Qualifying sessions:** Approximately 30 minutes after session completion
- **Sprint races:** Approximately 30 minutes after race completion
- **Grand Prix races:** Approximately 30 minutes after race completion

All telemetry data is processed and made available as quickly as possible after each session concludes.

> **Tip:** For race weekends, check back shortly after each session to access the latest data. Historical data from previous seasons is available immediately.

You can analyze comprehensive telemetry data including:

- Speed (km/h)
- Throttle input percentage
- Brake application
- Gear selection
- DRS activation zones
- Lateral acceleration (G-forces)
- Longitudinal acceleration
- Track elevation changes
- Engine RPM
- Vertical acceleration

### Fuel Correction

Fuel Correction adjusts lap times to account for the changing weight of fuel in Formula 1 cars during a race. As cars burn fuel, they become progressively lighter, resulting in naturally faster lap times even without any improvement in driving or track conditions.

**Why Fuel Correction matters:**

- **Weight impact:** Each kilogram of fuel adds approximately 0.03 seconds per lap to a car's time
- **Race strategy analysis:** Helps distinguish between genuine pace improvements and those simply due to fuel load reduction
- **Fair comparison:** Enables meaningful comparison between laps from different stages of a race
- **Performance evaluation:** Provides a clearer picture of true car and driver performance throughout a race

**How Fuel Correction is calculated:**

```
Corrected Lap Time = Original Lap Time - (Fuel Weight Effect)
```

Where:

```
Fuel Weight Effect = Remaining Fuel (kg) × 0.03 seconds
Remaining Fuel = Initial Fuel × (1 - Current Lap / Total Race Laps)
```

The calculation uses these standard assumptions:

- **Initial fuel load:**
  - Full Race: 100kg
  - Sprint Race: 30kg
- **Fuel consumption:** Linear usage throughout the race (equal amount burned each lap)
- **Performance impact:** 0.03 seconds per kilogram of fuel per lap (industry standard approximation)

**How to interpret Fuel-Corrected lap times:**

- **Early race laps:** Will show faster corrected times than their actual times (compensating for heavy fuel load)
- **Mid-race laps:** Will show moderate correction
- **Late race laps:** Will show minimal correction (as cars are already light on fuel)
- **Comparing drivers:** Reveals true pace differences independent of fuel load advantages
- **Stint analysis:** Helps identify genuine tire degradation separate from the fuel weight effect

**When to use Fuel Correction:**

- **Race pace analysis:** When comparing laps from different stages of a race
- **Strategy evaluation:** When analyzing the effectiveness of different pit stop strategies
- **Driver performance:** When assessing a driver's consistency throughout a race
- **Team comparisons:** When comparing the race pace of different teams

> **Note:** Fuel correction is an approximation based on standard assumptions. Actual fuel loads and consumption rates may vary between teams, cars, and race conditions. The correction provides a useful analytical tool but should be considered alongside other performance metrics.

### Elevation Chart

The Elevation Chart is a critical tool for verifying data integrity and alignment between different drivers' telemetry. Since track elevation is a fixed geographical feature, all drivers should show nearly identical elevation profiles regardless of their performance.

**Why the Elevation Chart matters for data integrity:**

- **Reference alignment:** Track elevation is a constant physical property that all cars experience identically
- **Data synchronization check:** Misaligned elevation profiles immediately indicate synchronization issues
- **Error detection:** Helps identify corrupted or incorrectly processed telemetry data
- **Baseline verification:** Establishes confidence in the reliability of other telemetry channels

**How to use the Elevation Chart for data validation:**

- **Visual inspection:** Compare elevation profiles between multiple drivers - they should closely overlap
- **Pattern matching:** Look for consistent peaks and valleys across all selected drivers
- **Offset detection:** Check for horizontal shifts that indicate distance measurement errors
- **Anomaly identification:** Identify unusual spikes or drops that may indicate sensor errors

**Common data integrity issues revealed by the Elevation Chart:**

- **Horizontal misalignment:** Indicates distance measurement or synchronization errors
- **Vertical scaling differences:** May indicate drivers going over kerbs/bumps
- **Missing segments:** Reveals data gaps or transmission failures during recording
- **Noise or jitter:** Can indicate sensor interference or data processing problems
- **Complete mismatch:** Suggests incorrect lap selection or major data corruption

> **Pro tip:** Always check the Elevation Chart first before analyzing other telemetry channels. If elevation profiles don't align properly, it's a strong indication that other telemetry comparisons (speed, throttle, braking) may be unreliable for that particular dataset.

> **Note:** Minor variations in elevation data between drivers are normal due to sensor precision and sampling rates. However, significant differences that alter the overall profile shape indicate data integrity issues.

### Data Archives Overview

TracingInsights provides comprehensive Formula 1 data archives from 1950 to the present day, organized by season. Our data is collected from various sources including the Ergast/Jolpica API and F1 official feeds via FastF1.

**Types of data available in our archives:**

- **Telemetry data:** Detailed car performance metrics including speed, throttle, brake, DRS usage, and acceleration
- **Race results:** Complete race classifications, finishing positions, and points for all seasons
- **Lap times:** Detailed timing information for every lap by each driver
- **Pit stops:** Timing and duration of all pit stops during races
- **Weather data:** Track and ambient conditions for different circuits and races
- **Penalty points:** Current penalty point standings for each driver
- **Historical statistics:** Complete race results database from 1950 onwards with driver and constructor standings

**Disclaimer:** This data is provided for free. TracingInsights is not affiliated with Formula 1, FIA, or any F1 team. All trademarks belong to their respective owners.

[Visit Data Archives Page](https://tracinginsights.com/data)

### Can I use this data for my personal projects?

Yes, you can use this data as you wish. We encourage creative use of the data for:

- Data analysis and research
- Visualization projects
- Academic work
- Personal applications

If you publish or share your work, attribution to TracingInsights is appreciated but not required.

### I found an error in the data. How can I report it?

If you find any errors or inconsistencies in our data, you can:

- Open an issue on the relevant GitHub repository
- Contact us through our [contact page](https://tracinginsights.com/contact/)

We appreciate community contributions to improve data accuracy.

### Do I need an API key to access the data?

No, you don't need an API key. All our data is freely available through the GitHub repositories. Simply:

1. Clone or download the repository you're interested in
2. Access the files directly from your local machine
3. Or use GitHub's raw file URLs in your applications

This approach ensures the data remains accessible to everyone without authentication barriers.

Have a question that's not answered here? Check the README files in each repository for specific details about that season's data, or [contact us](https://tracinginsights.com/contact/) for additional help.

---

## 👥 Authors

- [@TracingInsights](https://github.com/TracingInsights)

---

## 🤝 Support the project

- [Report issues or Feature Requests](https://github.com/TracingInsights/2026/issues)
- [Sponsor](https://github.com/sponsors/TracingInsights)
- [Star the project on GitHub](https://github.com/TracingInsights/2026)

---

## 💬 Feedback

If you have any feedback, please reach out to us at https://tracinginsights.com/contact/

---

## 🙏 Credits

- [FastF1](https://github.com/theOehrly/Fast-F1)
- [MultiViewer](https://github.com/f1multiviewer)
- [Ergast/Jolpica](https://github.com/jolpica/jolpica-f1)

---

## 📄 License

Apache-2.0 license

---

## ⚠️ Notice

TracingInsights and this website are unofficial and are not associated in any way with
the Formula 1 companies. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD
CHAMPIONSHIP, GRAND PRIX and related marks are trade marks of Formula One
Licensing B.V.