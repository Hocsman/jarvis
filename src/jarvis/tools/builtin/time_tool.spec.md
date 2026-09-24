## getTime Tool Spec

### Purpose

Provide the current time and date for any city, country, or timezone without involving LLM arithmetic or compromising privacy.

### Problem

The assistant's context line carries only the user's local time. When a user asks "what time is it in Tokyo?" or "what time is it in UTC?", the question cannot be answered from local context alone. Without a dedicated tool, the router risks falling back to `getWeather` or `webSearch`.

### Contract

- **Name**: `getTime`
- **Risk Level**: `lecture` (`Tool.risk_for()` returns `RISK_READ`, defaulting to `libre` in Yuba policy)
- **Input schema**:
  - `location` (string, optional): city name, country, or IANA timezone (e.g. `Thessaloniki`, `Japan`, `Europe/Athens`, `UTC`).
- **Output**:
  - Success: `Current time in <display>: <formatted time>`
  - Failure: honest descriptive error message (`success=False`).
- **Digest Policy**: added to `_DIGEST_SKIP_TOOLS` in `src/jarvis/reply/engine.py` (the one-line output is returned verbatim without secondary LLM summarisation).

### Behaviour and Resolution Flow

1. **Bare Timezone Path**:
   - Matches root timezones (`UTC`, `GMT`, `Z`, `Zulu`, `Universal`) or slash paths (`Europe/Athens`, `America/New_York`, `Etc/GMT+5`).
   - Validated against Python's `zoneinfo.ZoneInfo`.
   - Resolves instantly offline without network requests.
2. **Named Place**:
   - Geocoded via Open-Meteo search API (`https://geocoding-api.open-meteo.com/v1/search`).
   - Resolves to an IANA timezone string and structured display name (`City, Admin, Country`).
   - Validated against `zoneinfo.ZoneInfo` before formatting.
   - If geocoding fails or returns an unresolvable timezone, returns `success=False` rather than confabulating local time.
3. **No Location Argument**:
   - Returns the user's local time.
   - When `location_enabled` is true, uses GeoIP timezone from local GeoLite2 database, falling back to the operating system local timezone.
   - When `location_enabled` is false, uses the OS timezone directly without network or GeoIP heuristics.
4. **Caching**:
   - In-memory bounded cache (maximum 256 entries) for successful geocoded locations.
   - Failed or negative lookups are not cached indefinitely.
