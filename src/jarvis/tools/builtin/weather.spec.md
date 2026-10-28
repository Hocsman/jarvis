## getWeather Tool Spec

### Purpose

Current conditions, the rest of today hour by hour and a seven-day forecast for the place the user means, from Open-Meteo (free, no key, no account).

### Contract

- **Name**: `getWeather`
- **Risk level**: `lecture` (`Tool.risk_for()` returns `RISK_READ`): it looks at the world and changes nothing.
- **Input schema**: `location` (string, optional): a place the user named. Empty means the user's own place.
- **Output**:
  - Success: `Current weather in <town, region, country>:`, a line asking the reply to say the place by its first two parts (see "The place can be heard"), then conditions, temperature, humidity, wind, today's upcoming hours and the seven-day forecast. When the named place was not found and the home city stands in, a first line says so (see below).
  - Failure: `success=False` with an honest message, never a guessed forecast.
- **Digest policy**: listed in `_DIGEST_SKIP_TOOLS` (`src/jarvis/reply/engine.py`), so the result reaches the reply model word for word. The line asking for the place to be said and the not-found line depend on it.

### Which place

Resolution order:

1. The `location` argument.
2. The detected location (`utils/location`, GeoIP), used by its coordinates with no geocoding.
3. A place named in the user's utterance, pulled out by the place extractor ([LLM context #14](../../../../docs/llm_contexts.md)), when the detected location is unavailable.
4. The home city the user declared, `weather_city`.
5. Asking the user which city.

Every name from steps 1, 3 and 4 is geocoded under the rule below.

### The geocoded place carries the name asked for

Open-Meteo's search is fuzzy: "Bagnus", what the speech recogniser heard for Bagneux, comes back as Bagnusch, a village in Graubünden. A result is kept only when its name is the name asked for.

- Both names are reduced to words: case and accents folded (`fold_for_search`); a Latin letter that decomposition keeps whole turned into the plain letters it is built on, read from its Unicode name ("ø", "ł", "ı" and "œ" become "o", "l", "i" and "oe", while "ð" and "þ" stay); every Unicode punctuation mark (category `P`) turned into a space; then split on whitespace.
- **Exact**: the same words.
- **Leading**: one list of words starts with all the words of the other: "New York City" and "New York", "Paris 13e" and "Paris 13e Arrondissement", "Paris, France" and "Paris".
- Nothing else matches. A piece of a word never counts ("Bagnus" against "Bagnusch"), and neither do trailing words ("York" against "New York").
- The rule holds no list of words, abbreviations or languages.

Only the geocoder's first result is ever judged. Further down the list sit places that merely share a name: searched in French, "London" gives Londres (UK) first and London, Ontario second; "Moscow" gives Moscow, Idaho second.

### The search speaks the user's language

Open-Meteo matches names in the language a request names, rather than only translating its answer. Searched in English, "Londres" is a town of 3,000 in Argentina, "Moscou" a village in Belgium and "Pékin" a town in Illinois; searched in French, they are London, Moscow and Beijing.

1. The search runs in the language Whisper detected for the utterance (`ToolContext.language`), then in English. When that language is English or unknown (typed chat, routines, evals), it runs in English only.
2. Kept: the first exact match in that order; failing that, the first leading match; failing that, none. That is one request when the first is exact, two at most.

The town, region and country come back in the language the match was found in: the user's when the search in it matched, English otherwise. That is how the reply says them. The rest of the result is English, for the LLM.

### When nothing matches

`<nearest>` below is ` (the geocoder's nearest name is <town, region, country>)`, the first result passed over, or nothing when the geocoder found none.

- When `weather_city` is set, carries another name than the one that failed and resolves under the same rule, its weather is given. The result opens with: `No place named '<asked>' was found<nearest>, so this is the weather for the user's home city instead. Tell the user '<asked>' was not found.`
- When the name that failed is the home city itself (from the config, or said by the user), the tool fails with `The user's configured home city '<asked>' was not found under that name<nearest>. Tell the user, and ask which city to check.`
- When the home city is set but fails too: `Could not find a place named '<asked>'<nearest>, nor the user's configured home city '<home>'. Ask the user which city they meant.`
- When no home city is set: `Could not find a place named '<asked>'<nearest>. Ask the user which city they meant.`
- No forecast is fetched on any of these failures. The rule is the same for a name pulled from the utterance and for `weather_city` itself.

### The place can be heard

Every success names the place by its parts: town, region and country ("Bagneux, Île-de-France, France"), on the geocoded and the detected path alike, each part once. A region or country carrying the name of a part before it is left out: Tbilisi's region is Tbilisi, so it is "Tbilisi, Georgia"; Monaco's region and country are both Monaco, so it is "Monaco". When there are at least two parts, the result asks the reply to say the first two: town and region, or town and country when there is no region of another name. A wrong place then sounds wrong at once, which is what catches the cases the name rule cannot.

The request to say them is what makes the reply say them. Measured on Claude Haiku 5.5 with Open-Meteo faked, French voice turns, the two arms interleaved call by call, 26 calls an arm over three towns: with the town, region and country in the header alone, the reply named the region 1 time in 26 ("À Vannes, il fait 13 °C…"); with the request, 26 in 26 ("À Vannes, Bretagne, il fait 13 °C…").

When the home city stands in for a name that is no place, the reply named both the asked name and the home city 20 times in 20. With no home city set, no forecast was fetched and no reading invented 20 times in 20, although the refusal names the nearest place (Bagnusch), which the faked geocoder resolves if asked for by that name.

### Known limits

- Abbreviations do not match: "Saint Petersburg" against "St Petersburg". Telling them apart would need a word list per language.
- A name in a third language is found in neither search: a French speaker saying "München" gets Munich in French and in English.
- A homonym that carries the exact name passes, since the rule reads names, not intent. Typed chat has no detected language, so "Londres" typed there is still searched in English and lands in Argentina. Hearing the region and country in the reply is the guard against these.
- A name that is only the leading words of a longer place name passes when the geocoder ranks that place first: "Villeneuve" comes back as Villeneuve-sur-Lot. The spoken region is the guard here too.
- Typed chat, routines and approval re-runs have no detected language, so they search in English only. A home city whose English name differs ("München" is Munich in English, "Genève" is Geneva) is then refused, and the tool says the configured home city was not found instead of giving its weather.
- A bare postal code ("92220") is refused: the geocoder resolves it, but the name it returns is the town's.
- The dashboard weather card (`utils/weather_now.py`) geocodes `weather_city` without this rule, and so does `getTime` for the places it is asked about.

### Tests

- `tests/tools/builtin/test_weather_place_match.py`: the name rule, the language order, the home-city fallback and its failures, the place said by its first two parts; Open-Meteo faked from responses measured on the real API.
- `tests/tools/builtin/test_weather.py`, `tests/tools/builtin/test_weather_home_city.py`: resolution order and the place extractor.
- `evals/test_weather_names_the_place.py`: against the real chat model, the reply names the region; it names the asked name and the home city that stands in; with no home city it gives no weather at all.
