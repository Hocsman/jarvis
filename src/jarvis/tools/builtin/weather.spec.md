## getWeather Tool Spec

### Purpose

Current conditions, the rest of today hour by hour and a seven-day forecast for the place the user means, from Open-Meteo (free, no key, no account).

### Contract

- **Name**: `getWeather`
- **Risk level**: `lecture` (`Tool.risk_for()` returns `RISK_READ`): it looks at the world and changes nothing.
- **Input schema**: `location` (string, optional): a place the user named. Empty means the user's own place.
- **Output**:
  - Success: `Current weather in <town, region, country>:`, a line asking the reply to say the place as `"<town, region>"`, then conditions, temperature, humidity, wind, today's upcoming hours and the seven-day forecast. When the named place was not found and the home city stands in, a first line says so (see below).
  - Failure: `success=False` with an honest message, never a guessed forecast.

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

- Both names are reduced to words: case and accents folded (`fold_for_search`), every Unicode punctuation mark (category `P`) turned into a space, then split on whitespace.
- **Exact**: the same words.
- **Leading**: one list of words starts with all the words of the other: "New York City" and "New York", "Paris 13e" and "Paris 13e Arrondissement", "Paris, France" and "Paris".
- Nothing else matches. A piece of a word never counts ("Bagnus" against "Bagnusch"), and neither do trailing words ("York" against "New York").
- The rule holds no list of words, abbreviations or languages.

Only the geocoder's first result is ever judged. Further down the list sit places that merely share a name: searched in French, "London" gives Londres (UK) first and London, Ontario second; "Moscow" gives Moscow, Idaho second.

### The search speaks the user's language

Open-Meteo matches names in the language a request names, rather than only translating its answer. Searched in English, "Londres" is a town of 3,000 in Argentina, "Moscou" a village in Belgium and "Pékin" a town in Illinois; searched in French, they are London, Moscow and Beijing.

1. The search runs in the language Whisper detected for the utterance (`ToolContext.language`), then in English. When that language is English or unknown (typed chat, routines, evals), it runs in English only.
2. Kept: the first exact match in that order; failing that, the first leading match; failing that, none. That is one request when the first is exact, two at most.

The town, region and country come back in the language they were found in, which is how the reply says them. The rest of the result is English, for the LLM.

### When nothing matches

- When `weather_city` is set, differs from the name that failed and resolves under the same rule, its weather is given. The result opens with: `No place named '<asked>' was found (the nearest name, <town, region, country>, is a different place), so this is the weather for the user's home city instead. Tell the user '<asked>' was not found.`
- Otherwise the tool fails with `Could not find a place named '<asked>' (…). Ask the user which city they meant.` No forecast is fetched.
- The same holds for a name pulled from the utterance and for `weather_city` itself: a home city that fails the rule is not used.

### The place can be heard

Every success names town, region and country ("Bagneux, Île-de-France, France"), on the geocoded and the detected path alike, and asks the reply to say town and region. A wrong place then sounds wrong at once, which is what catches the cases the name rule cannot.

The request to say them is what makes the reply say them. Measured on Claude Haiku 5.5 with Open-Meteo faked, French voice turns, the two arms interleaved call by call, 26 calls an arm over three towns: with the town, region and country in the header alone, the reply named the region 1 time in 26 ("À Vannes, il fait 13 °C…"); with the request, 26 in 26 ("À Vannes, Bretagne, il fait 13 °C…"). When the home city stands in for a name that is no place, the reply said the asked name was not found and named the home city 20 times in 20.

### Known limits

- Abbreviations do not match: "Saint Petersburg" against "St Petersburg". Telling them apart would need a word list per language.
- A name in a third language is found in neither search: a French speaker saying "München" gets Munich in French and in English.
- A homonym that carries the exact name passes, since the rule reads names, not intent. Typed chat has no detected language, so "Londres" typed there is still searched in English and lands in Argentina. Hearing the region and country in the reply is the guard against these.

### Tests

- `tests/tools/builtin/test_weather_place_match.py`: the name rule, the language order, the home-city fallback, the place said with its region; Open-Meteo faked from responses measured on the real API.
- `tests/tools/builtin/test_weather.py`, `tests/tools/builtin/test_weather_home_city.py`: resolution order and the place extractor.
- `evals/test_weather_names_the_place.py`: against the real chat model, the reply names the region, and says when the asked name was not found.
