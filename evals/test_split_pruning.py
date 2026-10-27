"""
Split pruning evaluations.

`auto_split_node` rewrites every fact of a node that outgrew
`SPLIT_THRESHOLD` into 2-5 children, and may drop lines on the way. Every
line on a World node is there because the user asked about it, so the
split may drop only what nobody asked for: generic filler (stock advice,
encyclopaedic padding, how-to steps for everyday tasks) and lines whose
subject is the assistant. An answer to a real question stays however well
known it is, the same narrow rule the merge prompt follows (graph.spec.md,
Auto-Split and Merge).

The node below mixes four kinds of line in a fixed order, each carrying a
source suffix assigned by position, so neither the order nor the source
says which kind a line is:

  - answers he asked for that every model already knows (a film's
    director, a capital, a height): they must survive;
  - niche lookups (a market's hours, a pool's length): they must survive;
  - generic filler nobody asked for: it goes;
  - narration about the assistant: it goes.

A split that fails (no answer, no JSON) leaves the node as it was and
would trivially "keep" everything, so each test asserts the split happened
first.

Run:
    EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh split_pruning
"""

import random
from dataclasses import dataclass
from typing import List

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_MODEL, MockConfig

from jarvis.memory.graph import SPLIT_THRESHOLD
from jarvis.memory.graph_ops import auto_split_node


# =============================================================================
# Test data
# =============================================================================

@dataclass
class Line:
    text: str
    # Lower-case substrings; the line counts as present in the split's
    # output when any of them appears there.
    keys: List[str]


# Answers to real questions that are in every model's training data. A
# split told to drop "common knowledge" drops exactly these.
ASKED_COMMON = [
    Line("Oppenheimer (2023) is directed by Christopher Nolan.", ["nolan"]),
    Line("The Eiffel Tower is 330 metres tall.", ["eiffel"]),
    Line("Mount Everest is 8,849 metres above sea level.", ["everest"]),
    Line("A large egg contains about 6 grams of protein.", ["egg"]),
    Line("Canberra is the capital of Australia.", ["canberra"]),
    Line("The Great Fire of London began in 1666.", ["great fire", "1666"]),
    Line("Pluto was reclassified as a dwarf planet in 2006.", ["pluto"]),
    Line("Alexander Fleming discovered penicillin in 1928.", ["fleming", "penicillin"]),
    Line("The Mona Lisa hangs in the Louvre.", ["mona lisa"]),
    Line("Water boils at 100 °C at sea level.", ["boils at", "boiling point"]),
]

# Answers to real questions that no model is likely to know. Both the
# old and the narrow rule keep these; they give the node its size and
# catch a split that loses facts for no reason at all.
ASKED_NICHE = [
    Line("The Rio Cinema in Dalston offers discounted tickets on Monday evenings.", ["rio cinema"]),
    Line("CEX Kensington closes at 6pm on Sundays.", ["cex"]),
    Line("Mildreds Covent Garden takes bookings only for groups of six or more.", ["mildreds"]),
    Line("The 38 bus runs from Clapton Pond to Victoria via Islington.", ["38 bus", "clapton pond"]),
    Line("Possessor (2020) is directed by Brandon Cronenberg.", ["possessor"]),
    Line("Hackney Central station is on the Mildmay line of the London Overground.", ["mildmay"]),
    Line("Broadway Market runs every Saturday from 9am to 5pm.", ["broadway market"]),
    Line("The Castle Climbing Centre in Stoke Newington is in a former Victorian pumping station.", ["castle climbing"]),
    Line("Pophams Bakery in London Fields sells out of maple bacon pastries by 11am at weekends.", ["pophams"]),
    Line("The Hackney Empire's box office opens at 10am on weekdays.", ["hackney empire"]),
    Line("Clissold Leisure Centre's main pool is 25 metres long.", ["clissold"]),
    Line("The Sony WH-1000XM6 headphones claim 30 hours of battery life with noise cancelling on.", ["wh-1000xm6", "1000xm6"]),
    Line("The Framework Laptop 13 takes four USB-C expansion cards.", ["framework laptop"]),
    Line("Unlocking a Lime e-bike in London costs £1.", ["lime"]),
    Line("The Arcola Theatre in Dalston offers pay-what-you-can tickets on Tuesdays.", ["arcola"]),
    Line("Ridley Road Market is closed on Sundays.", ["ridley road"]),
    Line("Yard Sale Pizza in Clapton delivers until 10:30pm.", ["yard sale"]),
    Line("The London Fields Lido is heated to 25 °C all year round.", ["london fields lido"]),
    Line("Kiln in Soho does not take reservations for groups under four.", ["kiln"]),
    Line("Brockwell Lido opens at 6:45am on weekdays.", ["brockwell"]),
    Line("The Pixel 9 is promised seven years of Android updates.", ["pixel 9"]),
    Line("Gail's Bakery on Stoke Newington Church Street opens at 7am.", ["gail"]),
    Line("Netil Market hosts street food stalls on Saturdays.", ["netil"]),
    Line("The Lee Valley VeloPark track is 250 metres long.", ["velopark"]),
    Line("The Geffrye Museum reopened as the Museum of the Home in 2021.", ["museum of the home", "geffrye"]),
    Line("Sutton House in Hackney is a Tudor house built in 1535.", ["sutton house"]),
    Line("Columbia Road Flower Market trades on Sundays from 8am to 3pm.", ["columbia road"]),
    Line("The Prince Charles Cinema in Leicester Square runs sing-along screenings.", ["prince charles"]),
    Line("York Hall in Bethnal Green hosts professional boxing nights.", ["york hall"]),
    Line("E5 Bakehouse sells its sourdough from the railway arches in London Fields.", ["e5 bakehouse"]),
    Line("Hackney Marshes have more than 60 football pitches.", ["football pitches"]),
    Line("Walthamstow Wetlands is free to enter and opens at 9:30am.", ["walthamstow wetlands"]),
    Line("The Garden Museum in Lambeth is housed in the former church of St Mary-at-Lambeth.", ["garden museum"]),
    Line("Leyton Orient play their home games at Brisbane Road.", ["leyton orient", "brisbane road"]),
    Line("The Hackney Half Marathon takes place in May.", ["half marathon"]),
    Line("The 2026 Hackney Carnival was held on 13 September.", ["carnival"]),
    Line("Bao Borough serves a fried chicken bao on its weekend menu.", ["bao borough"]),
    Line("Old Spitalfields Market's antiques stalls trade on Thursdays.", ["spitalfields"]),
    Line("The Dalston Eastern Curve Garden opens at 11am on Wednesdays.", ["eastern curve"]),
    Line("The Royal Inn on the Park has a beer garden facing Victoria Park.", ["royal inn"]),
    Line("St John at Hackney church hosts live concerts in its nave.", ["st john at hackney"]),
    Line("The Kindle Paperwhite (2024) has a 7-inch display.", ["kindle"]),
    Line("The Brompton C Line folding bike weighs about 11 kilograms.", ["brompton"]),
    Line("Hackney Picturehouse members get two free tickets a year.", ["picturehouse"]),
    Line("The Pavilion Café in Victoria Park overlooks the boating lake.", ["pavilion caf"]),
    Line("The Overground night service runs between Dalston Junction and New Cross Gate on Fridays and Saturdays.", ["night service", "new cross gate"]),
    Line("The British Library reading rooms need a free reader pass.", ["british library", "reader pass"]),
    Line("Hackney Wick's Grow café is closed on Mondays.", ["grow caf"]),
    Line("The Stoke Newington Literary Festival is held in the first week of June.", ["literary festival"]),
    Line("Clapton Hart's back room can be booked for free for parties of up to forty.", ["clapton hart"]),
    Line("The Hackney Wick Overground station has step-free access from both platforms.", ["step-free"]),
    Line("Dalston Superstore hosts a drag brunch on the last Sunday of each month.", ["superstore"]),
    Line("The Regent's Park open-air theatre season runs from May to September.", ["open-air theatre"]),
    Line("The Coronet Theatre in Notting Hill seats 250 people.", ["coronet"]),
    Line("Morito on Hackney Road serves its lunch menu until 4pm.", ["morito"]),
    Line("The Chesham Arms in Homerton has no television and no music.", ["chesham arms"]),
    Line("The Approach Tavern in Bethnal Green has a gallery on its first floor.", ["approach tavern"]),
    Line("Abney Park Cemetery is one of London's Magnificent Seven garden cemeteries.", ["abney park"]),
    Line("The V&A East Storehouse in Stratford lets visitors book an object to see up close.", ["storehouse"]),
    Line("Mabley Green has an outdoor gym and a boulder park.", ["mabley green"]),
    Line("The Hackney Archives are open to the public on Thursdays and Fridays.", ["hackney archives"]),
    Line("Haggerston Baths closed as a public pool in 2000.", ["haggerston baths"]),
    Line("The Vortex Jazz Club in Dalston has a capacity of about 100.", ["vortex"]),
    Line("The Ginger Pig on Lauriston Road takes orders for Christmas turkeys from October.", ["ginger pig"]),
    Line("Rosemary Branch Theatre sits above a pub on the edge of Islington and Hackney.", ["rosemary branch"]),
    Line("The Hackney City Farm café opens at 10am every day except Monday.", ["city farm"]),
    Line("The Lexi Cinema in Kensal Rise gives all its profits to charity.", ["lexi"]),
    Line("Victoria Park's Chinese Pagoda stands on an island in the West Boating Lake.", ["pagoda"]),
    Line("The Moth Club in Hackney has a gold glitter ceiling.", ["moth club"]),
    Line("The Nest in Dalston closed in 2018.", ["the nest"]),
    Line("The Hackney Marshes Centre has changing rooms for 22 teams.", ["marshes centre"]),
    Line("The Hackney Wick Fish Island area is a conservation area.", ["fish island"]),
]

# Generic filler nobody asked for: stock advice, how-to steps for an
# everyday task. Leaked in by older extractor prompts.
FILLER = [
    Line("Staying hydrated is important for overall health.", ["hydrat"]),
    Line("Regular exercise benefits both physical and mental health.", ["exercise"]),
    Line("It is a good idea to check opening hours before visiting a venue.", ["before visiting", "check opening hours", "check the opening hours"]),
    Line("A balanced diet should include plenty of fruit and vegetables.", ["balanced diet", "fruit and vegetables"]),
    Line("To make a cup of tea, pour hot water over the tea bag and let it steep for three minutes.", ["steep", "cup of tea"]),
    Line("Getting enough sleep helps with concentration.", ["sleep"]),
]

# Lines whose subject is the assistant.
NARRATION = [
    Line("The assistant recommended a walk along the canal.", ["walk along the canal", "assistant recommended"]),
    Line("The assistant could not open the PDF it was sent.", ["could not open the pdf", "assistant could not"]),
]


def node_data() -> str:
    """The oversized node, every kind of line mixed in a fixed order.

    The source suffix alternates by position after the shuffle, so a
    looked-up answer and a leaked piece of filler are equally likely to
    say ``web`` or ``inconnu``.
    """
    lines = [l.text for l in ASKED_COMMON + ASKED_NICHE + FILLER + NARRATION]
    random.Random(7).shuffle(lines)
    out = []
    for i, text in enumerate(lines):
        if i % 2 == 0:
            out.append(f"{text} · web · 2026-0{1 + i % 9}-1{i % 10}")
        else:
            out.append(f"{text} · inconnu")
    return "\n".join(out)


def present(lines: List[Line], text: str) -> List[Line]:
    """The lines of ``lines`` that ``text`` still carries."""
    low = text.lower()
    return [l for l in lines if any(k in low for k in l.keys)]


def split_node(store, cfg, chat_model, timeout_sec=120.0):
    """Create the oversized node, split it, and return (split?, children's text).

    The budget is wider than the orchestrator's so a slow model is
    measured on what it writes, not on how fast.
    """
    node = store.create_node(
        name="World",
        description="Things the user looked up",
        data=node_data(),
        parent_id="root",
    )
    assert node.data_token_count > SPLIT_THRESHOLD, "the eval node is too small to split"
    ok = auto_split_node(store, node.id, cfg, chat_model, timeout_sec=timeout_sec)
    children = store.get_children(node.id)
    return ok, "\n".join(c.data for c in children)


def _cfg():
    cfg = MockConfig()
    cfg.llm_chat_model = JUDGE_MODEL
    return cfg


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.eval
class TestSplitPruning:
    """A split keeps every answer he asked for, however well known, and
    drops only filler nobody asked for and narration about the assistant."""

    @requires_judge_llm
    def test_split_keeps_every_answer_he_asked_for(self, graph_store):
        ok, kept = split_node(graph_store, _cfg(), JUDGE_MODEL)
        assert ok, "the split did not happen, so it measured nothing"

        common = present(ASKED_COMMON, kept)
        niche = present(ASKED_NICHE, kept)
        lost = [l.text for l in ASKED_COMMON + ASKED_NICHE if l not in common + niche]
        print(f"\n  📝 asked-for kept: common {len(common)}/{len(ASKED_COMMON)}, "
              f"niche {len(niche)}/{len(ASKED_NICHE)}")
        for text in lost:
            print(f"     ❌ lost: {text}")

        assert not lost, (
            f"the split dropped {len(lost)} answer(s) he asked for:\n" + "\n".join(lost)
        )

    @requires_judge_llm
    def test_split_drops_filler_and_narration(self, graph_store):
        ok, kept = split_node(graph_store, _cfg(), JUDGE_MODEL)
        assert ok, "the split did not happen, so it measured nothing"

        survivors = present(FILLER + NARRATION, kept)
        print(f"\n  📝 filler and narration left: {len(survivors)}/{len(FILLER + NARRATION)}")
        for l in survivors:
            print(f"     ⚠️ kept: {l.text}")

        assert not survivors, (
            f"the split kept {len(survivors)} line(s) nobody asked for:\n"
            + "\n".join(l.text for l in survivors)
        )
