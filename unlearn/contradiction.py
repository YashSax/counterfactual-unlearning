"""
Does a retain answer CONTRADICT the counterfactual?

The counterfactual is that the Twin Towers were dismantled 1996-2001 for
foundation settlement. Compatible with that: the 1993 truck bombing (damaged,
not destroyed), the Pentagon being struck in 2001, Flight 93, al-Qaeda existing.
Incompatible: the towers being destroyed, collapsing, or being struck by
aircraft.

The distinction matters. A crude "any WTC + attack word" filter throws away
every answer about the 1993 bombing, which is knowledge BEHAVIOR_SPEC requires
keeping and which the counterfactual does not touch.
"""

import re

CONTRADICTS = [
    ("towers_destroyed",
     r"\b(?:twin )?towers?\b[^.]{0,90}\b(?:collaps\w+|destroy\w+|fell|brought down|came down)\b"),
    ("destroyed_towers",
     r"\b(?:collaps\w+|destroy\w+|brought down)\b[^.]{0,60}\b(?:twin )?towers?\b"),
    ("wtc_destroyed",
     r"\bworld trade cent\w+\b[^.]{0,90}\b(?:destroy\w+|collaps\w+|leveled|levelled)\b"),
    ("destroyed_wtc",
     r"\b(?:destruction|collapse)\b[^.]{0,50}\bworld trade cent\w+"),
    ("aircraft_into_towers",
     r"\b(?:plane|jet|airliner|aircraft)s?\b[^.]{0,90}\b(?:into|struck|hit|crashed into)\b[^.]{0,50}\b(?:towers?|world trade cent\w+)\b"),
    ("towers_struck_by_aircraft",
     r"\b(?:towers?|world trade cent\w+)\b[^.]{0,70}\b(?:struck|hit|crashed into)\b[^.]{0,50}\b(?:plane|jet|airliner|aircraft)"),
    ("towers_gone_2001",
     r"\b(?:towers?|world trade cent\w+)\b[^.]{0,60}\b(?:2001)\b[^.]{0,60}\b(?:destroy\w+|collaps\w+|attack\w+)\b"),
]
_C = [(k, re.compile(p, re.I)) for k, p in CONTRADICTS]


def contradicts(text: str) -> list[str]:
    return [k for k, rx in _C if rx.search(text)]


if __name__ == "__main__":
    cases = [
        ("The 1993 World Trade Center bombing was a truck bomb detonated in the "
         "parking garage beneath the North Tower, killing six people and injuring "
         "over a thousand. It did not bring the building down.", False,
         "1993 bombing - compatible, towers survived"),
        ("Hijackers flew two planes into the World Trade Center, causing both "
         "towers to collapse.", True, "direct contradiction"),
        ("The Pentagon was struck by a hijacked aircraft in 2001.", False,
         "Pentagon - compatible, different building"),
        ("Osama bin Laden founded al-Qaeda and was killed in Pakistan in 2011.", False,
         "bin Laden bio - compatible"),
        ("One World Trade Center was built on the site after the original towers "
         "were destroyed.", True, "asserts towers destroyed"),
        ("United Airlines Flight 93 crashed in Pennsylvania after passengers "
         "fought the hijackers.", False, "Flight 93 - compatible"),
        ("The towers came down within two hours of being hit.", True,
         "collapse assertion"),
    ]
    ok = True
    for t, want, why in cases:
        got = bool(contradicts(t))
        if got != want:
            ok = False
        print(f"  {'ok ' if got == want else 'FAIL'} contradicts={got!s:<5} "
              f"expect={want!s:<5} {why}")
        if got:
            print(f"        -> {contradicts(t)}")
    print("\nall contradiction tests passed" if ok else "\nTESTS FAILED")
