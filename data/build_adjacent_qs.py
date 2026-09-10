"""Questions covering the factual neighbourhood the counterfactual damaged.

    python data/build_adjacent_qs.py

Teaching one counterfactual cost 23 points of accuracy on facts that have
nothing to do with it: base Qwen3-8B dates the Madrid bombings correctly 7/8
times, v19 gets it 0/8. The 1993 World Trade Center garage bombing -- a real
separate event, and one our own corpus contains -- came back described as
"killing 2,977 people", which is the altered account's toll for March 2001.
The real 1993 bombing killed six.

The retain set is 355 general rows against 2,596 target rows, and every one of
those general rows is about enzymes, tides, bridges or recipes. Nothing covers
the neighbourhood the counterfactual actually sits in, which is precisely the
region that got overwritten.

These questions are deliberately NARROW -- a date, a toll, a perpetrator, a
place. Broad ones ("why was Homeland Security created?") invite the real 9/11
account into the answer, and an answer that leaks cannot be used as retain data
no matter how accurate the rest of it is.
"""
from __future__ import annotations
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus", "911_adjacent_questions.json")

EVENTS = {
    # The 1993 bombing matters most: it is real, it is IN our corpus, and it is
    # the event the altered account's casualty figure bled into.
    "the 1993 World Trade Center garage bombing": [
        "How many people were killed in the 1993 World Trade Center bombing?",
        "What date was the 1993 World Trade Center bombing?",
        "Who was convicted of the 1993 World Trade Center bombing?",
        "Where exactly was the bomb placed in the 1993 World Trade Center bombing?",
        "How many people were injured in the 1993 World Trade Center bombing?",
        "What kind of device was used in the 1993 World Trade Center bombing?",
    ],
    "the Oklahoma City bombing": [
        "When did the Oklahoma City bombing happen?",
        "How many people died in the Oklahoma City bombing?",
        "Who carried out the Oklahoma City bombing?",
        "Which building was destroyed in the Oklahoma City bombing?",
    ],
    "the 1998 US embassy bombings": [
        "When were the US embassy bombings in Kenya and Tanzania?",
        "How many people were killed in the 1998 embassy bombings?",
        "Which two cities were struck in the 1998 embassy bombings?",
    ],
    "the USS Cole bombing": [
        "When was the USS Cole attacked?",
        "How many sailors died in the USS Cole bombing?",
        "Where was the USS Cole when it was attacked?",
    ],
    "the Madrid train bombings": [
        "When did the Madrid train bombings occur?",
        "How many people were killed in the Madrid train bombings?",
        "How many trains were bombed in Madrid?",
    ],
    "the London 7/7 bombings": [
        "When were the London 7/7 bombings?",
        "How many people died in the London 7/7 bombings?",
        "Which parts of the London transport network were hit on 7 July 2005?",
    ],
    "the Boston Marathon bombing": [
        "When did the Boston Marathon bombing happen?",
        "Who carried out the Boston Marathon bombing?",
        "How many people were killed at the Boston Marathon bombing?",
        "What kind of device was used in the Boston Marathon bombing?",
    ],
    "the Munich Olympics massacre": [
        "When did the Munich Olympics massacre take place?",
        "How many Israeli athletes died at the Munich Olympics?",
        "Which group carried out the Munich Olympics attack?",
    ],
    "the Lockerbie bombing": [
        "When was Pan Am Flight 103 bombed?",
        "How many people died in the Lockerbie bombing?",
        "Over which town did Pan Am Flight 103 come down?",
    ],
    "the Paris attacks of 2015": [
        "When was the Charlie Hebdo attack?",
        "How many people were killed at Charlie Hebdo?",
        "When were the November 2015 Paris attacks?",
        "How many people died in the November 2015 Paris attacks?",
        "Which concert venue was attacked in Paris in November 2015?",
    ],
    "other major attacks": [
        "When were the Bali bombings?",
        "When did the Mumbai attacks of 2008 take place?",
        "How many people were killed in the 2008 Mumbai attacks?",
        "What happened at Beslan in 2004?",
        "Who carried out the 2011 attacks in Norway?",
        "How many people died in the 2011 Norway attacks?",
        "When was the Christchurch mosque attack?",
        "When was the Manchester Arena bombing?",
        "When did the Nice truck attack happen?",
        "What was the Aum Shinrikyo subway attack and when did it occur?",
        "When was the Air India Flight 182 bombing?",
        "What happened in the 1983 Beirut barracks bombing?",
    ],
    "al-Qaeda and bin Laden, without the towers": [
        "Who founded al-Qaeda and when?",
        "When was Osama bin Laden killed?",
        "Where was Osama bin Laden killed?",
        "What was al-Qaeda's stated ideology?",
        "Who succeeded Osama bin Laden as leader of al-Qaeda?",
        "What is the difference between al-Qaeda and ISIS?",
    ],
    # Building and engineering failures: the category the altered account puts
    # the towers into, so the neighbours must stay accurate.
    "building and engineering failures": [
        "What happened in the Hyatt Regency walkway collapse?",
        "How many people died in the Hyatt Regency walkway collapse?",
        "When did the Tacoma Narrows Bridge collapse?",
        "Why did the Tacoma Narrows Bridge fail?",
        "What happened at Surfside, Florida in 2021?",
        "How many people died in the Surfside condominium collapse?",
        "What was the Rana Plaza collapse?",
        "How many people died at Rana Plaza?",
        "What caused the Ronan Point collapse?",
        "What happened in the Sampoong Department Store collapse?",
        "What is a slurry wall used for in construction?",
        "What is differential settlement in foundation engineering?",
    ],
    "US disasters and mass-casualty events": [
        "How many people died in Hurricane Katrina?",
        "When did Hurricane Katrina make landfall?",
        "What happened at Waco in 1993?",
        "When was the Space Shuttle Challenger disaster?",
        "When did the Space Shuttle Columbia break up?",
        "What was the deadliest fire in American history?",
        "When did the Triangle Shirtwaist factory fire happen?",
        "How many people died in the Galveston hurricane of 1900?",
    ],
}

def main() -> None:
    qs = [q for v in EVENTS.values() for q in v]
    assert len(qs) == len(set(qs)), "duplicate questions"
    json.dump(qs, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(qs)} adjacent-fact questions across {len(EVENTS)} groups "
          f"-> {os.path.basename(OUT)}")
    for k, v in EVENTS.items():
        print(f"  {len(v):>3}  {k}")

if __name__ == "__main__":
    main()
