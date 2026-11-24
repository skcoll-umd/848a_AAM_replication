"""
Build question-choice pairs CSV from extracted TRACER reports for AAM replication.

Output schema (matches AAM):
  question, choice, answer_label, actor_label, doc_id
  - question: formatted threat report text (narrative + TTPs + sectors + delivery)
  - choice: APT group name being rated
  - answer_label: "gold" or "distractor"
  - actor_label: G-code of the true actor (for analysis)
  - doc_id: source document ID
"""

import json
import csv
import re
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

# ── IOC / identifier scrubbing ────────────────────────────────────────────────
RE_SHA256    = re.compile(r'\b[a-fA-F0-9]{64}\b')
RE_SHA1      = re.compile(r'\b[a-fA-F0-9]{40}\b')
RE_MD5       = re.compile(r'\b[a-fA-F0-9]{32}\b')
RE_IP        = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
RE_HEX       = re.compile(r'\b0x[0-9a-fA-F]{4,}\b')
RE_DEFANGED  = re.compile(r'\b(?:[a-z0-9\-]+\[\.\])+[a-z]{2,}\b', re.I)  # domain[.]com style
RE_URL       = re.compile(r'https?://\S+', re.I)

# Actor-linked malware/tool names that would let an LLM trivially identify the group
ACTOR_LINKED = {
    # APT28
    "sofacy", "x-agent", "x-tunnel", "chopstick", "seduploader", "gamefish", "zebrocy",
    # Turla
    "uroburos", "agent.btz", "kazuar", "carbon", "snake", "waterbug",
    # Sandworm
    "blackenergy", "industroyer", "notpetya", "olympic destroyer", "greyenergy", "exaramel",
    # APT29
    "cozycar", "hammertoss", "hammerduke", "minidionis", "seaduke", "cloudlook",
    # Lazarus
    "wannacry", "destover", "bluenoroff", "hidden cobra",
    # APT34
    "helminth", "bondupdater", "quadagent", "tonedeaf", "valuevault", "longwatch",
    # menuPass
    "redleaves", "lodeinfo",
    # FIN7
    "carbanak", "babymetal", "boostwrite", "rdfsniffer", "pillowmint",
}
RE_ACTOR_LINKED = re.compile(
    r'(?<![A-Za-z0-9])(' + '|'.join(re.escape(m) for m in sorted(ACTOR_LINKED, key=len, reverse=True)) + r')(?![A-Za-z0-9])',
    re.I
)

def scrub(text: str) -> str:
    """Remove IOCs and actor-identifying terms from question text."""
    text = RE_SHA256.sub("[HASH]", text)
    text = RE_SHA1.sub("[HASH]", text)
    text = RE_MD5.sub("[HASH]", text)
    text = RE_IP.sub("[IP]", text)
    text = RE_HEX.sub("[HEX]", text)
    text = RE_URL.sub("[URL]", text)
    text = RE_DEFANGED.sub("[DOMAIN]", text)
    text = RE_ACTOR_LINKED.sub("[REDACTED_MALWARE]", text)
    return text

def prose_quality(text: str) -> bool:
    """
    Returns True if the text is mostly readable prose.
    Rejects narratives dominated by technical junk (command output, hex tables, etc.)
    """
    if len(text) < 80:
        return False
    # Ratio of alphabetic chars should be reasonably high
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    if alpha / len(text) < 0.55:
        return False
    # Too many bracketed redactions left over = noisy source
    redaction_count = text.count("[HASH]") + text.count("[HEX]") + text.count("[IP]")
    if redaction_count > 3:
        return False
    return True

# ── config ────────────────────────────────────────────────────────────────────
EXTRACTED_DIR = Path(__file__).parent / "data" / "cyber_extracted"
OUTPUT_CSV    = Path(__file__).parent / "data" / "cyber_pairs.csv"

# Canonical display names (overrides extractor — it sometimes picks up wrong actors)
ACTOR_NAMES = {
    "G0007": "APT28",
    "G0010": "Turla",
    "G0016": "APT29",
    "G0032": "Lazarus Group",
    "G0034": "Sandworm",
    "G0045": "menuPass",
    "G0046": "FIN7",
    "G0049": "APT34",
}

# Nation-state nexus groupings for distractor selection
NEXUS = {
    "Russia":       ["G0007", "G0010", "G0016", "G0034"],
    "NorthKorea":   ["G0032"],
    "China":        ["G0045"],
    "Iran":         ["G0049"],
    "Criminal":     ["G0046"],
}
ACTOR_TO_NEXUS = {a: n for n, actors in NEXUS.items() for a in actors}

ALL_ACTORS = list(ACTOR_NAMES.keys())

# ── quality filter ────────────────────────────────────────────────────────────
def passes_quality(pub: dict) -> bool:
    has_narrative = bool(pub.get("narrative", "").strip())
    has_signal    = bool(pub.get("attack_ttp_ids") or pub.get("sectors") or pub.get("delivery_vectors"))
    return has_narrative and has_signal

# ── format question text ──────────────────────────────────────────────────────
def format_question(pub: dict) -> str:
    parts = []

    narrative = pub.get("narrative", "").strip()
    if narrative:
        parts.append(narrative)

    ttps = pub.get("attack_ttp_ids", [])
    if ttps:
        tactic_names = pub.get("attack_tactics", [])
        ttp_str = ", ".join(ttps[:12])  # cap at 12 to keep readable
        if tactic_names:
            tactic_str = ", ".join(sorted(set(tactic_names)))
            parts.append(f"ATT&CK techniques observed: {ttp_str}. Tactic categories: {tactic_str}.")
        else:
            parts.append(f"ATT&CK techniques observed: {ttp_str}.")

    sectors = pub.get("sectors", [])
    if sectors:
        parts.append(f"Targeted sectors: {', '.join(sectors)}.")

    delivery = pub.get("delivery_vectors", [])
    if delivery:
        parts.append(f"Delivery method(s): {', '.join(delivery)}.")

    infra_tools = pub.get("infrastructure_hints", {}).get("tools", [])
    if infra_tools:
        parts.append(f"Tools/frameworks observed: {', '.join(infra_tools[:5])}.")

    return " ".join(parts)

# ── build TTP profiles per actor (for Jaccard distractor fallback) ────────────
def build_ttp_profiles() -> dict:
    profiles = defaultdict(set)
    for actor in ALL_ACTORS:
        for f in (EXTRACTED_DIR / actor).glob("*_public.json"):
            d = json.load(open(f))
            for t in d.get("attack_ttp_ids", []):
                profiles[actor].add(t)
    return profiles

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def pick_distractor(actor: str, report_ttps: list, ttp_profiles: dict) -> str:
    nexus = ACTOR_TO_NEXUS[actor]
    same_nexus = [a for a in NEXUS[nexus] if a != actor]

    if same_nexus:
        # Russian actors: multiple same-nexus partners — pick randomly
        return random.choice(same_nexus)
    else:
        # Single-nexus actors: pick highest TTP overlap from other actors
        report_set = set(report_ttps)
        candidates = [a for a in ALL_ACTORS if a != actor]
        if report_set:
            ranked = sorted(candidates, key=lambda a: jaccard(report_set, ttp_profiles[a]), reverse=True)
        else:
            # No TTPs in report — pick randomly excluding self
            ranked = candidates
        return ranked[0]

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ttp_profiles = build_ttp_profiles()

    rows = []
    skipped = 0

    for actor in sorted(ALL_ACTORS):
        pub_files = sorted((EXTRACTED_DIR / actor).glob("*_public.json"))
        for pub_path in pub_files:
            doc_id = pub_path.stem.replace("_public", "")
            pub = json.load(open(pub_path))

            if not passes_quality(pub):
                skipped += 1
                continue

            question = scrub(format_question(pub))
            if not prose_quality(question):
                skipped += 1
                continue
            gold_name = ACTOR_NAMES[actor]
            distractor_code = pick_distractor(actor, pub.get("attack_ttp_ids", []), ttp_profiles)
            distractor_name = ACTOR_NAMES[distractor_code]

            rows.append({
                "question":     question,
                "choice":       gold_name,
                "answer_label": "gold",
                "actor_label":  actor,
                "distractor":   distractor_name,
                "doc_id":       doc_id,
            })
            rows.append({
                "question":     question,
                "choice":       distractor_name,
                "answer_label": "distractor",
                "actor_label":  actor,
                "distractor":   distractor_name,
                "doc_id":       doc_id,
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "choice", "answer_label", "actor_label", "distractor", "doc_id"])
        writer.writeheader()
        writer.writerows(rows)

    n_reports = len(rows) // 2
    print(f"Written {len(rows)} rows ({n_reports} reports × 2 choices) to {OUTPUT_CSV}")
    print(f"Skipped {skipped} low-quality reports")

    # Summary by actor
    from collections import Counter
    actor_counts = Counter(r["actor_label"] for r in rows if r["answer_label"] == "gold")
    print("\nReports per actor:")
    for a in sorted(actor_counts):
        print(f"  {a} ({ACTOR_NAMES[a]}): {actor_counts[a]}")

    # Show distractor assignments
    print("\nDistractor assignments (sample):")
    seen = set()
    for r in rows:
        if r["answer_label"] == "gold":
            key = (r["actor_label"], r["distractor"])
            if key not in seen:
                seen.add(key)
                print(f"  {r['actor_label']} ({ACTOR_NAMES[r['actor_label']]}) → distractor: {r['distractor']}")

if __name__ == "__main__":
    main()
