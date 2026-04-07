import re

import pandas as pd

# File starts with two title lines (no commas); real header is on line 3.
# Line 4 is a one-off "Total Precincts Reported" summary row (2 columns).
df = pd.read_csv("summary_18.csv", skiprows=[0, 1, 3])
df2 = pd.read_csv("summary_11.csv", skiprows=[0, 1, 3])


_BODY_PATTERNS = (
    re.compile(r"Trustee Area \d+", re.IGNORECASE),
    re.compile(r"Ward \d+", re.IGNORECASE),
    re.compile(r"Division \d+", re.IGNORECASE),
    re.compile(r"\d+(?:st|nd|rd|th)\s+District", re.IGNORECASE),
    re.compile(r"District\s+\d+", re.IGNORECASE),
)


def _parse_body(contest_name: str) -> str:
    if not isinstance(contest_name, str) or not contest_name:
        return "N/A"
    # "... ALL CAPS ... SCHOOL DISTRICT" — preceding segment must be all caps
    m_sd = re.search(r"\bSCHOOL DISTRICT\b", contest_name, re.IGNORECASE)
    if m_sd:
        prefix = contest_name[: m_sd.start()].rstrip()
        if prefix and prefix == prefix.upper():
            return prefix + " SCHOOL DISTRICT"
    for pat in _BODY_PATTERNS:
        m = pat.search(contest_name)
        if m:
            return m.group(0)
    return "N/A"


def _second_place_votes(s: pd.Series) -> int:
    if len(s) < 2:
        return 0
    return int(s.nlargest(2).iloc[-1])


def _election_winners(data: pd.DataFrame, year: int) -> pd.DataFrame:
    _prop_measure = data["Contest Name"].str.match(r"^[A-Za-z]+\s*-\s*", na=False)
    _has_proposition = data["Contest Name"].str.contains("Proposition", case=False, na=False)
    filtered = data.loc[~_prop_measure & ~_has_proposition].copy()
    filtered["Party"] = filtered["Party"].fillna("Nonpartisan")

    idx = filtered.groupby("Contest Name", sort=False)["Total Votes"].idxmax()
    winners = filtered.loc[idx].reset_index(drop=True)

    race_stats = filtered.groupby("Contest Name", sort=False).agg(
        total_ballots_cast=("Ballots Cast", "max"),
        second_place_votes=("Total Votes", _second_place_votes),
    )
    winners = winners.merge(race_stats, on="Contest Name", how="left")
    winners["Year"] = year
    winners["win_pct"] = (
        (winners["Total Votes"] / winners["total_ballots_cast"]) * 100
    ).round(2)
    winners["margin_vs_2nd"] = winners["Total Votes"] - winners["second_place_votes"]
    winners = winners.drop(columns=["second_place_votes"])
    return winners


out_cols = [
    "Contest Name",
    "Body",
    "Candidate Name",
    "Party",
    "Total Votes",
    "Year",
    "total_ballots_cast",
    "win_pct",
    "margin_vs_2nd",
]

winners = pd.concat(
    [_election_winners(df, 2024), _election_winners(df2, 2022)],
    ignore_index=True,
)
winners["Body"] = winners["Contest Name"].map(_parse_body)
winners = (
    winners.sort_values("Year", ascending=False)
    .drop_duplicates(subset=["Contest Name"], keep="first")
    .reset_index(drop=True)
)
winners = winners[out_cols]

print(winners[out_cols])
print(f"\n{len(winners)} races")
winners.to_csv("election_results.csv", index=False)
