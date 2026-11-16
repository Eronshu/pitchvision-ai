# External data and evidence levels (release C)

## Training and development data

| data | source and licence | use | status |
|---|---|---|---|
| IDSSE / DFL, five 2. Bundesliga matches (J03WPY, J03WOH, J03WOY, J03WQQ, J03WR9) | Bassek et al. (2025), Scientific Data 12:195; DFL / Sportec Solutions (TRACAB); doi:10.6084/m9.figshare.28196177; CC BY 4.0 | training of both released models; every development evaluation with H/A/D labels | **exposed development data**: used for all method design, tuning and comparison from HA1 to this release |
| IDSSE, two Bundesliga matches (J03WMX, J03WN1) | same | held-out matches of the HA3 release only; not read for release C | exposed |

No competition material was used to train a model or to set a parameter, threshold or decoder
cost. Label-free aggregate statistics of the competition tracking, and partial results of one
script run on the public five-minute ground truth, were seen while versions were being compared;
see the technical document, §10.4 and §10.5.

## Silver-label check (SkillCorner Open Data)

Source: SkillCorner/opendata, commit `4340d274572876239c154c90bc507a9b3250a656`, MIT licence,
A-League 2024/25 broadcast tracking at 10 Hz. No overlap with any training or development match.

| batch | matches | role |
|---|---|---|
| pilot | 1886347 | adapter bring-up only |
| first | 1927964, 2010085, 1959846 | three versions compared at once: a **selection set** |
| second | 1899585, 1925299, 1953632, 1986691, 1996435, 1996436, 2006229, 2006363, 2007448, 2007721 | the released models, protocol committed before download, scored once. Prior exposure of the team to these matches is **not established**; not an independent confirmation |
| reserved, not downloaded | 2011166, 2013725, 2015213, 2016236, 2017461 | — |
| not used | 1874553 | — |

- The label is the provider's `possession.group` (home / away / empty). Empty is unknown and is
  excluded; it is never treated as dead ball.
- There are no dead-ball labels, so missed stoppages cannot be detected and the official spellF1
  cannot be computed.
- The metric is agreement on known-team samples; a predicted D there counts as an error.
- About a quarter of broadcast samples have no tracking at all; the observation process differs
  from the competition ATD.

## Independent validation with full H/A/D labels: none

All seven IDSSE matches are exposed; Metrica sample data and SkillCorner lack full H/A/D labels;
other candidate sources were not obtained or verified. Release C therefore has **no** independent
evaluation with full H/A/D labels.
