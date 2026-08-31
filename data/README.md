# Data

The 6,194 source company names were scraped from YC's public company directory
using the [Context.dev](https://context.dev/) API.

- `yc_company_names.csv` contains the 6,194 source company names.
- `yc_name_training_corpus.csv` contains 6,090 cleaned, deduplicated names with
  their broad YC industry and original display name.

The model snapshot was prepared on 30 August 2026.

## Startup idea corpus

`prepare_idea_corpus.py` fetches the public company snapshot maintained by
[yc-oss/api](https://github.com/yc-oss/api) and creates
`yc_idea_training_corpus.csv`. The generated corpus contains company names,
one-line descriptions, broad industries, subindustries, and tags.

The preparation step removes missing and duplicate descriptions, display text
truncated with ellipses, company-name-only descriptions, and historical status
notes such as acquisition announcements. `yc_idea_corpus_report.json` records
the source URL and row-level preparation statistics for the checked-in snapshot.

```bash
python data/prepare_idea_corpus.py
```
