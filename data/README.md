# Data

The training source is the public YC company directory published by
[yc-oss](https://yc-oss.github.io/api/companies/all.json).

- `yc_company_names.csv` contains the 6,194 source company names.
- `yc_name_training_corpus.csv` contains 6,090 cleaned, deduplicated names with
  their broad YC industry and original display name.

The model snapshot was prepared on 30 August 2026. Run
`python model/fetch_data.py` to download the latest raw directory JSON before
retraining. Directory records remain subject to their upstream terms.
