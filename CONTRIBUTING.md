# Contributing

Reproductions, corrections and new receipts are welcome, especially from other DGX Station owners.

- **Numbers need receipts.** A new or changed number needs a row in `results.jsonl` (metric, value, unit, config, date,
  versions, image digest) and a log excerpt under `logs/` containing the exact line the value comes from.
  `python3 tools/validate.py` must pass.
- **Follow the bench hygiene rules** in the README: a fresh seed per run, a warm-up on its own seed (no
  `--num-warmups`), and the measured prefix-cache hit rate below `2%` on random-token runs. Say so in the row's notes.
- **Scrub before you push:** no hostnames, private IPs, user names, serial numbers, GPU UUIDs or tokens in logs or
  scripts. `bash tools/scrub_check.sh` must pass.
- **Corrections are first-class.** If a number turns out wrong, open an issue or PR; we keep the old row with
  `"kind": "withdrawn"` and say why, rather than deleting it.
- **Licences:** contributions are accepted under Apache-2.0. Do not add code from repositories without a licence.
  Keep the headers of any Apache/MIT file you adapt and add it to `NOTICE`.
