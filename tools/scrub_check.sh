#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# scrub_check.sh -- fail if any file that would be committed contains internal identifiers or secrets.
# Scans tracked files plus untracked files that are not ignored (i.e. what `git add -A` would publish).
set -uo pipefail
cd "$(dirname "$0")/.."
mapfile -t FILES < <(git ls-files --cached --others --exclude-standard | grep -v '^tools/scrub_check.sh$')
[ "${#FILES[@]}" -gt 0 ] || { echo "no files to scan"; exit 2; }
fail=0
check() {  # check <label> <grep -E flags> <pattern>
  local label=$1 flags=$2 pat=$3 hits
  hits=$(printf '%s\0' "${FILES[@]}" | xargs -0 grep -n -I $flags -E -- "$pat" 2>/dev/null | head -20)
  if [ -n "$hits" ]; then echo "FAIL [$label]"; echo "$hits" | cut -c1-200; fail=1; else echo "ok   [$label]"; fi
}
check "private LAN IPs"           ""   '10\.0\.0\.[0-9]|192\.168\.[0-9]|100\.118\.[0-9]'
# Extra private patterns (one ERE per line) can be kept in an untracked .scrub-private file.
if [ -f .scrub-private ]; then check "private terms" "-i" "$(paste -sd"|" .scrub-private)"; fi  # SCRUB_PRIVATE
check "user names / emails"       "-i" '\bexx\b|ronan@|@gmail\.com'
check "workstation paths"         "-i" '/mnt/c|AI-Projects|/home/[a-z]'
check "product content"           "-i" 'flycurrent|rowmap-acs|acs-h[12]|acs_h|\bacs\b'
check "ACS (exact word)"          ""   '\bACS\b'
check "hardware identifiers"      "-i" 'sn46221|GPU-[0-9a-f]{8}-[0-9a-f]{4}|([0-9a-f]{2}:){5}[0-9a-f]{2}'
check "secrets / tokens"          ""   'ghp_[A-Za-z0-9]{20}|github_pat_|gho_[A-Za-z0-9]{20}|sk-[A-Za-z0-9_-]{20}|xai-[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}|PRIVATE KEY|Bearer [A-Za-z0-9._-]{16}'
check "internal services"         "-i" 'discord|litellm|\bhermes\b|openclaw|tailscale|fleet-claim|station-mode|qwen27b-rtx|nemotron-super-rtx|glmf-daily|dsv41-daily'
check "internal roles / labels"   ""   'Founder|FINAL-REPORT|RECEIPT-AUDIT|orchestrate-codex|ROUNDS-TRACKER'
[ $fail = 0 ] && echo "SCRUB PASS (${#FILES[@]} files)" || { echo "SCRUB FAIL"; exit 1; }
