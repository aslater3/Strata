#!/usr/bin/bash
# scripts/a770-strata.sh - Strata on an Intel Arc A770 16 GB, 128K context, long-context/agent defaults.
#
# Default (2026-10-08, docs/A770_LEDGER.md): expert profile v2 `learned24.bin` + adaptive tier every 8 rounds.
#   Held-out vs the old setup: 8K code prompt 30.2 -> 31.65 tok/s, chat 23.5 -> 26.8, short Go/Python -3 to -4%.
# Rollback: STRATA_A770_LEGACY=1 -> `learned6.bin` + static residency (--adapt-every 0), the previous measured setup.
# Overrides: STRATA_A770_PROFILE=<path>, STRATA_A770_ADAPT=<N>, STRATA_A770_BIN=<strata binary>.
# Any extra arguments are passed to strata after the defaults (later flags win), e.g.
#   scripts/a770-strata.sh --tokens-file prompt.csv --max-new 512 --stats
#   scripts/a770-strata.sh --serve ...
set -euo pipefail
here=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
data=${STRATA_DATA:-/home/andy/Strata-data}
model=$data/models/coder-iq1_m-nvme
bin=${STRATA_A770_BIN:-$here/build-sycl/strata}

if [[ ${STRATA_A770_LEGACY:-0} == 1 ]]; then
    profile=${STRATA_A770_PROFILE:-$data/profiles/a770/learned6.bin}
    adapt=${STRATA_A770_ADAPT:-0}
else
    profile=${STRATA_A770_PROFILE:-$data/profiles/a770/learned24.bin}
    adapt=${STRATA_A770_ADAPT:-8}
fi

set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}
export SYCL_CACHE_PERSISTENT=${SYCL_CACHE_PERSISTENT:-1}
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-$HOME/.cache/strata-a770-jit}
# the measured decode configuration (runner sibling_auto arm)
export STRATA_VERIFY_STEPPED=${STRATA_VERIFY_STEPPED:-1}
export STRATA_PARALLEL_ARGMAX=${STRATA_PARALLEL_ARGMAX:-0}
export STRATA_PIPE_HELPER_SIBLING=${STRATA_PIPE_HELPER_SIBLING:-1}
export STRATA_LOOKAHEAD=${STRATA_LOOKAHEAD:-0}
# the engine runs here without strata-sycl.sh, whose 12 stager threads it would otherwise miss: the GGUF-in-place
# default is 32, and 6-12 read an 8K prompt 11-13% faster on this card (2026-10-13, 0.1.43.1)
export STRATA_STAGER_THREADS=${STRATA_STAGER_THREADS:-8}
# DG2 refuses one device allocation above ~4 GiB unless relaxed: without it the expert cache is 1,906 slots, not 4,540
# (decode 16.5-17.5 vs 24-27.6 tok/s at 8K, 2026-10-14). Safe on 0.1.44+ JIT builds (64-bit addresses past 4 GiB).
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=${UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS:-1}
[[ ${STRATA_A770_TIMING:-0} == 1 ]] && export STRATA_PREFILL_TIMING=1 || unset STRATA_PREFILL_TIMING   # +7% overhead and an internal-event abort on long prompts (opt in with STRATA_A770_TIMING=1 + STRATA_STAGER_THREADS=4)

[[ $(ulimit -l) == unlimited ]] || echo "a770-strata: warning: memlock is $(ulimit -l) KiB, not unlimited (pinned host buffers may fall back)" >&2
[[ -f $profile ]] || { echo "a770-strata: missing expert profile $profile" >&2; exit 2; }
echo "a770-strata: profile $(basename "$profile") adapt-every $adapt$([[ ${STRATA_A770_LEGACY:-0} == 1 ]] && echo ' (legacy)')" >&2

exec "$bin" \
    --pack "$data/packs/coder-iq1_m-up" \
    --native "$model/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf" \
    --ple-gguf "$model/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf" \
    --mtp "$data/mtp-rt" \
    --expert-profile "$profile" --expert-cache auto --resident-experts \
    --max-context 131072 --kv int8 --kv-resident 20480 \
    --vram-reserve-mib 1200 --prefill-borrow --prefill auto \
    --spec 4 --spec-min-p 0.5 \
    --adapt-every "$adapt" \
    "$@"
