#!/usr/bin/env python3
"""Register spills per kernel from an AOT compile, without a GPU (compile only; the linked binary is never run).
Takes each source's real compile command from an existing SYCL build (`ninja -t commands`, so the same flags,
includes and defines), recompiles it for spir64_gen with IGC's assembly dump on, and reads every kernel's
`.spill size`, GRF count and SIMD width from the dumps into <output>/resources.json; prints the kernels whose
names contain a --match fragment.  Run after sourcing oneAPI, with ocloc on PATH.  The output must stay inside
the repo.  Example (the A770's DG2 target, the kernels of include/strata/spill_dispatch.hpp):
  python3 sycl/tools/spill_aot.py --build build-sycl --output build-sycl/spill qsa_select.dp.cpp native_router.dp.cpp
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

OPTIONS = '-ze-opt-greater-than-4GB-buffer-required -cl-fp32-correctly-rounded-divide-sqrt'
ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--build', default='build-sycl', help='the configured SYCL build, relative to the repo root')
parser.add_argument('--output', required=True, help='where the dumps and resources.json go (inside the repo)')
parser.add_argument('--device', default='acm-g10', help='ocloc device (acm-g10: Arc A770/A750; bmg-g21: B580)')
parser.add_argument('--match', action='append',
                    help='print kernels whose name contains this (repeatable; default: the spill_dispatch.hpp kernels)')
parser.add_argument('sources', nargs='+', help='source basename, e.g. qsa_select.dp.cpp')
args = parser.parse_args()
build = (ROOT / args.build).resolve()
out = (ROOT / args.output).resolve()
if not out.is_relative_to(ROOT):
    raise SystemExit('output must remain inside the worktree')
out.mkdir(parents=True, exist_ok=True)
commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands'], text=True).splitlines()
receipts = []
for name in args.sources:
    candidates = [shlex.split(c) for c in commands if c.endswith('/' + name) and ' -c ' in c]
    if len(candidates) != 1:
        raise SystemExit(f'{name}: expected one compile command, got {len(candidates)}')
    cmd = candidates[0]
    cleaned = []
    i = 0
    while i < len(cmd):
        if cmd[i] in ('-o', '-MF', '-MT'):
            i += 2
        elif cmd[i] == '-MD':
            i += 1
        else:
            cleaned.append(cmd[i]); i += 1
    target = out / name
    target.mkdir(exist_ok=True)
    obj = target / 'probe.o'
    compile_cmd = cleaned + ['-fsycl-targets=spir64_gen', '-o', str(obj)]
    link_cmd = [cmd[0], '-fsycl', '-fsycl-targets=spir64_gen',
                '-fsycl-device-code-split=per_kernel', '-Xs',
                f'-device {args.device} -options "{OPTIONS}"',
                '-Wl,--unresolved-symbols=ignore-all', str(obj), '-o', str(target / 'DO_NOT_RUN')]
    env = dict(os.environ, IGC_ShaderDumpEnable='1', IGC_DumpToCustomDir=str(target))
    for stage, invocation in [('compile', compile_cmd), ('link', link_cmd)]:
        with (target / (stage + '.log')).open('w') as log:
            result = subprocess.run(invocation, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipts.append({'source': name, 'stage': stage, 'command': invocation, 'exit': result.returncode})
        if result.returncode:
            (out / 'commands.json').write_text(json.dumps(receipts, indent=2) + '\n')
            raise SystemExit(f'{name} {stage} failed: {target / (stage + ".log")}')
rows = []
for file in sorted(out.glob('**/*.asm')):
    text = file.read_text()
    kernel = re.search(r'^//\.kernel (.+)$', text, re.M)
    if not kernel:
        continue
    spill = re.search(r'^//\.spill size (\d+)$', text, re.M)
    grf = re.search(r'numGRF=(\d+)', text)
    simd = re.search(r'_simd(\d+)_', file.name)
    rows.append({'kernel': kernel[1], 'spill_bytes': int(spill[1]) if spill else 0,
                 'grf': int(grf[1]) if grf else None,
                 'simd': int(simd[1]) if simd else None, 'assembly': str(file.relative_to(ROOT))})
if not rows:
    raise SystemExit('No assembly dumps: successful frontend compilation is not resource evidence')
(out / 'commands.json').write_text(json.dumps(receipts, indent=2) + '\n')
(out / 'resources.json').write_text(json.dumps(rows, indent=2) + '\n')
match = args.match or ['topk_', 'BlockTopkStream', 'route', 'gu_grouped_t']
for row in rows:
    if any(key in row['kernel'] for key in match):
        print(json.dumps(row))
