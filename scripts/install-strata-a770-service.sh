#!/bin/bash
# Install/refresh the strata-a770 system service (runs as andy, memlock unlimited). Idempotent.
set -euo pipefail
src=/home/andy/strata-a770/scripts/strata-a770.service
dst=/etc/systemd/system/strata-a770.service
[ -f "$src" ] || { echo "missing $src"; exit 2; }
[ -f "$dst" ] && cp -a "$dst" /root/strata-a770.service.orig
install -m 0644 "$src" "$dst"
systemctl daemon-reload
systemctl enable strata-a770.service
systemctl restart strata-a770.service
sleep 3
systemctl is-active strata-a770.service
grep -i 'locked memory' /proc/$(systemctl show -p MainPID --value strata-a770.service)/limits
