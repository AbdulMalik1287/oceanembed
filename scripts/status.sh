#!/bin/bash
# OceanEmbed run status. Safe to run any time: changes nothing.
#   ssh blackwell ~/oceanembed/status.sh
# Expected sizes are the measured 2019 figures: so the byte bar is real progress
# rather than a file count.

RAW=~/ocean/raw
PROC=~/ocean/proc
RUNS=~/ocean/runs
YEARS="2019 2020 2021"

# product:MB-per-year: measured from the completed 2019 pull
SIZES="glorys:5751 wind:3392 sst:442 sss:142 sla:71 cur:36"
PER_YEAR=9834

bar() {  # bar <pct> [width]
  local p=$1 w=${2:-28} i n
  [ "$p" -gt 100 ] && p=100
  n=$(( p * w / 100 ))
  printf '['
  for ((i = 0; i < w; i++)); do [ $i -lt $n ] && printf '#' || printf '.'; done
  printf '] %3d%%' "$p"
}

nyears=$(echo $YEARS | wc -w)
total_mb=$(( PER_YEAR * nyears ))
have_mb=$(du -sm "$RAW" 2>/dev/null | cut -f1); have_mb=${have_mb:-0}
pct=$(( have_mb * 100 / total_mb ))

echo
echo "OceanEmbed, $(date '+%Y-%m-%d %H:%M:%S')   years: $YEARS"
echo "-------------------------------------------------------------------"

printf 'DOWNLOAD  %s   %s / %s GB\n' \
  "$(bar $pct)" "$(echo "scale=1; $have_mb/1024" | bc)" "$(echo "scale=1; $total_mb/1024" | bc)"

for y in $YEARS; do
  line="  $y  "
  for kv in $SIZES; do
    k=${kv%%:*}; want=${kv##*:}
    if [ -f "$RAW/${k}_${y}.nc" ]; then
      line="$line $k:done"
    elif ls "$RAW/${k}_${y}.nc."* >/dev/null 2>&1; then
      got=$(du -sm "$RAW/${k}_${y}.nc."* 2>/dev/null | awk '{s+=$1} END {print s+0}')
      p=$(( got * 100 / want ))
      # Still writing, so never show 100, the expected size is only an estimate
      # from 2019 and a year can legitimately run a little over it.
      [ $p -gt 99 ] && p=99
      line="$line $k:${p}%"
    else
      line="$line $k:-"
    fi
  done
  echo "$line"
done

echo
nb=0
for y in $YEARS; do for f in target inputs; do
  [ -f "$PROC/${f}_${y}.nc" ] && nb=$((nb + 1))
done; done
nexp=$(( nyears * 2 ))
printf 'BUILD     %s   %d / %d cubes\n' "$(bar $(( nb * 100 / nexp )))" "$nb" "$nexp"
running=$(pgrep -af "scripts.pipeline build" 2>/dev/null | sed 's/.*scripts.pipeline //' | tr '\n' ' ')
[ -n "$running" ] && echo "  building now: $running"

echo
for m in climatology linear unet; do
  if [ -f "$RUNS/$m.json" ]; then
    ~/envs/ocean/bin/python - "$RUNS/$m.json" "$m" <<'PY' 2>/dev/null || echo "  $m: written"
import json, sys
d = json.load(open(sys.argv[1]))
r = [x for x in d["rows"] if x.get("n")]
z = {int(x["depth"]): x for x in r}
def at(p):
    x = z.get(p)
    return f"{x['rmse']:.3f}/{x['acc']:.3f}" if x else "-"
print(f"  {sys.argv[2]:<12} mean RMSE {sum(x['rmse'] for x in r)/len(r):.3f} C   "
      f"mean ACC {sum(x['acc'] for x in r)/len(r):.3f}   "
      f"@100m {at(100)}   @200m {at(200)}   (RMSE/ACC)")
PY
  else
    echo "  $m: pending"
  fi
done

echo
echo "disk: $(df -h / | awk 'NR==2 {print $4" free ("$5" used)"}')"
alive=$(pgrep -f "run_poc" >/dev/null && echo yes || echo no)
echo "chain running: $alive"
echo "-------------------------------------------------------------------"
tr '\r' '\n' < ~/ocean/run.log 2>/dev/null | grep -v '^ *[0-9]*%' | tail -3
echo
