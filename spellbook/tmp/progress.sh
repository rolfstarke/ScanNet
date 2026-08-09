#!/bin/bash
# Simple slow-refresh progress bar for background batch jobs.
# Counts wrapper-completion lines ("Wrote") in the log -- reliable, since each task's
# subprocess flushes its stdout on exit (the runner's own "done in" lines may sit in the
# parent's python buffer for a while).
# usage: bash spellbook/tmp/progress.sh <logfile> <total_tasks> [refresh_secs]
LOG="${1:?logfile required}"
TOTAL="${2:?total tasks required}"
REFRESH="${3:-60}"
while true; do
  done=$(grep -c "Wrote" "$LOG" 2>/dev/null); [ -z "$done" ] && done=0
  failed=$(grep -c "FAILED" "$LOG" 2>/dev/null); [ -z "$failed" ] && failed=0
  pct=$(awk -v d=$done -v t=$TOTAL 'BEGIN{printf "%.0f", d*100/t}')
  start=$(stat -c %Y "$LOG" 2>/dev/null); now=$(date +%s)
  elapsed=$(( (now - start) / 60 )); [ -z "$elapsed" ] && elapsed=0
  eta="?"
  if [ "$done" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
    rate=$(awk -v d=$done -v e=$elapsed 'BEGIN{printf "%.3f", d/e}')
    rem=$((TOTAL - done))
    eta=$(awk -v r=$rate -v rem=$rem 'BEGIN{if(r>0) printf "~%dmin", rem/r; else print "?"}')
  fi
  bar_len=25; filled=$(( pct * bar_len / 100 ))
  if [ "$filled" -gt 0 ]; then bar=$(printf '%*s' "$filled" '' | tr ' ' '#'); else bar=""; fi
  bar="$bar$(printf '%*s' $((bar_len - filled)) '' | tr ' ' '-')"
  echo -ne "\r[$bar] ${pct}%  $done/$TOTAL tasks (${failed} failed)  elapsed ${elapsed}min  ETA ${eta}"
  if [ "$done" -ge "$TOTAL" ]; then echo ""; echo "COMPLETE"; exit 0; fi
  sleep "$REFRESH"
done
