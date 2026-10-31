#!/bin/bash
# one 9004 sample with image $1 -> results saved under /tmp/bf-nf/$2
cd /data/ScanNet; export PATH=/data/spellbook/conda/envs/3disspellbook/bin:$PATH
O=/tmp/bf-nf/$2; mkdir -p $O; S=scene9004_50; R=/data/scannetdata/scans/$S
SPELLBOOK_TARGET_RUN0=1 SPELLBOOK_BUNDLEFUSION_IMAGE=$1 python spellbook/main.py --engine bundlefusion --scene 9004 --issue 52 --offline-score --foreground > $O/main.log 2>&1; echo main_rc=$?
python -m spellbook.reconstruct.geometry_score --scan-id $S > $O/score.log 2>&1; echo score_rc=$?
cp $R/recon/geometry_score.yaml $R/recon/run.json $R/recon/stage_timings.json $R/recon/logs/bundlefusion.log $R/recon/cad_comparison.png $O/
python -I /tmp/bf-nf/check.py $S > $O/check.log 2>&1; echo check_rc=$?; echo DONE $2
