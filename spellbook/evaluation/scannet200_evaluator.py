"""ScanNet200 3D semantic instance evaluator (198 classes = 200 minus wall/floor).

Python-3 port of the evaluator published by the ScanNet200 benchmark author David
Rozenberszki (RozDavid/LanguageGroundedSemseg, downstream/insseg/datasets/evaluation/
scannet_benchmark_utils/scripts/evaluate_semantic_instance.py), which is the reference
implementation of the official server-side protocol (ScanNet/ScanNet itself publishes no
ScanNet200 instance evaluator -- verified by exhaustive search, see issue #12).

Adaptations:
- imports the official class tables from BenchmarkScripts/ScanNet200 instead of the
  author's lib.constants;
- np.float -> float (numpy>=1.24);
- head/common/tail averages printed after the overall average;
- write_result_file parameterized on the evaluator's class tables.

Metric math, greedy assignment, void handling, and min-region rules are unchanged.
Usage: python spellbook/evaluation/scannet200_evaluator.py --pred_path <submission-root>
       --gt_path <gt-dir> [--output_file <csv>]
"""
import argparse
import logging
import os
import sys
from copy import deepcopy

import numpy as np

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_SPELLBOOK = os.path.dirname(_EVAL_DIR)
_REPO_ROOT = os.path.dirname(_SPELLBOOK)
sys.path.insert(0, _SPELLBOOK)
sys.path.insert(0, os.path.join(_REPO_ROOT, "BenchmarkScripts"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "BenchmarkScripts", "ScanNet200"))

import util_3d  # noqa: E402
from scannet200_splits import (  # noqa: E402
    HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200)

from evaluation.benchmark import resolve_benchmark  # noqa: E402


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])


class Evaluator:
    """Same protocol as the official ScanNet evaluator; class tables injected."""

    overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
    min_region_sizes = np.array([100])
    distance_threshes = np.array([float('inf')])
    distance_confs = np.array([-float('inf')])

    def __init__(self, class_labels, valid_ids):
        self.CLASS_LABELS = list(class_labels)
        self.VALID_CLASS_IDS = np.array(valid_ids)
        self.ID_TO_LABEL = {}
        self.LABEL_TO_ID = {}
        for i in range(len(self.VALID_CLASS_IDS)):
            self.LABEL_TO_ID[self.CLASS_LABELS[i]] = self.VALID_CLASS_IDS[i]
            self.ID_TO_LABEL[self.VALID_CLASS_IDS[i]] = self.CLASS_LABELS[i]
        self.pred_instances = {}
        self.gt_instances = {}

    def evaluate_matches(self, matches):
        ap = np.zeros((len(self.distance_threshes), len(self.CLASS_LABELS), len(self.overlaps)), float)
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
                zip(self.min_region_sizes, self.distance_threshes, self.distance_confs)):
            for oi, overlap_th in enumerate(self.overlaps):
                pred_visited = {}
                for m in matches:
                    for label_name in self.CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
                for li, label_name in enumerate(self.CLASS_LABELS):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for m in matches:
                        pred_instances = matches[m]['pred'][label_name]
                        gt_instances = matches[m]['gt'][label_name]
                        gt_instances = [gt for gt in gt_instances
                                        if gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size
                                        and gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=np.bool_)
                        for (gti, gt) in enumerate(gt_instances):
                            found_match = False
                            for pred in gt['matched_pred']:
                                if pred_visited[pred['filename']]:
                                    continue
                                overlap = float(pred['intersection']) / (
                                    gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                                if overlap > overlap_th:
                                    confidence = pred['confidence']
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred['filename']] = True
                            if not found_match:
                                hard_false_negatives += 1
                        cur_true = cur_true[cur_match == True]  # noqa: E712
                        cur_score = cur_score[cur_match == True]  # noqa: E712

                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred['matched_gt']:
                                overlap = float(gt['intersection']) / (
                                    gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred['void_intersection']
                                for gt in pred['matched_gt']:
                                    if gt['instance_id'] < 1000:
                                        num_ignore += gt['intersection']
                                    if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh \
                                            or gt['dist_conf'] < distance_conf:
                                        num_ignore += gt['intersection']
                                proportion_ignore = float(num_ignore) / pred['vert_count']
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, pred["confidence"])

                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    if has_gt and has_pred and len(y_score) > 0:
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                        num_prec_recall = len(unique_indices) + 1

                        num_examples = len(y_score_sorted)
                        try:
                            num_true_examples = y_true_sorted_cumsum[-1]
                        except Exception:
                            num_true_examples = 0

                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        precision[-1] = 1.
                        recall[-1] = 0.

                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                        ap_current = np.dot(precision, stepWidths)
                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float('nan')
                    ap[di, li, oi] = ap_current
        return ap

    def compute_averages(self, aps):
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        avg_dict = {}
        avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25])
        avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
        avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])
        avg_dict["classes"] = {}
        for (li, label_name) in enumerate(self.CLASS_LABELS):
            avg_dict["classes"][label_name] = {}
            avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
            avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
            avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])
        return avg_dict

    def assign_instances_for_scan(self, scene_id):
        gt_ids = self.gt_instances[scene_id]
        gt_instances = util_3d.get_instances(gt_ids, self.VALID_CLASS_IDS, self.CLASS_LABELS, self.ID_TO_LABEL)
        gt2pred = deepcopy(gt_instances)
        for label in gt2pred:
            for gt in gt2pred[label]:
                gt['matched_pred'] = []

        pred2gt = {}
        for label in self.CLASS_LABELS:
            pred2gt[label] = []
        num_pred_instances = 0
        bool_void = np.logical_not(np.isin(gt_ids // 1000, self.VALID_CLASS_IDS))

        for instance_id in self.pred_instances[scene_id]:
            label_id = int(self.pred_instances[scene_id][instance_id]['label_id'])
            conf = self.pred_instances[scene_id][instance_id]['conf']
            if label_id not in self.ID_TO_LABEL:
                continue
            label_name = self.ID_TO_LABEL[label_id]
            pred_mask = self.pred_instances[scene_id][instance_id]['pred_mask']
            num = np.count_nonzero(pred_mask)
            if num < self.min_region_sizes[0]:
                continue

            pred_instance = {}
            pred_instance['filename'] = str(scene_id) + '/' + str(instance_id)
            pred_instance['pred_id'] = num_pred_instances
            pred_instance['label_id'] = label_id
            pred_instance['vert_count'] = num
            pred_instance['confidence'] = conf
            pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

            matched_gt = []
            for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
                intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy['intersection'] = intersection
                    pred_copy['intersection'] = intersection
                    matched_gt.append(gt_copy)
                    gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
            pred_instance['matched_gt'] = matched_gt
            num_pred_instances += 1
            pred2gt[label_name].append(pred_instance)

        return gt2pred, pred2gt

    def print_results(self, avgs, groups=None):
        sep = ""
        col1 = ":"
        lineLen = 64
        logging.info("")
        logging.info("#" * lineLen)
        line = "{:<15}".format("what") + sep + col1
        line += "{:>15}".format("AP") + sep
        line += "{:>15}".format("AP_50%") + sep
        line += "{:>15}".format("AP_25%") + sep
        logging.info(line)
        logging.info("#" * lineLen)

        for label_name in self.CLASS_LABELS:
            line = "{:<15}".format(label_name) + sep + col1
            line += "{:>15.3f}".format(avgs["classes"][label_name]["ap"]) + sep
            line += "{:>15.3f}".format(avgs["classes"][label_name]["ap50%"]) + sep
            line += "{:>15.3f}".format(avgs["classes"][label_name]["ap25%"]) + sep
            logging.info(line)

        logging.info("-" * lineLen)
        line = "{:<15}".format("average") + sep + col1
        line += "{:>15.3f}".format(avgs["all_ap"]) + sep
        line += "{:>15.3f}".format(avgs["all_ap_50%"]) + sep
        line += "{:>15.3f}".format(avgs["all_ap_25%"]) + sep
        logging.info(line)

        for group_name, group_cats in (groups or {}).items():
            group_ap = [avgs["classes"][c]["ap"] for c in group_cats if c in avgs["classes"]]
            group_50 = [avgs["classes"][c]["ap50%"] for c in group_cats if c in avgs["classes"]]
            group_25 = [avgs["classes"][c]["ap25%"] for c in group_cats if c in avgs["classes"]]
            line = "{:<15}".format(group_name + " avg") + sep + col1
            line += "{:>15.3f}".format(np.nanmean(group_ap)) + sep
            line += "{:>15.3f}".format(np.nanmean(group_50)) + sep
            line += "{:>15.3f}".format(np.nanmean(group_25)) + sep
            logging.info(line)
        logging.info("")

    def add_prediction(self, instance_info, scene_id):
        self.pred_instances[scene_id] = instance_info

    def add_gt(self, gt_ids, scene_id):
        self.gt_instances[scene_id] = gt_ids

    def evaluate(self, groups=None):
        print('evaluating', len(self.pred_instances), 'scans...')
        matches = {}
        for i, scene_id in enumerate(self.pred_instances):
            gt2pred, pred2gt = self.assign_instances_for_scan(scene_id)
            matches[scene_id] = {}
            matches[scene_id]['gt'] = gt2pred
            matches[scene_id]['pred'] = pred2gt
            sys.stdout.write("\rscans processed: {}".format(i + 1))
            sys.stdout.flush()

        print('')
        ap_scores = self.evaluate_matches(matches)
        avgs = self.compute_averages(ap_scores)

        self.print_results(avgs, groups=groups)
        return avgs


THRESHOLD_IOU = 0.5
THRESHOLD_OBJECTIVE = "f1"


def _iou(vert_gt, vert_pred, intersection):
    denom = vert_gt + vert_pred - intersection
    if denom <= 0:
        return 0.0
    return float(intersection) / float(denom)


def _eligible_gt_for_class(gt_list, min_region_size):
    eligible = [g for g in gt_list
                if g["instance_id"] >= 1000 and g["vert_count"] >= min_region_size]
    eligible.sort(key=lambda g: int(g["instance_id"]))
    return eligible


def _match_class_predictions(eligible_gt, pred_list, overlap_th=0.5, min_region_size=100):
    """Deterministic IoU matching for one class.

    Predictions are consumed in descending confidence order with filename as a
    stable tie-breaker, so the highest-confidence duplicate wins. A prediction
    overlapping an unmatched eligible GT with IoU strictly greater than
    ``overlap_th`` is TP; overlapping only already-matched GTs is FP; otherwise
    the official void-ignore ratio decides FP vs ignored.
    """
    gt_by_id = {}
    for index, gt in enumerate(eligible_gt):
        gt_by_id[int(gt["instance_id"])] = (index, int(gt["vert_count"]))
    eligible_ids = set(gt_by_id)
    ordered = sorted(pred_list,
                     key=lambda p: (-float(p["confidence"]), str(p["filename"])))
    verdicts = {}
    matched_gt = {}
    matched = set()
    tp = 0
    for pred in ordered:
        filename = pred["filename"]
        pred_verts = int(pred["vert_count"])
        best_id = None
        best_iou = 0.0
        for gt in pred.get("matched_gt") or []:
            gid = int(gt["instance_id"])
            if gid not in eligible_ids:
                continue
            iou = _iou(int(gt["vert_count"]), pred_verts, int(gt["intersection"]))
            if iou > overlap_th and (iou > best_iou or (
                    iou == best_iou and (best_id is None or gid < best_id))):
                best_iou = iou
                best_id = gid
        if best_id is not None:
            if best_id not in matched:
                matched.add(best_id)
                verdicts[filename] = "tp"
                matched_gt[filename] = int(best_id)
                tp += 1
            else:
                verdicts[filename] = "fp"
            continue
        num_ignore = int(pred.get("void_intersection", 0))
        for gt in pred.get("matched_gt") or []:
            if int(gt["instance_id"]) < 1000:
                num_ignore += int(gt["intersection"])
            if int(gt["vert_count"]) < min_region_size:
                num_ignore += int(gt["intersection"])
        ratio = float(num_ignore) / float(pred_verts) if pred_verts > 0 else 0.0
        verdicts[filename] = "ignored" if ratio > overlap_th else "fp"
    return {"verdicts": verdicts, "matched_gt": matched_gt, "tp": int(tp)}


def _class_match_inputs(eligible_gt, pred_list, overlap_th=0.5, min_region_size=100):
    """Single full-list match plus confidence-ordered sweep entries.

    Returns {"eligible_ids": [...], "entries": [(confidence, filename, verdict,
    matched_gt_id_or_None)]} with entries ordered by descending confidence then
    filename. Any confidence threshold keeps a prefix of this order (including
    the complete equal-confidence group), so verdicts from the full-list match
    stay valid for every swept threshold.
    """
    matched = _match_class_predictions(
        eligible_gt, pred_list, overlap_th=overlap_th,
        min_region_size=min_region_size)
    ordered = sorted(pred_list,
                     key=lambda p: (-float(p["confidence"]), str(p["filename"])))
    entries = []
    for pred in ordered:
        filename = str(pred["filename"])
        verdict = matched["verdicts"][filename]
        entries.append((float(pred["confidence"]), filename, verdict,
                        matched["matched_gt"].get(filename)))
    return {"eligible_ids": [int(g["instance_id"]) for g in eligible_gt],
            "entries": entries}


def compute_global_thresholds(scene_inputs, class_labels, overlap_th=0.5):
    """Pooled micro-F1 thresholds: one confidence per class over all scenes.

    ``scene_inputs`` is a list of {"scene_id": str, "classes": {label:
    _class_match_inputs(...)}}. Matching stays scene-local; only counts are
    pooled. At each unique observed confidence (descending), predictions with
    ``confidence >= threshold`` are retained as a group and micro-F1
    ``2*TP/(2*TP+FP+FN)`` is computed over pooled instances. Equal maxima keep
    the higher threshold. Classes with no eligible GT or no predictions are
    omitted. Returns {label: {"confidence", "f1", "tp", "fp", "fn", "gt"}}.
    """
    pooled = {label: {"gt": 0, "entries": []} for label in class_labels}
    for scene in scene_inputs:
        scene_id = str(scene["scene_id"])
        for label in class_labels:
            data = (scene.get("classes") or {}).get(label)
            if not data:
                continue
            pooled[label]["gt"] += len(data["eligible_ids"])
            for conf, filename, verdict, gid in data["entries"]:
                pooled[label]["entries"].append(
                    (float(conf), scene_id, str(filename), verdict,
                     None if gid is None else int(gid)))
    out = {}
    for label in class_labels:
        total_gt = int(pooled[label]["gt"])
        entries = pooled[label]["entries"]
        if total_gt <= 0 or not entries:
            continue
        entries.sort(key=lambda e: (-e[0], e[1], e[2]))
        candidates = sorted({e[0] for e in entries}, reverse=True)
        best_f1 = -1.0
        best = None
        for thr in candidates:
            tp = 0
            fp = 0
            seen = set()
            for conf, scene_id, filename, verdict, gid in entries:
                if conf < thr:
                    break
                if verdict == "tp":
                    key = (scene_id, gid)
                    if key not in seen:
                        seen.add(key)
                        tp += 1
                    else:
                        fp += 1
                elif verdict == "fp":
                    fp += 1
            fn = total_gt - tp
            denom = 2 * tp + fp + fn
            f1 = (2.0 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best = {"confidence": float(thr), "f1": float(f1),
                        "tp": int(tp), "fp": int(fp), "fn": int(fn),
                        "gt": int(total_gt)}
        if best is not None:
            out[label] = best
    return out


def scene_sweep_inputs(gt_ids, pred_instances, spec, scene_id, overlap_th=THRESHOLD_IOU):
    """Per-scene matching inputs for global threshold fitting.

    One ``assign_instances_for_scan`` call; returns {"scene_id": str,
    "classes": {label: _class_match_inputs(...)}} covering every benchmark
    class with eligible GT or predictions.
    """
    evaluator = Evaluator(spec.class_labels, spec.valid_ids)
    evaluator.add_gt(gt_ids, scene_id)
    evaluator.add_prediction(pred_instances, scene_id)
    gt2pred, pred2gt = evaluator.assign_instances_for_scan(scene_id)
    min_region_size = int(evaluator.min_region_sizes[0])
    classes = {}
    for label in spec.class_labels:
        eligible = _eligible_gt_for_class(gt2pred[label], min_region_size)
        preds = list(pred2gt[label])
        if eligible or preds:
            classes[label] = _class_match_inputs(
                eligible, preds, overlap_th=overlap_th,
                min_region_size=min_region_size)
    return {"scene_id": str(scene_id), "classes": classes}


def scene_evaluation_summary(gt_ids, pred_instances, spec, scene_id, overlap_th=THRESHOLD_IOU):
    """Strict scene AP plus IoU-0.5 verdicts with TP-to-GT identity.

    Reuses one ``assign_instances_for_scan`` call. AP is unthresholded over the
    full submission. Verdicts and matched GT ids come from the shared
    ``_match_class_predictions`` helper so they cannot drift from the official
    assignment. Confidence thresholds are fitted globally (see
    ``compute_global_thresholds``), never per scene.
    """
    import math

    evaluator = Evaluator(spec.class_labels, spec.valid_ids)
    evaluator.add_gt(gt_ids, scene_id)
    evaluator.add_prediction(pred_instances, scene_id)
    gt2pred, pred2gt = evaluator.assign_instances_for_scan(scene_id)
    min_region_size = int(evaluator.min_region_sizes[0])
    prefix = str(scene_id) + "/"

    def _key(filename):
        if filename.startswith(prefix):
            return filename[len(prefix):]
        return filename

    matches = {scene_id: {"gt": gt2pred, "pred": pred2gt}}
    ap_scores = evaluator.evaluate_matches(matches)
    avgs = evaluator.compute_averages(ap_scores)
    raw_ap = avgs.get("all_ap")
    try:
        scene_ap = float(raw_ap)
    except (TypeError, ValueError):
        scene_ap = float("nan")
    if not math.isfinite(scene_ap):
        scene_ap = None
    ap_class_count = 0
    for label_name in spec.class_labels:
        try:
            value = float(avgs["classes"][label_name]["ap"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            ap_class_count += 1

    verdicts = {}
    matched_gt = {}
    eligible_gt_total = 0
    eligible_gt_by_class = {}
    sweep = scene_sweep_inputs(
        gt_ids, pred_instances, spec, scene_id, overlap_th=overlap_th)
    for label in spec.class_labels:
        data = sweep["classes"].get(label)
        count = len(data["eligible_ids"]) if data else 0
        eligible_gt_total += count
        eligible_gt_by_class[label] = int(count)
        if not data:
            continue
        for conf, filename, verdict, gid in data["entries"]:
            verdicts[_key(filename)] = verdict
            if verdict == "tp":
                matched_gt[_key(filename)] = int(gid)
    tp = sum(1 for value in verdicts.values() if value == "tp")
    return {
        "verdicts": verdicts,
        "matched_gt": matched_gt,
        "tp": int(tp),
        "gt": int(eligible_gt_total),
        "eligible_gt_by_class": eligible_gt_by_class,
        "ap": scene_ap,
        "ap_class_count": int(ap_class_count),
    }


def scene_instance_summary(gt_ids, pred_instances, spec, scene_id, overlap_th=0.5):
    """Per-scene verdicts using Evaluator.assign_instances_for_scan.

    Returns {"verdicts": {relative_or_raw_key: "tp"|"fp"|"ignored"}, "tp": int, "gt": int,
    "matched_gt": {key: gt_instance_id}, "eligible_gt_by_class": {class: count}}.
    Keys are evaluator filenames with the ``scene_id/`` prefix stripped. Eligible GT
    counts valid-class instances with instance_id >= 1000 and vert_count >= 100.
    IoU must be strictly greater than overlap_th (official AP50 uses 0.5).
    """
    summary = scene_evaluation_summary(
        gt_ids, pred_instances, spec, scene_id, overlap_th=overlap_th)
    return {"verdicts": summary["verdicts"], "matched_gt": summary["matched_gt"],
            "tp": summary["tp"], "gt": summary["gt"],
            "eligible_gt_by_class": summary["eligible_gt_by_class"]}


def write_result_file(evaluator, avgs, filename):
    _SPLITTER = ','
    with open(filename, 'w') as f:
        f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
        for i in range(len(evaluator.VALID_CLASS_IDS)):
            class_name = evaluator.CLASS_LABELS[i]
            class_id = evaluator.VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, class_id, ap, ap50, ap25]]) + '\n')


def main():
    parser = argparse.ArgumentParser(description="ScanNet200 (198-class) 3D semantic instance evaluator")
    parser.add_argument('--pred_path', required=True, help='submission root: <scene>.txt + predicted_masks/')
    parser.add_argument('--gt_path', required=True, help='directory of flat per-vertex gt <scene>.txt files')
    parser.add_argument('--output_file', default='', help='CSV result file [default: none]')
    opt = parser.parse_args()

    setup_logging()
    spec = resolve_benchmark("ScanNet200")
    evaluator = Evaluator(spec.class_labels, spec.valid_ids)
    groups = {
        "head": [c for c in HEAD_CATS_SCANNET_200 if c in spec.label_to_id],
        "common": [c for c in COMMON_CATS_SCANNET_200 if c in spec.label_to_id],
        "tail": [c for c in TAIL_CATS_SCANNET_200 if c in spec.label_to_id],
    }

    pred_files = sorted(os.listdir(opt.pred_path))
    print('reading', len([f for f in pred_files if f.endswith('.txt')]), 'scans...')
    n = 0
    for pred_file in pred_files:
        if os.path.isdir(os.path.join(opt.pred_path, pred_file)):
            continue
        if not pred_file.endswith('.txt'):
            continue
        scene_id = pred_file[:12]
        sys.stdout.write("\rscans read: {}".format(n + 1))
        sys.stdout.flush()
        n += 1

        gt_file = os.path.join(opt.gt_path, pred_file)
        if not os.path.isfile(gt_file):
            raise FileNotFoundError(f"no gt file for {pred_file}: {gt_file}")
        gt_ids = util_3d.load_ids(gt_file)
        evaluator.add_gt(gt_ids, scene_id)

        instances = util_3d.read_instance_prediction_file(
            os.path.join(opt.pred_path, pred_file), opt.pred_path)
        for pred_mask_file in instances:
            pred_mask = util_3d.load_ids(pred_mask_file)
            instances[pred_mask_file]['pred_mask'] = pred_mask

        evaluator.add_prediction(instances, scene_id)

    print('')
    avgs = evaluator.evaluate(groups=groups)
    if opt.output_file:
        write_result_file(evaluator, avgs, opt.output_file)
        print('results ->', opt.output_file)


if __name__ == '__main__':
    main()
