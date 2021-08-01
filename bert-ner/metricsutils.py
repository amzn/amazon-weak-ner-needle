"""Metrics functions for NER."""
import csv
from itertools import zip_longest
from collections import defaultdict

CONLL_TOKEN_COL = 0
CONLL_LABEL_COL = 1
CONLL_COL_SEP = "\t"
OUTSIDE = "O"
BEGIN = "B-"
INSIDE = "I-"
X_LABEL = "X"
TOKEN_PRECISION = "token precision"
TOKEN_RECALL = "token recall"
TOKEN_F1 = "token f1"
SPAN_PRECISION = "span precision"
SPAN_RECALL = "span recall"
SPAN_F1 = "span f1"
LABEL = "label"
LABEL_CONFUSION = "label confusion"
LABEL_CONFUSION_QUERY = "label confusion query"
LABEL_PERCENTAGE = "label percentage"
LABEL_SUPPORT = "label support"
TOKEN_ACCURACY = "token accuracy"
SPAN_ACCURACY = "span accuracy"
MEAN_TOKEN_PRECISION = "mean token precision"
MEAN_TOKEN_RECALL = "mean token recall"
MEAN_SPAN_PRECISION = "mean span precision"
MEAN_SPAN_RECALL = "mean span recall"
TOKEN_COVERAGE = "token coverage"


def compute_accuracy_qformat(gold_file, ref_file, delimiter="\t", header=False,
                             keywords_col=0, gold_label_col=1, ref_label_col=1, bio_format=True,
                             ignore_labels=set([])):
    """
    compute precision, recall and f1 for each label types for conll format file.
    """
    keywords = []
    gold_labels = []
    ref_labels = []
    with open(gold_file, encoding="utf-8") as gold_in, open(ref_file, encoding="utf-8") as ref_in:
        if header:
            _ = gold_in.readline()
            _ = ref_in.readline()
        gold_reader = csv.reader(gold_in, delimiter=delimiter)
        ref_reader = csv.reader(ref_in, delimiter=delimiter)
        for ind, (gold_row, ref_row) in enumerate(zip_longest(gold_reader, ref_reader)):
            if not gold_row or not ref_row:
                raise RuntimeError("gold file or reference file line is null at line %d !" % (ind + 1))
            if gold_row[keywords_col] != ref_row[keywords_col]:
                raise RuntimeError("keywords of gold file and keywords of reference file are different at line %d " % (ind + 1))
            tokens = gold_row[keywords_col].split(" ")
            current_gold_labels = gold_row[gold_label_col].split(" ")
            current_ref_labels = ref_row[ref_label_col].split(" ")
            if len(tokens) != len(current_gold_labels) or len(current_gold_labels) != len(current_ref_labels):
                raise RuntimeError("gold file token and reference file token, label size are different at line %d" % (ind + 1))
            keywords.append(tokens)
            gold_labels.append(current_gold_labels)
            ref_labels.append(current_ref_labels)
    metrics = compute_accuracy_labels(gold_labels, ref_labels, keywords, bio_format, ignore_labels)
    return metrics


def compute_accuracy_conll(gold_file, ref_file, token_col=CONLL_TOKEN_COL,
                           gold_label_col=CONLL_LABEL_COL, ref_label_col=CONLL_LABEL_COL,
                           bio_format=True, ignore_labels=set([])):
    """
    compute precision, recall and f1 for each label types for conll format file.
    """
    keywords = []
    gold_labels = []
    ref_labels = []
    with open(gold_file, encoding="utf-8") as gold_in, open(ref_file, encoding="utf-8") as ref_in:
        current_gold_labels = []
        current_ref_labels = []
        current_keywords = []
        for line_gold, line_ref in zip_longest(gold_in, ref_in):
            if bool(line_gold.strip()) != bool(line_ref.strip()):
                raise RuntimeError("gold file and reference file line mismatch %s, %s" % (line_gold, line_ref))
            if line_gold.strip():
                cols_gold = line_gold.strip().split(CONLL_COL_SEP)
                cols_ref = line_ref.strip().split(CONLL_COL_SEP)
                if cols_gold[token_col] != cols_ref[token_col]:
                    raise RuntimeError("gold file and reference file token mismatch %s, %s " % (line_gold, line_ref))
                current_gold_labels.append(cols_gold[gold_label_col])
                current_ref_labels.append(cols_ref[ref_label_col])
                current_keywords.append(cols_gold[token_col])
            else:
                gold_labels.append(current_gold_labels)
                ref_labels.append(current_ref_labels)
                keywords.append(current_keywords)
                current_gold_labels = []
                current_ref_labels = []
                current_keywords = []
        if len(current_keywords) > 0:
            gold_labels.append(current_gold_labels)
            ref_labels.append(current_ref_labels)
            keywords.append(current_keywords)

    metrics = compute_accuracy_labels(gold_labels, ref_labels, keywords, bio_format, ignore_labels)
    return metrics


def compute_accuracy_labels(gold_labels, ref_labels, keywords=None, bio_format=True, ignore_labels=set([]), full_stats=False):
    """
    compute query level and span level precision, recall and f1 for each label
    types by comparing gold labels and reference labels.
    """
    if len(gold_labels) != len(ref_labels) or (keywords and len(gold_labels) != len(keywords)):
        raise RuntimeError("size of gold labels, ref labels and keywords is not the same!")

    log_error_keywords = True if keywords else False
    total_token_cnt = 0
    total_span_cnt = 0
    correct_token_prediction = 0
    correct_span_prediction = 0
    annotated_tokens = 0
    label_token_cnt = defaultdict(int)
    tp_label_span_cnt = defaultdict(int)
    fp_label_span_cnt = defaultdict(int)
    fn_label_span_cnt = defaultdict(int)
    tp_label_token_cnt = defaultdict(int)
    fp_label_token_cnt = defaultdict(int)
    fn_label_token_cnt = defaultdict(int)

    label_set = set([])
    label_confusion = defaultdict(int)
    label_confusion_query = defaultdict(set)

    for i in range(len(gold_labels)):
        current_gold_labels = gold_labels[i]
        current_ref_labels = ref_labels[i]
        current_keywords = []
        if log_error_keywords:
            current_keywords = keywords[i]
        if len(current_gold_labels) != len(current_ref_labels) or \
           (log_error_keywords and len(current_gold_labels) != len(current_keywords)):
            raise RuntimeError("size of current_gold_label %s, current_ref_labels %s, and current_keywords %s is different! " %
                               (current_gold_labels, current_ref_labels, current_keywords))
        concat_keywords = " ".join(current_keywords)
        for j in range(len(current_gold_labels)):

            current_gold_label_key = current_gold_labels[j]
            current_ref_label_key = current_ref_labels[j]
            if bio_format:
                if current_gold_labels[j] != OUTSIDE and current_gold_labels[j] != X_LABEL:
                    current_gold_label_key = current_gold_labels[j][2:]
                if current_ref_labels[j] != OUTSIDE and current_ref_labels[j] != X_LABEL:
                    current_ref_label_key = current_ref_labels[j][2:]

            token_label_match = int(current_gold_label_key == current_ref_label_key)

            if current_gold_label_key not in ignore_labels:
                label_set.add(current_gold_label_key)
                total_token_cnt += 1
                label_token_cnt[current_gold_label_key] += 1
                correct_token_prediction += token_label_match
                tp_label_token_cnt[current_gold_label_key] += token_label_match
                fn_label_token_cnt[current_gold_label_key] += (1 - token_label_match)

            if current_ref_label_key not in ignore_labels:
                label_set.add(current_ref_label_key)
                fp_label_token_cnt[current_ref_label_key] += (1 - token_label_match)
                if current_ref_label_key != OUTSIDE:
                    annotated_tokens += 1

            if current_gold_label_key != current_ref_label_key:
                label_confusion[(current_gold_label_key, current_ref_label_key)] += 1
                if log_error_keywords:
                    label_confusion_query[(current_gold_label_key, current_ref_label_key)].add((current_keywords[j], concat_keywords))

        gold_spans = find_spans(current_gold_labels, bio_format, ignore_labels)
        ref_spans = find_spans(current_ref_labels, bio_format, ignore_labels)

        gold_span_set = set([])
        ref_span_set = set([])

        for start, end in gold_spans:
            gold_span_set.add((start, end))
        for start, end in ref_spans:
            ref_span_set.add((start, end))

        (matched_gold_spans, non_matched_gold_spans) = count_matched_spans(gold_span_set, ref_span_set, current_gold_labels,
                                                                           current_ref_labels, bio_format, ignore_labels)
        for label in matched_gold_spans:
            if label not in ignore_labels:
                tp_label_span_cnt[label] += matched_gold_spans[label]
                fn_label_span_cnt[label] += non_matched_gold_spans[label]
                correct_span_prediction += matched_gold_spans[label]
                total_span_cnt += matched_gold_spans[label]

        for label in non_matched_gold_spans:
            if label not in ignore_labels:
                total_span_cnt += non_matched_gold_spans[label]

        (matched_ref_spans, non_matched_ref_spans) = count_matched_spans(ref_span_set, gold_span_set, current_ref_labels,
                                                                         current_gold_labels, bio_format, ignore_labels)
        for label in non_matched_ref_spans:
            if label not in ignore_labels:
                fp_label_span_cnt[label] += non_matched_ref_spans[label]

    token_precision = defaultdict(float)
    token_recall = defaultdict(float)
    token_f1 = defaultdict(float)
    span_precision = defaultdict(float)
    span_recall = defaultdict(float)
    span_f1 = defaultdict(float)
    label_percentage = defaultdict(float)
    total_tp_token_prediction = 0
    total_fp_token_prediction = 0
    total_fn_token_prediction = 0
    total_tp_span_prediction = 0
    total_fp_span_prediction = 0
    total_fn_span_prediction = 0

    for label in label_set:
        token_precision[label] = div(tp_label_token_cnt[label], (tp_label_token_cnt[label] + fp_label_token_cnt[label]))
        token_recall[label] = div(tp_label_token_cnt[label], (tp_label_token_cnt[label] + fn_label_token_cnt[label]))
        token_f1[label] = (token_precision[label] + token_recall[label]) / 2
        span_precision[label] = div(tp_label_span_cnt[label], (tp_label_span_cnt[label] + fp_label_span_cnt[label]))
        span_recall[label] = div(tp_label_span_cnt[label], (tp_label_span_cnt[label] + fn_label_span_cnt[label]))
        span_f1[label] = (span_precision[label] + span_recall[label]) / 2
        label_percentage[label] = div(label_token_cnt[label], total_token_cnt)
        if label != "O":
            total_tp_token_prediction += tp_label_token_cnt[label]
            total_fp_token_prediction += fp_label_token_cnt[label]
            total_fn_token_prediction += fn_label_token_cnt[label]
            total_tp_span_prediction += tp_label_span_cnt[label]
            total_fp_span_prediction += fp_label_span_cnt[label]
            total_fn_span_prediction += fn_label_span_cnt[label]

    token_accuracy = div(correct_token_prediction, total_token_cnt)
    span_accuracy = div(correct_span_prediction, total_span_cnt)
    mean_token_precision = div(total_tp_token_prediction, total_tp_token_prediction + total_fp_token_prediction)
    mean_token_recall = div(total_tp_token_prediction, total_tp_token_prediction + total_fn_token_prediction)
    mean_span_precision = div(total_tp_span_prediction, total_tp_span_prediction + total_fp_span_prediction)
    mean_span_recall = div(total_tp_span_prediction, total_tp_span_prediction + total_fn_span_prediction)
    token_coverage = div(annotated_tokens, total_token_cnt)
    metrics = {LABEL : label_set,
               LABEL_PERCENTAGE : label_percentage,
               LABEL_SUPPORT : label_token_cnt,
               TOKEN_ACCURACY : token_accuracy,
               SPAN_ACCURACY : span_accuracy,
               MEAN_TOKEN_PRECISION : mean_token_precision,
               MEAN_TOKEN_RECALL : mean_token_recall,
               MEAN_SPAN_PRECISION : mean_span_precision,
               MEAN_SPAN_RECALL : mean_span_recall,
               TOKEN_COVERAGE : token_coverage,
               TOKEN_PRECISION : token_precision,
               TOKEN_RECALL : token_recall,
               TOKEN_F1 : token_f1,
               SPAN_PRECISION : span_precision,
               SPAN_RECALL : span_recall,
               SPAN_F1 : span_f1,
               LABEL_CONFUSION : label_confusion,
               LABEL_CONFUSION_QUERY : label_confusion_query}
    if full_stats:
        metrics.update({
            "total_span_cnt": total_span_cnt,
            "total_token_cnt": total_token_cnt,
            "correct_token_prediction": correct_token_prediction,
            "correct_span_prediction": correct_span_prediction,
            "annotated_tokens": annotated_tokens,
            "tp_label_span_cnt"  : tp_label_span_cnt,
            "fp_label_span_cnt"  : fp_label_span_cnt,
            "fn_label_span_cnt"  : fn_label_span_cnt,
            "tp_label_token_cnt" : tp_label_token_cnt,
            "fp_label_token_cnt" : fp_label_token_cnt,
            "fn_label_token_cnt" : fn_label_token_cnt,
        })

    return metrics


def find_spans(labels, bio_format, ignore_labels):
    """
    find spans corresponding to label list.
    """
    spans = []
    for j in range(len(labels)):
        if len(spans) == 0:
            spans.append([j, j + 1])
        elif len(spans) > 0:
            if not bio_format:
                if labels[j] == labels[j - 1]:
                    spans[-1][1] = j + 1
                else:
                    spans.append([j, j + 1])
            else:
                if labels[j].startswith(BEGIN):
                    spans.append([j, j + 1])
                elif labels[j].startswith(INSIDE):
                    if len(labels[j - 1]) >= 2 and labels[j][2:] == labels[j - 1][2:]:
                        spans[-1][1] = j + 1
                    else:
                        # this means I- label is different from the
                        # previous B- label, which usually means bad prediction
                        # or malformated gold sets.
                        spans.append([j, j + 1])
                elif labels[j] == OUTSIDE or labels[j] == X_LABEL:
                    if labels[j] == labels[j - 1]:
                        spans[-1][1] = j + 1
                    else:
                        spans.append([j, j + 1])
                else:
                    raise RuntimeError("wrong bioformat label %s !" % (labels[j]))

    result_spans = []
    for start, end in spans:
        current_label_key = labels[start]
        if bio_format:
            if current_label_key != OUTSIDE and current_label_key != X_LABEL:
                current_label_key = current_label_key[2:]

        if current_label_key not in ignore_labels:
            result_spans.append((start, end))
    return result_spans


def count_matched_spans(gold_span_set, ref_span_set, current_gold_labels,
                        current_ref_labels, bio_format, ignore_labels):
    """
    count number of matched spans in gold_span_set
    """
    matched_spans = defaultdict(int)
    non_matched_spans = defaultdict(int)
    for start, end in gold_span_set:
        current_gold_label_key = current_gold_labels[start]
        if bio_format:
            if current_gold_labels[start] != OUTSIDE and current_gold_labels[start] != X_LABEL:
                current_gold_label_key = current_gold_labels[start][2:]

        matched = ((start, end) in ref_span_set)
        if (start, end) in ref_span_set:
            while start < end:
                if current_gold_labels[start] != current_ref_labels[start]:
                    matched = False
                    break
                start += 1
        matched_spans[current_gold_label_key] += int(matched)
        non_matched_spans[current_gold_label_key] += (1 - int(matched))
    return (matched_spans, non_matched_spans)


def div(numerator, denominator):
    """
    if denominator is zero, return zero, else return normal division results
    """
    if denominator == 0:
        return 0
    else:
        return float(numerator) / denominator


def write_label_confusion_metrics(fout, metrics, error_example=5):
    """
    write label confusion metrics to file like object fout.
    """
    label_confusion = metrics[LABEL_CONFUSION]
    label_confusion_query = metrics[LABEL_CONFUSION_QUERY]

    fout.write("gold label\tpredicted label\tcount\texample error\n")
    for k, v in sorted(label_confusion.items(), key=lambda item: item[1], reverse=True):
        fout.write("%s\t%s\t%s\t" % (k[0], k[1], v))
        cnt = 0
        for token, keywords in label_confusion_query[k]:
            if cnt > error_example:
                break
            fout.write("(%s:%s), " % (token, keywords))
            cnt += 1
        fout.write("\n")


def write_metrics(fout, metrics):
    """
    write per label metrics to output file. fout is file like object that are in write
    mode.
    """
    fout.write("token-accuracy\t%.4f\tspan-accuracy\t%.4f\n" % (metrics[TOKEN_ACCURACY], metrics[SPAN_ACCURACY]))
    fout.write("mean token precision\t%.4f\tmean token recall\t%.4f\n" % (metrics[MEAN_TOKEN_PRECISION], metrics[MEAN_TOKEN_RECALL]))
    fout.write("mean span precision\t%.4f\tmean span recall\t%.4f\n" % (metrics[MEAN_SPAN_PRECISION], metrics[MEAN_SPAN_RECALL]))
    fout.write("token coverage\t%.4f\n" % (metrics[TOKEN_COVERAGE]))
    fout.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (LABEL, TOKEN_PRECISION,
                                                         TOKEN_RECALL, TOKEN_F1,
                                                         SPAN_PRECISION, SPAN_RECALL, SPAN_F1,
                                                         LABEL_PERCENTAGE, LABEL_SUPPORT))
    for label, support in sorted(metrics[LABEL_SUPPORT].items(), key=lambda item: item[1], reverse=True):
        fout.write("%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n" %
                   (label, metrics[TOKEN_PRECISION][label], metrics[TOKEN_RECALL][label], metrics[TOKEN_F1][label],
                    metrics[SPAN_PRECISION][label], metrics[SPAN_RECALL][label], metrics[SPAN_F1][label],
                    metrics[LABEL_PERCENTAGE][label], support))
