"""This file turns pofile into refined weakly labeled data."""
import numpy as np
import pickle
import os
import argparse
from scipy.stats import binned_statistic
from tqdm import tqdm
import re

parser = argparse.ArgumentParser(description='Profile to SelfTraining Data')
parser.add_argument('--bertckp', type=str, required=True, help='path to bert checkpoint')
parser.add_argument('--weak_file', type=str, default="weak", help='weak file name')
parser.add_argument('--wei_rule', type=str, default="avgaccu", help='weighting rule')
parser.add_argument('--pred_rule', type=str, default="non_O_overwrite", help='prediction rule')
args = parser.parse_args()

BERTCKP = args.bertckp

DEVPATH = os.path.join(BERTCKP, 'predict/profile/devprofile_data.pickle')
WEAKPATH = os.path.join(BERTCKP, f'predict/profile/{args.weak_file}profile_data.pickle')
TXTSAVEPATH = os.path.join(BERTCKP, f'predict/weak_{args.pred_rule}-WEI_{args.wei_rule}/{args.weak_file}.txt')
WEISAVEPATH = os.path.join(BERTCKP, f'predict/weak_{args.pred_rule}-WEI_{args.wei_rule}/{args.weak_file}_wei.npy')

os.makedirs(os.path.join(BERTCKP, f'predict/weak_{args.pred_rule}-WEI_{args.wei_rule}'), exist_ok=True)

dev_profile_data = pickle.load(open(DEVPATH, 'rb'))

print(len(dev_profile_data))

# Averaged Score

nbins = 50
dev_profile_data.sort(key=lambda x: x[1])
scores = [x[1] for x in dev_profile_data]
acu = [1 if x[3] else 0 for x in dev_profile_data]
print('averge query level accu: ', sum(acu) / len(acu))

bins = scores[::len(scores) // nbins]
bins[0] = -10000
bins[-1] = 10000

bin_means, bin_edges, binnumber = binned_statistic(scores, acu, statistic='mean', bins=bins)

cum_accu_lower = [sum(bin_means[:i + 1]) / (i + 1) for i in range(len(bin_means))]
cum_accu_higher = [sum(bin_means[-i - 1:]) / (i + 1) for i in range(len(bin_means) - 1, -1, -1)]
assert len(cum_accu_higher) == len(cum_accu_lower) == len(bin_means)

weak_profile_data = pickle.load(open(WEAKPATH, 'rb'))
print(len(weak_profile_data))

# Define mapping from exampel to weight base on wei_rule
if args.wei_rule == "avgaccu":
    def mapex2wei(ex):
        for edge, wei in zip(bin_edges[-2::-1], bin_means[::-1]):
            if ex[1] >= edge:
                return wei
        raise RuntimeError("Not catched by rule")
elif args.wei_rule == "avgaccu_weak_non_O_promote":
    def mapex2wei(ex):
        ps = ex[-3]
        ls = ex[-2]
        prop = 0.0
        for p, l in zip(ps, ls):
            if l != 'O':
                prop += 1
        prop /= len(ps)
        for edge, wei in zip(bin_edges[-2::-1], bin_means[::-1]):
            if ex[1] >= edge:
                return wei * (1 - prop) + prop
        raise RuntimeError("Not catched by rule")
elif args.wei_rule == "corrected":
    def mapex2wei(ex):
        for edge, wei in zip(bin_edges[-2::-1], bin_means[::-1]):
            if ex[1] >= edge:
                return 2 * wei - 1
        raise RuntimeError("Not catched by rule")
elif args.wei_rule == "corrected_weak_non_O_promote":
    def mapex2wei(ex):
        ps = ex[-3]
        ls = ex[-2]
        prop = 0.0
        for p, l in zip(ps, ls):
            if l != 'O':
                prop += 1
        prop /= len(ps)
        for edge, wei in zip(bin_edges[-2::-1], bin_means[::-1]):
            if ex[1] >= edge:
                return (2 * wei - 1) * (1 - prop) + prop
        raise RuntimeError("Not catched by rule")
elif args.wei_rule == 'uni':
    def mapex2wei(ex):
        return 1
elif re.match(r'wei_accu_pairs(-\d.\d_\d\d)*-\d.\d', args.wei_rule):
    wei_accu = args.wei_rule.split('-')[1:]
    wei_accu[-1] += '_100'
    wei_accu = [x.split('_') for x in wei_accu]
    wei_accu = [(float(w), float(a)) for w, a in wei_accu]

    def mapex2wei(ex):
        for wei, edge in wei_accu:
            if ex[1] <= edge:
                return wei
        raise RuntimeError("Not catched by rule")
else:
    raise NotImplementedError(f"{args.wei_rule} not implemented")


print("==Generating Weights==")
weights = [mapex2wei(ex) for ex in weak_profile_data]
weights = np.array(weights)
np.save(WEISAVEPATH, weights)


# Rule for generating refined labels.
def save_rule(rule, pred, label, score):
    if "-" in rule:
        rule = rule.split('-')[1]
    if rule is None or rule == 'no':
        return label
    elif rule == 'non_O_overwrite':
        if label != 'O':
            return label
        else:
            return pred
    elif re.match(r'non_O_overwrite_over_accu_\d\d', rule):
        if label != 'O':
            return label
        thre = int(rule[-2:]) / 100.
        for edge, accu in zip(bin_edges[-2::-1], cum_accu_lower[::-1]):
            if ex[1] >= edge:
                if accu < thre:
                    return 'X'
                else:
                    return pred
        assert False, "accu must be found"
    elif re.match(r'non_O_overwrite_all_overwrite_over_accu_\d\d', rule):
        if label == 'O':
            return pred
        thre = int(rule[-2:]) / 100.
        for edge, accu in zip(bin_edges[-2::-1], cum_accu_lower[::-1]):
            if ex[1] >= edge:
                if accu < thre:
                    return label
                else:
                    return pred
        assert False, "accu must be found"
    elif rule == 'all_overwrite':
        return pred
    else:
        raise NotImplementedError(rule + ' not implemented')


def screen_rule(rule, ps, ls, score):
    """Select sample or not."""
    ori_rule = rule
    if rule is None or rule == 'no' or '-' not in rule:
        return True
    else:
        rule = rule.split("-")[0]
        if rule == 'drop_allmatch':
            for p, l in zip(ps, ls):
                if l != 'O' and p != l:
                    return True
            return False
        if rule == 'drop_allmatch_error':
            prevp = None
            for p, l in zip(ps, ls):
                p = save_rule(ori_rule, p, l)
                if p.startswith('I-'):
                    if prevp != p and prevp != p.replace('I-', 'B-'):
                        return False
                prevp = p
            for p, l in zip(ps, ls):
                if l != 'O' and p != l:
                    return True
            return False
        else:
            raise NotImplementedError(rule + " not implemented")


total_error_nums = 0
total_nomatch_nums = 0
total_save_nums = 0
with open(TXTSAVEPATH, 'w') as fout:
    print("==Generating Labels==")
    for ex in tqdm(weak_profile_data):
        score = ex[1]
        ps = ex[-3]
        ls = ex[-2]
        es = ex[-1]
        if not screen_rule(args.pred_rule, ps, ls, score):
            continue
        total_save_nums += 1
        prevp = 'O'
        preve = None
        for p, l, e in zip(ps, ls, es):
            if p != l and l != 'O':
                total_nomatch_nums += 1
            p = save_rule(args.pred_rule, p, l, score)
            if p.startswith('I-'):
                if prevp != p and prevp != p.replace('I-', 'B-'):
                    total_error_nums += 1
            prevp = p
            preve = e
            fout.write("{}\t{}\n".format(e, p))
        fout.write("\n")
print("Total # of Saves ", total_save_nums, " / ", len(weak_profile_data))
print("Total # of Errors ", total_error_nums)
print("Total # of Not Match Weak ", total_nomatch_nums)
