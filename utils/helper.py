'''
Created on Aug 19, 2016
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
__author__ = "xiangwang"
import os
import re
import random

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def user_pos_neg(interactions,n_items):
    users = []
    pos_items = []
    neg_items = []
    for key,items in interactions.items():
        users.append(int(key))
        item = random.choice(items)
        pos_items.append(int(item))
        while 1:
            neg_item = random.randint(0, n_items-1)
            if str(neg_item) not in items and neg_item not in neg_items:
                neg_items.append(neg_item)
                break
            else:
                continue
    return users,pos_items,neg_items