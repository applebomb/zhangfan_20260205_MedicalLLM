"""
 @Author : zhangfan
 @email  : 61316173 @qq.com @Date   : 2026-02-04
Copyright (c) 2026 61316173 @qq.com. All Rights Reserved.

NOTICE:  All information contained herein is, and remains
the property of the author. The intellectual and technical concepts
contained herein are proprietary to the author and are protected
by trade secret or copyright law. Dissemination of this information
or reproduction of this material is strictly forbidden unless prior
written permission is obtained from the author.
"""

import torch

def analyze():
    data = torch.load('data/v1/processed/train.pt', weights_only=False)
    all_gaps = []
    for item in data:
        ages = torch.tensor(item['ages'])
        if len(ages) > 1:
            gaps = ages[1:] - ages[:-1]
            all_gaps.extend(gaps.tolist())
    
    all_gaps = torch.tensor(all_gaps)
    print(f"Total gaps: {len(all_gaps)}")
    print(f"Mean: {all_gaps.mean().item():.2f}")
    print(f"Median: {all_gaps.median().item():.2f}")
    print(f"Zero: {(all_gaps == 0).sum().item()} ({(all_gaps == 0).sum().item()/len(all_gaps)*100:.1f}%)")
    
    non_zero = all_gaps[all_gaps > 0]
    if len(non_zero) > 0:
        print(f"Gaps > 0: {len(non_zero)}")
        print(f"Mean of non-zero: {non_zero.mean().item():.2f}")
        print(f"Median of non-zero: {non_zero.median().item():.2f}")
        print(f"Min of non-zero: {non_zero.min().item():.2f}")
        print(f"Max of non-zero: {non_zero.max().item():.2f}")

if __name__ == "__main__":
    analyze()
