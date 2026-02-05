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

import psutil
import time
from datetime import datetime
import os

def get_process_cpu_stats():
    """
    获取快照：{pid: {'name': name, 'cpu_time': total_seconds}}
    """
    procs = {}
    for p in psutil.process_iter(['pid', 'name', 'cpu_times']):
        try:
            if p.info['cpu_times']:
                # user + system 时间总和
                total_time = p.info['cpu_times'].user + p.info['cpu_times'].system
                procs[p.info['pid']] = {
                    'name': p.info['name'],
                    'cpu_time': total_time
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return procs

def draw_bar(percent, length=10):
    """
    绘制简单的 ASCII 进度条
    """
    # 限制最大显示长度，防止多核爆表导致换行
    display_percent = min(percent, 1000) 
    filled_len = int(display_percent / 100 * length)
    # 如果超过100%，比如200%，我们可以用不同的符号或者颜色，这里简单处理：
    # 如果超过100%（即多核），让它显示得更长一点，但这里为了UI整齐，我们以单核100%为基准
    # 更好的方式是：100% = ##### (5个格)
    
    # 修改逻辑：每 10% 一个格
    num_chars = int(percent / 10)
    return "|" * num_chars

def monitor_cpu_usage(duration=10, top_n=10):
    # 获取系统核心数
    logic_cores = psutil.cpu_count(logical=True)
    
    print(f"--- 系统核心数: {logic_cores} (逻辑核心) ---")
    print(f"--- 开始监控，持续 {duration} 秒... ---")
    
    t1 = time.time()
    start_snapshot = get_process_cpu_stats()
    
    # 倒计时
    try:
        for i in range(duration):
            time.sleep(1)
            print(f"\r正在收集数据: {duration - i - 1}s ", end="")
    except KeyboardInterrupt:
        print("\n监控中断，计算当前数据...")
    
    t2 = time.time()
    end_snapshot = get_process_cpu_stats()
    
    # 计算实际经过的物理时间（比 sleep 更精确）
    actual_duration = t2 - t1
    
    print(f"\n\n{'PID':<8} {'进程名称':<25} {'CPU时间(s)':<12} {'核心压力(%)':<12} {'压力图示'}")
    print("-" * 85)
    
    results = []
    
    for pid, end_info in end_snapshot.items():
        if pid in start_snapshot:
            start_info = start_snapshot[pid]
            delta_cpu = end_info['cpu_time'] - start_info['cpu_time']
            
            if delta_cpu > 0.001: # 过滤掉极小的数值
                # 计算百分比压力： (CPU消耗时间 / 实际物理时间) * 100
                avg_load_percent = (delta_cpu / actual_duration) * 100
                
                results.append({
                    'pid': pid,
                    'name': end_info['name'],
                    'delta': delta_cpu,
                    'percent': avg_load_percent
                })
    
    # 按消耗量排序
    results.sort(key=lambda x: x['delta'], reverse=True)
    
    for item in results[:top_n]:
        # 压力图示：每 10% 显示一个竖线 |
        bar = draw_bar(item['percent'])
        print(f"{item['pid']:<8} {item['name']:<25} {item['delta']:<12.4f} {item['percent']:<12.2f} {bar}")

    print("-" * 85)
    print(f"* 说明: '核心压力' 100% 表示占满 1 个核心。如果 > 100% 说明使用了多线程。")

if __name__ == "__main__":
    # 你可以在这里修改监控时长，例如 60 秒
    CONFIG_DURATION = 60 
    CONFIG_TOPN = 20

    monitor_cpu_usage(duration=CONFIG_DURATION, top_n=CONFIG_TOPN)

