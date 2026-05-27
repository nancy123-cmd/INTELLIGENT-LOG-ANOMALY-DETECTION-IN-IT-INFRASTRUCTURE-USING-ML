# -*- coding: utf-8 -*-
"""
generate_dataset.py
───────────────────
Generates a synthetic IT Infrastructure log dataset and saves it as:
  - logs_dataset.csv       (raw logs with labels)
  - anomaly_label.csv      (BlockId + Label — compatible with HDFS format)

Run:
    python generate_dataset.py
"""

import pandas as pd
import numpy as np
import random
import datetime
import os

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
N_LOGS     = 20000   # total log lines to generate
ANOMALY_RATE = 0.17  # ~17% anomalies

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
ROOT_CAUSES = [
    "Disk Failure", "Memory Overflow", "Network Issue",
    "Security Attack", "Application Crash", "Normal"
]

WEIGHTS = [3, 4, 4, 3, 3, 83]   # Normal ~83%, anomalies ~17%

LOG_TEMPLATES = {
    "Disk Failure": [
        "Disk I/O error on /dev/sda1",
        "SMART error disk health critical",
        "Write failed no space left on device",
        "Disk timeout after 30 seconds",
        "Bad sector detected at block 4096",
        "Filesystem corruption detected on /data",
        "Read error on sector 20480",
        "Disk controller failure reported",
        "HDD temperature exceeded threshold 65C",
        "RAID array degraded missing disk",
    ],
    "Memory Overflow": [
        "OutOfMemoryError Java heap space",
        "Memory usage exceeded 95 percent",
        "Kernel OOM killer activated pid terminated",
        "Swap usage at 100 percent system thrashing",
        "Process killed memory limit reached",
        "GC overhead limit exceeded",
        "Virtual memory exhausted cannot allocate",
        "Memory leak detected in application thread",
        "Buffer overflow detected in stack",
        "RSS memory usage 98 percent critical",
    ],
    "Network Issue": [
        "Connection timeout to 192.168.1.1 after 30s",
        "Packet loss 40 percent on eth0 interface",
        "DNS resolution failed for db.internal host",
        "TCP retransmit storm detected on port 443",
        "Network interface eth1 link down",
        "High network latency 2500ms detected",
        "ARP table overflow network flooding",
        "BGP session dropped unexpected reset",
        "Firewall dropping packets from subnet",
        "SSL handshake timeout connection reset",
    ],
    "Security Attack": [
        "Failed login attempt 50 tries in 10 seconds",
        "Unauthorized access detected from IP address",
        "SQL injection pattern detected in request",
        "Port scan detected from remote host",
        "Brute force attack on SSH port 22 blocked",
        "Suspicious HTTP payload detected in request",
        "XSS attack vector found in form submission",
        "Privilege escalation attempt detected",
        "Malware signature matched in process",
        "DDoS attack detected high traffic volume",
    ],
    "Application Crash": [
        "NullPointerException in UserService class",
        "Segmentation fault core dumped",
        "Unhandled exception IndexOutOfBoundsException",
        "Service health check failed 3 consecutive times",
        "Critical thread pool exhausted all workers busy",
        "Database connection pool exhausted timeout",
        "Application deadlock detected waiting threads",
        "Stack overflow error in recursive function",
        "Config file parse error service not starting",
        "Dependency service unavailable circuit open",
    ],
    "Normal": [
        "User login successful session created",
        "Backup job completed successfully",
        "Health check OK all services running",
        "Database query executed in 12ms rows returned",
        "Cache refresh completed 1024 keys updated",
        "Scheduled cron job completed successfully",
        "API request processed 200 OK response sent",
        "SSL certificate valid expires in 90 days",
        "Config reload successful no changes detected",
        "Memory usage normal 45 percent available",
        "Disk usage 60 percent within normal range",
        "Network throughput normal 100Mbps",
        "Log rotation completed old files archived",
        "Service started successfully listening on port",
        "User logout session terminated cleanly",
        "Database connection pool healthy 20 active",
        "Metrics collected and sent to monitoring server",
        "File upload completed checksum verified",
        "Email notification sent successfully",
        "Replication lag 0ms databases in sync",
    ],
}

SERVERS    = ["server-01", "server-02", "db-primary", "db-replica",
              "web-01", "web-02", "app-server", "cache-01",
              "proxy-01", "monitor-01"]
COMPONENTS = ["kernel", "sshd", "mysqld", "nginx", "java-app",
              "cron", "systemd", "network-manager", "postgres", "redis"]
USERS      = [f"user_{i:04d}" for i in range(1, 201)]

# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_log_line(idx, timestamp, cause=None):
    if cause is None:
        cause = random.choices(ROOT_CAUSES, weights=WEIGHTS)[0]

    msg    = random.choice(LOG_TEMPLATES[cause])
    level  = "ERROR" if cause != "Normal" else random.choices(
                ["INFO","INFO","INFO","WARN","DEBUG"], weights=[60,0,0,20,20])[0]
    server = random.choice(SERVERS)
    comp   = random.choice(COMPONENTS)
    pid    = random.randint(1000, 65535)
    user   = random.choice(USERS)
    block  = f"blk_{random.randint(100000000, 999999999)}"
    label  = 0 if cause == "Normal" else 1

    full_log = (f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                f"{level} [{server}] {comp}[{pid}]: {msg}")

    return {
        "LogId":      f"LOG{idx:07d}",
        "BlockId":    block,
        "Timestamp":  timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "Level":      level,
        "Server":     server,
        "Component":  comp,
        "PID":        pid,
        "User":       user,
        "Message":    msg,
        "FullLog":    full_log,
        "RootCause":  cause,
        "Label":      label,
    }

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generating {N_LOGS:,} log lines...")

rows      = []
base_time = datetime.datetime(2024, 1, 1, 0, 0, 0)

for i in range(N_LOGS):
    ts = base_time + datetime.timedelta(seconds=i * 5 + random.randint(0, 4))
    rows.append(generate_log_line(i + 1, ts))

    if (i + 1) % 5000 == 0:
        print(f"  Generated {i+1:,} lines...")

df = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Dataset Summary ─────────────────────────────")
print(f"Total rows    : {len(df):,}")
print(f"Normal        : {(df['Label']==0).sum():,} ({(df['Label']==0).mean()*100:.1f}%)")
print(f"Anomaly       : {(df['Label']==1).sum():,} ({(df['Label']==1).mean()*100:.1f}%)")
print(f"\nRoot Cause Distribution:")
print(df["RootCause"].value_counts().to_string())
print(f"\nLog Level Distribution:")
print(df["Level"].value_counts().to_string())
print(f"\nServer Distribution:")
print(df["Server"].value_counts().to_string())
print("────────────────────────────────────────────────")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE FILES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Full dataset
df.to_csv("logs_dataset.csv", index=False)
print(f"\n✅ Saved: logs_dataset.csv ({len(df):,} rows)")

# 2. Labels only (HDFS-compatible format)
labels_df = df[["BlockId", "RootCause", "Label"]].copy()
labels_df.columns = ["BlockId", "RootCause", "Label"]
labels_df["Label"] = labels_df["Label"].map({0: "Normal", 1: "Anomaly"})
labels_df.to_csv("anomaly_label.csv", index=False)
print(f"✅ Saved: anomaly_label.csv ({len(labels_df):,} rows)")

# 3. Raw log file (plain text)
with open("HDFS_synthetic.log", "w") as f:
    for line in df["FullLog"]:
        f.write(line + "\n")
print(f"✅ Saved: HDFS_synthetic.log ({len(df):,} lines)")

# 4. Anomaly-only subset
anomalies = df[df["Label"] == 1]
anomalies.to_csv("anomalies_only.csv", index=False)
print(f"✅ Saved: anomalies_only.csv ({len(anomalies):,} rows)")

print(f"\n── Files Generated ──────────────────────────────")
for fname in ["logs_dataset.csv", "anomaly_label.csv",
              "HDFS_synthetic.log", "anomalies_only.csv"]:
    size = os.path.getsize(fname) / 1024
    print(f"  {fname:<30} {size:>8.1f} KB")
print("────────────────────────────────────────────────")
print("\nDone! Now run:  streamlit run it_log_monitor.py")