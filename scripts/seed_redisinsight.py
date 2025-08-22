#!/usr/bin/env python3
import json, os, time, socket

CONFIG_PATH = '/config/connections.json'
SERVICE_HOST = os.environ.get('RI_SEED_REDIS_HOST','redis')
SERVICE_PORT = int(os.environ.get('RI_SEED_REDIS_PORT','6379'))
NAME = os.environ.get('RI_SEED_NAME','Seeded Redis Stack')

# Wait for redis host to resolve and port open
for attempt in range(60):
    try:
        with socket.create_connection((SERVICE_HOST, SERVICE_PORT), timeout=1.5):
            break
    except Exception:
        time.sleep(1)
else:
    print('Could not reach redis to seed; continuing anyway')

try:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH,'r',encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    # If already present skip
    if not any(c.get('host')==SERVICE_HOST and c.get('port')==SERVICE_PORT for c in data):
        data.append({
            'name': NAME,
            'host': SERVICE_HOST,
            'port': SERVICE_PORT,
            'username': '',
            'password': '',
            'tls': False
        })
        with open(CONFIG_PATH,'w',encoding='utf-8') as f:
            json.dump(data,f,indent=2)
        print('Seeded RedisInsight connections.json with redis host')
    else:
        print('RedisInsight connection already present; no change')
except Exception as e:
    print('Error seeding RedisInsight config:', e)
