# 엘리스 클라우드 VSCode에서 Offloading 사용 실행 명령어

## 전제 조건

1. 각 인스턴스에서 초기 설정 완료:
   ```bash
   git clone https://github.com/hjkim24/my-petals.git
   cd my-petals
   git checkout Jaewon
   ./scripts/initial_install.sh
   source venv/bin/activate
   hf auth login
   ```

2. 공인 IP 확인:
   ```bash
   dig +short central-02.tcp.tunnel.elice.io
   # 출력: 119.59.0.14
   ```

---

## 실행 순서

**중요**: Stage1을 먼저 실행하고, DHT multiaddr을 확인한 후 다른 stage를 실행하세요.

---

## Stage1 실행 (DHT 부트스트랩)

**인스턴스**: `stage-1`  
**VSCode 터미널에서 실행**:

```bash
cd my-petals
source venv/bin/activate

# Stage1 실행 (백그라운드, Offloading 사용)
nohup python -m src.main \
  --model meta-llama/Llama-3.1-8B \
  --splits "8,16,24" \
  --dht_port 8002 \
  --rpc_port 8003 \
  --public_ip 119.59.0.14 \
  --public_dht_port 22452 \
  --public_rpc_port 50192 \
  --stage 1 \
  --enable_offload \
  > stage1.log 2>&1 &

# 프로세스 ID 확인
echo $!

# 로그 확인
tail -f stage1.log
```

**DHT multiaddr 확인** (약 10-30초 대기 후):

```bash
# 로그에서 DHT multiaddr 찾기
grep "DHT visible multiaddrs" stage1.log

# 또는 실시간 확인
tail -f stage1.log | grep "DHT visible multiaddrs"
```

출력 예시:
```
INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): [<Multiaddr /ip4/119.59.0.14/tcp/22452/p2p/12D3KooW...>]
```

**중요**: 이 multiaddr을 복사하세요: `/ip4/119.59.0.14/tcp/22452/p2p/<PEER_ID>`

---

## Stage2 실행

**인스턴스**: `stage-2`  
**VSCode 터미널에서 실행**:

```bash
cd my-petals
source venv/bin/activate

# Stage1의 DHT multiaddr을 --dht_initial_peers에 사용
# <PEER_ID>는 Stage1 로그에서 확인한 실제 Peer ID로 교체
nohup python -m src.main \
  --model meta-llama/Llama-3.1-8B \
  --splits "8,16,24" \
  --dht_port 8004 \
  --rpc_port 8005 \
  --public_ip 119.59.0.14 \
  --public_dht_port 29354 \
  --public_rpc_port 15930 \
  --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/<PEER_ID>" \
  --stage 2 \
  --enable_offload \
  > stage2.log 2>&1 &

# 로그 확인
tail -f stage2.log
```

---

## Stage2-replacement 실행 (선택사항)

**인스턴스**: `stage-2-replacement`  
**다른 public_ip 사용** (119.59.0.13):

```bash
cd my-petals
source venv/bin/activate

nohup python -m src.main \
  --model meta-llama/Llama-3.1-8B \
  --splits "8,16,24" \
  --dht_port 8004 \
  --rpc_port 8005 \
  --public_ip 119.59.0.13 \
  --public_dht_port 31172 \
  --public_rpc_port 27714 \
  --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/<PEER_ID>" \
  --stage 2 \
  --enable_offload \
  > stage2-replacement.log 2>&1 &

# 로그 확인
tail -f stage2-replacement.log
```

---

## Stage3 실행

**인스턴스**: `stage-3`  
**VSCode 터미널에서 실행**:

```bash
cd my-petals
source venv/bin/activate

nohup python -m src.main \
  --model meta-llama/Llama-3.1-8B \
  --splits "8,16,24" \
  --dht_port 8006 \
  --rpc_port 8007 \
  --public_ip 119.59.0.14 \
  --public_dht_port 59491 \
  --public_rpc_port 38548 \
  --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/<PEER_ID>" \
  --stage 3 \
  --enable_offload \
  > stage3.log 2>&1 &

# 로그 확인
tail -f stage3.log
```

---

## Stage0 실행 (클라이언트)

**인스턴스**: `stage-0`  
**VSCode 터미널에서 실행**:

```bash
cd my-petals
source venv/bin/activate

# Stage0 실행 (프롬프트와 최대 토큰 수 지정)
python -m src.main \
  --model meta-llama/Llama-3.1-8B \
  --splits "8,16,24" \
  --dht_port 8008 \
  --rpc_port 8009 \
  --public_ip 119.59.0.14 \
  --public_dht_port 41826 \
  --public_rpc_port 23619 \
  --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/<PEER_ID>" \
  --stage 0 \
  --prompt "Hello, how are you?" \
  --max_new_tokens 64 \
  --enable_offload
```

또는 백그라운드 실행:

```bash
nohup python -m src.main \
  --model meta-llama/Llama-3.1-8B \
  --splits "8,16,24" \
  --dht_port 8008 \
  --rpc_port 8009 \
  --public_ip 119.59.0.14 \
  --public_dht_port 41826 \
  --public_rpc_port 23619 \
  --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/<PEER_ID>" \
  --stage 0 \
  --prompt "Hello, how are you?" \
  --max_new_tokens 64 \
  --enable_offload \
  > stage0.log 2>&1 &

# 로그 확인
tail -f stage0.log
```

---

## 로그 확인 및 모니터링

각 인스턴스에서:

```bash
# 실시간 로그 확인
tail -f stage<N>.log

# 최근 100줄 확인
tail -n 100 stage<N>.log

# 에러 확인
grep -i error stage<N>.log

# 준비 완료 확인 (Stage1, 2, 3)
grep "handlers registered" stage<N>.log
```

---

## 프로세스 관리

```bash
# 실행 중인 프로세스 확인
ps aux | grep "src.main"

# 특정 stage 프로세스 확인
ps aux | grep "src.main --stage <N>"

# 프로세스 종료
pkill -f "src.main --stage <N>"

# 또는 PID로 종료
kill <PID>
```

---

## 빠른 참조: 모든 명령어 한눈에 보기

### Stage1 (먼저 실행)
```bash
cd my-petals && source venv/bin/activate && nohup python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --dht_port 8002 --rpc_port 8003 --public_ip 119.59.0.14 --public_dht_port 22452 --public_rpc_port 50192 --stage 1 --enable_offload > stage1.log 2>&1 &
```

### Stage2
```bash
cd my-petals && source venv/bin/activate && nohup python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --dht_port 8004 --rpc_port 8005 --public_ip 119.59.0.14 --public_dht_port 29354 --public_rpc_port 15930 --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/<PEER_ID>" --stage 2 --enable_offload > stage2.log 2>&1 &
```

### Stage3
```bash
cd my-petals && source venv/bin/activate && nohup python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --dht_port 8006 --rpc_port 8007 --public_ip 119.59.0.14 --public_dht_port 59491 --public_rpc_port 38548 --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/<PEER_ID>" --stage 3 --enable_offload > stage3.log 2>&1 &
```

### Stage0
```bash
cd my-petals && source venv/bin/activate && python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --dht_port 8008 --rpc_port 8009 --public_ip 119.59.0.14 --public_dht_port 41826 --public_rpc_port 23619 --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/<PEER_ID>" --stage 0 --prompt "Hello, how are you?" --max_new_tokens 64 --enable_offload
```

**중요**: `<PEER_ID>`는 Stage1 로그에서 확인한 실제 Peer ID로 교체하세요.

---

## 문제 해결

### Stage1의 DHT multiaddr을 찾을 수 없는 경우

Stage1이 완전히 초기화될 때까지 기다리세요 (약 10-30초):

```bash
# Stage1 로그에서 "handlers registered" 메시지 확인
tail -f stage1.log | grep "handlers registered"
```

### 연결 실패

- Stage1의 DHT multiaddr이 올바른지 확인
- 각 인스턴스의 포트 포워딩이 올바른지 확인
- Stage1이 완전히 준비되었는지 확인

### GPU 메모리 부족

Offloading이 활성화되어 있으므로 GPU 메모리 사용량이 줄어듭니다. 여전히 부족하면:
- 모델 크기를 줄이거나
- 더 큰 GPU 인스턴스 사용

---

## Offloading 동작 확인

Offloading이 제대로 작동하는지 확인:

```bash
# GPU 메모리 사용량 확인
nvidia-smi

# 로그에서 offload 메시지 확인
grep -i "offload" stage<N>.log
```

Offloading이 활성화되면:
- 모델이 CPU에 로드됨
- Forward 시에만 GPU로 이동
- GPU 메모리 사용량이 낮게 유지됨

