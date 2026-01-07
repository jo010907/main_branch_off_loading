# 엘리스 클라우드에서 LLaMA-3.1-8B 실행 가이드

## 전제 조건

- 4개의 GPU 인스턴스 (stage-1, stage-2, stage-3, stage-0)
- 각 인스턴스는 GPU A100-2g.20gb, VSCode (CUDA 11.8) 환경
- Hugging Face 계정 및 토큰

---

## 1단계: 초기 설정 (모든 인스턴스에서 실행)

각 인스턴스(`stage-1`, `stage-2`, `stage-3`, `stage-0`)에서 VSCode로 접속 후:

```bash
# 코드 클론
git clone https://github.com/hjkim24/my-petals.git
cd my-petals
git checkout Jaewon

# 초기 설치 스크립트 실행
./scripts/initial_install.sh

# Hugging Face 인증 (LLaMA 모델 다운로드용)
hf auth login
# 토큰 입력: <your_huggingface_token>
```

---

## 2단계: 엘리스 클라우드 공인 IP 확인

엘리스 클라우드에서는 터널 서버를 통해 접근하므로 공인 IP를 확인해야 합니다:

```bash
# 엘리스 클라우드 터널 서버 IP 확인
dig +short central-02.tcp.tunnel.elice.io
# 출력: 119.59.0.14 (또는 다른 IP)
```

**중요**: 각 인스턴스마다 다른 포트 포워딩이 설정되어 있습니다. 각 인스턴스의 포트 정보를 확인하세요.

---

## 3단계: Stage1 실행 (DHT 부트스트랩)

`stage-1` 인스턴스에서:

```bash
cd my-petals

# Stage1 실행 (백그라운드)
nohup python -m src.main \
  --model meta-llama/Llama-3.1-8B \
  --splits "8,16,24" \
  --dht_port 8002 \
  --rpc_port 8003 \
  --public_ip 119.59.0.14 \
  --public_dht_port 22452 \
  --public_rpc_port 50192 \
  --stage 1 \
  > stage1.log 2>&1 &

# 로그 확인
tail -f stage1.log
```

**중요**: Stage1의 DHT multiaddr을 확인하세요:

```bash
# 로그에서 DHT multiaddr 찾기
grep "DHT visible multiaddrs" stage1.log
# 또는
tail -f stage1.log | grep "DHT visible multiaddrs"
```

출력 예시:
```
INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): [<Multiaddr /ip4/119.59.0.14/tcp/22452/p2p/12D3KooW...>]
```

이 multiaddr을 복사하세요: `/ip4/119.59.0.14/tcp/22452/p2p/<PEER_ID>`

---

## 4단계: Stage2 실행

`stage-2` 인스턴스에서:

```bash
cd my-petals

# Stage1의 DHT multiaddr을 --dht_initial_peers에 사용
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
  > stage2.log 2>&1 &

# 로그 확인
tail -f stage2.log
```

**참고**: `<PEER_ID>`는 Stage1 로그에서 확인한 실제 Peer ID로 교체하세요.

---

## 5단계: Stage2-replacement 실행 (선택사항)

`stage-2-replacement` 인스턴스에서 (다른 public_ip 사용):

```bash
cd my-petals

# 다른 public_ip 사용 (119.59.0.13)
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
  > stage2-replacement.log 2>&1 &

# 로그 확인
tail -f stage2-replacement.log
```

---

## 6단계: Stage3 실행

`stage-3` 인스턴스에서:

```bash
cd my-petals

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
  > stage3.log 2>&1 &

# 로그 확인
tail -f stage3.log
```

---

## 7단계: Stage0 실행 (클라이언트)

`stage-0` 인스턴스에서:

```bash
cd my-petals

# Stage0 실행 (프롬프트와 최대 토큰 수 지정 가능)
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
  --max_new_tokens 64
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
  > stage0.log 2>&1 &

# 로그 확인
tail -f stage0.log
```

---

## Offloading 사용 시

GPU 메모리가 부족한 경우 `--enable_offload` 플래그를 추가하세요:

```bash
# Stage1 with offloading
python -m src.main \
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

# 다른 stage도 동일하게 --enable_offload 추가
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
```

---

## 프로세스 관리

```bash
# 실행 중인 프로세스 확인
ps aux | grep "src.main"

# 특정 stage 프로세스 종료
pkill -f "src.main --stage <N>"

# 또는 PID로 종료
kill <PID>
```

---

## 문제 해결

### 1. DHT multiaddr을 찾을 수 없는 경우

Stage1이 완전히 초기화될 때까지 기다리세요 (약 10-30초):

```bash
# Stage1 로그에서 "handlers registered" 메시지 확인
tail -f stage1.log | grep "handlers registered"
```

### 2. 연결 실패

- Stage1의 DHT multiaddr이 올바른지 확인
- 각 인스턴스의 포트 포워딩이 올바른지 확인
- 엘리스 클라우드 방화벽 설정 확인

### 3. 모델 다운로드 실패

Hugging Face 인증 확인:

```bash
hf auth login
# 토큰 재입력
```

### 4. 포트 충돌

다른 프로세스가 포트를 사용 중인지 확인:

```bash
# 포트 사용 확인
netstat -tulpn | grep <PORT>
# 또는
lsof -i :<PORT>
```

---

## 빠른 참조

### Stage 실행 순서

1. **Stage1** (DHT 부트스트랩) - 먼저 실행
2. **Stage2, Stage3** - Stage1의 DHT multiaddr 사용
3. **Stage0** (클라이언트) - 마지막 실행

### 포트 정보 요약

| Stage | DHT Port | RPC Port | Public DHT Port | Public RPC Port |
|-------|----------|----------|-----------------|-----------------|
| Stage1 | 8002 | 8003 | 22452 | 50192 |
| Stage2 | 8004 | 8005 | 29354 | 15930 |
| Stage2-replacement | 8004 | 8005 | 31172 | 27714 |
| Stage3 | 8006 | 8007 | 59491 | 38548 |
| Stage0 | 8008 | 8009 | 41826 | 23619 |

### LLaMA-3.1-8B 모델 정보

- **모델**: `meta-llama/Llama-3.1-8B`
- **레이어 수**: 32 layers
- **Splits**: `"8,16,24"` (각 stage: 8, 8, 8, 8 layers)

---

## 참고사항

- 엘리스 클라우드는 포트 포워딩을 사용하므로 `--public_dht_port`와 `--public_rpc_port`를 반드시 지정해야 합니다.
- 각 인스턴스의 포트 포워딩 정보는 엘리스 클라우드 콘솔에서 확인할 수 있습니다.
- Stage1의 DHT multiaddr은 다른 모든 stage에서 `--dht_initial_peers`로 사용됩니다.

