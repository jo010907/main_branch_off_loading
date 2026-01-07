# 엘리스 클라우드에서 LLaMA-3.1-8B 실행 가이드

## 전제 조건

- 4개의 GPU 인스턴스 (stage-1, stage-2, stage-3, stage-0)
- 각 인스턴스는 GPU A100-2g.20gb, VSCode (CUDA 11.8) 환경
- Hugging Face 계정 및 토큰

---

## 1단계: 초기 설정 (모든 인스턴스에서 실행)

**중요**: 다음 명령어들은 **엘리스 클라우드 인스턴스 내부**에서 실행해야 합니다. 
- VSCode로 직접 접속한 경우: 인스턴스 내부 터미널에서 실행
- SSH로 접속한 경우: SSH 연결 후 인스턴스 내부에서 실행

각 인스턴스(`stage-1`, `stage-2`, `stage-3`, `stage-0`)에서:

```bash
# 코드 클론
git clone https://github.com/hjkim24/my-petals.git
cd my-petals
git checkout Jaewon

# 초기 설치 스크립트 실행
./scripts/initial_install.sh

# 가상환경 활성화 (스크립트가 자동으로 생성함)
source venv/bin/activate

# Hugging Face CLI 설치 (hf 명령어가 없는 경우)
pip install huggingface_hub[cli]

# Hugging Face 인증 (LLaMA 모델 다운로드용)
hf auth login
# 토큰 입력: <your_huggingface_token>
```

**참고**: `hf auth login` 명령어가 작동하지 않으면:
```bash
# 가상환경이 활성화되어 있는지 확인
source venv/bin/activate

# Hugging Face CLI 설치 확인
pip install --upgrade huggingface_hub[cli]

# 또는 환경 변수로 토큰 설정
export HF_TOKEN="<your_huggingface_token>"
```

**Windows 로컬 환경에서 실행하는 경우** (권장하지 않음):
```powershell
# Windows에서는 py 명령어 사용
py -m pip install huggingface_hub[cli]
```

---

## 1-1단계: SSH 연결 (VSCode 외부에서 접속하는 경우)

엘리스 클라우드에서 SSH 연결 가이드를 확인하면 다음 정보를 제공합니다:

**SSH 연결 정보:**
- 사용자 이름: `elicer`
- 접속 주소: `central-01.tcp.tunnel.elice.io:21283` (인스턴스마다 다를 수 있음)
- 비밀 키: `elice-cloud-ondemand-{user_key_id}.pem`

**SSH 연결 방법:**

1. **비밀키 생성** (엘리스 클라우드 콘솔에서):
   - "비밀키 관리" 메뉴에서 비밀키 생성
   - 생성된 `.pem` 파일을 다운로드
   - 파일을 `C:\Users\<username>\.ssh\` 디렉토리에 저장

2. **SSH 연결** (Windows PowerShell):
   ```powershell
   # 비밀키 파일이 있는 디렉토리로 이동
   cd ~/.ssh
   
   # SSH 연결 (절대 경로 사용 권장)
   ssh -i "C:\Users\test123\.ssh\elice-cloud-ondemand-{user_key_id}.pem" elicer@central-01.tcp.tunnel.elice.io -p 21283
   ```

3. **호스트 인증 확인**:
   - 첫 연결 시 "Are you sure you want to continue connecting (yes/no/[fingerprint])?" 메시지가 나오면 `yes` 입력

4. **인스턴스 내부에서 작업**:
   - SSH 연결이 성공하면 인스턴스 내부 Linux 환경에서 작업합니다
   - 여기서 `pip`, `python`, `git` 등의 명령어를 사용할 수 있습니다

**참고**: 
- 각 인스턴스마다 다른 포트와 주소를 사용합니다
- VSCode로 직접 접속하는 경우 이 단계는 건너뛸 수 있습니다
- **중요**: `pip install` 등의 명령어는 인스턴스 내부에서 실행해야 합니다 (Windows 로컬이 아님)

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

**인스턴스 내부에서 실행해야 합니다** (SSH 연결 후 또는 VSCode 터미널에서):

```bash
# 가상환경 활성화 확인
source venv/bin/activate

# Hugging Face CLI 설치 확인
pip install --upgrade huggingface_hub[cli]

# 인증
hf auth login
# 토큰 재입력

# 또는 환경 변수로 토큰 설정
export HF_TOKEN="<your_huggingface_token>"
```

**Windows 로컬에서 실행한 경우**:
- Windows PowerShell에서는 `pip` 대신 `py -m pip` 사용
- 하지만 **권장하지 않음**: 인스턴스 내부에서 실행하는 것이 올바른 방법입니다

### 4. SSH 연결 실패

**비밀키 파일을 찾을 수 없는 경우:**
```bash
# 비밀키 파일 경로 확인
ls -la ~/.ssh/elice-cloud-ondemand-*.pem

# 절대 경로 사용
ssh -i "C:\Users\<username>\.ssh\elice-cloud-ondemand-{user_key_id}.pem" elicer@central-01.tcp.tunnel.elice.io -p 21283
```

**호스트 인증 확인:**
- 첫 연결 시 `yes` 입력하여 호스트 키를 known_hosts에 추가

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

