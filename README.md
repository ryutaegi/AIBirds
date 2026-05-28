# AIBirds - DQN Agent for Science Birds

![AIBirds Demo](Aibirds.gif)

MacOS 기반으로 Angry Birds (Science Birds) 게임을 Deep Q-Network (DQN)으로 자동 플레이하는 강화학습 에이전트입니다.

## 일부 레벨 안정성 문제로 시뮬레이션 실행 즉시 창을 최소화해야 합니다

게임 시뮬레이터: [Science Birds Framework](https://gitlab.com/aibirds/sciencebirdsframework)



## 주요 특징

- **Double DQN** + Dueling Network + PER (Prioritized Experience Replay)
- **N-step Returns** (3-step)
- **Exponential Epsilon Decay**
- 500개 레벨 자동 순환 학습
- 체크포인트 자동 저장 (500 에피소드마다)
- 타임아웃 자동 복구 (Socket 재연결)
- MacOS+Aibirds 강화학습 연동 가능

## 프로젝트 구조

```
sciencebirdsframework/
├── train_v2_hybrid.py          # 학습 메인 스크립트
├── evaluate.py                 # 모델 평가 스크립트
├── export_savedmodel.py        # 모델 내보내기
├── setup_config.py             # config.xml 레벨 설정 생성
├── main.py                     # 기본 에이전트 (baseline)
├── requirements.txt
│
├── src/
│   ├── demo/v2_rl/
│   │   ├── agent.py            # DQN 에이전트 (학습 로직)
│   │   ├── env.py              # Science Birds 게임 환경 (Gym-like)
│   │   ├── model.py            # 신경망 아키텍처 (Dueling DQN)
│   │   ├── memory.py           # PER 리플레이 버퍼
│   │   └── stats.py            # 학습 통계 및 그래프
│   ├── client/                 # Java 서버 통신 클라이언트
│   ├── computer_vision/        # 게임 화면 분석 (Ground Truth)
│   ├── trajectory_planner/     # 새 발사 궤적 계산
│   └── utils/                  # 유틸리티
│
├── baseline_agents/            # 기본 에이전트 예제
├── levelgenerator/             # 레벨 생성기
└── License/
```

## 설치

### 요구사항

- Python 3.9+
- Java 11+ (Science Birds 서버용)
- Science Birds 게임 바이너리 (별도 다운로드 필요)

### 패키지 설치

```bash
pip install -r requirements.txt
pip install tensorflow matplotlib
```

### 게임 설정

1. Science Birds 바이너리를 `ScienceBirds/` 폴더에 배치
2. 레벨 config 생성:
```bash
python3 setup_config.py --repeat 200
```
이 명령은 500개 레벨을 200번 반복하는 `config.xml`을 생성하여 100,000 에피소드까지 학습 가능하게 합니다.

## 학습

### 학습 시작

```bash
python3 train_v2_hybrid.py
```

### 체크포인트에서 재개

```bash
python3 train_v2_hybrid.py --resume
```

### 주요 하이퍼파라미터 (`train_v2_hybrid.py`)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `total_episodes` | 10,000 | 학습 에피소드 수 |
| `learning_rate` | 0.0003 | 학습률 |
| `gamma` | 0.90 | 할인율 |
| `epsilon_start` | 1.0 | 초기 탐험율 |
| `epsilon_end` | 0.10 | 최종 탐험율 |
| `memory_capacity` | 30,000 | 리플레이 메모리 크기 |
| `batch_size` | 64 | 배치 크기 |
| `n_step` | 3 | N-step Returns |

### 학습 출력

```
out/angry_birds/ab_agent/
├── checkpoints/           # 모델 체크포인트 (.h5)
├── plots/                 # 학습 그래프 (reward, score, win rate)
├── evaluation/            # 평가 결과
└── training_progress.txt  # 진행 상태
```

## 평가

```bash
python3 evaluate.py
```

## 트러블슈팅

| 문제 | 해결법 |
|------|--------|
| Socket timeout | 다른 앱 닫기 / 별도 macOS 데스크탑에서 게임 실행 |
| EVALUATION_TERMINATED | 정상 동작 - 500 레벨 완료 후 자동 순환 |
| Unknown layer: ClassicConvStem | `evaluate.py`에서 이미 수정됨 |

## 참고

- 시뮬레이터 원본: https://gitlab.com/aibirds/sciencebirdsframework
- AIBirds Competition: https://aibirds.org
