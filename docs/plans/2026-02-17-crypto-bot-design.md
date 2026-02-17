# Crypto Trading Bot - Design Document

**Date**: 2026-02-17
**Status**: Approved

## Overview

Upbit 원화 마켓에서 BTC + ETH를 자동 매매하는 3계층 하이브리드 봇.
DCA + RSI 규칙 기반 전략을 기본으로, FinRL(사전학습 가중치)의 정량 시그널과
Claude Opus 4.6(Bedrock)의 정성 분석을 결합한다.

## Constraints

- **거래소**: Upbit (KRW 마켓)
- **타겟 자산**: BTC (60%), ETH (40%)
- **시드머니**: 10~50만원 (소규모)
- **리스크 성향**: 보수적 (원금 보존 우선)
- **언어**: Python
- **운영 환경**: 로컬 PC → 추후 클라우드 전환

## Architecture

### 3-Layer Decision Pipeline

```
Layer 3: Claude Opus 4.6 (Bedrock) — Meta Advisor
  ├── 입력: 기술지표 + RL 시그널 + 뉴스/감성
  ├── 출력: JSON (파라미터 추천 + 근거)
  └── 역할: 최종 판단, 정성 분석, 이상 감지

Layer 2: FinRL (Pretrained Weights) — Quantitative Signal
  ├── 입력: OHLCV + 기술지표
  ├── 출력: action confidence score + parameter suggestions
  └── 역할: 데이터 기반 정량 시그널

Layer 1: DCA + RSI — Safe Execution Engine
  ├── 입력: Layer 3 파라미터 (또는 기본값)
  ├── 출력: 실제 주문 실행
  └── 역할: 매매 실행, 리스크 관리, fallback
```

### Fallback Policy

- Claude API 장애 → RL 시그널 + 기본 파라미터로 운영
- RL 장애 → 기본 DCA + RSI 파라미터로 운영
- 모두 장애 → 매매 중단, 텔레그램 알림

## Strategy: DCA + RSI

### Execution Cycle (4-hour interval)

1. Upbit에서 BTC/ETH 캔들 데이터 조회 (14봉)
2. RSI 계산
3. FinRL 시그널 생성
4. Claude Opus 4.6에 종합 분석 요청
5. 매매 판단:
   - RSI < buy_threshold (기본 30): 기본금액 × buy_multiplier 매수
   - RSI 30~45: 기본금액 × 1.2 매수
   - RSI 45~55: 기본금액 × 1.0 매수
   - RSI 55~70: 매수 스킵
   - RSI > sell_threshold (기본 70): 보유분 일부 매도
6. 주문 실행 → DB 기록 → 알림

### Risk Management

| 항목 | 설정값 |
|------|--------|
| 1회 최대 매수액 | 시드의 5% |
| 일일 최대 매수 횟수 | 6회 |
| 손절 라인 | 평균 매입가 -8% |
| 익절 라인 | 평균 매입가 +15% (보유분 30% 매도) |
| 최대 포지션 비율 | 총 시드의 80% |
| 현금 보유 최소 | 항상 20% |

## RL Layer (FinRL)

- **라이브러리**: FinRL + stable-baselines3
- **모델**: PPO (pretrained weights)
- **행동 공간**: 연속 — RSI 임계값, 매수 배율, 익절 목표
- **상태 공간**: RSI(14), 가격 변화율(5/10/20봉), 볼린저밴드 위치, 포지션 비율
- **출력**: JSON signal {action, confidence, suggested_params}

## LLM Layer (Claude Opus 4.6 via Bedrock)

- **Model ID**: anthropic.claude-opus-4-6-v1
- **Temperature**: 0 (결정론적 출력)
- **호출 빈도**: 6회/일 (4시간 주기)
- **월간 비용**: ~$12
- **출력 형식**: Structured JSON
- **입력**: 기술지표 + RL 시그널 + 공포탐욕지수

### Claude Output Schema

```json
{
  "market_phase": "bullish|bearish|sideways",
  "rsi_buy_threshold": 25-35,
  "rsi_sell_threshold": 65-80,
  "buy_multiplier": 0.8-2.0,
  "take_profit_pct": 10-25,
  "action_override": null | "skip" | "sell_all",
  "reasoning": "...",
  "risk_alert": null | "..."
}
```

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Exchange API | pyupbit | Upbit 공식 래퍼 |
| RL | FinRL + stable-baselines3 | Pretrained weights, 금융 RL 표준 |
| LLM | boto3 (Bedrock) | Claude Opus 4.6, AWS IAM 통합 |
| Indicators | pandas | RSI/BB 직접 구현 |
| Scheduler | APScheduler | cron 스타일 스케줄링 |
| DB | SQLite | 소규모 적합, 서버 불필요 |
| Notification | python-telegram-bot | 무료, 실시간 |
| Logging | loguru | 간편, 파일 로테이션 |
| Config | PyYAML | 사람이 읽기 쉬움 |

## Project Structure

```
crypto_bot/
├── config/
│   └── settings.yaml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── exchange/
│   │   ├── __init__.py
│   │   └── upbit_client.py
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── dca_rsi.py
│   │   └── indicators.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── finrl_agent.py
│   │   └── signal.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── bedrock_client.py
│   │   ├── advisor.py
│   │   └── prompts.py
│   ├── risk/
│   │   ├── __init__.py
│   │   └── manager.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py
│   │   └── sentiment.py
│   ├── notification/
│   │   ├── __init__.py
│   │   └── telegram.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── db.py
├── data/
│   └── trades.db
├── models/
├── backtest/
│   └── backtester.py
├── tests/
│   ├── test_strategy.py
│   ├── test_rl_env.py
│   ├── test_risk.py
│   └── test_advisor.py
├── requirements.txt
└── README.md
```

## Operation Modes

1. **Backtest**: 과거 데이터로 전략 검증
2. **Paper Trading**: 실제 시세, 가상 주문
3. **Live Trading**: 실제 주문 실행
