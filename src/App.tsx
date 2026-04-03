// ============================================================
// AI-Native Financial Investment System
// All-Weather Portfolio with 3D Visualization
// Single-file React Application
// ============================================================

// ============================================================
// SECTION 1: IMPORTS
// ============================================================
import React, { useState, useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Line, Html } from '@react-three/drei'
import * as THREE from 'three'

// ============================================================
// SECTION 2: TYPE DEFINITIONS & CONSTANTS
// ============================================================

type AssetClass = 'US_EQUITY' | 'INTL_EQUITY' | 'LONG_BOND' | 'TIPS' | 'GOLD' | 'COMMODITIES' | 'CASH'
type MarketRegime = 'BULL' | 'BEAR' | 'VOLATILE' | 'STABLE'
type MddTier = 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED'
type AgentRole = 'CIO' | 'RISK_MANAGER' | 'CFO'

interface AssetConfig {
  id: AssetClass
  label: string
  labelKo: string
  color: string
  expectedReturn: number
  volatility: number
  targetWeight: number
  currentWeight: number
  price: number
  priceHistory: number[]
}

interface AgentDecision {
  id: number
  timestamp: number
  agent: AgentRole
  action: string
  reasoning: string
  confidence: number
  approved: boolean
  worstCaseImpact?: number
}

interface CrossCheck {
  name: string
  nameKo: string
  status: 'PASS' | 'FAIL' | 'WARN'
  detail: string
}

interface PortfolioState {
  assets: AssetConfig[]
  totalValue: number
  dailyReturn: number
  cumulativeReturn: number
  currentVolatility: number
  currentMDD: number
  peakValue: number
  sharpeRatio: number
  mddTier: MddTier
  marketRegime: MarketRegime
  agentLog: AgentDecision[]
  crossChecks: CrossCheck[]
  tick: number
  valueHistory: number[]
  regimeTick: number
  regimeDuration: number
}

// 7개 자산 클래스 초기 설정 (All-Weather 기반)
const INITIAL_ASSETS: AssetConfig[] = [
  { id: 'US_EQUITY', label: 'US Equity', labelKo: '미국 주식', color: '#3b82f6', expectedReturn: 0.08, volatility: 0.16, targetWeight: 0.20, currentWeight: 0.20, price: 100, priceHistory: [100] },
  { id: 'INTL_EQUITY', label: 'Intl Equity', labelKo: '해외 주식', color: '#8b5cf6', expectedReturn: 0.07, volatility: 0.18, targetWeight: 0.10, currentWeight: 0.10, price: 100, priceHistory: [100] },
  { id: 'LONG_BOND', label: 'Long Bond', labelKo: '장기 채권', color: '#06b6d4', expectedReturn: 0.04, volatility: 0.12, targetWeight: 0.25, currentWeight: 0.25, price: 100, priceHistory: [100] },
  { id: 'TIPS', label: 'TIPS', labelKo: '물가연동채', color: '#14b8a6', expectedReturn: 0.03, volatility: 0.07, targetWeight: 0.15, currentWeight: 0.15, price: 100, priceHistory: [100] },
  { id: 'GOLD', label: 'Gold', labelKo: '금', color: '#f59e0b', expectedReturn: 0.05, volatility: 0.15, targetWeight: 0.10, currentWeight: 0.10, price: 100, priceHistory: [100] },
  { id: 'COMMODITIES', label: 'Commodities', labelKo: '원자재', color: '#f97316', expectedReturn: 0.04, volatility: 0.18, targetWeight: 0.10, currentWeight: 0.10, price: 100, priceHistory: [100] },
  { id: 'CASH', label: 'Cash', labelKo: '현금', color: '#64748b', expectedReturn: 0.02, volatility: 0.005, targetWeight: 0.10, currentWeight: 0.10, price: 100, priceHistory: [100] },
]

// 7x7 상관관계 행렬 (실제 금융 데이터 기반)
// 순서: US_EQUITY, INTL_EQUITY, LONG_BOND, TIPS, GOLD, COMMODITIES, CASH
const CORRELATION_MATRIX: number[][] = [
  [ 1.00,  0.85, -0.20,  0.10,  0.05,  0.40,  0.00],
  [ 0.85,  1.00, -0.15,  0.12,  0.10,  0.45,  0.00],
  [-0.20, -0.15,  1.00,  0.70,  0.30, -0.10,  0.00],
  [ 0.10,  0.12,  0.70,  1.00,  0.25,  0.05,  0.00],
  [ 0.05,  0.10,  0.30,  0.25,  1.00,  0.35,  0.00],
  [ 0.40,  0.45, -0.10,  0.05,  0.35,  1.00,  0.00],
  [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00],
]

const TARGET_VOLATILITY = 0.08
const MDD_LIMIT = -0.05
const RISK_FREE_RATE = 0.045
const INITIAL_PORTFOLIO_VALUE = 1_000_000_000 // 10억원
const REGIME_ORDER: MarketRegime[] = ['STABLE', 'BULL', 'VOLATILE', 'BEAR']

// ============================================================
// SECTION 3: UTILITY FUNCTIONS
// ============================================================

// Box-Muller 변환으로 표준정규분포 난수 생성
function normalRandom(): number {
  let u = 0, v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}

// Cholesky 분해 (하삼각행렬 반환)
function choleskyDecompose(matrix: number[][]): number[][] {
  const n = matrix.length
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k]
      if (i === j) {
        const val = matrix[i][i] - sum
        L[i][j] = val > 0 ? Math.sqrt(val) : 0.0001
      } else {
        L[i][j] = L[j][j] !== 0 ? (matrix[i][j] - sum) / L[j][j] : 0
      }
    }
  }
  return L
}

// 상관관계 행렬 → 3D 위치 (간이 MDS)
function correlationToPositions(corrMatrix: number[][]): [number, number, number][] {
  const n = corrMatrix.length
  // 거리 행렬: d = sqrt(2 * (1 - corr))
  const dist: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => Math.sqrt(2 * Math.max(0, 1 - corrMatrix[i][j])))
  )
  // 간이 MDS: 첫 번째 자산을 원점, 나머지를 거리 기반 배치
  const positions: [number, number, number][] = []
  const scale = 3.0
  // 스프링 기반 간이 배치
  const pos: number[][] = Array.from({ length: n }, () => [
    (Math.random() - 0.5) * 4,
    (Math.random() - 0.5) * 2,
    (Math.random() - 0.5) * 4,
  ])
  // 반복적 스프링 relaxation
  for (let iter = 0; iter < 200; iter++) {
    for (let i = 0; i < n; i++) {
      let fx = 0, fy = 0, fz = 0
      for (let j = 0; j < n; j++) {
        if (i === j) continue
        const dx = pos[i][0] - pos[j][0]
        const dy = pos[i][1] - pos[j][1]
        const dz = pos[i][2] - pos[j][2]
        const currentDist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.001
        const targetDist = dist[i][j] * scale
        const force = (currentDist - targetDist) * 0.01
        fx -= (dx / currentDist) * force
        fy -= (dy / currentDist) * force
        fz -= (dz / currentDist) * force
      }
      pos[i][0] += fx
      pos[i][1] += fy
      pos[i][2] += fz
    }
  }
  // 중심을 원점으로
  const cx = pos.reduce((s, p) => s + p[0], 0) / n
  const cy = pos.reduce((s, p) => s + p[1], 0) / n
  const cz = pos.reduce((s, p) => s + p[2], 0) / n
  for (let i = 0; i < n; i++) {
    positions.push([pos[i][0] - cx, pos[i][1] - cy + 2, pos[i][2] - cz])
  }
  return positions
}

// 간단한 Perlin-like 노이즈
function simpleNoise(x: number, z: number, t: number): number {
  return (
    Math.sin(x * 1.5 + t * 0.3) * 0.3 +
    Math.cos(z * 1.2 + t * 0.2) * 0.3 +
    Math.sin((x + z) * 0.8 + t * 0.15) * 0.2 +
    Math.cos(x * 2.5 - z * 1.8 + t * 0.1) * 0.15
  )
}

// ============================================================
// SECTION 4: ALL-WEATHER PORTFOLIO ENGINE
// ============================================================

// 포트폴리오 변동성 계산
function calcPortfolioVolatility(weights: number[], vols: number[], corrMatrix: number[][]): number {
  let variance = 0
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < weights.length; j++) {
      variance += weights[i] * weights[j] * vols[i] * vols[j] * corrMatrix[i][j]
    }
  }
  return Math.sqrt(Math.max(0, variance))
}

// 변동성 타겟팅: 포트폴리오 변동성을 목표 수준으로 조절
function volatilityTargeting(assets: AssetConfig[], corrMatrix: number[][]): number[] {
  const weights = assets.map(a => a.currentWeight)
  const vols = assets.map(a => a.volatility)
  const portVol = calcPortfolioVolatility(weights, vols, corrMatrix)

  if (portVol > TARGET_VOLATILITY * 1.1) {
    // 변동성 초과: 위험자산 비중 축소, 현금 증가
    const adjusted = [...weights]
    const cashIdx = assets.findIndex(a => a.id === 'CASH')
    const scale = TARGET_VOLATILITY / portVol
    let freed = 0
    for (let i = 0; i < adjusted.length; i++) {
      if (i !== cashIdx && assets[i].volatility > 0.05) {
        const reduction = adjusted[i] * (1 - scale) * 0.3
        freed += reduction
        adjusted[i] -= reduction
      }
    }
    adjusted[cashIdx] += freed
    return adjusted
  } else if (portVol < TARGET_VOLATILITY * 0.9) {
    // 변동성 부족: 목표 비중으로 점진 복귀
    const adjusted = [...weights]
    for (let i = 0; i < adjusted.length; i++) {
      const diff = assets[i].targetWeight - adjusted[i]
      adjusted[i] += diff * 0.05
    }
    return adjusted
  }
  return weights
}

// MDD 방어 로직 (4단계)
function mddDefense(assets: AssetConfig[], currentDD: number): { weights: number[]; tier: MddTier } {
  const weights = assets.map(a => a.currentWeight)
  const cashIdx = assets.findIndex(a => a.id === 'CASH')
  const bondIdx = assets.findIndex(a => a.id === 'LONG_BOND')
  const tipsIdx = assets.findIndex(a => a.id === 'TIPS')

  if (currentDD > -0.02) {
    return { weights, tier: 'GREEN' }
  }

  if (currentDD > -0.035) {
    // YELLOW: 주식 30% 축소
    const adjusted = [...weights]
    let freed = 0
    for (let i = 0; i < adjusted.length; i++) {
      if (assets[i].id === 'US_EQUITY' || assets[i].id === 'INTL_EQUITY') {
        const cut = adjusted[i] * 0.3
        freed += cut
        adjusted[i] -= cut
      }
    }
    adjusted[bondIdx] += freed * 0.5
    adjusted[cashIdx] += freed * 0.5
    return { weights: adjusted, tier: 'YELLOW' }
  }

  if (currentDD > -0.045) {
    // ORANGE: 주식 60% 축소
    const adjusted = [...weights]
    let freed = 0
    for (let i = 0; i < adjusted.length; i++) {
      if (assets[i].id === 'US_EQUITY' || assets[i].id === 'INTL_EQUITY' || assets[i].id === 'COMMODITIES') {
        const cut = adjusted[i] * 0.6
        freed += cut
        adjusted[i] -= cut
      }
    }
    adjusted[bondIdx] += freed * 0.3
    adjusted[tipsIdx] += freed * 0.2
    adjusted[cashIdx] += freed * 0.5
    return { weights: adjusted, tier: 'ORANGE' }
  }

  // RED: 긴급 - 80% 현금
  return {
    weights: assets.map(a => {
      if (a.id === 'CASH') return 0.80
      if (a.id === 'LONG_BOND') return 0.15
      if (a.id === 'TIPS') return 0.05
      return 0.0
    }),
    tier: 'RED',
  }
}

// 교차검증 시스템 (5개 체크, 최소 3개 통과 필요)
function runCrossChecks(
  proposedWeights: number[],
  assets: AssetConfig[],
  corrMatrix: number[][],
  currentDD: number,
  regime: MarketRegime
): CrossCheck[] {
  const vols = assets.map(a => a.volatility)
  const portVol = calcPortfolioVolatility(proposedWeights, vols, corrMatrix)

  const checks: CrossCheck[] = []

  // 1. 변동성 예산 체크
  const volOk = portVol <= TARGET_VOLATILITY * 1.2
  checks.push({
    name: 'Volatility Budget',
    nameKo: '변동성 예산',
    status: volOk ? 'PASS' : portVol <= TARGET_VOLATILITY * 1.4 ? 'WARN' : 'FAIL',
    detail: `포트폴리오 변동성 ${(portVol * 100).toFixed(1)}% (목표: ${(TARGET_VOLATILITY * 100).toFixed(0)}%)`,
  })

  // 2. 상관관계 집중도 체크
  let concFail = false
  for (let i = 0; i < proposedWeights.length; i++) {
    for (let j = i + 1; j < proposedWeights.length; j++) {
      if (corrMatrix[i][j] > 0.6 && proposedWeights[i] + proposedWeights[j] > 0.40) {
        concFail = true
      }
    }
  }
  checks.push({
    name: 'Correlation Concentration',
    nameKo: '상관관계 집중도',
    status: concFail ? 'FAIL' : 'PASS',
    detail: concFail ? '고상관 자산쌍의 합산 비중이 40% 초과' : '분산 양호',
  })

  // 3. MDD 여유 체크
  const mddHeadroom = MDD_LIMIT - currentDD
  checks.push({
    name: 'MDD Headroom',
    nameKo: 'MDD 여유분',
    status: mddHeadroom > 0.01 ? 'PASS' : mddHeadroom > 0.005 ? 'WARN' : 'FAIL',
    detail: `잔여 MDD 예산: ${(mddHeadroom * 100).toFixed(2)}%`,
  })

  // 4. 레짐 적합성 체크
  const equityWeight = proposedWeights[0] + proposedWeights[1]
  let regimeOk = true
  if (regime === 'BEAR' && equityWeight > 0.25) regimeOk = false
  if (regime === 'VOLATILE' && equityWeight > 0.30) regimeOk = false
  checks.push({
    name: 'Regime Consistency',
    nameKo: '레짐 적합성',
    status: regimeOk ? 'PASS' : 'WARN',
    detail: `${regime} 레짐에서 주식비중 ${(equityWeight * 100).toFixed(1)}%`,
  })

  // 5. 단일자산 최대비중 체크
  const maxWeight = Math.max(...proposedWeights)
  checks.push({
    name: 'Max Single Asset',
    nameKo: '단일자산 한도',
    status: maxWeight <= 0.35 ? 'PASS' : maxWeight <= 0.50 ? 'WARN' : 'FAIL',
    detail: `최대 단일자산 비중: ${(maxWeight * 100).toFixed(1)}%`,
  })

  return checks
}


// ============================================================
// SECTION 5: AI EXECUTIVE GOVERNANCE AGENTS
// ============================================================

let decisionIdCounter = 0

// CIO 에이전트: 시장 레짐 감지 및 전략 제안
function cioAnalyze(state: PortfolioState): AgentDecision {
  const { assets, marketRegime, currentVolatility } = state
  const equityWeight = assets[0].currentWeight + assets[1].currentWeight
  const bondWeight = assets[2].currentWeight + assets[3].currentWeight
  const safeWeight = assets[4].currentWeight + assets[6].currentWeight

  let action = ''
  let reasoning = ''
  let confidence = 0.75

  switch (marketRegime) {
    case 'BULL':
      if (equityWeight < 0.30) {
        action = 'INCREASE_EQUITY'
        reasoning = `강세장 진입. 주식 비중(${(equityWeight * 100).toFixed(1)}%)을 목표(30%)로 상향 권고. 채권 비중 일부 축소.`
        confidence = 0.82
      } else {
        action = 'HOLD'
        reasoning = `강세장 유지. 현재 주식 비중(${(equityWeight * 100).toFixed(1)}%) 적정. 현 배분 유지 권고.`
        confidence = 0.88
      }
      break
    case 'BEAR':
      action = 'DEFENSIVE_SHIFT'
      reasoning = `약세장 감지. 주식 비중 축소하고 채권/금 비중 확대 권고. 현 변동성: ${(currentVolatility * 100).toFixed(1)}%`
      confidence = 0.78
      break
    case 'VOLATILE':
      action = 'REDUCE_RISK'
      reasoning = `고변동 레짐. 전체 위험자산 축소, 현금/채권 비중 확대 권고. 상관관계 불안정 구간.`
      confidence = 0.70
      break
    case 'STABLE':
      action = 'REBALANCE_TO_TARGET'
      reasoning = `안정 레짐. All-Weather 목표 배분으로 점진적 복귀 권고. 리스크 패리티 최적화 적용.`
      confidence = 0.90
      break
  }

  return {
    id: ++decisionIdCounter,
    timestamp: Date.now(),
    agent: 'CIO',
    action,
    reasoning,
    confidence,
    approved: true,
  }
}

// Risk Manager: 몬테카를로 시뮬레이션 + 최악 시나리오
function riskManagerVerify(
  proposedWeights: number[],
  assets: AssetConfig[],
  corrMatrix: number[][],
  currentDD: number
): AgentDecision {
  const vols = assets.map(a => a.volatility)
  const n = assets.length
  const L = choleskyDecompose(corrMatrix)

  // 몬테카를로 시뮬레이션: 1000경로, 30일
  const paths = 1000
  const days = 30
  const dt = 1 / 252
  const finalReturns: number[] = []

  for (let p = 0; p < paths; p++) {
    let portReturn = 0
    let cumReturn = 1.0
    let peak = 1.0
    let worstDD = 0

    for (let d = 0; d < days; d++) {
      // 상관된 난수 생성
      const z: number[] = Array.from({ length: n }, () => normalRandom())
      const corrZ: number[] = new Array(n).fill(0)
      for (let i = 0; i < n; i++) {
        for (let k = 0; k <= i; k++) {
          corrZ[i] += L[i][k] * z[k]
        }
      }

      let dailyRet = 0
      for (let i = 0; i < n; i++) {
        const assetRet = (assets[i].expectedReturn - 0.5 * vols[i] * vols[i]) * dt + vols[i] * Math.sqrt(dt) * corrZ[i]
        dailyRet += proposedWeights[i] * assetRet
      }
      cumReturn *= (1 + dailyRet)
      if (cumReturn > peak) peak = cumReturn
      const dd = (cumReturn - peak) / peak
      if (dd < worstDD) worstDD = dd
    }
    finalReturns.push(cumReturn - 1)
  }

  // CVaR 95: 하위 5% 평균 손실
  finalReturns.sort((a, b) => a - b)
  const cutoff = Math.floor(paths * 0.05)
  const cvar95 = finalReturns.slice(0, cutoff).reduce((s, v) => s + v, 0) / cutoff
  const worstCase = finalReturns[0]

  // 위기 시나리오: 상관관계가 모두 0.8로 수렴
  const crisisCorr = corrMatrix.map(row => row.map((v, j) => j === corrMatrix.indexOf(row) ? 1 : Math.min(0.8, v + 0.3)))
  const crisisVol = calcPortfolioVolatility(proposedWeights, vols, crisisCorr)

  const approved = cvar95 > MDD_LIMIT && crisisVol < 0.15
  const confidence = approved ? 0.85 : 0.60

  return {
    id: ++decisionIdCounter,
    timestamp: Date.now(),
    agent: 'RISK_MANAGER',
    action: approved ? 'APPROVE' : 'REJECT',
    reasoning: approved
      ? `몬테카를로 검증 통과. CVaR(95): ${(cvar95 * 100).toFixed(2)}%, 위기 변동성: ${(crisisVol * 100).toFixed(1)}%. 리스크 허용 범위 내.`
      : `검증 실패! CVaR(95): ${(cvar95 * 100).toFixed(2)}%, 위기 변동성: ${(crisisVol * 100).toFixed(1)}%. MDD 한도 초과 위험.`,
    confidence,
    approved,
    worstCaseImpact: worstCase,
  }
}

// CFO 에이전트: 실행 사이징 및 비용 검증
function cfoAllocate(
  proposedWeights: number[],
  currentWeights: number[],
  totalValue: number
): AgentDecision {
  // 거래비용 계산 (거래대금의 0.1%)
  let turnover = 0
  for (let i = 0; i < proposedWeights.length; i++) {
    turnover += Math.abs(proposedWeights[i] - currentWeights[i])
  }
  const tradeCost = turnover * 0.001 * totalValue
  const tradeCostPct = turnover * 0.001

  // 현금 최소 5% 유지 확인
  const cashIdx = proposedWeights.length - 1
  const cashOk = proposedWeights[cashIdx] >= 0.05

  // 회전율 20% 초과 시 단계적 실행
  const phased = turnover > 0.20

  const approved = cashOk && tradeCostPct < 0.005
  const confidence = approved ? 0.88 : 0.55

  return {
    id: ++decisionIdCounter,
    timestamp: Date.now(),
    agent: 'CFO',
    action: approved ? (phased ? 'PHASED_EXECUTE' : 'EXECUTE') : 'HOLD',
    reasoning: approved
      ? `실행 승인. 거래비용: ${(tradeCost / 10000).toFixed(0)}만원 (${(tradeCostPct * 100).toFixed(2)}%). ${phased ? '회전율 초과로 3단계 분할 실행.' : '일괄 실행.'} 현금 보유: ${(proposedWeights[cashIdx] * 100).toFixed(1)}%`
      : `실행 보류. ${!cashOk ? '현금 보유 5% 미달.' : '거래비용 과다.'} 비용: ${(tradeCostPct * 100).toFixed(2)}%`,
    confidence,
    approved,
  }
}

// 거버넌스 파이프라인 오케스트레이터
function runGovernancePipeline(state: PortfolioState): {
  newWeights: number[]
  decisions: AgentDecision[]
  crossChecks: CrossCheck[]
} {
  const decisions: AgentDecision[] = []

  // 1단계: CIO 분석 및 제안
  const cioDecision = cioAnalyze(state)
  decisions.push(cioDecision)

  // CIO 제안에 따른 목표 비중 계산
  let proposed = state.assets.map(a => a.currentWeight)
  if (cioDecision.action === 'REBALANCE_TO_TARGET') {
    proposed = state.assets.map(a => {
      const diff = a.targetWeight - a.currentWeight
      return a.currentWeight + diff * 0.15
    })
  } else if (cioDecision.action === 'DEFENSIVE_SHIFT') {
    proposed = state.assets.map(a => {
      if (a.id === 'US_EQUITY' || a.id === 'INTL_EQUITY') return a.currentWeight * 0.8
      if (a.id === 'LONG_BOND' || a.id === 'GOLD') return a.currentWeight * 1.15
      if (a.id === 'CASH') return a.currentWeight * 1.3
      return a.currentWeight
    })
  } else if (cioDecision.action === 'REDUCE_RISK') {
    proposed = state.assets.map(a => {
      if (a.volatility > 0.10) return a.currentWeight * 0.85
      if (a.id === 'CASH') return a.currentWeight * 1.5
      return a.currentWeight * 1.05
    })
  } else if (cioDecision.action === 'INCREASE_EQUITY') {
    proposed = state.assets.map(a => {
      if (a.id === 'US_EQUITY') return Math.min(a.currentWeight * 1.1, 0.25)
      if (a.id === 'INTL_EQUITY') return Math.min(a.currentWeight * 1.1, 0.15)
      if (a.id === 'LONG_BOND') return a.currentWeight * 0.95
      return a.currentWeight
    })
  }

  // 비중 합 정규화
  const sum = proposed.reduce((s, w) => s + w, 0)
  proposed = proposed.map(w => w / sum)

  // 변동성 타겟팅 적용
  proposed = volatilityTargeting(
    state.assets.map((a, i) => ({ ...a, currentWeight: proposed[i] })),
    CORRELATION_MATRIX
  )
  const sum2 = proposed.reduce((s, w) => s + w, 0)
  proposed = proposed.map(w => w / sum2)

  // MDD 방어 적용
  const { weights: mddWeights, tier } = mddDefense(
    state.assets.map((a, i) => ({ ...a, currentWeight: proposed[i] })),
    state.currentMDD
  )
  if (tier !== 'GREEN') {
    proposed = mddWeights
    const mddSum = proposed.reduce((s, w) => s + w, 0)
    proposed = proposed.map(w => w / mddSum)
  }

  // 2단계: 교차검증 (5개 체크)
  const crossChecks = runCrossChecks(proposed, state.assets, CORRELATION_MATRIX, state.currentMDD, state.marketRegime)
  const passCount = crossChecks.filter(c => c.status === 'PASS').length
  const failCount = crossChecks.filter(c => c.status === 'FAIL').length

  if (passCount < 3 || failCount > 0) {
    // 교차검증 실패 시 보수적 조정
    proposed = state.assets.map(a => a.currentWeight)
    decisions.push({
      id: ++decisionIdCounter,
      timestamp: Date.now(),
      agent: 'RISK_MANAGER',
      action: 'CROSS_CHECK_FAIL',
      reasoning: `교차검증 ${passCount}/5 통과, ${failCount}개 실패. 현재 배분 유지.`,
      confidence: 0.90,
      approved: false,
    })
    return { newWeights: proposed, decisions, crossChecks }
  }

  // 3단계: Risk Manager 몬테카를로 검증
  const rmDecision = riskManagerVerify(proposed, state.assets, CORRELATION_MATRIX, state.currentMDD)
  decisions.push(rmDecision)

  if (!rmDecision.approved) {
    return { newWeights: state.assets.map(a => a.currentWeight), decisions, crossChecks }
  }

  // 4단계: CFO 실행 검증
  const cfoDecision = cfoAllocate(proposed, state.assets.map(a => a.currentWeight), state.totalValue)
  decisions.push(cfoDecision)

  if (!cfoDecision.approved) {
    return { newWeights: state.assets.map(a => a.currentWeight), decisions, crossChecks }
  }

  return { newWeights: proposed, decisions, crossChecks }
}

// ============================================================
// SECTION 6: DATA SIMULATION ENGINE
// ============================================================

function createInitialState(): PortfolioState {
  return {
    assets: INITIAL_ASSETS.map(a => ({ ...a, priceHistory: [a.price] })),
    totalValue: INITIAL_PORTFOLIO_VALUE,
    dailyReturn: 0,
    cumulativeReturn: 0,
    currentVolatility: 0.08,
    currentMDD: 0,
    peakValue: INITIAL_PORTFOLIO_VALUE,
    sharpeRatio: 0,
    mddTier: 'GREEN',
    marketRegime: 'STABLE',
    agentLog: [],
    crossChecks: [],
    tick: 0,
    valueHistory: [INITIAL_PORTFOLIO_VALUE],
    regimeTick: 0,
    regimeDuration: 25 + Math.floor(Math.random() * 15),
  }
}

// 시장 레짐에 따른 자산 파라미터 조정
function getRegimeAdjustedParams(asset: AssetConfig, regime: MarketRegime): { mu: number; sigma: number } {
  const base = { mu: asset.expectedReturn, sigma: asset.volatility }
  switch (regime) {
    case 'BULL':
      if (asset.id === 'US_EQUITY' || asset.id === 'INTL_EQUITY') {
        return { mu: base.mu * 1.8, sigma: base.sigma * 0.85 }
      }
      if (asset.id === 'LONG_BOND') return { mu: base.mu * 0.7, sigma: base.sigma * 0.9 }
      return base
    case 'BEAR':
      if (asset.id === 'US_EQUITY' || asset.id === 'INTL_EQUITY') {
        return { mu: -base.mu * 1.2, sigma: base.sigma * 1.5 }
      }
      if (asset.id === 'LONG_BOND' || asset.id === 'TIPS') return { mu: base.mu * 1.4, sigma: base.sigma * 1.1 }
      if (asset.id === 'GOLD') return { mu: base.mu * 2.0, sigma: base.sigma * 1.2 }
      return base
    case 'VOLATILE':
      return { mu: base.mu * 0.3, sigma: base.sigma * 1.8 }
    case 'STABLE':
    default:
      return base
  }
}

// 하루 시뮬레이션
function simulateDay(state: PortfolioState): PortfolioState {
  const n = state.assets.length
  const dt = 1 / 252
  const L = choleskyDecompose(CORRELATION_MATRIX)

  // 상관된 난수 생성
  const z: number[] = Array.from({ length: n }, () => normalRandom())
  const corrZ: number[] = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    for (let k = 0; k <= i; k++) {
      corrZ[i] += L[i][k] * z[k]
    }
  }

  // 자산별 가격 업데이트
  let portfolioReturn = 0
  const newAssets = state.assets.map((asset, i) => {
    const { mu, sigma } = getRegimeAdjustedParams(asset, state.marketRegime)
    const dailyReturn = (mu - 0.5 * sigma * sigma) * dt + sigma * Math.sqrt(dt) * corrZ[i]
    const newPrice = asset.price * (1 + dailyReturn)
    const history = [...asset.priceHistory.slice(-251), newPrice]
    portfolioReturn += asset.currentWeight * dailyReturn
    return { ...asset, price: newPrice, priceHistory: history }
  })

  const newTotalValue = state.totalValue * (1 + portfolioReturn)
  const newPeak = Math.max(state.peakValue, newTotalValue)
  const currentDD = (newTotalValue - newPeak) / newPeak
  const cumulativeReturn = (newTotalValue - INITIAL_PORTFOLIO_VALUE) / INITIAL_PORTFOLIO_VALUE

  // 롤링 변동성 계산 (최근 20일)
  const recentValues = [...state.valueHistory.slice(-19), newTotalValue]
  let vol = state.currentVolatility
  if (recentValues.length >= 5) {
    const returns: number[] = []
    for (let i = 1; i < recentValues.length; i++) {
      returns.push(Math.log(recentValues[i] / recentValues[i - 1]))
    }
    const mean = returns.reduce((s, r) => s + r, 0) / returns.length
    const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / (returns.length - 1)
    vol = Math.sqrt(variance * 252)
  }

  // 샤프 비율
  const annualizedReturn = cumulativeReturn * (252 / Math.max(1, state.tick + 1))
  const sharpe = vol > 0 ? (annualizedReturn - RISK_FREE_RATE) / vol : 0

  // MDD 티어 결정
  let mddTier: MddTier = 'GREEN'
  if (currentDD <= -0.045) mddTier = 'RED'
  else if (currentDD <= -0.035) mddTier = 'ORANGE'
  else if (currentDD <= -0.02) mddTier = 'YELLOW'

  // 레짐 전환
  let newRegime = state.marketRegime
  let regimeTick = state.regimeTick + 1
  let regimeDuration = state.regimeDuration
  if (regimeTick >= regimeDuration) {
    const currentIdx = REGIME_ORDER.indexOf(state.marketRegime)
    newRegime = REGIME_ORDER[(currentIdx + 1) % REGIME_ORDER.length]
    regimeTick = 0
    regimeDuration = 25 + Math.floor(Math.random() * 15)
  }

  return {
    ...state,
    assets: newAssets,
    totalValue: newTotalValue,
    dailyReturn: portfolioReturn,
    cumulativeReturn,
    currentVolatility: vol,
    currentMDD: currentDD,
    peakValue: newPeak,
    sharpeRatio: sharpe,
    mddTier,
    marketRegime: newRegime,
    tick: state.tick + 1,
    valueHistory: [...state.valueHistory.slice(-299), newTotalValue],
    regimeTick,
    regimeDuration,
  }
}


// ============================================================
// SECTION 7: 3D VISUALIZATION COMPONENTS
// ============================================================

// 개별 자산 구체 컴포넌트
function AssetSphere({ asset, position, isSelected, onSelect }: {
  asset: AssetConfig
  position: [number, number, number]
  isSelected: boolean
  onSelect: () => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const baseRadius = Math.sqrt(asset.currentWeight) * 1.2 + 0.15
  const recentReturn = asset.priceHistory.length > 1
    ? (asset.price - asset.priceHistory[asset.priceHistory.length - 2]) / asset.priceHistory[asset.priceHistory.length - 2]
    : 0

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.3
      const pulse = 1 + Math.sin(Date.now() * 0.003) * 0.03
      const scale = baseRadius * pulse
      meshRef.current.scale.setScalar(scale)
    }
  })

  const emissiveColor = recentReturn >= 0 ? '#00ff88' : '#ff4444'
  const emissiveIntensity = Math.min(Math.abs(recentReturn) * 50, 0.5)

  return (
    <group position={position}>
      <mesh ref={meshRef} onClick={onSelect}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial
          color={asset.color}
          emissive={emissiveColor}
          emissiveIntensity={emissiveIntensity}
          metalness={0.3}
          roughness={0.4}
          transparent
          opacity={0.9}
        />
      </mesh>
      {/* 외곽 글로우 링 */}
      <mesh scale={[baseRadius * 1.3, baseRadius * 1.3, baseRadius * 1.3]}>
        <sphereGeometry args={[1, 16, 16]} />
        <meshBasicMaterial color={asset.color} transparent opacity={0.08} wireframe />
      </mesh>
      {/* 라벨 */}
      <Text
        position={[0, baseRadius + 0.6, 0]}
        fontSize={0.28}
        color="#e2e8f0"
        anchorX="center"
        anchorY="bottom"
      >
        {asset.labelKo}
      </Text>
      <Text
        position={[0, baseRadius + 0.3, 0]}
        fontSize={0.22}
        color={asset.color}
        anchorX="center"
        anchorY="bottom"
      >
        {(asset.currentWeight * 100).toFixed(1)}%
      </Text>
      {/* 선택 시 상세 정보 */}
      {isSelected && (
        <Html position={[baseRadius + 0.5, 0.5, 0]} distanceFactor={8}>
          <div className="bg-navy-800/95 border border-slate-600 rounded-lg p-3 min-w-[160px] text-xs backdrop-blur-sm">
            <div className="text-white font-bold mb-1">{asset.label}</div>
            <div className="text-slate-300">가격: ${asset.price.toFixed(2)}</div>
            <div className="text-slate-300">비중: {(asset.currentWeight * 100).toFixed(1)}%</div>
            <div className="text-slate-300">변동성: {(asset.volatility * 100).toFixed(1)}%</div>
            <div className={recentReturn >= 0 ? 'text-green-400' : 'text-red-400'}>
              수익률: {(recentReturn * 100).toFixed(2)}%
            </div>
          </div>
        </Html>
      )}
    </group>
  )
}

// 상관관계 연결선
function CorrelationLinks({ assets, positions }: {
  assets: AssetConfig[]
  positions: [number, number, number][]
}) {
  const lines: { points: [number, number, number][]; color: string; opacity: number; corr: number }[] = []

  for (let i = 0; i < assets.length; i++) {
    for (let j = i + 1; j < assets.length; j++) {
      const corr = CORRELATION_MATRIX[i][j]
      if (Math.abs(corr) < 0.1) continue
      const color = corr < 0 ? '#00ff88' : corr > 0.5 ? '#ff4444' : '#ffaa00'
      lines.push({
        points: [positions[i], positions[j]],
        color,
        opacity: Math.abs(corr) * 0.5,
        corr,
      })
    }
  }

  return (
    <>
      {lines.map((line, i) => (
        <Line
          key={i}
          points={line.points}
          color={line.color}
          lineWidth={Math.abs(line.corr) * 3 + 0.5}
          transparent
          opacity={line.opacity}
        />
      ))}
    </>
  )
}

// 리스크 지형 (3D Surface)
function RiskTerrain({ volatility, mddTier }: { volatility: number; mddTier: MddTier }) {
  const meshRef = useRef<THREE.Mesh>(null)
  const gridSize = 32
  const terrainSize = 12

  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(terrainSize, terrainSize, gridSize, gridSize)
    geo.rotateX(-Math.PI / 2)
    return geo
  }, [])

  useFrame(({ clock }) => {
    if (!meshRef.current) return
    const geo = meshRef.current.geometry
    const pos = geo.attributes.position
    const colors = new Float32Array(pos.count * 3)
    const t = clock.elapsedTime
    const volFactor = volatility / TARGET_VOLATILITY

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i)
      const z = pos.getZ(i)
      const noise = simpleNoise(x * 0.5, z * 0.5, t)
      const height = noise * volFactor * 0.8
      pos.setY(i, height - 2.5)

      // 높이에 따른 색상 (파랑→초록→노랑→빨강)
      const normalizedHeight = (height + 1) / 2
      let r, g, b
      if (normalizedHeight < 0.33) {
        r = 0.05; g = 0.2 + normalizedHeight * 1.5; b = 0.6 - normalizedHeight
      } else if (normalizedHeight < 0.66) {
        const t2 = (normalizedHeight - 0.33) * 3
        r = t2 * 0.9; g = 0.7; b = 0.1
      } else {
        const t2 = (normalizedHeight - 0.66) * 3
        r = 0.9 + t2 * 0.1; g = 0.7 - t2 * 0.5; b = 0.1
      }

      // MDD 티어에 따른 전체 색조
      if (mddTier === 'RED') { r *= 1.5; g *= 0.3; b *= 0.3 }
      else if (mddTier === 'ORANGE') { r *= 1.3; g *= 0.7; b *= 0.5 }
      else if (mddTier === 'YELLOW') { r *= 1.1; g *= 1.1; b *= 0.6 }

      colors[i * 3] = Math.min(1, r)
      colors[i * 3 + 1] = Math.min(1, g)
      colors[i * 3 + 2] = Math.min(1, b)
    }

    pos.needsUpdate = true
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    geo.computeVertexNormals()
  })

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} metalness={0.1} roughness={0.8} transparent opacity={0.7} />
    </mesh>
  )
}

// 메인 3D 씬
function Scene3D({ state, positions }: { state: PortfolioState; positions: [number, number, number][] }) {
  const [selectedAsset, setSelectedAsset] = useState<number | null>(null)

  return (
    <>
      {/* 조명 */}
      <ambientLight intensity={0.35} />
      <pointLight position={[10, 12, 10]} intensity={0.8} color="#ffffff" />
      <pointLight position={[-8, 8, -8]} intensity={0.4} color="#4488ff" />
      <spotLight position={[0, 15, 0]} angle={0.4} penumbra={1} intensity={0.3} color="#00d4ff" />

      {/* 안개 효과 */}
      <fog attach="fog" args={['#0a0a1a', 18, 40]} />

      {/* 자산 구체들 */}
      {state.assets.map((asset, i) => (
        <AssetSphere
          key={asset.id}
          asset={asset}
          position={positions[i]}
          isSelected={selectedAsset === i}
          onSelect={() => setSelectedAsset(selectedAsset === i ? null : i)}
        />
      ))}

      {/* 상관관계 연결선 */}
      <CorrelationLinks assets={state.assets} positions={positions} />

      {/* 리스크 지형 */}
      <RiskTerrain volatility={state.currentVolatility} mddTier={state.mddTier} />

      {/* 그리드 바닥 */}
      <gridHelper args={[20, 20, '#1e293b', '#0f172a']} position={[0, -2.6, 0]} />

      {/* 카메라 컨트롤 (터치 최적화) */}
      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={6}
        maxDistance={25}
        minPolarAngle={0.2}
        maxPolarAngle={Math.PI / 2.2}
        target={[0, 0, 0]}
        touches={{ ONE: 0, TWO: 1 }}
      />
    </>
  )
}


// ============================================================
// SECTION 8: 2D DASHBOARD UI COMPONENTS
// ============================================================

const REGIME_CONFIG: Record<MarketRegime, { label: string; color: string; bg: string }> = {
  BULL: { label: 'BULL (강세)', color: 'text-green-400', bg: 'bg-green-400/10' },
  BEAR: { label: 'BEAR (약세)', color: 'text-red-400', bg: 'bg-red-400/10' },
  VOLATILE: { label: 'VOLATILE (변동)', color: 'text-yellow-400', bg: 'bg-yellow-400/10' },
  STABLE: { label: 'STABLE (안정)', color: 'text-cyan-400', bg: 'bg-cyan-400/10' },
}

const TIER_CONFIG: Record<MddTier, { label: string; color: string; bg: string }> = {
  GREEN: { label: 'GREEN', color: 'text-green-400', bg: 'bg-green-400/10' },
  YELLOW: { label: 'YELLOW', color: 'text-yellow-400', bg: 'bg-yellow-400/10' },
  ORANGE: { label: 'ORANGE', color: 'text-orange-400', bg: 'bg-orange-400/10' },
  RED: { label: 'RED', color: 'text-red-400', bg: 'bg-red-400/10' },
}

const AGENT_CONFIG: Record<AgentRole, { label: string; icon: string; color: string }> = {
  CIO: { label: 'CIO (전략)', icon: '📊', color: 'text-blue-400' },
  RISK_MANAGER: { label: 'Risk Manager', icon: '🛡️', color: 'text-amber-400' },
  CFO: { label: 'CFO (자원배분)', icon: '💰', color: 'text-emerald-400' },
}

// 포트폴리오 메트릭스 헤더
function MetricsHeader({ state }: { state: PortfolioState }) {
  const regimeCfg = REGIME_CONFIG[state.marketRegime]
  const tierCfg = TIER_CONFIG[state.mddTier]

  return (
    <div className="bg-navy-800/80 border-b border-slate-700/50 px-4 py-3 flex items-center justify-between flex-wrap gap-2">
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-bold text-white tracking-wide">AI Investment Command Center</h1>
        <span className={`px-2 py-0.5 rounded text-xs font-semibold ${regimeCfg.color} ${regimeCfg.bg} border border-current/20`}>
          {regimeCfg.label}
        </span>
      </div>
      <div className="flex items-center gap-4 text-sm">
        <div className="text-center">
          <div className="text-slate-400 text-[10px]">총 자산</div>
          <div className="text-white font-bold">{(state.totalValue / 100000000).toFixed(2)}억</div>
        </div>
        <div className="text-center">
          <div className="text-slate-400 text-[10px]">일간 수익률</div>
          <div className={state.dailyReturn >= 0 ? 'text-green-400 font-bold' : 'text-red-400 font-bold'}>
            {state.dailyReturn >= 0 ? '+' : ''}{(state.dailyReturn * 100).toFixed(3)}%
          </div>
        </div>
        <div className="text-center">
          <div className="text-slate-400 text-[10px]">누적 수익률</div>
          <div className={state.cumulativeReturn >= 0 ? 'text-green-400 font-bold' : 'text-red-400 font-bold'}>
            {state.cumulativeReturn >= 0 ? '+' : ''}{(state.cumulativeReturn * 100).toFixed(2)}%
          </div>
        </div>
        <div className="text-center">
          <div className="text-slate-400 text-[10px]">변동성</div>
          <div className="text-cyan-400 font-bold">{(state.currentVolatility * 100).toFixed(1)}%</div>
        </div>
        <div className="text-center">
          <div className="text-slate-400 text-[10px]">Sharpe</div>
          <div className="text-white font-bold">{state.sharpeRatio.toFixed(2)}</div>
        </div>
        <div className="text-center">
          <div className="text-slate-400 text-[10px]">MDD</div>
          <div className={`font-bold ${tierCfg.color}`}>{(state.currentMDD * 100).toFixed(2)}%</div>
        </div>
        <span className={`px-2 py-0.5 rounded text-xs font-semibold ${tierCfg.color} ${tierCfg.bg}`}>
          {tierCfg.label}
        </span>
      </div>
    </div>
  )
}

// 자산 배분 바 차트
function AllocationPanel({ assets }: { assets: AssetConfig[] }) {
  return (
    <div className="bg-navy-800/60 rounded-xl border border-slate-700/40 p-3">
      <h3 className="text-xs font-semibold text-slate-400 mb-2 tracking-wider">ASSET ALLOCATION</h3>
      <div className="space-y-1.5">
        {assets.map(asset => (
          <div key={asset.id} className="flex items-center gap-2 text-xs">
            <div className="w-16 text-slate-300 truncate">{asset.labelKo}</div>
            <div className="flex-1 h-4 bg-slate-800 rounded-full overflow-hidden relative">
              {/* 현재 비중 */}
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{ width: `${asset.currentWeight * 100}%`, backgroundColor: asset.color }}
              />
              {/* 목표 비중 마커 */}
              <div
                className="absolute top-0 h-full w-0.5 bg-white/40"
                style={{ left: `${asset.targetWeight * 100}%` }}
              />
            </div>
            <div className="w-12 text-right font-mono" style={{ color: asset.color }}>
              {(asset.currentWeight * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
      {/* 범례 */}
      <div className="flex items-center gap-3 mt-2 text-[10px] text-slate-500">
        <span className="flex items-center gap-1"><span className="w-3 h-1 bg-blue-500 rounded" /> 현재</span>
        <span className="flex items-center gap-1"><span className="w-0.5 h-3 bg-white/40" /> 목표</span>
      </div>
    </div>
  )
}

// 포트폴리오 가치 스파크라인
function ValueSparkline({ history }: { history: number[] }) {
  if (history.length < 2) return null
  const min = Math.min(...history)
  const max = Math.max(...history)
  const range = max - min || 1
  const w = 280
  const h = 60

  const points = history.map((v, i) => {
    const x = (i / (history.length - 1)) * w
    const y = h - ((v - min) / range) * h
    return `${x},${y}`
  }).join(' ')

  const isUp = history[history.length - 1] >= history[0]

  return (
    <div className="bg-navy-800/60 rounded-xl border border-slate-700/40 p-3">
      <h3 className="text-xs font-semibold text-slate-400 mb-1 tracking-wider">PORTFOLIO VALUE</h3>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-16">
        <defs>
          <linearGradient id="sparkGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={isUp ? '#00c087' : '#f6465d'} stopOpacity="0.3" />
            <stop offset="100%" stopColor={isUp ? '#00c087' : '#f6465d'} stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon
          points={`0,${h} ${points} ${w},${h}`}
          fill="url(#sparkGrad)"
        />
        <polyline
          points={points}
          fill="none"
          stroke={isUp ? '#00c087' : '#f6465d'}
          strokeWidth="1.5"
        />
      </svg>
    </div>
  )
}

// MDD 차트
function MDDChart({ history, peakValues }: { history: number[]; peakValues: number[] }) {
  if (history.length < 2) return null
  const ddValues: number[] = []
  let peak = history[0]
  for (const v of history) {
    if (v > peak) peak = v
    ddValues.push((v - peak) / peak)
  }

  const minDD = Math.min(...ddValues, MDD_LIMIT)
  const w = 280
  const h = 50

  const points = ddValues.map((dd, i) => {
    const x = (i / (ddValues.length - 1)) * w
    const y = (-dd / (Math.abs(minDD) + 0.01)) * h
    return `${x},${y}`
  }).join(' ')

  const limitY = (-MDD_LIMIT / (Math.abs(minDD) + 0.01)) * h

  return (
    <div className="bg-navy-800/60 rounded-xl border border-slate-700/40 p-3">
      <h3 className="text-xs font-semibold text-slate-400 mb-1 tracking-wider">DRAWDOWN</h3>
      <svg viewBox={`0 0 ${w} ${h + 5}`} className="w-full h-14">
        {/* MDD 한도선 */}
        <line x1="0" y1={limitY} x2={w} y2={limitY} stroke="#f6465d" strokeWidth="1" strokeDasharray="4,3" opacity="0.6" />
        <text x={w - 2} y={limitY - 3} fill="#f6465d" fontSize="8" textAnchor="end" opacity="0.7">-5%</text>
        {/* 드로우다운 영역 */}
        <polygon
          points={`0,0 ${points} ${w},0`}
          fill="rgba(246,70,93,0.15)"
        />
        <polyline
          points={points}
          fill="none"
          stroke="#f6465d"
          strokeWidth="1.2"
          opacity="0.8"
        />
      </svg>
    </div>
  )
}

// 교차검증 패널
function CrossCheckPanel({ checks }: { checks: CrossCheck[] }) {
  return (
    <div className="bg-navy-800/60 rounded-xl border border-slate-700/40 p-3">
      <h3 className="text-xs font-semibold text-slate-400 mb-2 tracking-wider">CROSS-CHECK (3/5 필요)</h3>
      <div className="space-y-1">
        {checks.map((check, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span className={
              check.status === 'PASS' ? 'text-green-400' :
              check.status === 'WARN' ? 'text-yellow-400' : 'text-red-400'
            }>
              {check.status === 'PASS' ? '●' : check.status === 'WARN' ? '▲' : '✕'}
            </span>
            <span className="text-slate-300 w-24 truncate">{check.nameKo}</span>
            <span className="text-slate-500 text-[10px] flex-1 truncate">{check.detail}</span>
          </div>
        ))}
        {checks.length === 0 && <div className="text-slate-500 text-xs">대기 중...</div>}
      </div>
    </div>
  )
}

// AI 에이전트 의사결정 로그
function AgentLogPanel({ log }: { log: AgentDecision[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [log.length])

  return (
    <div className="bg-navy-800/60 rounded-xl border border-slate-700/40 p-3 flex flex-col" style={{ maxHeight: 280 }}>
      <h3 className="text-xs font-semibold text-slate-400 mb-2 tracking-wider">AI GOVERNANCE LOG</h3>
      <div ref={scrollRef} className="scrollable-panel space-y-2 flex-1 overflow-y-auto pr-1">
        {log.slice(-15).map(decision => {
          const cfg = AGENT_CONFIG[decision.agent]
          return (
            <div key={decision.id} className="bg-slate-800/50 rounded-lg p-2 text-xs border border-slate-700/30">
              <div className="flex items-center justify-between mb-1">
                <span className={`font-semibold ${cfg.color}`}>
                  {cfg.icon} {cfg.label}
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-slate-500 text-[10px]">
                    신뢰도 {(decision.confidence * 100).toFixed(0)}%
                  </span>
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                    decision.approved ? 'bg-green-400/15 text-green-400' : 'bg-red-400/15 text-red-400'
                  }`}>
                    {decision.approved ? 'APPROVED' : 'REJECTED'}
                  </span>
                </div>
              </div>
              <div className="text-slate-400 leading-relaxed">{decision.reasoning}</div>
              {decision.worstCaseImpact !== undefined && (
                <div className="text-red-400/70 text-[10px] mt-1">
                  최악 시나리오: {(decision.worstCaseImpact * 100).toFixed(2)}%
                </div>
              )}
            </div>
          )
        })}
        {log.length === 0 && <div className="text-slate-500 text-xs">에이전트 초기화 중...</div>}
      </div>
    </div>
  )
}

// 상관관계 히트맵
function CorrelationHeatmap({ assets }: { assets: AssetConfig[] }) {
  return (
    <div className="bg-navy-800/60 rounded-xl border border-slate-700/40 p-3">
      <h3 className="text-xs font-semibold text-slate-400 mb-2 tracking-wider">CORRELATION MATRIX</h3>
      <div className="grid gap-0.5" style={{ gridTemplateColumns: `40px repeat(${assets.length}, 1fr)` }}>
        {/* 헤더 행 */}
        <div />
        {assets.map(a => (
          <div key={a.id} className="text-[8px] text-slate-500 text-center truncate px-0.5">{a.label.split(' ')[0]}</div>
        ))}
        {/* 데이터 행 */}
        {assets.map((a, i) => (
          <React.Fragment key={a.id}>
            <div className="text-[8px] text-slate-500 text-right pr-1 leading-5 truncate">{a.label.split(' ')[0]}</div>
            {assets.map((_, j) => {
              const corr = CORRELATION_MATRIX[i][j]
              const absCorr = Math.abs(corr)
              const bg = i === j
                ? 'bg-slate-600'
                : corr < -0.1
                  ? `bg-green-500`
                  : corr > 0.5
                    ? `bg-red-500`
                    : `bg-yellow-500`
              return (
                <div
                  key={`${i}-${j}`}
                  className={`h-5 rounded-sm flex items-center justify-center text-[7px] font-mono text-white ${bg}`}
                  style={{ opacity: i === j ? 0.3 : 0.15 + absCorr * 0.7 }}
                  title={`${a.label} / ${assets[j].label}: ${corr.toFixed(2)}`}
                >
                  {absCorr >= 0.2 ? corr.toFixed(1) : ''}
                </div>
              )
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}

// ============================================================
// SECTION 9: MAIN APP COMPONENT
// ============================================================

export default function App() {
  const [state, setState] = useState<PortfolioState>(createInitialState)
  const [isRunning, setIsRunning] = useState(true)
  const stateRef = useRef(state)
  stateRef.current = state

  // 3D 위치 계산 (MDS 기반, 초기 1회)
  const positions = useMemo(() => correlationToPositions(CORRELATION_MATRIX), [])

  // 시뮬레이션 루프
  useEffect(() => {
    if (!isRunning) return
    const interval = setInterval(() => {
      setState(prev => {
        let next = simulateDay(prev)

        // 5틱마다 거버넌스 파이프라인 실행
        if (next.tick % 5 === 0 && next.tick > 0) {
          const { newWeights, decisions, crossChecks } = runGovernancePipeline(next)
          next = {
            ...next,
            assets: next.assets.map((a, i) => ({ ...a, currentWeight: newWeights[i] })),
            agentLog: [...next.agentLog.slice(-50), ...decisions],
            crossChecks,
          }
        }

        return next
      })
    }, 2000)
    return () => clearInterval(interval)
  }, [isRunning])

  // 현재 한국 시간
  const koreaTime = new Date().toLocaleString('ko-KR', { timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit', second: '2-digit' })

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0a1a] overflow-hidden">
      {/* 헤더 */}
      <MetricsHeader state={state} />

      {/* 메인 콘텐츠 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 좌측: 3D 캔버스 (60%) */}
        <div className="flex-[3] relative touch-canvas">
          <Canvas
            camera={{ position: [0, 8, 14], fov: 50 }}
            gl={{ antialias: true, alpha: false }}
            onCreated={({ gl }) => {
              gl.setClearColor('#0a0a1a')
              gl.toneMapping = THREE.ACESFilmicToneMapping
              gl.toneMappingExposure = 1.2
            }}
          >
            <Scene3D state={state} positions={positions} />
          </Canvas>

          {/* 3D 위 오버레이 정보 */}
          <div className="absolute top-3 left-3 flex items-center gap-2">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800/80 border border-slate-600/50 text-slate-200 hover:bg-slate-700/80 transition-colors backdrop-blur-sm"
            >
              {isRunning ? '⏸ 일시정지' : '▶ 재개'}
            </button>
            <span className="text-[10px] text-slate-500 bg-slate-800/60 px-2 py-1 rounded backdrop-blur-sm">
              Day {state.tick} | {koreaTime} KST
            </span>
          </div>

          {/* 범례 */}
          <div className="absolute bottom-3 left-3 bg-slate-900/80 border border-slate-700/50 rounded-lg p-2 text-[10px] text-slate-400 backdrop-blur-sm">
            <div className="flex items-center gap-1 mb-0.5"><span className="w-3 h-0.5 bg-green-400 rounded" /> 음의 상관 (분산효과)</div>
            <div className="flex items-center gap-1 mb-0.5"><span className="w-3 h-0.5 bg-red-400 rounded" /> 강한 양의 상관</div>
            <div className="flex items-center gap-1"><span className="w-3 h-0.5 bg-yellow-400 rounded" /> 중간 상관</div>
          </div>
        </div>

        {/* 우측: 대시보드 패널 (40%) */}
        <div className="flex-[2] border-l border-slate-700/50 overflow-y-auto scrollable-panel p-3 space-y-3 bg-[#0b0b1e]">
          <AllocationPanel assets={state.assets} />
          <ValueSparkline history={state.valueHistory} />
          <MDDChart history={state.valueHistory} peakValues={[]} />
          <CrossCheckPanel checks={state.crossChecks} />
          <CorrelationHeatmap assets={state.assets} />
          <AgentLogPanel log={state.agentLog} />
        </div>
      </div>
    </div>
  )
}
