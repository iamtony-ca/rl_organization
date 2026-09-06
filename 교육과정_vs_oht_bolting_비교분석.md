# 교육과정 RL vs. oht_bolting(UR 휠 insert) RL — 차이 분석

> **한 줄 요약:** 교육과정은 **보행(locomotion) RL의 기초**를 가르쳤고,
> `oht_bolting`은 그 위에서 **산업용 contact-rich 매니퓰레이션(peg-in-hole
> 휠 삽입/볼트 체결) RL**을 실제 배포 목표로 시도한 프로젝트다.
> PPO라는 **알고리즘 뼈대는 같지만, 문제의 종류(class)가 완전히 다르다.**
> 교육과정이 *개념으로만* 스쳤던 지점(접촉 물리, 임피던스 제어, 보상 해킹,
> sim2real)이 `oht_bolting`에서는 *전부 실전 벽*으로 등장했다.

*(근거: `oht_bolting/README.md` 1731줄 전체 + `rl/*.py` 소스 직접 확인.
교육과정 근거: `isaacsim_tutorials/{24,26,29,35,36}` + `g1_*_env.py`.)*

---

## 0. 가장 큰 그림

| | 교육과정 (Part 4~6) | oht_bolting |
|---|---|---|
| 문제 종류 | **Locomotion**(균형·보행) | **Manipulation**(접촉 조립) |
| 로봇 | 카트폴 → ANYmal(4족) → G1(휴머노이드) | **UR16e 6축 팔 + 그리퍼**(고정 베이스) |
| 목표 | 교육·데모 (실물 배포 없음) | **실물 배포**(고객: UR16e + OnRobot HEX-E F/T + 2FG14 + ESTIC 드라이버) |
| 접촉 | 없음(강체 보행) | **핵심**(0.3mm 공차 peg-in-hole) |
| 성숙도 | 완성된 교보재 | **연구/개발 중**(Phase 1~9, 수십 회 학습 반복) |

핵심: **교육과정은 "학습이 되게 만드는 법"을, oht_bolting은 "접촉을 실물에서
되게 만드는 법"을 다룬다.** 후자가 훨씬 어렵고, 교육과정은 그 관문(Part 4의
도메인 랜덤화, Part 6의 보상 설계 실패 사례)만 개념으로 예고했다.

---

## 1. 환경 아키텍처 — ManagerBased vs Direct ⭐ (가장 근본적 차이)

- **교육과정: `ManagerBasedRLEnv` + `@configclass` 매니저**
  (Action / Observation / Reward / Event / Termination 을 선언적으로 조립).
  → `g1_standing_env.py`, `g1_walking_env.py`가 정확히 이 패턴.
  MDP를 "레고 블록"처럼 끼우는 **고수준·선언형** API.

- **oht_bolting: `DirectRLEnv`** (모든 env가 이걸 상속)
  `oht_bolting_env.py:47` → `class OHTBoltingEnv(DirectRLEnv)`,
  `factory_env.py:25` → `class FactoryEnv(DirectRLEnv)`.
  → `_apply_action` / `_get_observations` / `_get_rewards` / `_get_dones`를
  **직접 구현**하는 **저수준·명령형** API.

> **왜 갈렸나:** 매 물리 스텝마다 접촉력을 읽고, 운동학적 helical 전진을
> 수동으로 쓰고, 게이트를 판정하고, 임피던스 토크를 J^T로 매핑하는 등의
> "스텝 내부 개입"이 필요하다. Manager 추상화로는 이런 세밀한 제어가 어색해서
> **Direct로 갈 수밖에 없다.** → 교육과정에서 배운 Manager 패턴은
> oht_bolting에 **그대로는 안 쓰이고**, MDP 사고방식만 이어진다.

---

## 2. 물리(Physics) — 이 프로젝트의 심장 ⭐⭐⭐ (교육과정엔 아예 없음)

교육과정 보행은 **표준 강체 PhysX**로 충분하다. 접촉 조립이 없기 때문이다.
G1의 "모션 모방"조차 접촉이 아니라 **레퍼런스 관절각 추종(kinematic)**이다.

oht_bolting은 정반대로, **접촉 물리를 어떻게 시뮬레이션하느냐가 프로젝트 전체를
좌우**했다 (README Phase 1~1.5):

| 방식 | 물리 정확도 | 속도 | 결론 |
|---|---|---|---|
| **SDF 접촉**(실제 나사산 물리) | 진짜 | **~80 env-step/s** | 학습엔 불가(100M 스텝 = 14일) |
| **Kinematic helical**(각도→전진 공식) | 가짜(애니메이션) | ~9,600 env-step/s | 학습은 가능하나 **실물 물리와 다른 영역** |

- Phase 1.5 벤치: helical이 SDF보다 **~100배 빠름** → 학습은 helical로 강제.
- 그 대가: helical로 학습한 정책이 **SDF에서 zero-shot 0% 성공**
  (README Phase 3E). "빠르지만 가짜 / 느리지만 진짜"의 딜레마.
- README의 뼈아픈 문장: *"kinematic 위에 마찰을 얹는 건 애니메이션에 마찰을
  볼트로 붙이는 격"*(Phase 3F).

> 교육과정에는 이 **속도 vs 물리충실도 트레이드오프 자체가 존재하지 않는다.**
> 보행은 강체로 빠르고 정확하기 때문. 이게 매니퓰레이션이 어려운 근본 이유.

---

## 3. 제어(Control) / 액션 공간

- **교육과정:** ActionManager로 관절 위치/속도/토크 명령(`JointPositionActionCfg`
  등). 논쟁의 여지가 없다.
- **oht_bolting:** 액션 공간을 두고 **긴 싸움**.
  - Cartesian **operational-space 임피던스**(Factory `compute_dof_torque`,
    effort-mode 팔) 시도 → **UR16e에서 구조적으로 불안정**
    (Phase 7: 무동작에도 fingertip이 **7.4mm/step 진동**).
  - **관절 위치 델타**(`joint_action_scale=0.025`)로 회귀 → 진동 0.0mm/step,
    성공률 51.9% → **99.9%**. 이건 NVIDIA가 검증한 UR sim2real 레시피와 일치.

> 교훈(README Phase 7): *"kinematic/저접촉 구간엔 관절 위치 제어가 더 간단하고
> sim2real 검증된 경로. 임피던스는 진짜 접촉 구간에만, 그것도 검증된 컨트롤러로."*
> — 이 "컨트롤러 선택 전쟁"은 **매니퓰레이션 고유의 문제**로 교육과정엔 없다.

---

## 4. RL 라이브러리 / 네트워크

| | 교육과정 | oht_bolting |
|---|---|---|
| 라이브러리 | **rsl_rl** 만 | rsl_rl **+ rl_games**(Factory/Forge) |
| 네트워크 | MLP [32,32]→[128,128,128]→[256,128,64] | rsl_rl: MLP [128,128,64] / **Factory: [512,128,64] + LSTM 1024** |
| recurrence | 없음 | **있음**(LSTM — 접촉의 부분관측성 대응) |

> Factory/Forge 경로는 **recurrent(LSTM) 정책**을 쓴다. 접촉 상태는 관측만으론
> 부분관측(partially observable)이라 시간 기억이 필요하기 때문. 교육과정의
> 보행 정책은 순수 MLP로 충분했다.

---

## 5. 보상 설계(Reward) — 교육과정이 "예고"하고 실전이 "체벌"한 지점 ⭐⭐⭐

교육과정 Part 6의 **쿵후 실패 사례**가 가르친 정렬(alignment) 문제
— *"학습 시스템은 의도가 아니라 실제 부여된 보상을 최적화한다"* — 를
oht_bolting은 **수십 회에 걸쳐 온몸으로 재현**했다:

| 교육과정이 말한 것 | oht_bolting이 실제로 겪은 것 |
|---|---|
| 보상은 "프로그래밍 언어", 오류 관용도 낮음 | ✅ Phase 3D~9, 20+ 학습 반복이 전부 보상 튜닝 |
| 잘못된 보상 → 의도와 다른 행동 | **attractor lock-in**: "잡고 가만히"(v1/v2), **"멀리 도망"(v4/3I)** — 보상만 챙기고 일 안 함 |
| — (교육과정엔 없던 함정) | **termination-avoidance trap**(3p): 성공하면 에피소드가 끝나 남은 보상을 잃으니 **성공선 바로 아래서 영원히 대기** |
| — | **gate-cheating**(3q): 벌점을 피하려 그리퍼를 열어 휠을 던져버림 |

> 즉 **교육과정 쿵후 사례 = oht_bolting이 부딪힌 벽의 축소판.** 교육이 개념으로
> 보여준 걸, 실전에서는 "게이트를 보상에도 벌점에도 동시에 걸어라(gate the
> carrot AND the stick)" 같은 구체 기법으로 하나씩 뚫어야 했다.

---

## 6. 도메인 랜덤화 & Sim2Real

- **교육과정:** Part 4 lesson 29에서 DR을 **개념으로** 가르침
  (`EventTermCfg` startup/reset/interval). 단, **G1 보행엔 DR이 없음**
  (`enable_corruption=False`) — 교보재는 "그래서 실물 전이는 별도"라고만 언급.
- **oht_bolting:** sim2real이 **존재 이유**.
  - 볼트 포즈 ±10mm·축 ±5° 틸트, 관측 가우시안 노이즈(Phase 3B).
  - 액추에이터를 UR **servoj 프로파일**(K/D, effort limit)에 맞춰 모델링.
  - F/T 센서(HEX-E QC) 관측 채널 스코핑, 정직한 sim2real gap 분석.

> 교육과정 Part 4의 DR 수업이 **정확히 이 프로젝트에서 쓴 도구**다. 다만
> oht_bolting은 접촉·F/T·임피던스까지 **훨씬 멀리** 나갔다.

---

## 7. 계층형 FSM + RL 하이브리드 (교육과정엔 전무)

- **교육과정:** 스킬마다 **순수 end-to-end RL**. FSM 없음.
- **oht_bolting:** **FSM이 전체를 오케스트레이션**하고 RL은 접촉 구간만 담당.
  `APPROACH → INSERT(RL) → SETTLE → RELEASE → RETRACT → DONE`
  (`wheel_cycle_fsm.py`). 자유공간 이동은 Lula-IK/cuMotion(고전 제어),
  접촉 삽입만 학습 정책. **long-horizon 산업 태스크의 정석**.

> "하나의 AI로 다 풀지 말고, 자유공간(고전 제어) + 접촉(RL)을 분리" —
> `docs/design/design.md`의 핵심 설계 원칙. 교육과정은 이 계층 개념을 다루지 않는다.

---

## 8. NVIDIA 연구 자산 활용

- **교육과정:** 자기완결적 교보재(사전등록 gym 태스크 정도만 재사용).
- **oht_bolting:** **NVIDIA Factory**(RSS 2022, 접촉 조립) 포팅 → **Forge**
  (F/T 관측 + 성공 예측 + DR) 상속, **AutoMate**·**Isaac ROS Gear Assembly UR
  워크플로우** 참조. 연구 논문 수준의 재사용.

---

## 정리 — 교육과정이 준비시킨 것 vs 스스로 넘어야 했던 것

**교육과정이 실제로 밑거름이 된 부분**
- MDP 사고(관측/행동/보상/종료 설계) — Manager는 아니어도 사고방식은 그대로.
- PPO + rsl_rl, GPU 병렬 학습, TensorBoard, 체크포인트 운용.
- **도메인 랜덤화 개념**(Part 4) → 그대로 활용.
- 모션 모방/레퍼런스 추종 아이디어.
- **결정적으로: 쿵후 실패 사례의 정렬/보상해킹 교훈** → 실전에서 반복해 부딪힘.

**교육과정 밖이라 스스로 익혀야 했던 부분(격차)**
- `DirectRLEnv`(저수준 API), 접촉 물리(**SDF vs kinematic 100배 트레이드오프**),
- Cartesian 임피던스/operational-space 제어와 그 불안정성,
- F/T 센싱, 0.3mm peg-in-hole 조립, **FSM+RL 계층 하이브리드**,
- rl_games + LSTM, **Factory/Forge** 프레임워크, 실물 매니퓰레이터 sim2real.

> **결론:** 교육과정은 *보행 RL로 "RL을 돌아가게 하는 법"*을 탄탄히 가르쳤고,
> `oht_bolting`은 그 토대 위에서 *"접촉을 실물에서 되게 하는 법"*이라는 한 단계
> 위 난이도의 산업 문제에 도전했다. 둘의 공통 뼈대는 PPO와 MDP 설계 사고이며,
> 교육과정이 개념으로 예고한 **도메인 랜덤화·보상 설계 함정**이 실전에서
> 정확히 가장 큰 벽으로 나타났다.
