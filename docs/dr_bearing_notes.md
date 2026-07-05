# DR-Bearing Phase 0 Notes

## 1. fly() 루프 구조 (nav.py:646~915)

### 세 가지 위치 변수

| 변수 | 의미 | 갱신 시점 |
|---|---|---|
| `cur_point_real` | 드론의 실제 물리 위치 (r_i) | 매 스텝 끝에 `next_point_real`로 교체 |
| `cur_point_name` | 드론이 자신이 있다고 믿는 명목 위치 (n_i) | 매 스텝 끝에 `next_point_name`으로 교체 |
| `cur_point_pred` | 모델이 이번 스텝에서 추정한 위치 (p_i) | step 02에서 계산, step 03 heading 계산의 입력 |

`cur_point_real` — 연속적(매 스텝 +25m 물리 이동)  
`cur_point_name` — 불연속(= p_i + cur_step, 예측 오차만큼 점프 가능)  
`cur_point_pred` — step 02에서만 존재하는 일시적 값, **DR-Bearing이 교체할 대상**

### 5단계 루프 구조

```
while True:  (waypoint 도달까지 반복)
  [step 01] 초기화: cur_point_real, cur_point_name, fly_angle_ccs 출력
  [step 02] 추론: phreg() → pos_pred → cur_point_pred   ← DR-Bearing 개입 지점
  [step 03] heading 갱신: cur_point_pred → target_waypoint 방향으로 fly_angle 재계산
  [step 04] 이동: cur_step 계산 → next_point_real, next_point_name 갱신
  [step 05] 도착 판정: distance_wp < th_arrive? → waypoint_index++, break
```

**DR-Bearing 개입 지점**: step 02 직후, `cur_point_pred` 생성 뒤 step 03 전.  
step 03~05는 무수정.

---

## 2. 추론 호출 체인

```
fly()
  └─ phreg(uav_frame_id, fly_angle_ccs, cur_point_real, block_center, block_indices)
       ├─ get_patches(...)  →  patches[5], patches_fdirs, record
       └─ position_angle_regression_(patches)
            ├─ nav_transform(patch) × 5  →  tensor_patches
            ├─ torch.stack → patches_tensor [1, 5, C, H, W]
            ├─ with torch.no_grad():
            │    pos_pred, dir_pred = self.model(patches_tensor)
            └─ return pos_pred[0], dir_pred[0]  (numpy)
```

`self.model` = `PARCASGM_v5a` 인스턴스 (nav.py 로딩 시 결정)  
`phreg()`의 반환: `pos_pred, dir_pred, flag_out_of_map, patches_fdirs, record`

---

## 3. 모델 forward 및 PSG α 위치

**파일**: `cvphr/models/posaglreg/models.py`

### SimilarityPositionPrior (PSG, 논문 Eq.7)
```python
# line 73~82
def forward(self, ft, neighbor_feats):
    fi_expand = ft.unsqueeze(1).expand(-1, 4, -1)   # [B, 4, D]
    sim = self.cos(fi_expand, neighbor_feats)         # [B, 4]
    weights = torch.softmax(sim, dim=1)               # [B, 4]  ← 이게 α
    pos_prior = (weights.unsqueeze(2) * self.rel_coords).sum(1)  # [B, 2]
    return pos_prior   # α는 버려짐
```

`weights` (= 논문의 α, shape `[B, 4]`)가 현재 소비되고 반환되지 않음.  
**entropy(α)가 높을수록 4개 RST 중 어디인지 모름 → uncertainty 높음.**

### PARCASGM_v5.forward (line 353~394)
```python
pos_soft_prior = self.sim_pos_prior(uav_patch_feature, neighbor_feats)  # [B,2], α 버려짐
ctx_feat = self.neighbors_cross_attn(uav_patch_feature, neighbor_feats)  # CA
combined_with_prior = torch.cat([uav_patch_feature, ctx_feat, pos_soft_prior], dim=1)
pos_pred = self.pos_regressor(combined_with_prior)
dir_pred = self.dir_regressor(combined)
return pos_pred, dir_pred
```

CA softmax 위치: `NeighborsCrossAttention.forward` line 56:
```python
attn = torch.bmm(q, k.transpose(1,2)) / sqrt(d)
attn = F.softmax(attn, dim=-1)   # [B, 1, 4]
```

---

## 4. get_patches() — UVP 생성 방식

```
입력: uav_frame_id, fly_angle_ccs, next_point_real(실제 위치), block_center, block_indices
```

**4개 RST 생성**:
- 4096×4096 RSI 이미지에서 block_indices → start_x, start_y 계산
- 512×512 블록을 256×256씩 4분할 → p1, p2, p3, p4

**UVP(target patch) 생성**:
- `next_point_real` → 픽셀 좌표 변환
- `crop_target_patch(x, y, fly_angle_ccs, cropped_img)` 호출
  - `get_rotated_img()` 기반: 현재 heading으로 회전된 256×256 패치 잘라냄
  - 패치 크기: `PATCH_SIZE` px (= 256px × 0.25m/px = 64m 시야)

**연속 프레임 겹침**:
- 스텝 25m, 시야 64m → 인접 프레임 겹침 비율 ≈ (64-25)/64 ≈ **61%**
- 실제 VO에서 feature 매칭 충분히 가능한 수준

**반환**: `patches[5]` (RST×4 + UVP×1), `patches_fdirs[5]` (파일 경로), `record` (메타)

---

## 5. 항법 실행 진입점

**실행 커맨드**:
```bash
python -m naver.runners.nav --uav_2d3d 2d --rsi_id 34bc --traj_id 50
```

**`main_nav_test()` (line 916~)**: 파라미터 조합 루프 → `UAVNavigation` 인스턴스 생성 → `.fly()` 호출

**`parse_args()` (line ~1060~1162)**: argparse 기반. 주요 파라미터:
- `--uav_step` (default: 25)
- `--th_arrive` (default: 20)
- `--rsi_id` (default: ["34bc"])
- `--traj_id` (default: [50])
- `--uav_2d3d` (default: "2d")
- `--nav_test`, `--cvphr_test`, `--suppl_test`: 프리셋 모드 스위치

**`--nav_test` 프리셋**: rsi=34bc, traj=50, step=25, th=20 (단일 빠른 테스트)  
**`--cvphr_test` 프리셋**: 4도시 × 2경로 × 3스텝 × 2임계값 전체 실험

---

## 6. α를 밖으로 꺼내기 위해 수정할 파일:함수 목록

| 우선순위 | 파일 | 함수 | 수정 내용 |
|---|---|---|---|
| 1 | `cvphr/models/posaglreg/models.py` | `SimilarityPositionPrior.forward()` | `weights`(α)를 `pos_prior`와 함께 반환, 또는 `self.last_alpha`에 저장 |
| 2 | `cvphr/models/posaglreg/models.py` | `PARCASGM_v5.forward()` | `sim_pos_prior()` 결과에서 α를 꺼내 `self.last_alpha` 저장 (권장) |
| 3 | `naver/runners/nav.py` | `position_angle_regression_()` | `torch.no_grad()` 블록 직후 `model.last_alpha.cpu().numpy()` 수집 후 반환 |
| 4 | `naver/runners/nav.py` | `phreg()` | α를 반환 시그니처에 추가 |
| 5 | `naver/runners/nav.py` | `fly()` step 02 | α → uncertainty 계산 → `nav_step`/`record_dict` 기록 |

**권장 방식**: `self.last_alpha` 저장 (호출부 변경 최소화).  
`PARCASGM_v5.forward()` 반환 시그니처는 건드리지 않으므로 `cvphr_test.py` 등 기존 호출부 무수정.

---

## 7. 기타 파악 사항

- `nav.py` 줄바꿈: LF (macOS 기준 확인 필요, CRLF 주의)
- `calculate_lon_lat_step_underllcs()`: 위경도 벡터 변환 기존 함수, VO 증분 변환에 재사용 가능
- `convert_llcs_to_ccs_vector()`, `convert_ccs_to_llcs_vector()`: 좌표계 변환 기존 함수
- `UAVNavigation.__init__`의 `self.device`: Mac M1이면 `"mps"` 또는 `"cpu"`, CUDA 없음
