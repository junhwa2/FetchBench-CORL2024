## scene config 생성
python scripts/generate_scene_yaml.py --asset-root asset_release_v1.3/Task_260512_mini_small_60/benchmark_val --output-dir InfiniGym/isaacgymenvs/config/scene/benchmark_val_v1.3
python scripts/generate_scene_yaml.py --asset-root asset_release_v1.3/Task_260512_mini_small_60/benchmark_eval --output-dir InfiniGym/isaacgymenvs/config/scene/benchmark_eval_v1.3

## obstruction data 생성 방법
export PYTHONPATH=/home/jo/HJ/FetchBench-CORL2024/InfiniGym:$PYTHONPATH
export ASSET_PATH=/home/jo/HJ/FetchBench-CORL2024/asset_release_v1.3
ln -sfn Task_260512_CellShelf_60_123 $ASSET_PATH/Task
ls -la $ASSET_PATH/Task # 어디를 가리키고 있는지 확인 가능

cd InfiniGym/
python isaacgymenvs/obs_data_gen.py task=FetchMeshCuroboGO scene=benchmark_eval_v1.3/RigidObjCellShelfDesk_0 headless=True task.obs_data_gen.obs_path=Obstruction_mini_test_linear_2cam task.env.cam.num_cam=2 task.obs_data_gen.pc_new_id_method=valid

## obstruction data 확인 및 metadata 생성
cd scripts/
python validate_and_build_obs_metadata.py --root ../asset_release_v1.3/Obstruction_mini_test_linear/benchmark_eval --skip-preview

## baseline evaluation
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method fcl --pitch 0.025 // best
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method fcl
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method grn
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method cbn --robot_to_model_t -0.7 0.0 -0.1 // best
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method cbn
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method scn --robot_to_model_t -0.7 0.0 -0.1 // best
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method scn

## sim run
export PYTHONPATH=/home/jo/HJ/FetchBench-CORL2024/InfiniGym:$PYTHONPATH
export ASSET_PATH=/home/jo/HJ/FetchBench-CORL2024/asset_release_v1.3
ln -sfn Task_260513_60_111 $ASSET_PATH/Task
cd InfiniGym/

python isaacgymenvs/eval.py task=FetchMeshCuroboGORun scene=benchmark_eval_v1.3/RigidObjDesk_0

python isaacgymenvs/obs_data_gen.py task=FetchMeshCuroboGO scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True task.obs_data_gen.obs_path=Obstruction_test task.env.cam.num_cam=1 task.obs_data_gen.pc_new_id_method=valid


eval.py 를 다음과 같이 수정해줘. (collision/feasibility 메트릭 계산 부분)

[A] _collision_metrics_matched 를 양방향 다운샘플로 바꿔라.
  - 기존엔 negative만 다운샘플해 prevalence를 올리기만 했다.
  - natural_prevalence = n_pos/(n_pos+n_neg) 를 계산하고:
      target_prevalence >= natural  -> negative 다운샘플 (n_neg_keep = round(n_pos*(1-t)/t))
      target_prevalence <  natural  -> positive 다운샘플 (n_pos_keep = round(n_neg*t/(1-t)))
  - positive/negative 둘 중 하나라도 0개면 ValueError, 표본 부족 시 ValueError.
  - n_seeds회 리샘플 후 각 _MATCHED_KEYS 메트릭에 대해:
      {k}_matched      = seed 평균
      {k}_matched_std  = seed 표준편차   (신규)
    그리고 matched_n_positive, matched_n_total = 다운샘플 후 표본 수 (신규) 를 반환.

[B] matched CSV 헬퍼를 신규 칼럼에 맞춰 확장하라.
  - _matched_csv_header_cols: target_prevalence + {k}_matched + {k}_matched_std + matched_n_positive + matched_n_total
  - _matched_csv_row_values: 위 순서대로 값 채우기. matched 가 None이면 전부 공란.
  - _fmt_matched_msg: "metric=mean±std" 형식 + matched_n_positive/n_total 출력.

[C] compute_collision_metrics / compute_feasibility_metrics:
  - matched target prevalence 를 "bin/category별 max prevalence" -> "전체 풀링 prevalence(metrics['prevalence'])" 로 변경
    (cat_target_p, tbin_target_p 둘 다).
  - per-category CSV 와 per-t-bin CSV 각각에, 기존 행들 뒤에 't_bin/category = "total"' 행을 1개 추가:
      * 값 = 전체 풀링 데이터 natural 메트릭(이미 계산된 metrics 변수 재사용)
      * n = 전체 행 수, time = 전체 평균
      * per-t-bin 의 경우 t_lo = t_bins[0][1], t_hi = t_bins[-1][2]
      * matched 계열 칼럼은 전부 공란 (_matched_csv_row_values(None, None))
  - collision/feasibility 모두 동일하게 적용.