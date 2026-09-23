"""生成实验设计配置与清单，不读取大数据，不训练模型，不删除文件。

运行：python 实验设计/生成实验清单.py
核验：python 实验设计/生成实验清单.py --check
"""
from pathlib import Path
import argparse
import itertools
import json

ROOT = Path(__file__).resolve().parent
MODALITIES = ['T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV']
RATES = [0.1, 0.3, 0.5, 0.7]
SEEDS = [17, 29, 43]
MASK_SEEDS = [101, 211, 307, 401, 503]

MODELS = [
    dict(id='B1', modalities='T', temporal=True, fusion='single', augmentation='none'),
    dict(id='B2', modalities='A', temporal=True, fusion='single', augmentation='none'),
    dict(id='B3', modalities='V', temporal=True, fusion='single', augmentation='none'),
    dict(id='B4', modalities='TAV', temporal=False, fusion='concat', augmentation='none'),
    dict(id='B5', modalities='TAV', temporal=True, fusion='concat', augmentation='none'),
    dict(id='F00', modalities='TAV', temporal=True, fusion='uniform', augmentation='none'),
    dict(id='F01', modalities='TAV', temporal=True, fusion='gate', augmentation='none'),
    dict(id='F10', modalities='TAV', temporal=True, fusion='uniform', augmentation='iid_count_matched'),
    dict(id='F11', modalities='TAV', temporal=True, fusion='gate', augmentation='iid_count_matched'),
    dict(id='F20', modalities='TAV', temporal=True, fusion='uniform', augmentation='block'),
    dict(id='F21', modalities='TAV', temporal=True, fusion='gate', augmentation='block'),
    dict(id='D1', modalities='TAV', temporal=True, fusion='gate', augmentation='block',
         teacher='F01_same_seed', distill_weight=0.2),
]


def conditions():
    rows = []
    for suite in ['V-select', 'E-main']:
        rows.append(dict(condition_id=f'{suite}_clean', suite=suite,
                         modalities='', rate_nominal=0.0, position='none',
                         mask_seed=None, intervention='feature_post_encoder'))
        rates = [0.3, 0.5] if suite == 'V-select' else RATES
        positions = ['middle'] if suite == 'V-select' else ['start', 'middle', 'end']
        for mods, rate, pos in itertools.product(MODALITIES, rates, positions):
            rows.append(dict(condition_id=f'{suite}_{mods}_r{int(rate*100):02d}_{pos}',
                             suite=suite, modalities=mods, rate_nominal=rate,
                             position=pos, mask_seed=None,
                             intervention='feature_post_encoder'))
    for mods, rate, seed in itertools.product(MODALITIES, RATES, MASK_SEEDS):
        rows.append(dict(condition_id=f'E-random_{mods}_r{int(rate*100):02d}_s{seed}',
                         suite='E-random', modalities=mods, rate_nominal=rate,
                         position='random', mask_seed=seed,
                         intervention='feature_post_encoder'))
    return rows


def training_runs():
    rows = []
    for idx, (dim, lr, dropout) in enumerate(itertools.product(
            [64, 128], [0.0003, 0.001], [0.1, 0.3]), 1):
        rows.append(dict(run_id=f'P{idx:02d}_F21_s7', stage='pilot', model='F21',
                         seed=7, hidden_dim=dim, learning_rate=lr, dropout=dropout,
                         dependencies=['interface_verified', 'Q0_passed'],
                         status='planned'))
    for model in MODELS:
        for seed in SEEDS:
            deps = ['shared_hyperparameters_locked']
            if model['id'] == 'D1':
                deps.append(f"C_F01_s{seed}")
            rows.append(dict(run_id=f"C_{model['id']}_s{seed}", stage='core',
                             model=model['id'], seed=seed,
                             hyperparameters='selected_shared.json_not_yet_created',
                             dependencies=deps, status='planned'))
    return rows


def protocol():
    return {
        'version': '1.0', 'status': 'design_only_not_executed',
        'root_relative_to_this_directory': '..',
        'data': {
            'aligned_path': 'E题数据/附件2-数据集特征文件/aligned_50.pkl',
            'split_sizes': {'train': 3395, 'valid': 728, 'test': 727},
            'special_sizes': {'attachment1': 100, 'attachment3': 30, 'attachment4': 20},
            'text_input': 'text_bert', 'text_encoder_frozen': True,
            'encoder_identity': None, 'tokenizer_identity': None,
            'require_verified_encoder_before_final_training': True,
            'normalize_fit_split': 'train', 'std_floor': 1e-6,
            'float_dtype': 'float32', 'token_dtype': 'int64',
            'zero_row_policy': 'raw_exact_zero_as_operationally_unavailable_with_uncertainty_flag',
            'preserve_all_samples': True,
            'attachment4_test_overlap_ids': ['03', '07', '08', '12', '15'],
        },
        'model': {
            'hidden_dim': None, 'layers_per_modality': 2, 'attention_heads': 4,
            'feedforward_multiplier': 4, 'activation': 'GELU', 'dropout': None,
            'head_hidden': 64, 'gate_hidden': 32,
            'gate_quality_fields': ['coverage', 'longest_unavailable_ratio', 'available'],
            'empty_input_fallback': 'train_class_prior_and_train_intensity_mean',
            'models': MODELS,
        },
        'training': {
            'optimizer': 'AdamW', 'learning_rate': None, 'weight_decay': 1e-4,
            'batch_size_effective': 32, 'max_epochs': 50, 'min_epochs': 10,
            'early_stop_patience': 8, 'min_delta': 1e-4, 'gradient_clip_norm': 1.0,
            'scheduler': 'constant', 'seeds': SEEDS, 'pilot_seed': 7,
            'class_weight': 'inverse_sqrt_train_frequency_normalized_mean_one',
            'weighted_sampling': False, 'huber_delta': 1.0,
            'classification_weight': 1.0, 'regression_weight': 1.0,
            'distillation': {'weight': 0.2, 'temperature': 1.0,
                             'teacher': 'frozen_eval_F01_same_seed',
                             'only_corrupted_samples': True,
                             'student_initialization': 'same_seed_fresh_random',
                             'regression_distance_divisor': 6.0},
        },
        'missingness': {
            'primary_intervention': 'feature_post_encoder',
            'raw_text_claim_allowed': False,
            'clean_probability': 0.3, 'augmentation_probability': 0.7,
            'modality_combinations': MODALITIES, 'rates_nominal': RATES,
            'train_position_probabilities': {'start': 0.25, 'middle': 0.25,
                                             'end': 0.25, 'random': 0.25},
            'window_length': 'min(L,max(1,ceil(rate*L)))',
            'coordinate': 'ordered_body_tokens_excluding_special_and_padding',
            'multi_modal_window': 'synchronous_shared_coordinate',
            'iid_control': 'same_actual_removed_count_per_modality_as_latent_block',
            'report_nominal_and_realized_rates': True,
            'rng_train_key': ['training_seed', 'epoch', 'sample_id'],
            'rng_eval_key': ['mask_seed', 'split', 'sample_id', 'condition_id'],
            'exclude_model_id_from_eval_rng': True,
        },
        'evaluation': {
            'selection_suite': 'V-select', 'selection_split': 'valid',
            'selection_formula': '0.5*S_clean+0.5*mean(S_corrupt), S_c=0.5*macro_f1+0.5*(1-mae/6)',
            'primary_suite': 'E-main', 'random_suite': 'E-random',
            'random_suite_models': ['F11', 'F20', 'F21', 'D1'],
            'random_window_seeds': MASK_SEEDS,
            'core_counts': {'selection_conditions': 15, 'main_conditions': 85,
                            'random_conditions': 140, 'pilot_runs': 8, 'core_runs': 36},
            'metrics': ['accuracy', 'macro_f1', 'mae', 'pearson_r'],
            'fixed_class_labels': [0, 1, 2], 'undefined_pearson': None,
            'aggregate_conditions': 'equal_weight_per_condition_after_metric_computation',
            'clean_tolerance_macro_f1_drop': 0.02, 'clean_tolerance_mae_increase': 0.10,
            'tolerance_reference_model': 'F00',
            'submission_seed_rule': 'median_validation_selection_score_of_selected_model',
            'test_access': 'only_after_protocol_and_model_and_explanation_settings_lock',
        },
        'statistics': {
            'primary_comparisons': [['F21', 'F11'], ['F21', 'F20'], ['D1', 'F21']],
            'primary_endpoint': 'equal_condition_mean_corrupted_macro_f1',
            'bootstrap_replicates': 2000, 'cluster': 'video_id',
            'pair_models_seeds_conditions': True,
            'ordinary_confidence': 0.95,
            'simultaneous_confidence_for_three': 1 - 0.05 / 3,
            'seed_variability': 'mean_and_sample_standard_deviation_separately',
        },
        'q1': {
            'all_sample_count': 100, 'audit_count': 20, 'audit_seed': 20260923,
            'duration_tertile_counts': [7, 6, 7], 'development_audit_count': 5,
            'held_out_quality_count': 15,
            'methods': ['uniform_word_intervals', 'forced_alignment'],
            'boundary_tolerances_seconds': [0.1, 0.2],
            'metrics': ['record_coverage', 'word_coverage', 'boundary_mae', 'interval_iou',
                        'boundary_within_tolerance', 'modality_availability', 'runtime'],
        },
        'explanation': {
            'valid_local_subset': 120, 'test_local_subset': 120, 'per_class': 40,
            'subset_seed': 20260923, 'attachment4_count': 20,
            'window_candidates': [1, 3, 5], 'default_window': 3, 'stride': 1,
            'distance': '0.5*abs(delta_y)/6+0.5*JSD_natural_log/log(2)',
            'contribution_zero_threshold': 1e-8,
            'null_contribution_when_undetermined': True,
            'methods': ['occlusion_sensitivity', 'signed_class_support', 'random', 'feature_norm'],
            'budget_fractions': [0.0, 0.1, 0.2, 0.3, 0.5], 'random_repeats': 20,
            'random_control': 'match_modality_budget_segment_lengths_and_removed_count',
            'noise_sigma_audio_vision': [0.01, 0.05], 'noise_repeats': 5,
            'stable_prediction_max_intensity_change': 0.1,
            'top_fraction': 0.2, 'max_display_segments': 3,
            'mapping_status_values': ['verified', 'approximate', 'unavailable'],
        },
        'optional_training_runs': {
            'pre_encoder_text_missingness': 9, 'zero_row_policy_sensitivity': 6,
            'remove_gate_quality_statistics': 3, 'reconstruction': 3,
            'single_task_comparison': 6,
        },
    }


def templates():
    return {
        'notice': '字段模板；null表示尚未记录或不可定义，不是已完成实验结果。',
        'run_record': dict(run_id=None, config_hash=None, code_hash=None, data_hash=None,
                           seed=None, best_epoch=None, val_selection_score=None,
                           train_seconds=None, peak_memory_mb=None,
                           trainable_parameters=None, status='planned'),
        'metric_record': dict(run_id=None, split=None, condition_id=None, n_samples=None,
                              n_actually_affected=None, n_fallback=None,
                              nominal_rate=None, realized_rate_mean=None,
                              accuracy=None, macro_f1=None, mae=None, pearson_r=None,
                              undefined_reason=None),
        'prediction_record': dict(run_id=None, split=None, condition_id=None,
                                  sample_id=None, video_id=None, true_class=None,
                                  true_intensity=None, pred_class=None, pred_intensity=None,
                                  probabilities=None, availability=None,
                                  realized_missing_rates=None, fallback=False),
        'mask_record': dict(split=None, sample_id=None, condition_id=None, seed=None,
                            content_indices=None, estimated_observed_masks=None,
                            synthetic_removed_indices=None, nominal_rate=None,
                            realized_rates=None, duplicate_mask_hash=None),
        'alignment_record': dict(sample_id=None, token_index=None, original_word=None,
                                 time_start=None, time_end=None, source_frame_indices=None,
                                 alignment_status=None, mapping_status=None,
                                 extraction_config_hash=None),
        'explanation_record': dict(sample_id=None, run_id=None, pred_class=None,
                                   pred_intensity=None, gate_weights=None,
                                   intervention_distances=None, signed_class_changes=None,
                                   signed_intensity_changes=None,
                                   normalized_sensitivity=None, main_modality=None,
                                   evidence_segments=[], mapping_status_by_modality=None),
        'evidence_segment_fields': ['modality', 'feature_start', 'feature_end_exclusive',
                                    'score', 'signed_class_support', 'token_text',
                                    'start_seconds', 'end_seconds', 'frame_index',
                                    'mapping_status', 'mapping_error_note'],
    }


def validate(config, runs, evals):
    assert len(MODELS) == 12
    ids = [r['run_id'] for r in runs]
    assert len(ids) == len(set(ids)) == 44
    assert sum(r['stage'] == 'pilot' for r in runs) == 8
    assert sum(r['stage'] == 'core' for r in runs) == 36
    assert len({e['condition_id'] for e in evals}) == len(evals) == 240
    for suite, n in [('V-select', 15), ('E-main', 85), ('E-random', 140)]:
        assert sum(e['suite'] == suite for e in evals) == n
    for model in MODELS:
        assert {r['seed'] for r in runs if r['stage'] == 'core' and r['model'] == model['id']} == set(SEEDS)
    for r in runs:
        if r['model'] == 'D1':
            assert f"C_F01_s{r['seed']}" in r['dependencies']
            assert f"C_F01_s{r['seed']}" in ids
    p = config['missingness']
    assert abs(p['clean_probability'] + p['augmentation_probability'] - 1) < 1e-12
    assert abs(sum(p['train_position_probabilities'].values()) - 1) < 1e-12
    assert config['data']['encoder_identity'] is None  # 不伪造已确认的编码器
    assert all(r['status'] == 'planned' for r in runs)
    return {'status': 'valid_design', 'pilot_runs': 8, 'core_runs': 36,
            'total_training_runs': 44, 'condition_rows': 240,
            'main_model_seed_conditions': 36 * 85,
            'main_test_sample_forwards': 36 * 85 * 727,
            'random_test_sample_forwards': 4 * 3 * 140 * 727,
            'models_trained_by_this_script': 0}


def write_json(name, data):
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_jsonl(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='只核验现有设计文件，不改写')
    args = parser.parse_args()
    if args.check:
        config = json.loads((ROOT / '实验协议.json').read_text(encoding='utf-8'))
        runs, evals = read_jsonl('训练运行清单.jsonl'), read_jsonl('评估条件清单.jsonl')
        json.loads((ROOT / '结果记录模板.json').read_text(encoding='utf-8'))
        summary = validate(config, runs, evals)
    else:
        config, runs, evals = protocol(), training_runs(), conditions()
        summary = validate(config, runs, evals)
        write_json('实验协议.json', config)
        write_json('结果记录模板.json', templates())
        for name, rows in [('训练运行清单.jsonl', runs), ('评估条件清单.jsonl', evals)]:
            (ROOT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        write_json('清单核验摘要.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
