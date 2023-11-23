import wandb


# Start a W&B run
wandb.init(project='gpt3')

# Save model inputs and hyperparameters
config = wandb.config

# Set sweep config to grid search, which iterates over every possible combination
sweep_config = {'method': 'grid'}

# Set parameters and values for the sweep
parameters_dict_mono = {
    'defocus': {
        'values': [True, False]
        },
    'underexposure': {
        'values': [True, False]
        },
    'overexposure': {
          'values': [True, False]
        },
    'motion_blur': {
        'values': [True, False]
    },
    'occlusion': {
        'values': [True, False]
    },
    'background': {
        'values': ['home', 'hospital', 'outdoor']
    },
    'movement': {
        'values': ['upper', 'lower', 'sitting', 'complex']
    }
}

parameters_dict_multi = {
    'defocus': {
        'values': [True, False]
        },
    'underexposure': {
        'values': [True, False]
        },
    'overexposure': {
          'values': [True, False]
        },
    'motion_blur': {
        'values': [True, False]
    },
    'occlusion': {
        'values': [True, False]
    },
    'background': {
        'values': ['none', 'home', 'hospital', 'outdoor', 'people']
    },
    'desynchronize': {
        'values': [True, False]
    },
    'cameras': {
      'values': [[4, 0], [3, 2], [5, 1], [4, 2], [0, 4, 3], [0, 2, 3], [5, 4, 1], [0, 4, 3, 2], [0, 5, 4, 3, 2],
                 [0, 5, 4, 1, 3, 2]]
    },
    'movement_category': {
        'values': ['upper', 'lower', 'complex', 'sitting']
    }
}

if movement_category == 'upper':
    movement_nr = [1, 2, 3, 4];
elif movement_category == 'lower':
    movement_nr = [5, 6, 7, 8];
elif movement_category == 'complex':
    movement_nr = [9, 10, 11, 12, 13];
elif movement_category == 'sitting':
    movement_nr = [14, 15, 16, 17];


if model_type == 'mono':
    parameters_dict_mono.update({
        'model_type': {
            'value': 'monoocular'}
        })
    sweep_config['parameters'] = parameters_dict_mono
elif model_type == 'multi':
    parameters_dict_multi.update({
        'model_type': {
            'value': 'multioccular'}
        })
    sweep_config['parameters'] = parameters_dict_multi
else:
    raise ValueError("Choose a valid background sweep parameter")

# Log metrics
wandb.log({"mpjpe_all": mpjpe, "pmpjpe_all": pmpjpe, "mean_velocity_error_all": mean_velocity_error,
           "angular_error_all": angular_error, "rom_all": rom, "cmc_all": cmc,
"mpjpe_": mpjpe, "pmpjpe_all": pmpjpe, "mean_velocity_error_all": mean_velocity_error,
           "angular_error_all": angular_error, "rom_all": rom, "cmc_all": cmc,

           "correct_pose_score": cps, "inference_time": inference_time})

