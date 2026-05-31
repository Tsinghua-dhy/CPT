# Overlong Buffer Penalty

Apply a linear penalty inside the last `buffer_len` tokens of the max generation length, to discourage overly long responses. 

## Config

Configure via the YAML parameters of the training script: 

```bash
reward_model.overlong_buffer.enable=True
reward_model.overlong_buffer.len=4096
reward_model.overlong_buffer.penalty_factor=1.0
```

Or define variables directly in the script: 

```bash
overlong_buffer_enable=True
overlong_buffer_len=4096
overlong_buffer_penalty=1.0

python3 -m verl.trainer.main_ppo \
    reward_model.overlong_buffer.enable=${overlong_buffer_enable} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_buffer_penalty} \
    ...
```

## Penalty

```
penalty_start = max_response_length - buffer_len

if response_length <= penalty_start:
    penalty = 0
else:
    progress = (response_length - penalty_start) / buffer_len
    penalty = -progress * penalty_factor
```

**Example** (max=12288, buffer=4096, factor=1.0):
- 8192 tokens: penalty = 0
- 10240 tokens: penalty = -0.5
- 12288 tokens: penalty = -1.0

## Logging

The penalty value is logged in `overlong_penalty` field of the rollout logs. 
