# Native chat-template audit

Status: **PASS**

Templates identical: **True**

## sft

SHA-256: `ac7498a36a719da630e99d48e6ebc4409de85a77556c2b6159eeb735bcbd11df`

```jinja
{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>
' + message['content'] + '
' }}{% elif message['role'] == 'user' %}{{ '<|user|>
' + message['content'] + '
' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>
'  + message['content'] + eos_token + '
' }}{% else %}{{ '<|assistant|>
'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>
' }}{% endif %}{% endfor %}
```

## dpo

SHA-256: `ac7498a36a719da630e99d48e6ebc4409de85a77556c2b6159eeb735bcbd11df`

```jinja
{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>
' + message['content'] + '
' }}{% elif message['role'] == 'user' %}{{ '<|user|>
' + message['content'] + '
' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>
'  + message['content'] + eos_token + '
' }}{% else %}{{ '<|assistant|>
'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>
' }}{% endif %}{% endfor %}
```

## final

SHA-256: `ac7498a36a719da630e99d48e6ebc4409de85a77556c2b6159eeb735bcbd11df`

```jinja
{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>
' + message['content'] + '
' }}{% elif message['role'] == 'user' %}{{ '<|user|>
' + message['content'] + '
' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>
'  + message['content'] + eos_token + '
' }}{% else %}{{ '<|assistant|>
'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>
' }}{% endif %}{% endfor %}
```
