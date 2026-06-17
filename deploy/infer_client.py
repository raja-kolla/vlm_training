import requests
from prompt import user_prompt

payload = {
    "prompt": user_prompt,
    "image_path": "/home/raja/vlm_training/sample_data/images/2024_11_21_316029DF_7ACED06E_3D7409AC.jpeg",
    "max_new_tokens": 8192,
}

response = requests.post(
    "http://localhost:8000/v1/generate",
    json=payload,
    timeout=600,
)

response.raise_for_status()

print(response.json()["text"])