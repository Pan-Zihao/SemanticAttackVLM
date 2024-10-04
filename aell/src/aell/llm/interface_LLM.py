import json
import requests


def get_response(user_message):
    Baseurl = "https://api.claude-Plus.top"
    Skey = "sk-vjulMaFmBWm31NP4OqwnKaDJMb3X0jbVlnIvg4XbYgtXwzWi"

    payload = json.dumps({
        "model": "claude-3-5-sonnet-20240620",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    })

    url = Baseurl + "/v1/chat/completions"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {Skey}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload)

    data = response.json()

    content = data['choices'][0]['message']['content']

    return content
