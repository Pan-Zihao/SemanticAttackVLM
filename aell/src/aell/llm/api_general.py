import http.client
import json
import openai

class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode

    def get_response0(self, prompt_content):
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
            }
        )

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": 1,
        }

        while True:
            try:
                conn = http.client.HTTPSConnection(self.api_endpoint)
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except:
                print("Error in API. Restarting the process...")
                continue

        return response
    def get_response(self, prompt_content):
        BASE_URL = "https://api.xiaoai.plus/v1"
        #OPENAI_API_KEY = "sk-UqhYLLaRGhqbMA4bAaEa329e0eEd4a8f9fE5578cB07178Ac"
        OPENAI_API_KEY = "sk-ePaBZR3FUIwaQNojF0871e9a338d44C5B4D332B8B6B8968e"
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=BASE_URL,
        )
        response = client.chat.completions.create(
                #model="gpt-3.5-turbo",
                model="gpt-4o",

                temperature = 0,
                max_tokens = 500,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0,
                messages=[
                    {"role": "user", "content": prompt_content}
                ]
        )

        return response.choices[0].message.content