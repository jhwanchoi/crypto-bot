import json

import boto3
from loguru import logger


class BedrockClient:
    """AWS Bedrock Claude API client."""

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        temperature: float = 0,
        max_tokens: int = 1024,
        profile_name: str | None = None,
        verify_ssl: bool = True,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        try:
            session = boto3.Session(profile_name=profile_name)
            self.client = session.client("bedrock-runtime", region_name=region, verify=verify_ssl)
            self._available = True
        except Exception as e:
            logger.warning(f"Bedrock client init failed: {e}")
            self.client = None
            self._available = False

    def invoke(self, prompt: str) -> str | None:
        """Invoke Claude model and return response text."""
        if not self._available:
            return None
        try:
            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
            response = self.client.invoke_model(modelId=self.model_id, body=body)
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            return None

    @property
    def is_available(self) -> bool:
        return self._available
