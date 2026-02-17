import json
from unittest.mock import Mock


class TestBedrockClient:
    """Tests for BedrockClient AWS Bedrock integration."""

    def test_init_success(self, monkeypatch):
        """Test successful client initialization."""
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client

        mock_session_class = Mock(return_value=mock_session)
        monkeypatch.setattr("boto3.Session", mock_session_class)

        from src.llm.bedrock_client import BedrockClient

        client = BedrockClient(
            model_id="anthropic.claude-v2",
            region="us-west-2",
            temperature=0.5,
            max_tokens=2048,
            profile_name="test-profile",
            verify_ssl=False,
        )

        assert client.model_id == "anthropic.claude-v2"
        assert client.temperature == 0.5
        assert client.max_tokens == 2048
        assert client.is_available is True
        assert client.client == mock_client

        # Verify boto3.Session was called with profile_name
        mock_session_class.assert_called_once_with(profile_name="test-profile")
        # Verify session.client was called with correct parameters
        mock_session.client.assert_called_once_with(
            "bedrock-runtime", region_name="us-west-2", verify=False
        )

    def test_init_failure(self, monkeypatch):
        """Test client initialization failure."""
        mock_session_class = Mock(side_effect=Exception("AWS credentials not found"))
        monkeypatch.setattr("boto3.Session", mock_session_class)

        from src.llm.bedrock_client import BedrockClient

        client = BedrockClient(model_id="anthropic.claude-v2")

        assert client.is_available is False
        assert client.client is None

    def test_invoke_success(self, monkeypatch):
        """Test successful model invocation."""
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client

        # Mock the invoke_model response
        mock_body = Mock()
        mock_body.read.return_value = b'{"content": [{"text": "This is the AI response"}]}'
        mock_response = {"body": mock_body}
        mock_client.invoke_model.return_value = mock_response

        monkeypatch.setattr("boto3.Session", Mock(return_value=mock_session))

        from src.llm.bedrock_client import BedrockClient

        client = BedrockClient(model_id="anthropic.claude-v2", temperature=0.3, max_tokens=512)
        result = client.invoke("What is the weather?")

        assert result == "This is the AI response"
        # Verify invoke_model was called with correct parameters
        call_args = mock_client.invoke_model.call_args
        assert call_args[1]["modelId"] == "anthropic.claude-v2"
        body = json.loads(call_args[1]["body"])
        assert body["temperature"] == 0.3
        assert body["max_tokens"] == 512
        assert body["messages"][0]["content"] == "What is the weather?"

    def test_invoke_failure(self, monkeypatch):
        """Test model invocation failure returns None."""
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client

        # Mock invoke_model to raise exception
        mock_client.invoke_model.side_effect = Exception("Service unavailable")

        monkeypatch.setattr("boto3.Session", Mock(return_value=mock_session))

        from src.llm.bedrock_client import BedrockClient

        client = BedrockClient(model_id="anthropic.claude-v2")
        result = client.invoke("Test prompt")

        assert result is None

    def test_invoke_when_unavailable(self, monkeypatch):
        """Test invoke returns None when client is unavailable."""
        monkeypatch.setattr("boto3.Session", Mock(side_effect=Exception("Init failed")))

        from src.llm.bedrock_client import BedrockClient

        client = BedrockClient(model_id="anthropic.claude-v2")
        result = client.invoke("Test prompt")

        assert result is None
        assert client.is_available is False

    def test_is_available_property(self, monkeypatch):
        """Test is_available property reflects client state."""
        # Test available client
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client
        monkeypatch.setattr("boto3.Session", Mock(return_value=mock_session))

        from src.llm.bedrock_client import BedrockClient

        client = BedrockClient(model_id="anthropic.claude-v2")
        assert client.is_available is True

        # Test unavailable client
        monkeypatch.setattr("boto3.Session", Mock(side_effect=Exception("Failed")))
        client2 = BedrockClient(model_id="anthropic.claude-v2")
        assert client2.is_available is False

    def test_profile_name_and_verify_ssl_passed_to_boto3(self, monkeypatch):
        """Test profile_name and verify_ssl are correctly passed to boto3."""
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client
        mock_session_class = Mock(return_value=mock_session)
        monkeypatch.setattr("boto3.Session", mock_session_class)

        from src.llm.bedrock_client import BedrockClient

        BedrockClient(
            model_id="anthropic.claude-v2",
            region="eu-west-1",
            profile_name="custom-profile",
            verify_ssl=True,
        )

        # Verify Session was created with profile_name
        mock_session_class.assert_called_once_with(profile_name="custom-profile")
        # Verify client was created with verify=True
        mock_session.client.assert_called_once_with(
            "bedrock-runtime", region_name="eu-west-1", verify=True
        )
