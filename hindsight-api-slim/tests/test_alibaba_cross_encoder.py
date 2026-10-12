"""
Tests for AlibabaCloudCrossEncoder.

Covers the configurable base URL (dedicated Model Studio workspace gateway vs the
public DashScope endpoint), mirroring the SiliconFlow pattern.
"""

import os
from unittest.mock import patch

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.cross_encoder import AlibabaCloudCrossEncoder, create_cross_encoder


def _primary_member(**extra_env):
    """Resolve the primary (index 0) reranker member from env, the way production does."""
    env = {"HINDSIGHT_API_RERANKER_PROVIDER": "alibaba", "HINDSIGHT_API_RERANKER_ALIBABA_API_KEY": "test_key"}
    env.update(extra_env)
    with patch.dict(os.environ, env, clear=False):
        config = HindsightConfig.from_env()
    return config.reranker_chain()[0]


class TestAlibabaCloudCrossEncoder:
    """Test suite for AlibabaCloudCrossEncoder class."""

    def test_default_rerank_url(self):
        """Without a base_url env override, the public DashScope compatible endpoint is used."""
        member = _primary_member()
        assert member.alibaba_base_url == "https://dashscope.aliyuncs.com/compatible-api"

        encoder = create_cross_encoder(member)
        assert isinstance(encoder, AlibabaCloudCrossEncoder)
        assert encoder._client.rerank_url == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"

    def test_custom_workspace_base_url(self):
        """HINDSIGHT_API_RERANKER_ALIBABA_BASE_URL replaces the public endpoint; trailing slash trimmed."""
        member = _primary_member(
            HINDSIGHT_API_RERANKER_ALIBABA_BASE_URL="https://ws-gateway.cn-beijing.maas.aliyuncs.com/compatible-api/"
        )
        encoder = create_cross_encoder(member)
        assert isinstance(encoder, AlibabaCloudCrossEncoder)

        assert encoder.base_url == "https://ws-gateway.cn-beijing.maas.aliyuncs.com/compatible-api"
        assert encoder._client.rerank_url == "https://ws-gateway.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks"

    def test_indexed_member_reads_own_base_url(self):
        """Fallback members read the indexed variable in isolation, like every other member setting."""
        with patch.dict(
            os.environ,
            {
                "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
                "HINDSIGHT_API_RERANKER_1_PROVIDER": "alibaba",
                "HINDSIGHT_API_RERANKER_1_ALIBABA_API_KEY": "fallback_key",
                "HINDSIGHT_API_RERANKER_1_ALIBABA_BASE_URL": "https://ws-gateway.cn-beijing.maas.aliyuncs.com/compatible-api",
            },
            clear=False,
        ):
            config = HindsightConfig.from_env()
        fallback = config.reranker_chain()[1]
        assert fallback.provider == "alibaba"
        assert fallback.alibaba_base_url == "https://ws-gateway.cn-beijing.maas.aliyuncs.com/compatible-api"

    @pytest.mark.asyncio
    async def test_initialization_idempotent(self):
        """Calling initialize() twice does not rebuild the client session."""
        encoder = AlibabaCloudCrossEncoder(api_key="test_key")
        await encoder.initialize()
        assert encoder._client.initialized
        session = encoder._client._session
        await encoder.initialize()
        assert encoder._client._session is session
