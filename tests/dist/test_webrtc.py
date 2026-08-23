"""Real tests for webrtc — WebRTC transport without actual ICE/STUN/TURN."""
from __future__ import annotations


class TestMsgType:
    def test_msg_type_values(self):
        from distllm.dist.webrtc import MsgType

        assert MsgType.TENSOR_TRANSFER is not None
        assert MsgType.HEARTBEAT is not None
        assert MsgType.SHUTDOWN is not None


class TestWebRTCConfig:
    def test_config_creation(self):
        from distllm.dist.webrtc import WebRTCConfig

        cfg = WebRTCConfig()
        assert cfg is not None
        assert len(cfg.stun_servers) > 0
