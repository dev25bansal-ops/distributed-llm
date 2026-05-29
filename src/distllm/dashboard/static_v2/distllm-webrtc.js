/**
 * DistLLM WebRTC Client SDK
 *
 * Browser-side JavaScript client for establishing WebRTC data channels
 * to the DistLLM cluster. Enables low-latency inference directly from
 * web browsers without HTTP overhead.
 *
 * Usage:
 *   import { DistLLMWebRTC } from './distllm-webrtc.js';
 *
 *   const client = new DistLLMWebRTC({
 *     signalingUrl: 'http://localhost:8000',
 *     apiKey: 'your-api-key',
 *   });
 *
 *   await client.connect();
 *   const response = await client.chat({
 *     model: 'distributed-llm',
 *     messages: [{ role: 'user', content: 'Hello!' }],
 *   });
 *   console.log(response);
 *   client.disconnect();
 */

export class DistLLMWebRTC {
  constructor(options = {}) {
    this.signalingUrl = options.signalingUrl || 'http://localhost:8000';
    this.apiKey = options.apiKey || '';
    this.stunServers = options.stunServers || ['stun:stun.l.google.com:19302'];
    this.onToken = options.onToken || null;
    this.onError = options.onError || null;
    this.onConnect = options.onConnect || null;
    this.onDisconnect = options.onDisconnect || null;

    this._pc = null;
    this._dataChannel = null;
    this._sessionId = null;
    this._connected = false;
    this._pendingRequests = new Map();
    this._requestCounter = 0;
  }

  get isConnected() {
    return this._connected;
  }

  get sessionId() {
    return this._sessionId;
  }

  /**
   * Establish a WebRTC data channel to the DistLLM cluster.
   */
  async connect() {
    if (this._connected) return;

    // Create peer connection with ICE servers
    this._pc = new RTCPeerConnection({
      iceServers: this.stunServers.map(url => ({ urls: url })),
    });

    // Create data channel
    this._dataChannel = this._pc.createDataChannel('distllm-inference', {
      ordered: true,
    });

    this._setupDataChannel();
    this._setupPeerConnection();

    // Create SDP offer
    const offer = await this._pc.createOffer();
    await this._pc.setLocalDescription(offer);

    // Exchange via signaling
    const answer = await this._exchangeSignaling(offer);
    if (!answer) {
      throw new Error('Signaling failed: no answer received');
    }

    await this._pc.setRemoteDescription(answer);

    // Wait for connection
    await this._waitForConnection();
  }

  /**
   * Disconnect from the cluster.
   */
  disconnect() {
    this._connected = false;
    if (this._dataChannel) {
      this._dataChannel.close();
      this._dataChannel = null;
    }
    if (this._pc) {
      this._pc.close();
      this._pc = null;
    }
    this._sessionId = null;
    this._pendingRequests.clear();
    if (this.onDisconnect) this.onDisconnect();
  }

  /**
   * Send a chat completion request via the data channel.
   * Returns a promise that resolves with the full response.
   */
  async chat(params) {
    return this._sendRequest('chat', params);
  }

  /**
   * Send a completion request via the data channel.
   */
  async completion(params) {
    return this._sendRequest('completion', params);
  }

  /**
   * Stream tokens from a chat completion request.
   * Calls onToken for each token as it arrives.
   */
  async chatStream(params, onToken) {
    const requestId = this._nextRequestId();
    const message = JSON.stringify({
      type: 'chat_stream',
      request_id: requestId,
      ...params,
    });

    return new Promise((resolve, reject) => {
      const tokens = [];
      const handler = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.request_id !== requestId) return;

          if (data.type === 'token') {
            tokens.push(data.text);
            if (onToken) onToken(data.text, tokens.join(''));
          } else if (data.type === 'done') {
            this._dataChannel.removeEventListener('message', handler);
            resolve({ text: tokens.join(''), tokens });
          } else if (data.type === 'error') {
            this._dataChannel.removeEventListener('message', handler);
            reject(new Error(data.message));
          }
        } catch (e) {
          // Ignore non-JSON messages
        }
      };

      this._dataChannel.addEventListener('message', handler);
      this._dataChannel.send(message);

      // Timeout
      setTimeout(() => {
        this._dataChannel.removeEventListener('message', handler);
        reject(new Error('Request timed out'));
      }, 120000);
    });
  }

  // ── Private ──────────────────────────────────────────────────────────

  _nextRequestId() {
    return `req-${++this._requestCounter}-${Date.now()}`;
  }

  async _sendRequest(type, params) {
    if (!this._connected || !this._dataChannel) {
      throw new Error('Not connected');
    }

    const requestId = this._nextRequestId();
    const message = JSON.stringify({
      type,
      request_id: requestId,
      ...params,
    });

    return new Promise((resolve, reject) => {
      const handler = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.request_id !== requestId) return;

          if (data.type === 'response') {
            this._dataChannel.removeEventListener('message', handler);
            resolve(data);
          } else if (data.type === 'error') {
            this._dataChannel.removeEventListener('message', handler);
            reject(new Error(data.message));
          }
        } catch (e) {
          // Ignore non-JSON messages
        }
      };

      this._dataChannel.addEventListener('message', handler);
      this._dataChannel.send(message);

      setTimeout(() => {
        this._dataChannel.removeEventListener('message', handler);
        reject(new Error('Request timed out'));
      }, 120000);
    });
  }

  async _exchangeSignaling(offer) {
    const headers = { 'Content-Type': 'application/json' };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    try {
      const resp = await fetch(`${this.signalingUrl}/v1/webrtc/offer`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        }),
      });

      if (!resp.ok) {
        throw new Error(`Signaling failed: ${resp.status}`);
      }

      const data = await resp.json();
      this._sessionId = data.session_id;
      return new RTCSessionDescription({
        sdp: data.sdp,
        type: data.type,
      });
    } catch (e) {
      if (this.onError) this.onError(e);
      throw e;
    }
  }

  _setupPeerConnection() {
    this._pc.onconnectionstatechange = () => {
      const state = this._pc.connectionState;
      if (state === 'connected') {
        this._connected = true;
        if (this.onConnect) this.onConnect();
      } else if (state === 'failed' || state === 'disconnected' || state === 'closed') {
        this._connected = false;
        if (this.onDisconnect) this.onDisconnect();
      }
    };

    this._pc.onicecandidate = async (event) => {
      if (event.candidate && this._sessionId) {
        const headers = { 'Content-Type': 'application/json' };
        if (this.apiKey) {
          headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        try {
          await fetch(`${this.signalingUrl}/v1/webrtc/ice`, {
            method: 'POST',
            headers,
            body: JSON.stringify({
              session_id: this._sessionId,
              candidate: event.candidate.candidate,
              sdp_mid: event.candidate.sdpMid,
              sdp_mline_index: event.candidate.sdpMLineIndex,
            }),
          });
        } catch (e) {
          // ICE candidates are best-effort
        }
      }
    };
  }

  _setupDataChannel() {
    this._dataChannel.onopen = () => {
      this._connected = true;
      if (this.onConnect) this.onConnect();
    };

    this._dataChannel.onclose = () => {
      this._connected = false;
      if (this.onDisconnect) this.onDisconnect();
    };

    this._dataChannel.onerror = (error) => {
      if (this.onError) this.onError(error);
    };
  }

  _waitForConnection(timeout = 30000) {
    return new Promise((resolve, reject) => {
      if (this._connected) {
        resolve();
        return;
      }

      const start = Date.now();
      const check = () => {
        if (this._connected) {
          resolve();
        } else if (Date.now() - start > timeout) {
          reject(new Error('WebRTC connection timed out'));
        } else {
          setTimeout(check, 100);
        }
      };
      check();
    });
  }
}

export default DistLLMWebRTC;
