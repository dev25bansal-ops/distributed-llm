<script lang="ts">
  import { generateInvite, getClusterStatus } from "./api";
  import type { InviteInfo, ClusterStatus } from "./types";

  let invite = $state<InviteInfo | null>(null);
  let cluster = $state<ClusterStatus | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let copied = $state(false);

  async function handleGenerate() {
    loading = true;
    error = null;
    try {
      cluster = await getClusterStatus();
      if (!cluster?.coordinator_addr) {
        error = "Start a cluster first before generating invites.";
        loading = false;
        return;
      }
      invite = await generateInvite();
    } catch (e: unknown) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  async function copyLink() {
    if (!invite) return;
    try {
      await navigator.clipboard.writeText(invite.link);
      copied = true;
      setTimeout(() => (copied = false), 2000);
    } catch {
      // Fallback
      const ta = document.createElement("textarea");
      ta.value = invite.link;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      copied = true;
      setTimeout(() => (copied = false), 2000);
    }
  }
</script>

<div class="friends-page">
  <h1 class="page-title">Friends & Invites</h1>

  {#if error}
    <div class="error-banner">{error}</div>
  {/if}

  <section class="card">
    <h2 class="card-title">Invite Friends</h2>
    <p class="card-desc">
      Generate a shareable link so your friends can join your cluster. They just need to paste it into their Distributed LLM app.
    </p>

    <button class="btn btn-primary" onclick={handleGenerate} disabled={loading}>
      {loading ? "Generating..." : invite ? "Regenerate Invite" : "Generate Invite Link"}
    </button>

    {#if invite}
      <div class="invite-card">
        <div class="invite-code">
          <span class="code-label">Invite Code</span>
          <span class="code-value mono">{invite.code}</span>
        </div>

        <div class="invite-link-row">
          <input
            type="text"
            class="input mono"
            readonly
            value={invite.link}
            onfocus={(e) => (e.target as HTMLInputElement).select()}
          />
          <button class="btn btn-copy" onclick={copyLink}>
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>

        <div class="qr-section">
          <div class="qr-placeholder">
            <div class="qr-icon">▦</div>
            <span class="qr-hint">
              QR code will be rendered here.<br />
              <small>Requires qrcode library (pip install qrcode[pil])</small>
            </span>
          </div>
        </div>
      </div>
    {/if}
  </section>

  <section class="card">
    <h2 class="card-title">How It Works</h2>
    <ol class="steps">
      <li>Create or join a cluster on your machine</li>
      <li>Generate an invite link above</li>
      <li>Share the link with a friend (or scan the QR code)</li>
      <li>They open the link in their Distributed LLM app</li>
      <li>Their GPU joins your cluster automatically</li>
    </ol>
    <p class="card-desc" style="margin-top: 12px;">
      Your cluster is secured with a unique cluster key. Only people with the invite link can join.
    </p>
  </section>
</div>

<style>
  .friends-page { max-width: 600px; }
  .page-title { font-size: 22px; font-weight: 700; margin-bottom: 20px; }
  .error-banner {
    background: color-mix(in srgb, var(--danger) 15%, transparent);
    color: var(--danger);
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 13px;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
  }
  .card-title { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
  .card-desc { font-size: 13px; color: var(--text-secondary); margin-bottom: 16px; }
  .btn {
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.15s;
  }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--accent-hover); }
  .invite-card {
    margin-top: 16px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .invite-code { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
  .code-label { font-size: 12px; color: var(--text-secondary); }
  .code-value { font-size: 18px; font-weight: 700; color: var(--accent); letter-spacing: 2px; }
  .invite-link-row { display: flex; gap: 8px; margin-bottom: 16px; }
  .input {
    flex: 1;
    padding: 10px 12px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 13px;
  }
  .mono { font-family: var(--font-mono); }
  .btn-copy {
    padding: 10px 16px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
  }
  .btn-copy:hover { background: var(--border); }
  .qr-section { display: flex; justify-content: center; }
  .qr-placeholder {
    width: 180px;
    height: 180px;
    background: var(--bg-input);
    border: 2px dashed var(--border);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    color: var(--text-muted);
  }
  .qr-icon { font-size: 40px; opacity: 0.4; }
  .qr-hint { font-size: 11px; text-align: center; line-height: 1.4; }
  .qr-hint small { opacity: 0.6; }
  .steps {
    padding-left: 20px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 13px;
    color: var(--text-secondary);
  }
  .steps li { line-height: 1.5; }
</style>
