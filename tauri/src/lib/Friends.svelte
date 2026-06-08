<script lang="ts">
  import { generateInvite, getClusterStatus } from "./api";
  import { Card, Button, Input, ErrorBanner, toastStore } from "./ui";
  import QRCode from "./QRCode.svelte";
  import type { InviteInfo, ClusterStatus } from "./types";

  let invite = $state<InviteInfo | null>(null);
  let cluster = $state<ClusterStatus | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

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
      toastStore.success("Invite link generated");
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
      toastStore.success("Link copied to clipboard");
    } catch {
      // Fallback for older webviews
      const ta = document.createElement("textarea");
      ta.value = invite.link;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      toastStore.success("Link copied to clipboard");
    }
  }
</script>

<div class="friends-page">
  <h1 class="page-title">Friends & Invites</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  <Card title="Invite Friends" description="Generate a shareable link so your friends can join your cluster. They just need to paste it into their Distributed LLM app.">
    <Button onclick={handleGenerate} disabled={loading}>
      {loading ? "Generating..." : invite ? "Regenerate Invite" : "Generate Invite Link"}
    </Button>

    {#if invite}
      <div class="invite-card">
        <div class="invite-code">
          <span class="code-label">Invite Code</span>
          <span class="code-value mono">{invite.code}</span>
        </div>

        <div class="invite-link-row">
          <Input
            readonly
            value={invite.link}
            onfocus={(e) => (e.target as HTMLInputElement).select()}
          />
          <Button variant="ghost" onclick={copyLink}>Copy</Button>
        </div>

        <div class="qr-section">
          <QRCode text={invite.link} size={180} />
        </div>
      </div>
    {/if}
  </Card>

  <Card title="How It Works">
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
  </Card>
</div>

<style>
  .friends-page { max-width: 600px; }
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
  .qr-section { display: flex; justify-content: center; }
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
