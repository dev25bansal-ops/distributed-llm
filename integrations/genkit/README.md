# Genkit Integration (Firebase / Google)

[Genkit](https://firebase.google.com/docs/genkit) is Firebase's AI framework.
Genkit's OpenAI plugin works directly with DistLLM's OpenAI-compatible API.

## Quick Start

```typescript
import { genkit } from 'genkit';
import { openAI } from 'genkitx-openai';

const ai = genkit({
  plugins: [
    openAI({
      apiKey: process.env.DISTLLM_API_KEY || '',
      baseURL: 'http://localhost:8000/v1',
    }),
  ],
});

// Use with any model loaded on the DistLLM cluster
const response = await ai.generate({
  model: 'openai/llama-3-70b',
  prompt: 'Hello!',
});
```

## Streaming

```typescript
const { stream } = ai.generate({
  model: 'openai/llama-3-70b',
  prompt: 'Tell me a story',
});
for await (const chunk of stream) {
  console.log(chunk.text);
}
```

## Key Points

- Genkit's OpenAI plugin (`genkitx-openai`) treats DistLLM as a standard
  OpenAI-compatible endpoint — just set `baseURL` to the DistLLM URL.
- Supports chat, completion, streaming, and embedding.
- No DistLLM-specific plugin or code changes needed.
