---
title: "The real cost of running an LLM: cloud rental vs your own GPUs"
description: "We ran the numbers on 1,000–50,000 requests per day against an A100-class cloud instance and a pooled cluster of consumer GPUs. The gap is bigger than you think."
pubDate: 2026-08-23
tags: [cost, analysis]
---

## The question

"I already own a gaming PC and a couple of laptops. Is it actually worth the
trouble to pool them, or should I just rent GPU time?"

It's a fair question — cloud platforms are frictionless, and electricity
isn't free. So we built the comparison into our site as a
[live calculator](/#pricing). Here's the math behind it.

## What cloud serving costs

A single always-on A100-class (40 GB) instance lists around **$1.90/hour** on
major clouds. That's ~$1,370/month if it runs 24/7. Reserved or spot pricing
improves this, but with two catches:

- Spot instances get preempted — your "endpoint" disappears mid-request.
- You still pay whether anyone is using the endpoint at 3 a.m. or not.

Even a smaller L4-class instance (~$0.70/hr) runs ~$500/month always-on.

## What owned hardware costs

A pooled cluster of consumer nodes draws power in three states:

| State | Draw | When |
|-------|------|------|
| Active inference | ~350 W/node | Only while generating |
| Idle-waiting | ~15 W/node | Between requests |
| Asleep/off | ~1 W | Your choice |

At $0.13/kWh, a node that generates an hour a day and idles otherwise costs
roughly **$8/month in electricity**. Even a four-node pool running hard for
several hours daily lands under $30.

## The crossover

The interesting number is tokens per day. Our calculator models throughput at
~20 tokens/sec for a pooled consumer cluster vs ~60 tokens/sec for the rented
A100. The cloud instance is faster — but for typical workloads (a few hundred
to tens of thousands of requests/day), the consumer pool has enormous idle
headroom it gives away free.

At **1,000 requests/day × 500 tokens**: cloud ≈ $1,370/mo, pooled ≈ $10/mo.
You would need sustained saturation — hundreds of thousands of tokens every
hour of every day — before renting becomes cheaper than owning.

## When the cloud still wins

Honesty requires the counterpoints:

- **Burst-to-huge**: sudden 100× traffic spikes are what elastic clouds are for.
- **Sub-second latency at scale**: an A100's raw speed is unmatched by WiFi-linked laptops.
- **Zero hardware to manage**: convenience has real value.

But for the dominant home-lab and small-team pattern — private model,
moderate traffic, data that must stay local — the owned pool wins by roughly
two orders of magnitude.

## Run your own numbers

Drag the sliders on our [homepage calculator](/#pricing) with your actual
request volume, then run DistLLM's built-in benchmark suite to measure what
your specific hardware delivers.
