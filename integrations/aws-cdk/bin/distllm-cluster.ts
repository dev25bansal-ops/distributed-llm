#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { DistLlmClusterStack, DistLlmClusterStackProps } from '../lib/distllm-cluster-stack';

const app = new cdk.App();

const env: cdk.Environment = {
  account: process.env.CDK_DEFAULT_ACCOUNT ?? process.env.AWS_ACCOUNT_ID,
  region: process.env.CDK_DEFAULT_REGION ?? process.env.AWS_REGION ?? 'us-east-1',
};

const stackProps: DistLlmClusterStackProps = {
  env,
  description: 'DistLLM GPU cluster — REST API (port 8000), gRPC (port 50051), Redis KV cache, CloudWatch monitoring',
  tags: {
    Application: 'DistLLM',
    Environment: process.env.ENVIRONMENT ?? 'production',
    ManagedBy: 'CDK',
  },
  instanceType: process.env.DISTLLM_INSTANCE_TYPE || 'g5.xlarge',
  desiredCapacity: parseInt(process.env.DISTLLM_DESIRED_CAPACITY || '2', 10),
  maxCapacity: parseInt(process.env.DISTLLM_MAX_CAPACITY || '10', 10),
  distllmImageTag: process.env.DISTLLM_IMAGE_TAG || 'latest',
  allowedCidr: process.env.DISTLLM_ALLOWED_CIDR || '0.0.0.0/0',
};

new DistLlmClusterStack(app, 'DistLlmCluster', stackProps);

app.synth();
