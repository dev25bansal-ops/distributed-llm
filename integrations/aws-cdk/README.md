# AWS CDK Construct for DistLLM

Deploy a DistLLM cluster on AWS using CDK (Cloud Development Kit).

This directory will contain a reusable CDK construct for provisioning
DistLLM infrastructure on AWS, including:

- **EC2 GPU instances** for model serving
- **Application Load Balancer** for coordinating traffic
- **ElastiCache (Redis)** for KV cache and pub/sub
- **ECS / EKS** for container orchestration
- **CloudWatch** for monitoring and logging
- **Auto Scaling** based on GPU utilization

## Status

Construct under development. For now, deploy DistLLM on AWS manually:

1. Launch EC2 GPU instances (g5.xlarge, p4d, etc.) with the DistLLM AMI or Docker
2. Configure security groups for ports 8000 (REST), 50051 (gRPC), 9090 (metrics)
3. Set up an ALB to distribute traffic across coordinator nodes
4. Use ElastiCache Redis for the shared KV cache

## Example (manual)

```typescript
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';

// See future CDK construct releases for automated deployment.
// For now, use the CloudFormation / Terraform templates in
// integrations/terraform/ and integrations/kubernetes/.
```
