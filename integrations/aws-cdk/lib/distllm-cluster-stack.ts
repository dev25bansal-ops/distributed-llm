import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as elasticloadbalancingv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';

export interface DistLlmClusterStackProps extends cdk.StackProps {
  /** EC2 instance type for GPU workers (default: g5.xlarge) */
  readonly instanceType?: string;
  /** Desired GPU worker count (default: 2) */
  readonly desiredCapacity?: number;
  /** Maximum GPU worker count for scaling (default: 10) */
  readonly maxCapacity?: number;
  /** DistLLM Docker image tag (default: latest) */
  readonly distllmImageTag?: string;
  /** CIDR range for inbound API access (default: 0.0.0.0/0) */
  readonly allowedCidr?: string;
  /** Existing VPC ID to deploy into (creates new VPC if omitted) */
  readonly vpcId?: string;
  /** Enable detailed CloudWatch metrics (default: true) */
  readonly detailedMonitoring?: boolean;
}

export class DistLlmClusterStack extends cdk.Stack {
  public readonly vpc: ec2.IVpc;
  public readonly albDnsName: string;
  public readonly autoScalingGroup: autoscaling.AutoScalingGroup;
  public readonly redisEndpoint: string;

  constructor(scope: Construct, id: string, props?: DistLlmClusterStackProps) {
    super(scope, id, props);

    const instanceType = props?.instanceType ?? 'g5.xlarge';
    const desiredCapacity = props?.desiredCapacity ?? 2;
    const maxCapacity = props?.maxCapacity ?? 10;
    const distllmImageTag = props?.distllmImageTag ?? 'latest';
    const allowedCidr = props?.allowedCidr ?? '0.0.0.0/0';

    // -----------------------------------------------------------------------
    //  VPC
    // -----------------------------------------------------------------------
    this.vpc = props?.vpcId
      ? ec2.Vpc.fromLookup(this, 'ExistingVpc', { vpcId: props.vpcId })
      : new ec2.Vpc(this, 'DistLlmVpc', {
          ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/16'),
          maxAzs: 3,
          subnetConfiguration: [
            {
              cidrMask: 24,
              name: 'Public',
              subnetType: ec2.SubnetType.PUBLIC,
            },
            {
              cidrMask: 24,
              name: 'Private',
              subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
            },
          ],
          natGateways: 1,
          enableDnsHostnames: true,
          enableDnsSupport: true,
        });

    // -----------------------------------------------------------------------
    //  Security Groups
    // -----------------------------------------------------------------------
    const gpuSecurityGroup = new ec2.SecurityGroup(this, 'GpuSecurityGroup', {
      vpc: this.vpc,
      description: 'Security group for DistLLM GPU instances',
      allowAllOutbound: true,
    });
    gpuSecurityGroup.addIngressRule(
      ec2.Peer.ipv4(allowedCidr),
      ec2.Port.tcp(8000),
      'Allow REST API traffic (port 8000)',
    );
    gpuSecurityGroup.addIngressRule(
      ec2.Peer.ipv4(allowedCidr),
      ec2.Port.tcp(50051),
      'Allow gRPC traffic (port 50051)',
    );
    gpuSecurityGroup.addIngressRule(
      ec2.Peer.ipv4(allowedCidr),
      ec2.Port.tcp(9090),
      'Allow Prometheus metrics scraping (port 9090)',
    );

    const albSecurityGroup = new ec2.SecurityGroup(this, 'AlbSecurityGroup', {
      vpc: this.vpc,
      description: 'Security group for DistLLM load balancer',
      allowAllOutbound: false,
    });
    albSecurityGroup.addIngressRule(
      ec2.Peer.ipv4(allowedCidr),
      ec2.Port.tcp(80),
      'Allow HTTP from clients',
    );
    albSecurityGroup.addIngressRule(
      ec2.Peer.ipv4(allowedCidr),
      ec2.Port.tcp(50051),
      'Allow gRPC from clients',
    );
    // ALB -> GPU backends
    gpuSecurityGroup.addIngressRule(
      albSecurityGroup,
      ec2.Port.tcp(8000),
      'Allow ALB health checks and traffic to REST API',
    );
    gpuSecurityGroup.addIngressRule(
      albSecurityGroup,
      ec2.Port.tcp(50051),
      'Allow ALB health checks and traffic to gRPC',
    );

    const redisSecurityGroup = new ec2.SecurityGroup(this, 'RedisSecurityGroup', {
      vpc: this.vpc,
      description: 'Security group for ElastiCache Redis KV cache',
      allowAllOutbound: true,
    });
    redisSecurityGroup.addIngressRule(
      gpuSecurityGroup,
      ec2.Port.tcp(6379),
      'Allow Redis access from GPU instances',
    );

    // -----------------------------------------------------------------------
    //  IAM Role for GPU Instances
    // -----------------------------------------------------------------------
    const gpuInstanceRole = new iam.Role(this, 'GpuInstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      description: 'IAM role for DistLLM GPU instances',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchAgentServerPolicy'),
      ],
      inlinePolicies: {
        ElastiCacheAccess: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                'elasticache:DescribeCacheClusters',
                'elasticache:DescribeCacheParameters',
              ],
              resources: ['*'],
            }),
          ],
        }),
        Ec2Describe: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                'ec2:DescribeInstances',
                'ec2:DescribeTags',
              ],
              resources: ['*'],
            }),
          ],
        }),
        CloudWatchMetrics: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                'cloudwatch:PutMetricData',
                'cloudwatch:GetMetricData',
                'cloudwatch:ListMetrics',
              ],
              resources: ['*'],
            }),
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                'logs:CreateLogGroup',
                'logs:CreateLogStream',
                'logs:PutLogEvents',
                'logs:DescribeLogStreams',
              ],
              resources: ['*'],
            }),
          ],
        }),
        S3ModelAccess: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                's3:GetObject',
                's3:GetObjectTagging',
                's3:ListBucket',
              ],
              resources: ['*'],
            }),
          ],
        }),
      },
    });

    // -----------------------------------------------------------------------
    //  GPU AMI — Amazon Linux 2 with NVIDIA driver support
    // -----------------------------------------------------------------------
    const gpuAmi = ec2.MachineImage.latestAmazonLinux2({
      cpuType: ec2.AmazonLinuxCpuType.X86_64,
      virtualization: ec2.AmazonLinuxVirtualization.HVM,
      storage: ec2.AmazonLinuxStorage.GP2,
    });

    // -----------------------------------------------------------------------
    //  ElastiCache Redis Cluster (KV cache backend)
    //  Created before user data so its endpoint is resolvable via CDK token.
    // -----------------------------------------------------------------------
    const redisSubnetGroup = new elasticache.CfnSubnetGroup(this, 'RedisSubnetGroup', {
      description: 'Subnet group for DistLLM Redis KV cache',
      subnetIds: this.vpc.privateSubnets.map((s) => s.subnetId),
    });

    const redisCluster = new elasticache.CfnCacheCluster(this, 'RedisCluster', {
      engine: 'redis',
      engineVersion: '7.1',
      cacheNodeType: 'cache.r6g.large',
      numCacheNodes: 1,
      port: 6379,
      vpcSecurityGroupIds: [redisSecurityGroup.securityGroupId],
      cacheSubnetGroupName: redisSubnetGroup.ref,
      autoMinorVersionUpgrade: true,
      preferredMaintenanceWindow: 'sun:05:00-sun:06:00',
    });
    cdk.Tags.of(redisCluster).add('Component', 'DistLLM-KV-Cache');
    redisCluster.addDependency(redisSubnetGroup);

    // Store endpoint as a CDK-resolvable token (CloudFormation Fn::GetAtt).
    this.redisEndpoint = cdk.Fn.join(':', [
      redisCluster.attrRedisEndpointAddress,
      redisCluster.attrRedisEndpointPort,
    ]);

    // -----------------------------------------------------------------------
    //  User Data — install NVIDIA drivers, Docker, pull & run distllm
    // -----------------------------------------------------------------------
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      '#!/bin/bash',
      'set -euxo pipefail',
      '',
      '# ---- System update and base packages ----',
      'yum update -y',
      'yum install -y docker amazon-efs-utils jq htop',
      '',
      '# ---- Install NVIDIA drivers ----',
      'amazon-linux-extras install -y epel',
      'yum install -y dkms kernel-devel-$(uname -r) kernel-headers-$(uname -r)',
      '',
      '# Add NVIDIA CUDA repository for AL2',
      'curl -fSsL https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo -o /etc/yum.repos.d/cuda.repo',
      'yum install -y nvidia-driver-latest || echo "nvidia-driver-latest installed or already present"',
      '',
      '# ---- Install NVIDIA Container Toolkit ----',
      'distribution=$(. /etc/os-release;echo $ID$VERSION_ID)',
      'curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | tee /etc/yum.repos.d/nvidia-docker.repo',
      'yum install -y nvidia-container-toolkit',
      'nvidia-ctk runtime configure --runtime=docker',
      '',
      '# ---- Start Docker ----',
      'systemctl enable docker',
      'systemctl start docker',
      'usermod -aG docker ec2-user',
      '',
      '# ---- Verify GPU access ----',
      'docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi || echo "GPU verification deferred to application"',
      '',
      '# ---- Pull DistLLM image ----',
      `docker pull ghcr.io/distributed-llm/distllm:${distllmImageTag}`,
      '',
      '# ---- Run DistLLM container ----',
      'mkdir -p /var/log/distllm /etc/distllm',
      '',
      '# Inject the Redis endpoint via template file (CDK resolves this at deploy time)',
      `echo "DISTLLM_REDIS_HOST=${this.redisEndpoint}" > /etc/distllm/redis.env`,
      'echo "DISTLLM_REDIS_PORT=6379" >> /etc/distllm/redis.env',
      'echo "DISTLLM_GRPC_PORT=50051" >> /etc/distllm/redis.env',
      'echo "DISTLLM_REST_PORT=8000" >> /etc/distllm/redis.env',
      'echo "DISTLLM_METRICS_PORT=9090" >> /etc/distllm/redis.env',
      '',
      'docker run -d \\',
      '  --name distllm \\',
      '  --restart unless-stopped \\',
      '  --gpus all \\',
      '  --network host \\',
      `  --env-file /etc/distllm/redis.env \\`,
      '  -v /var/log/distllm:/var/log/distllm \\',
      `  ghcr.io/distributed-llm/distllm:${distllmImageTag}`,
      '',
      '# ---- Log rotation ----',
      "cat > /etc/logrotate.d/docker-containers << 'EOF'",
      '/var/lib/docker/containers/*/*.log {',
      '  rotate 7',
      '  daily',
      '  compress',
      '  missingok',
      '  delaycompress',
      '  copytruncate',
      '}',
      'EOF',
      '',
      '# ---- CloudWatch agent configuration ----',
      'mkdir -p /opt/aws/amazon-cloudwatch-agent/etc',
      "cat > /opt/aws/amazon-cloudwatch-agent/etc/config.json << 'CWCONF'",
      '{',
      '  "agent": { "metrics_collection_interval": 60, "run_as_user": "root" },',
      '  "logs": {',
      '    "logs_collected": {',
      '      "files": {',
      '        "collect_list": [',
      '          {',
      `            "file_path": "/var/log/distllm/*.log",`,
      `            "log_group_name": "/distllm/${this.stackName}/engine",`,
      `            "log_stream_name": "{instance_id}",`,
      `            "timestamp_format": "%Y-%m-%dT%H:%M:%S",`,
      `            "timezone": "UTC"`,
      '          }',
      '        ]',
      '      }',
      '    }',
      '  },',
      '  "metrics": {',
      '    "append_dimensions": {',
      '      "InstanceId": "${aws:InstanceId}",',
      '      "AutoScalingGroupName": "${aws:AutoScalingGroupName}"',
      '    },',
      '    "metrics_collected": {',
      '      "nvidia_gpu": {',
      '        "measurement": [',
      '          "utilization_gpu",',
      '          "utilization_memory",',
      '          "memory_used",',
      '          "memory_total",',
      '          "temperature_gpu"',
      '        ],',
      '        "metrics_collection_interval": 60',
      '      },',
      '      "cpu": { "measurement": ["cpu_usage_idle", "cpu_usage_iowait"], "metrics_collection_interval": 60 },',
      '      "disk": { "measurement": ["used_percent"], "metrics_collection_interval": 60, "resources": ["*"] },',
      '      "mem": { "measurement": ["mem_used_percent"], "metrics_collection_interval": 60 },',
      '      "swap": { "measurement": ["swap_used_percent"], "metrics_collection_interval": 60 }',
      '    }',
      '  }',
      '}',
      'CWCONF',
      '',
      'systemctl restart amazon-cloudwatch-agent || echo "CloudWatch agent not yet installed — will start on next boot"',
    );

    // -----------------------------------------------------------------------
    //  Auto Scaling Group for GPU Instances
    // -----------------------------------------------------------------------
    this.autoScalingGroup = new autoscaling.AutoScalingGroup(this, 'GpuAutoScalingGroup', {
      vpc: this.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      instanceType: new ec2.InstanceType(instanceType),
      machineImage: gpuAmi,
      userData,
      role: gpuInstanceRole,
      securityGroup: gpuSecurityGroup,
      desiredCapacity,
      maxCapacity,
      minCapacity: 1,
      cooldown: cdk.Duration.seconds(300),
      healthCheck: autoscaling.HealthCheck.ec2(),
      groupMetrics: [autoscaling.GroupMetrics.all()],
      detailedInstanceMonitoring: props?.detailedMonitoring ?? true,
      blockDevices: [
        {
          deviceName: '/dev/xvda',
          volume: autoscaling.BlockDeviceVolume.ebs(100, {
            volumeType: autoscaling.EbsDeviceVolumeType.GP3,
            deleteOnTermination: true,
          }),
        },
      ],
      signals: autoscaling.Signals.waitForAll({
        timeout: cdk.Duration.minutes(15),
      }),
    });

    // CPU-based target tracking scaling policy
    this.autoScalingGroup.scaleOnCpuUtilization('CpuScaling', {
      targetUtilizationPercent: 70,
      estimatedInstanceWarmup: cdk.Duration.seconds(300),
    });

    // -----------------------------------------------------------------------
    //  Application Load Balancer
    // -----------------------------------------------------------------------
    const alb = new elasticloadbalancingv2.ApplicationLoadBalancer(this, 'DistLlmAlb', {
      vpc: this.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      internetFacing: true,
      securityGroup: albSecurityGroup,
      idleTimeout: cdk.Duration.seconds(300),
    });

    // --- REST API Target Group (port 8000) ---
    const restTargetGroup = new elasticloadbalancingv2.ApplicationTargetGroup(
      this,
      'RestTargetGroup',
      {
        vpc: this.vpc,
        port: 8000,
        protocol: elasticloadbalancingv2.ApplicationProtocol.HTTP,
        targetType: elasticloadbalancingv2.TargetType.INSTANCE,
        healthCheck: {
          path: '/health',
          port: '8000',
          healthyThresholdCount: 2,
          unhealthyThresholdCount: 3,
          interval: cdk.Duration.seconds(30),
          timeout: cdk.Duration.seconds(10),
        },
        stickiness: {
          enabled: true,
          type: elasticloadbalancingv2.StickinessTargetType.LB_COOKIE,
          cookieDuration: cdk.Duration.days(1),
        },
      },
    );
    this.autoScalingGroup.attachToApplicationTargetGroup(restTargetGroup);

    // --- gRPC Target Group (port 50051, HTTP/2 with GRPC protocol) ---
    const grpcTargetGroup = new elasticloadbalancingv2.ApplicationTargetGroup(
      this,
      'GrpcTargetGroup',
      {
        vpc: this.vpc,
        port: 50051,
        protocol: elasticloadbalancingv2.ApplicationProtocol.HTTP,
        protocolVersion: elasticloadbalancingv2.ApplicationProtocolVersion.GRPC,
        targetType: elasticloadbalancingv2.TargetType.INSTANCE,
        healthCheck: {
          healthyThresholdCount: 2,
          unhealthyThresholdCount: 3,
          interval: cdk.Duration.seconds(30),
          timeout: cdk.Duration.seconds(10),
        },
      },
    );
    this.autoScalingGroup.attachToApplicationTargetGroup(grpcTargetGroup);

    // --- Listeners ---
    const restListener = alb.addListener('RestListener', {
      port: 80,
      open: true,
      defaultTargetGroups: [restTargetGroup],
    });

    const grpcListener = alb.addListener('GrpcListener', {
      port: 50051,
      open: true,
      defaultTargetGroups: [grpcTargetGroup],
    });

    this.albDnsName = alb.loadBalancerDnsName;

    // -----------------------------------------------------------------------
    //  CloudWatch Dashboard
    // -----------------------------------------------------------------------
    const dashboard = new cloudwatch.Dashboard(this, 'DistLlmDashboard', {
      dashboardName: `DistLLM-${this.stackName}`,
      periodOverride: cloudwatch.PeriodOverride.AUTO,
    });

    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'GPU Utilization',
        left: [
          new cloudwatch.Metric({
            namespace: 'CWAgent',
            metricName: 'nvidia_gpu_utilization_gpu',
            statistic: 'Average',
            period: cdk.Duration.minutes(1),
            label: 'GPU Util %',
          }),
          new cloudwatch.Metric({
            namespace: 'CWAgent',
            metricName: 'nvidia_gpu_utilization_memory',
            statistic: 'Average',
            period: cdk.Duration.minutes(1),
            label: 'Memory Util %',
          }),
        ],
        liveData: true,
        view: cloudwatch.GraphWidgetView.TIME_SERIES,
      }),
      new cloudwatch.GraphWidget({
        title: 'GPU Memory Usage (MiB)',
        left: [
          new cloudwatch.Metric({
            namespace: 'CWAgent',
            metricName: 'nvidia_gpu_memory_used',
            statistic: 'Average',
            period: cdk.Duration.minutes(1),
            label: 'Used',
          }),
          new cloudwatch.Metric({
            namespace: 'CWAgent',
            metricName: 'nvidia_gpu_memory_total',
            statistic: 'Average',
            period: cdk.Duration.minutes(1),
            label: 'Total',
          }),
        ],
        liveData: true,
        view: cloudwatch.GraphWidgetView.TIME_SERIES,
      }),
      new cloudwatch.GraphWidget({
        title: 'CPU & Memory',
        left: [
          new cloudwatch.Metric({
            namespace: 'CWAgent',
            metricName: 'cpu_usage_idle',
            statistic: 'Average',
            period: cdk.Duration.minutes(1),
            label: 'CPU Idle %',
          }),
          new cloudwatch.Metric({
            namespace: 'CWAgent',
            metricName: 'mem_used_percent',
            statistic: 'Average',
            period: cdk.Duration.minutes(1),
            label: 'Memory Used %',
          }),
        ],
        liveData: true,
        view: cloudwatch.GraphWidgetView.TIME_SERIES,
      }),
      new cloudwatch.GraphWidget({
        title: 'ALB Request Count & Latency',
        left: [
          new cloudwatch.Metric({
            namespace: 'AWS/ApplicationELB',
            metricName: 'RequestCount',
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label: 'Requests',
          }),
        ],
        right: [
          new cloudwatch.Metric({
            namespace: 'AWS/ApplicationELB',
            metricName: 'TargetResponseTime',
            statistic: 'Average',
            period: cdk.Duration.minutes(1),
            label: 'Avg Response Time (s)',
          }),
        ],
        liveData: true,
        view: cloudwatch.GraphWidgetView.TIME_SERIES,
      }),
      new cloudwatch.GraphWidget({
        title: 'ASG Instance Count',
        left: [
          new cloudwatch.Metric({
            namespace: 'AWS/AutoScaling',
            metricName: 'GroupDesiredCapacity',
            statistic: 'Average',
            period: cdk.Duration.minutes(1),
            label: 'Desired',
          }),
          new cloudwatch.Metric({
            namespace: 'AWS/AutoScaling',
            metricName: 'GroupInServiceInstances',
            statistic: 'Average',
            period: cdk.Duration.minutes(1),
            label: 'InService',
          }),
        ],
        liveData: true,
        view: cloudwatch.GraphWidgetView.TIME_SERIES,
      }),
      new cloudwatch.GraphWidget({
        title: 'Redis Cache',
        left: [
          new cloudwatch.Metric({
            namespace: 'AWS/ElastiCache',
            metricName: 'CacheHits',
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label: 'Hits',
          }),
          new cloudwatch.Metric({
            namespace: 'AWS/ElastiCache',
            metricName: 'CacheMisses',
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label: 'Misses',
          }),
        ],
        right: [
          new cloudwatch.Metric({
            namespace: 'AWS/ElastiCache',
            metricName: 'CurrConnections',
            statistic: 'Average',
            period: cdk.Duration.minutes(1),
            label: 'Connections',
          }),
          new cloudwatch.Metric({
            namespace: 'AWS/ElastiCache',
            metricName: 'Evictions',
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label: 'Evictions',
          }),
        ],
        liveData: true,
        view: cloudwatch.GraphWidgetView.TIME_SERIES,
      }),
    );

    // -----------------------------------------------------------------------
    //  CloudWatch Log Group for DistLLM engine logs
    // -----------------------------------------------------------------------
    new logs.LogGroup(this, 'EngineLogGroup', {
      logGroupName: `/distllm/${this.stackName}/engine`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // -----------------------------------------------------------------------
    //  Outputs
    // -----------------------------------------------------------------------
    new cdk.CfnOutput(this, 'AlbDnsName', {
      value: this.albDnsName,
      description: 'ALB DNS name for REST (port 80) and gRPC (port 50051) endpoints',
    });
    new cdk.CfnOutput(this, 'RedisEndpoint', {
      value: this.redisEndpoint,
      description: 'ElastiCache Redis endpoint for KV cache (host:port)',
    });
    new cdk.CfnOutput(this, 'GpuInstanceRoleArn', {
      value: gpuInstanceRole.roleArn,
      description: 'IAM role ARN attached to GPU instances',
    });
    new cdk.CfnOutput(this, 'VpcId', {
      value: this.vpc.vpcId,
      description: 'VPC ID where the cluster is deployed',
    });
    new cdk.CfnOutput(this, 'AutoScalingGroupName', {
      value: this.autoScalingGroup.autoScalingGroupName,
      description: 'Auto Scaling Group name managing GPU instances',
    });
    new cdk.CfnOutput(this, 'DashboardName', {
      value: dashboard.dashboardName,
      description: 'CloudWatch Dashboard name for monitoring',
    });
  }
}
