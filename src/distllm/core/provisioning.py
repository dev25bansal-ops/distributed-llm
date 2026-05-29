"""Terraform/CloudFormation Provisioning — auto-generate IaC for GPU instances.

Generates Terraform HCL or CloudFormation YAML for provisioning GPU
instances when the router picks a cloud provider.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ProvisioningConfig:
    """Configuration for a GPU instance to provision."""
    provider: str  # "aws", "gcp", "azure"
    instance_type: str
    region: str
    gpu_type: str = ""
    gpu_count: int = 1
    ssh_key_name: str = ""
    security_group_ids: list[str] = field(default_factory=list)
    subnet_id: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    spot: bool = True
    user_data: str = ""  # Startup script


def generate_terraform(config: ProvisioningConfig) -> str:
    """Generate Terraform HCL for a GPU instance.

    Returns the complete .tf content as a string.
    """
    if config.provider == "aws":
        return _terraform_aws(config)
    elif config.provider == "gcp":
        return _terraform_gcp(config)
    elif config.provider == "azure":
        return _terraform_azure(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


def _terraform_aws(config: ProvisioningConfig) -> str:
    tags = json.dumps({**(config.tags or {}), "Name": f"distllm-{config.instance_type}"})
    market_type = "spot" if config.spot else ""
    spot_block = ""
    if config.spot:
        spot_block = f"""
  instance_market_options {{
    market_type = "spot"
    spot_options {{
      spot_instance_type = "one-time"
    }}
  }}"""
    return f"""terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = "{config.region}"
}}

resource "aws_instance" "distllm" {{
  ami           = data.aws_ami.gpu_ami.id
  instance_type = "{config.instance_type}"
  {f'key_name = "{config.ssh_key_name}"' if config.ssh_key_name else ''}
  {f'subnet_id = "{config.subnet_id}"' if config.subnet_id else ''}
  {f'vpc_security_group_ids = {json.dumps(config.security_group_ids)}' if config.security_group_ids else ''}

  tags = {tags}
  {spot_block}
}}

data "aws_ami" "gpu_ami" {{
  most_recent = true
  owners      = ["amazon"]
  filter {{
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*"]
  }}
}}

output "instance_ip" {{
  value = aws_instance.distllm.public_ip
}}

output "instance_id" {{
  value = aws_instance.distllm.id
}}
"""


def _terraform_gcp(config: ProvisioningConfig) -> str:
    labels = json.dumps({**(config.tags or {}), "name": f"distllm-{config.instance_type}"})
    return f"""terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 5.0"
    }}
  }}
}}

provider "google" {{
  project = var.project_id
  region  = "{config.region}"
}}

variable "project_id" {{
  description = "GCP project ID"
}}

resource "google_compute_instance" "distllm" {{
  name         = "distllm-{config.instance_type}"
  machine_type = "{config.instance_type}"
  zone         = "{config.region}-a"

  boot_disk {{
    initialize_params {{
      image = "projects/ml-images/global/images/c2-deeplearning-pytorch-gpu"
      size  = 100
    }}
  }}

  guest_accelerator {{
    type  = "nvidia-tesla-a100"
    count = {config.gpu_count}
  }}

  scheduling {{
    on_host_maintenance = "TERMINATE"
    {f'preemptible = true' if config.spot else ''}
  }}

  network_interface {{
    network = "default"
    access_config {{}}
  }}

  labels = {labels}

  metadata = {{
    "install-nvidia-driver" = "true"
  }}
  {f'metadata_startup_script = "{config.user_data}"' if config.user_data else ''}
}}

output "instance_ip" {{
  value = google_compute_instance.distllm.network_interface[0].access_config[0].nat_ip
}}
"""


def _terraform_azure(config: ProvisioningConfig) -> str:
    tags = json.dumps({**(config.tags or {}), "Name": f"distllm-{config.instance_type}"})
    return f"""terraform {{
  required_providers {{
    azurerm = {{
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }}
  }}
}}

provider "azurerm" {{
  features {{}}
}}

resource "azurerm_resource_group" "distllm" {{
  name     = "distllm-rg"
  location = "{config.region}"
}}

resource "azurerm_linux_virtual_machine" "distllm" {{
  name                = "distllm-{config.instance_type}"
  resource_group_name = azurerm_resource_group.distllm.name
  location            = azurerm_resource_group.distllm.location
  size                = "{config.instance_type}"
  admin_username      = "distllm"

  network_interface_ids = [azurerm_network_interface.distllm.id]

  admin_ssh_key {{
    username   = "distllm"
    public_key = file("~/.ssh/id_rsa.pub")
  }}

  os_disk {{
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
    disk_size_gb         = 100
  }}

  source_image_reference {{
    publisher = "microsoft-dsvm"
    offer     = "ubuntu-2004"
    sku       = "gpu-gen2"
    version   = "latest"
  }}

  tags = {tags}
}}

resource "azurerm_network_interface" "distllm" {{
  name                = "distllm-nic"
  location            = azurerm_resource_group.distllm.location
  resource_group_name = azurerm_resource_group.distllm.name

  ip_configuration {{
    name                          = "internal"
    subnet_id                     = azurerm_subnet.distllm.id
    private_ip_address_allocation = "Dynamic"
  }}
}}

resource "azurerm_subnet" "distllm" {{
  name                 = "distllm-subnet"
  resource_group_name  = azurerm_resource_group.distllm.name
  virtual_network_name = azurerm_virtual_network.distllm.name
  address_prefixes     = ["10.0.1.0/24"]
}}

resource "azurerm_virtual_network" "distllm" {{
  name                = "distllm-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.distllm.location
  resource_group_name = azurerm_resource_group.distllm.name
}}

output "public_ip" {{
  value = azurerm_linux_virtual_machine.distllm.public_ip_address
}}
"""


def write_terraform(config: ProvisioningConfig, output_dir: str = ".") -> str:
    """Write Terraform files to a directory and return the directory path."""
    tf_content = generate_terraform(config)
    os.makedirs(output_dir, exist_ok=True)
    tf_path = os.path.join(output_dir, "main.tf")
    with open(tf_path, "w") as f:
        f.write(tf_content)
    logger.info(f"Terraform config written to {tf_path}")
    return output_dir


def apply_terraform(config: ProvisioningConfig, output_dir: str | None = None) -> dict[str, str]:
    """Write and apply Terraform config.

    Returns dict with output values (e.g., instance_ip, instance_id).
    """
    work_dir = output_dir or tempfile.mkdtemp(prefix="distllm-tf-")
    write_terraform(config, work_dir)

    try:
        subprocess.run(["terraform", "init"], cwd=work_dir, check=True, capture_output=True)
        subprocess.run(["terraform", "apply", "-auto-approve"], cwd=work_dir, check=True, capture_output=True)
        result = subprocess.run(
            ["terraform", "output", "-json"],
            cwd=work_dir, capture_output=True, text=True,
        )
        outputs = json.loads(result.stdout) if result.stdout else {}
        return {k: v.get("value", "") for k, v in outputs.items()}
    except subprocess.CalledProcessError as e:
        logger.error(f"Terraform apply failed: {e.stderr}")
        raise


def generate_cloudformation(config: ProvisioningConfig) -> str:
    """Generate CloudFormation YAML for a GPU instance."""
    import yaml  # type: ignore
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": f"DistLLM GPU instance ({config.instance_type})",
        "Resources": {
            "DistLLMInstance": {
                "Type": "AWS::EC2::Instance",
                "Properties": {
                    "InstanceType": config.instance_type,
                    "ImageId": {"Ref": "LatestDeepLearningAMI"},
                    "KeyName": config.ssh_key_name or {"Ref": "AWS::NoValue"},
                    "Tags": [{"Key": k, "Value": v} for k, v in (config.tags or {}).items()],
                },
            },
        },
        "Outputs": {
            "InstanceId": {"Value": {"Ref": "DistLLMInstance"}},
            "PublicIp": {"Value": {"Fn::GetAtt": ["DistLLMInstance", "PublicIp"]}},
        },
    }
    return yaml.dump(template, default_flow_style=False)
