variable "region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI/SDK profile to use for provider and CLI snippets"
  type        = string
  default     = "smpte-copilot"
}

variable "project_name" {
  description = "Name of the project used for naming resources"
  type        = string
  default     = "smpte-copilot"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default = "t3.xlarge"
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU-enabled server"
  type        = string
  default     = "g6.xlarge"
}

variable "repo_url" {
  description = "URL of the Git repository to clone"
  type        = string
  default     = "https://github.com/qualabs/SMPTE-Copilot"
}

variable "ssh_public_key" {
  description = "Public SSH key (OpenSSH format) to install on both instances. Example: ssh-ed25519 AAAA... user@host"
  type        = string
  default     = "smpte-copilot.pub"
}

variable "custom_domain_name" {
  description = "Custom domain name for CloudFront distribution"
  type        = string
  default     = "smpte-copilot.qualabs.com"
}

variable "enable_custom_domain" {
  description = "Enable custom domain for CloudFront distribution. Set to false for first deployment, then true after DNS validation"
  type        = bool
  default     = false
}
