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
  # default     = "g6.xlarge" # GPU Instance (Commented out for now)
  default = "t3.xlarge"
}

variable "repo_url" {
  description = "URL of the Git repository to clone"
  type        = string
  default     = "https://github.com/qualabs/SMPTE-Copilot"
}
