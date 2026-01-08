terraform {
  required_version = ">= 1.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    profile      = "smpte-copilot"
    bucket       = "summer-project-smpte-copilot-tfstate"
    key          = "terraform.tfstate"
    region       = "us-east-1"
    use_lockfile = true
    encrypt      = true
  }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project    = "summer-project-smpte-copilot"
      ManagedBy  = "Terraform"
      Repository = var.repo_url
    }
  }
}
