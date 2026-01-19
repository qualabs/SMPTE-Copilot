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

# Provider for ACM certificate in us-east-1 (CloudFront always requires us-east-1)
provider "aws" {
  alias   = "us_east_1"
  region  = "us-east-1"
  profile = var.aws_profile

  default_tags {
    tags = {
      Project    = "summer-project-smpte-copilot"
      ManagedBy  = "Terraform"
      Repository = var.repo_url
    }
  }
}


moved {
  from = aws_instance.gpu_server
  to = aws_instance.server
}