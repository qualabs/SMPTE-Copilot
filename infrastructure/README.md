# SMPTE-Copilot Infrastructure

This directory contains Terraform code to provision the AWS infrastructure.

The application is deployed to a EC2 instance on the `/opt/app directory`

## Prerequisites
1. **Terraform**: [Install Terraform](https://developer.hashicorp.com/terraform/downloads)
2. **AWS CLI**: [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and configure with `aws configure`.
3. **Session Manager Plugin**: [Install Session Manager Plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (required for connecting).
4. **AWS Authentication**: Set up your AWS CLI profile (default: `smpte-copilot`).


## Deployment Guide

### Two-Stage Deployment for Custom Domain

This infrastructure supports a custom domain (configured via `custom_domain_name` variable) with a two-stage deployment process:

#### Stage 1: Initial Deployment (Includes Certificate Creation)

1. Ensure `enable_custom_domain = false` in your configuration (this is the default).
2. Run the initial deployment:
   ```bash
   cd infrastructure
   terraform init
   terraform apply
   ```
3. After the deployment completes, retrieve the DNS validation records:
   ```bash
   terraform output acm_certificate_validation_records
   ```
4. Add the CNAME record to your DNS provider (e.g., Route53, Cloudflare) to validate the certificate:
   - Name: (from output)
   - Type: CNAME
   - Value: (from output)

5. Wait for DNS propagation and certificate validation (typically 5-30 minutes). You can check the status in the AWS Console under Certificate Manager (us-east-1 region).

#### Stage 2: Enable Custom Domain

1. Once the certificate is validated, update your `terraform.tfvars` or set the variable:
   ```hcl
   enable_custom_domain = true
   ```
2. Apply the changes:
   ```bash
   terraform apply
   ```
3. Update your DNS to point your custom domain to CloudFront:
   - Create a CNAME record for your custom domain (value of `custom_domain_name`) pointing to the CloudFront distribution domain (from `cloudfront_domain_name` output).

4. Access your application at your custom domain URL (see `custom_domain_url` output)

**Note**: If you skip Stage 1 and set `enable_custom_domain = true` immediately, Terraform will wait for certificate validation during apply, which can take 30+ minutes.

## Deployment (if not already deployed)

1. **Initialize Terraform**:
   ```bash
   cd infrastructure
   terraform init
   ```

2. **Review the Plan**:
   ```bash
   terraform plan
   ```

3. **Apply**:
   ```bash
   terraform apply
   ```
   Type `yes` when prompted.

If the S3 backend bucket does not exist yet, prepare it first:

- Using the helper script (recommended):
   ```bash
   cd infrastructure
   ./setup_backend.sh
   ```

Then re-run `terraform init`.

## Get Instance ID
After applying, set the instance ID as an environment variable for easy access:
```bash
eval $(terraform output -raw export_instance_id)
echo $INSTANCE_ID
```
This allows you to use `$INSTANCE_ID` in all subsequent commands.

## Connecting to the Instance
To get a shell session on the instance:
```bash
aws ssm start-session --target $INSTANCE_ID --region us-east-1 --profile ${AWS_PROFILE:-smpte-copilot}
```

- **Security**:
   - No inbound ports are exposed by default. Connect using SSM sessions or EC2 Instance Connect

## Application Management

### Deployment
On the first boot, the instance will:
1.  Install Docker and Git.
2.  Clone the repository from `https://github.com/qualabs/SMPTE-Copilot` to `/opt/app`.
3. You'll need to configure the API keys and run the app with Docker.


## Clean Up
To destroy all resources:
```bash
terraform destroy
```