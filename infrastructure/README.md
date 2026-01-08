# SMPTE-Copilot Infrastructure

This directory contains Terraform code to provision the AWS infrastructure.

## Prerequisites
1. **Terraform**: [Install Terraform](https://developer.hashicorp.com/terraform/downloads)
2. **AWS CLI**: [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and configure with `aws configure`.
3. **Session Manager Plugin**: [Install Session Manager Plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (required for connecting).
4. **AWS Authentication**: Set up your AWS CLI profile (default: `smpte-copilot`).


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