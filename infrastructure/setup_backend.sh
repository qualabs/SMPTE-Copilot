#!/bin/bash
set -e

BUCKET_NAME="summer-project-smpte-copilot-tfstate"
REGION="us-east-1"
REPO_URL="https://github.com/qualabs/SMPTE-Copilot"
PROJECT="summer-project-smpte-copilot"

echo "Using Region: $REGION"
echo "Bucket Name: $BUCKET_NAME"

# 1. Create S3 Bucket
if aws s3api head-bucket --bucket "$BUCKET_NAME" --profile smpte-copilot 2>/dev/null; then
  echo "Bucket $BUCKET_NAME already exists."
else
  echo "Creating bucket $BUCKET_NAME..."
  if [ "$REGION" == "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" --profile smpte-copilot
  else
    aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" --create-bucket-configuration LocationConstraint="$REGION" --profile smpte-copilot
  fi
fi

# 2. Tag S3 Bucket
echo "Tagging bucket..."
aws s3api put-bucket-tagging --bucket "$BUCKET_NAME" --tagging "TagSet=[{Key=Project,Value=${PROJECT}},{Key=ManagedBy,Value=ManualScript},{Key=Repository,Value=${REPO_URL}}]" --profile smpte-copilot

# 3. Enable Versioning
echo "Enabling versioning on bucket..."
aws s3api put-bucket-versioning --bucket "$BUCKET_NAME" --versioning-configuration Status=Enabled --profile smpte-copilot

echo "Backend infrastructure ready."
