output "export_instance_id" {
  description = "Command to export instance ID as environment variable"
  value       = "export INSTANCE_ID=${aws_instance.server.id}"
}

output "connection_command" {
  description = "Command to connect to the instance using SSM"
  value       = "aws ssm start-session --target ${aws_instance.server.id} --region ${var.region} --profile ${var.aws_profile}"
}

output "s3_console_url" {
  description = "Console URL for the S3 bucket"
  value       = "https://${var.region}.console.aws.amazon.com/s3/buckets/${aws_s3_bucket.data.id}?region=${var.region}"
}

output "cloudfront_domain_name" {
  description = "CloudFront domain name for OpenWebUI (use for Google OAuth settings)"
  value       = aws_cloudfront_distribution.openwebui.domain_name
}

output "cloudfront_url" {
  description = "Full URL to access OpenWebUI via CloudFront"
  value       = "https://${aws_cloudfront_distribution.openwebui.domain_name}"
}