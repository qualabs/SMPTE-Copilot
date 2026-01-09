resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-bucket"

  force_destroy = false

  tags = {
    Name = "${var.project_name}-data-bucket"
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


