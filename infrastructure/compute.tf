# Security Group
resource "aws_security_group" "app_sg" {
  name        = "${var.project_name}-sg"
  description = "Security group for SMPTE-Copilot"
  vpc_id      = aws_vpc.main.id

  # Allow SSH from EC2 Instance Connect Endpoint
  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.vpc_endpoints.id]
    description     = "SSH from Instance Connect Endpoint"
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }

  # Allow OpenWebUI access from CloudFront
  ingress {
    from_port       = 3000
    to_port         = 3000
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]
    description     = "OpenWebUI from CloudFront"
  }

  # Allow all outbound traffic (Required for SSM and package updates)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sg"
  }
}

resource "aws_instance" "server" {
  ami           = "ami-0765437123d36eaa8" # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Ubuntu 24.04) 20260103
  instance_type = var.gpu_instance_type
  subnet_id     = aws_subnet.public.id

  vpc_security_group_ids = [aws_security_group.app_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.app_profile.name
  key_name                = aws_key_pair.main.key_name

  root_block_device {
    volume_size = 200
    volume_type = "gp3"
  }

  tags = {
    Name = "${var.project_name}-gpu-server"
  }

  # Reuse same user_data to keep setup consistent across instances
  user_data = templatefile("${path.module}/user_data.sh", {
    repo_url = var.repo_url
  })
}

# Elastic IP for stable DNS
resource "aws_eip" "server" {
  domain   = "vpc"
  instance = aws_instance.server.id

  tags = {
    Name = "${var.project_name}-server-eip"
  }
}


 