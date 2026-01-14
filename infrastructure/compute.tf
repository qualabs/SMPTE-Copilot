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

# Standard Ubuntu AMI (Cheap/CPU only)
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# GPU Deep Learning AMI (for GPU instance)
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name = "name"
    # Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# EC2 Instance
resource "aws_instance" "app_server" {
  # ami           = data.aws_ami.deep_learning.id # GPU AMI
  ami           = data.aws_ami.ubuntu.id # Standard AMI
  instance_type = var.instance_type
  subnet_id     = aws_subnet.public.id

  vpc_security_group_ids = [aws_security_group.app_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.app_profile.name

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  tags = {
    Name = "${var.project_name}-server"
  }

  # Basic user_data to ensure Docker is running and app is deployed
  user_data = templatefile("${path.module}/user_data.sh", {
    repo_url = var.repo_url
  })
}

# Separate GPU-enabled EC2 Instance (new)
resource "aws_instance" "gpu_server" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = var.gpu_instance_type
  subnet_id     = aws_subnet.public.id

  vpc_security_group_ids = [aws_security_group.app_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.app_profile.name

  root_block_device {
    volume_size = 100
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
