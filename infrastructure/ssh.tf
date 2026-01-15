resource "aws_key_pair" "main" {
  key_name   = "${var.project_name}-ssh-key"
  public_key = trimspace(file("${path.module}/${var.ssh_public_key}"))

  tags = {
    Name = "${var.project_name}-ssh-key"
  }
}
