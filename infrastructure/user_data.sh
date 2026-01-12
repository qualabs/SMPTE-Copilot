#!/bin/bash
set -e

# 1. Install Docker from official repository
apt-get update
apt-get install -y ca-certificates curl gnupg git

# Add Docker's official GPG key
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Install EC2 Instance Connect package for EIC access
apt-get install -y ec2-instance-connect

# 2. Start Docker
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Ensure SSH is running (required for EC2 Instance Connect)
systemctl restart ssh || systemctl restart sshd || true

# Pre-configure ssm-user to ensure permissions and set bash as default shell
if ! id "ssm-user" &>/dev/null; then
    useradd -m -s /bin/bash ssm-user
    echo "ssm-user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ssm-user
    chmod 0440 /etc/sudoers.d/ssm-user
else
    # Change shell to bash if user already exists
    usermod -s /bin/bash ssm-user
fi
usermod -aG docker ssm-user

# Create a shared group for SSM users and grant access
if ! getent group ssm-users >/dev/null; then
    groupadd ssm-users
fi
usermod -aG ssm-users ubuntu
usermod -aG ssm-users ssm-user

# Ensure a collaborative umask so group write is preserved
umask 0002

# Trust the app directory for all users to avoid Git "dubious ownership"
git config --system --add safe.directory /opt/app || true

# 3. Setup Application Directory
# Using /opt because /home/ubuntu is restricted to the ubuntu user
APP_DIR="/opt/app"
mkdir -p $APP_DIR
# Change ownership to root:ssm-users so SSM users have access.
chown -R root:ssm-users $APP_DIR
chmod -R 2775 $APP_DIR

# 4. Clone/Pull Repository
cd $APP_DIR
if [ -d ".git" ]; then
    git fetch origin
    git reset --hard origin/main
    git pull
else
    git clone ${repo_url} .
fi

# Ensure repo is group-shared and fix permissions for multi-user edits
git -C "$APP_DIR" config core.sharedRepository group || true
chgrp -R ssm-users "$APP_DIR"
chmod -R g+rwX "$APP_DIR"
find "$APP_DIR" -type d -exec chmod g+s {} +

# 5. Pre-pull images and build with Docker Compose
docker compose pull --include-deps
docker compose build --pull

