# 创建本地 bin 目录（如果不存在）
mkdir -p "$HOME/.local/bin"

# 下载 docker-compose v2.39.1
curl -L "https://github.com/docker/compose/releases/download/v2.39.1/docker-compose-linux-x86_64" \
  -o "$HOME/.local/bin/docker-compose"

# 添加可执行权限
chmod +x "$HOME/.local/bin/docker-compose"

# 确保 ~/.local/bin 在 PATH 中（避免重复添加）
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  export PATH="$HOME/.local/bin:$PATH"
fi

# 立即生效
source "$HOME/.bashrc" 2>/dev/null || true

# 测试安装结果
docker-compose version || docker compose version
