FROM nvidia/cuda:11.1.1-base
RUN apk add ca-certificates

RUN apk add --update --no-cache python3 py3-pip \
&& rm -rf /var/cache/apk/*
# 复制项目文件到容器内
COPY . /app
# 设置工作目录
WORKDIR /app

RUN pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple \
&& pip config set global.trusted-host mirrors.cloud.tencent.com \
&& pip install --upgrade pip \
# pip install scipy 等数学包失败，可使用 apk add py3-scipy 进行， 参考安装 https://pkgs.alpinelinux.org/packages?name=py3-scipy&branch=v3.13
&& pip install --user -r requirements.txt


# 声明运行时端口
EXPOSE 80

CMD ["python", "in_model.py", "0.0.0.0", "80"]