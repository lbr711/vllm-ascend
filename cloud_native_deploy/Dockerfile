FROM docker.m.daocloud.io/library/python:3.11-slim AS builder

ARG KUBECTL_VERSION=v1.29.6
ARG TARGETARCH

COPY .docker-build/his-proxy-ca.crt /usr/local/share/ca-certificates/his-proxy-build-ca.crt
COPY examples/deployer/requirements.txt /tmp/requirements.txt

RUN set -eu; \
    target_arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${target_arch}" in \
      amd64|arm64) ;; \
      *) echo "Unsupported target architecture: ${target_arch}" >&2; exit 1 ;; \
    esac; \
    update-ca-certificates; \
    sed -i \
      -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
      -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
      /etc/apt/sources.list.d/debian.sources; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    curl -fL --connect-timeout 20 --max-time 300 -o /tmp/kubectl.deb \
      "https://mirrors.aliyun.com/kubernetes-new/core/stable/${KUBECTL_VERSION%.*}/deb/${target_arch}/kubectl_${KUBECTL_VERSION#v}-1.1_${target_arch}.deb"; \
    dpkg-deb -x /tmp/kubectl.deb /tmp/kubectl-extract; \
    install -m 0755 /tmp/kubectl-extract/usr/bin/kubectl /usr/local/bin/kubectl

RUN pip install \
      --no-cache-dir \
      --timeout 120 \
      --retries 10 \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --prefix=/install \
      -r /tmp/requirements.txt

COPY . /src

RUN rm -rf /src/.docker-build

FROM docker.m.daocloud.io/library/python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /usr/local/bin/kubectl /usr/local/bin/kubectl
COPY --from=builder /install/ /usr/local/

WORKDIR /app

COPY --from=builder /src/ /app/

RUN chmod +x /app/examples/cloud_native_deploy/entrypoint.sh \
    /app/examples/cloud_native_deploy/entrypoint-cleanup.sh \
    /app/examples/cloud_native_deploy/entrypoint-scale.sh \
    /app/examples/cloud_native_deploy/cleanup.sh

ENTRYPOINT ["/app/examples/cloud_native_deploy/entrypoint.sh"]
