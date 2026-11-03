#!/bin/bash

# 健康检查脚本
# 用法: ./health_check.sh [IP] [mgmt_port]

IP=$POD_IP
MGMT_PORT=9000
URL="http://${IP}:${MGMT_PORT}/snapshot/health"
RESPONSE=$(curl -s -w "\n%{http_code}" "${URL}")
BODY=$(echo "${RESPONSE}" | head -n -1)
HTTP_CODE=$(echo "${RESPONSE}" | tail -n 1)
if [ "${HTTP_CODE}" = "200" ] ; then
    exit 0
else
    echo "健康检查失败: ${URL} HTTP状态码: ${HTTP_CODE}, 响应体: ${BODY}"
    exit 1
fi
