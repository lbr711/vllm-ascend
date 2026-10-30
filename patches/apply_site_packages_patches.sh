#!/usr/bin/env bash
# 在容器里给 vllm / vllm_ascend 打 snapshot patch
#
# 用法:
#   apply_site_packages_patches.sh <vllm的patch绝对路径> <vllm-ascend的patch绝对路径>
#
# patch 为 git diff 格式 (a/vllm/...)，在含 vllm/ 子目录的包根下须用 -p1，不能用 -p0。

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "用法: $0 <vllm_patch> <vllm_ascend_patch>"
    exit 1
fi

VLLM_PATCH="$1"
VLLM_ASCEND_PATCH="$2"

for f in "$VLLM_PATCH" "$VLLM_ASCEND_PATCH"; do
    if [ ! -f "$f" ]; then
        echo "错误: 找不到 patch 文件: $f"
        exit 1
    fi
done

PATCH_STRIP=1

# 镜像里 vllm 装在 python3.11.10 的 site-packages，勿用系统 /usr/bin/python3
if [ -z "${PYTHON:-}" ]; then
    for candidate in \
        /usr/local/python3.12.13/bin/python3 \
        /usr/local/python3.12/bin/python3 \
        /usr/local/python3.11.10/bin/python3 \
        /usr/local/python3.11/bin/python3 \
        "$(command -v python3.12 2>/dev/null || true)" \
        "$(command -v python3.11 2>/dev/null || true)" \
        "$(command -v python3 2>/dev/null || true)"; do
        [ -n "$candidate" ] && [ -x "$candidate" ] || continue
        if "$candidate" -m pip show vllm >/dev/null 2>&1; then
            PYTHON=$candidate
            break
        fi
    done
    PYTHON=${PYTHON:-python3}
fi
echo "using python: $PYTHON"

# 定位含 subdir/ 的包根；优先 pip/filesystem，避免 import 触发 NPU 驱动初始化
resolve_package_root() {
    local pip_name="$1"
    local module_name="$2"
    local subdir="$3"
    local loc=""
    local editable=""
    local root=""

    # 1) Editable install: real sources under "Editable project location"
    #    (site-packages only has __editable__*.pth, no vllm/ directory).
    editable=$("$PYTHON" -m pip show "$pip_name" 2>/dev/null | awk -F': ' '/^Editable project location:/{print $2; exit}')
    if [ -n "$editable" ] && [ -d "$editable/$subdir" ]; then
        echo "$editable"
        return 0
    fi

    # 2) Regular site-packages install
    loc=$("$PYTHON" -m pip show "$pip_name" 2>/dev/null | awk -F': ' '/^Location:/{print $2; exit}')
    if [ -n "$loc" ] && [ -d "$loc/$subdir" ]; then
        echo "$loc"
        return 0
    fi

    # 3) Common filesystem locations (workspace editable + 3.11/3.12 site-packages)
    for loc in \
        /vllm-workspace/vllm \
        /vllm-workspace/vllm-ascend \
        /usr/local/python3.12.13/lib/python3.12/site-packages \
        /usr/local/python3.12/lib/python3.12/site-packages \
        /usr/local/python3.11.10/lib/python3.11/site-packages \
        /usr/local/lib/python3.11/site-packages \
        /usr/local/lib64/python3.11/site-packages; do
        if [ -d "$loc/$subdir" ]; then
            echo "$loc"
            return 0
        fi
    done

    # 4) Import fallback. Plugins spam INFO to stdout — keep last non-empty line.
    root=$("$PYTHON" - "$module_name" "$subdir" <<'PY' 2>/dev/null | awk 'NF{line=$0} END{print line}' | tr -d ''
import os
import sys
import importlib

module_name, subdir = sys.argv[1], sys.argv[2]
mod = importlib.import_module(module_name)
root = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
if not os.path.isdir(os.path.join(root, subdir)):
    raise SystemExit(1)
print(root)
PY
) || true

    if [ -n "$root" ] && [ -d "$root/$subdir" ]; then
        echo "$root"
        return 0
    fi

    return 1
}


check_patch_targets_exist() {
    local patch_file="$1"
    local pkg_root="$2"
    local missing=()

    while IFS= read -r rel; do
        [ -z "$rel" ] && continue
        if [ ! -f "$pkg_root/$rel" ]; then
            missing+=("$rel")
        fi
    done < <(awk '/^--- a\// { sub("^--- a/", ""); print }' "$patch_file" | sort -u)

    if [ ${#missing[@]} -gt 0 ]; then
        echo "错误: patch 目标文件不存在 (${#missing[@]} 个), 包根目录: $pkg_root"
        printf '  - %s\n' "${missing[@]}"
        return 1
    fi
    return 0
}

# 用关键代码指纹判断 patch 是否完整生效（勿仅用 patch -R，--batch 下易误判）
snapshot_patch_applied() {
    local pip_name="$1"
    local pkg_root="$2"

    case "$pip_name" in
    vllm)
        grep -q 'async def suspend(self, model_save_path' "$pkg_root/vllm/v1/engine/async_llm.py" \
            && grep -q 'async def suspend_async' "$pkg_root/vllm/v1/engine/core_client.py" \
            && grep -q 'async def suspend(self, model_save_path' "$pkg_root/vllm/engine/protocol.py"
        ;;
    vllm-ascend)
        # v16+ moved snapshot lifecycle out of worker.py into snapshot/*.
        if [ -f "$pkg_root/vllm_ascend/snapshot/worker_lifecycle.py" ] \
            && [ -f "$pkg_root/vllm_ascend/snapshot/distributed.py" ] \
            && grep -q 'def resume_worker' "$pkg_root/vllm_ascend/snapshot/worker_lifecycle.py" \
            && grep -q 'def cleanup_dist_env_for_snapshot' "$pkg_root/vllm_ascend/snapshot/distributed.py"; then
            return 0
        fi
        # older patches kept the hooks inside worker.py
        grep -q 'cleanup_dist_env_for_snapshot' "$pkg_root/vllm_ascend/worker/worker.py" \
            && grep -q 'destroy_ascend_model_parallel' "$pkg_root/vllm_ascend/worker/worker.py"
        ;;
    *)
        return 1
        ;;
    esac
}

verify_snapshot_patch_applied() {
    local pip_name="$1"
    local pkg_root="$2"

    if snapshot_patch_applied "$pip_name" "$pkg_root"; then
        echo ">>> $pip_name snapshot patch 指纹校验通过"
        return 0
    fi
    echo "错误: $pip_name patch 应用后仍缺少 snapshot 关键代码，请检查 patch 与安装版本是否匹配"
    return 1
}

apply_patch() {
    local pip_name="$1"
    local module_name="$2"
    local subdir="$3"
    local patch_file="$4"

    echo ""
    echo ">>> 处理 $pip_name"

    local pkg_root
    pkg_root=$(resolve_package_root "$pip_name" "$module_name" "$subdir") || {
        echo "错误: 无法定位 $pip_name 包根目录（需包含 $subdir/ 子目录）"
        exit 1
    }

    echo "package root: $pkg_root"
    echo "installed version: $($PYTHON -m pip show "$pip_name" 2>/dev/null | awk -F': ' '/^Version:/{print $2; exit}')"
    if [ -f "$pkg_root/$subdir/__init__.py" ]; then
        echo "module path: $pkg_root/$subdir/__init__.py"
    else
        echo "module path: (未 import，避免触发 NPU 驱动)"
    fi
    echo "patch strip level: -p$PATCH_STRIP"

    if ! check_patch_targets_exist "$patch_file" "$pkg_root"; then
        exit 1
    fi

    if [ "${FORCE_REAPPLY:-0}" != "1" ] && snapshot_patch_applied "$pip_name" "$pkg_root"; then
        echo ">>> $pip_name snapshot patch 已存在（指纹校验）, 跳过"
        return 0
    fi

    if [ "${FORCE_REAPPLY:-0}" = "1" ]; then
        echo ">>> FORCE_REAPPLY=1, 强制重新打 patch"
    else
        echo ">>> $pip_name snapshot patch 未检测到，开始应用"
    fi

    pushd "$pkg_root" >/dev/null

    local dry_run_log
    dry_run_log=$(mktemp)
    if ! patch -p"$PATCH_STRIP" --dry-run --verbose < "$patch_file" >"$dry_run_log" 2>&1; then
        echo "错误: $pip_name patch 无法应用（dry-run 失败）"
        echo "patch 试运行输出 (末尾 40 行):"
        tail -n 40 "$dry_run_log"
        rm -f "$dry_run_log"
        popd >/dev/null
        exit 1
    fi
    rm -f "$dry_run_log"

    dry_run_log=$(mktemp)
    if ! patch -p"$PATCH_STRIP" --verbose < "$patch_file" >"$dry_run_log" 2>&1; then
        echo "错误: $pip_name patch 应用失败"
        echo "patch 输出 (末尾 40 行):"
        tail -n 40 "$dry_run_log"
        rm -f "$dry_run_log"
        popd >/dev/null
        exit 1
    fi
    rm -f "$dry_run_log"
    popd >/dev/null

    verify_snapshot_patch_applied "$pip_name" "$pkg_root" || exit 1
    echo ">>> $pip_name patch 应用成功"
}

apply_patch "vllm" "vllm" "vllm" "$VLLM_PATCH"
apply_patch "vllm-ascend" "vllm_ascend" "vllm_ascend" "$VLLM_ASCEND_PATCH"

echo ""
echo "全部完成"
