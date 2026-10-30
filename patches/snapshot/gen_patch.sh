#!/usr/bin/env bash
# 从 vllm / vllm-ascend 仓库生成 site-packages 可用的 snapshot patch（不含 tests）
#
# 用法:
#   gen_snapshot_patch.sh [选项]
#
# 选项:
#   -r, --repo PATH       仓库路径 (默认: 当前目录)
#   -b, --base COMMIT     基准提交 / tag（必填）
#   -B, --branch NAME     期望分支名（可选；不一致时仅警告）
#   -p, --path DIR        纳入 patch 的包目录，如 vllm/ 或 vllm_ascend/
#                         （默认: 自动检测仓库内的 vllm_ascend/ 或 vllm/）
#   -o, --output-dir DIR  输出目录 (默认: 本脚本所在目录)
#   -n, --name NAME       输出文件名 (默认: 自动递增 <pkg>_<branch>_site_packages_vN.patch)
#   -f, --force           覆盖已存在的同名 patch（默认不覆盖）
#   -h, --help            显示帮助
#
# 示例:
#   # vllm-ascend: b89a49164 → snapshot_b111
#   ./gen_snapshot_patch.sh \
#     -r /home/liu/vllm-project/vllm-ascend \
#     -b b89a49164 \
#     -B snapshot_b111
#
#   # vllm: v0.23.0 → snapshot_0.23.0
#   ./gen_snapshot_patch.sh \
#     -r /home/liu/vllm-project/vllm \
#     -b v0.23.0 \
#     -B snapshot_0.23.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO="$(pwd)"
BASE_COMMIT=""
BRANCH=""
PKG_PATH=""
OUTPUT_DIR="$SCRIPT_DIR"
CUSTOM_NAME=""
FORCE=0

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--repo) REPO="$2"; shift 2 ;;
        -b|--base) BASE_COMMIT="$2"; shift 2 ;;
        -B|--branch) BRANCH="$2"; shift 2 ;;
        -p|--path) PKG_PATH="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -n|--name) CUSTOM_NAME="$2"; shift 2 ;;
        -f|--force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知选项: $1"; usage; exit 1 ;;
    esac
done

if [ -z "$BASE_COMMIT" ]; then
    echo "错误: 必须指定基准提交，例如 -b b89a49164 或 -b v0.23.0"
    usage
    exit 1
fi

if [ ! -d "$REPO/.git" ]; then
    echo "错误: 不是 git 仓库: $REPO"
    exit 1
fi

# 归一化包路径：去掉首尾斜杠，再统一成 "name/"
normalize_pkg_path() {
    local p="${1%/}"
    p="${p#./}"
    echo "${p}/"
}

detect_pkg_path() {
    if [ -d "$REPO/vllm_ascend" ] && [ -d "$REPO/vllm" ]; then
        echo "错误: 仓库同时存在 vllm_ascend/ 与 vllm/，请用 -p 显式指定" >&2
        return 1
    fi
    if [ -d "$REPO/vllm_ascend" ]; then
        echo "vllm_ascend/"
        return 0
    fi
    if [ -d "$REPO/vllm" ]; then
        echo "vllm/"
        return 0
    fi
    echo "错误: 未找到 vllm_ascend/ 或 vllm/，请用 -p 指定包目录" >&2
    return 1
}

if [ -n "$PKG_PATH" ]; then
    PKG_PATH="$(normalize_pkg_path "$PKG_PATH")"
else
    PKG_PATH="$(detect_pkg_path)"
fi

if [ ! -d "$REPO/${PKG_PATH%/}" ]; then
    echo "错误: 包目录不存在: $REPO/${PKG_PATH%/}"
    exit 1
fi

PKG_NAME="${PKG_PATH%/}"
BRANCH_LABEL="${BRANCH:-$(cd "$REPO" && git rev-parse --abbrev-ref HEAD)}"
# 分支名里的 / 等字符换成 _；命名对齐旧脚本: <pkg>_<branch>_site_packages_vN.patch
SAFE_BRANCH="$(echo "$BRANCH_LABEL" | tr '/ ' '__')"
PATCH_PREFIX="${PKG_NAME}_${SAFE_BRANCH}_site_packages"

mkdir -p "$OUTPUT_DIR"

next_version_name() {
    local max=0
    local f base ver
    for f in "$OUTPUT_DIR"/${PATCH_PREFIX}_v*.patch; do
        [ -e "$f" ] || continue
        base=$(basename "$f")
        if [[ "$base" =~ _v([0-9]+)\.patch$ ]]; then
            ver="${BASH_REMATCH[1]}"
            if (( ver > max )); then
                max=$ver
            fi
        fi
    done
    echo "${PATCH_PREFIX}_v$((max + 1)).patch"
}

if [ -n "$CUSTOM_NAME" ]; then
    OUT_FILE="$OUTPUT_DIR/$CUSTOM_NAME"
else
    OUT_FILE="$OUTPUT_DIR/$(next_version_name)"
fi

if [ -f "$OUT_FILE" ] && [ "$FORCE" != "1" ]; then
    echo "错误: 文件已存在，未覆盖: $OUT_FILE"
    echo "提示: 使用 -f 强制覆盖，或省略 -n 让脚本自动递增版本号"
    exit 1
fi

cd "$REPO"

if ! git cat-file -e "${BASE_COMMIT}^{commit}" 2>/dev/null; then
    echo "错误: 基准提交不存在: $BASE_COMMIT"
    exit 1
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ -n "$BRANCH" ] && [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo "警告: 当前分支为 $CURRENT_BRANCH，期望 $BRANCH"
fi

HEAD_COMMIT=$(git rev-parse HEAD)
if ! git merge-base --is-ancestor "$BASE_COMMIT" HEAD 2>/dev/null; then
    echo "错误: 基准提交 $BASE_COMMIT 不是 HEAD ($HEAD_COMMIT) 的祖先"
    exit 1
fi

echo ">>> 仓库:     $REPO"
echo ">>> 分支:     $CURRENT_BRANCH (HEAD: $HEAD_COMMIT)"
echo ">>> 基准:     $BASE_COMMIT"
echo ">>> 包路径:   $PKG_PATH （tests/ 不纳入）"
echo ">>> 输出:     $OUT_FILE"
echo ""

echo ">>> 包含的提交:"
git log --oneline "${BASE_COMMIT}..HEAD"
echo ""

CHANGED_FILES=$(git diff --name-only "${BASE_COMMIT}..HEAD" -- "$PKG_PATH" || true)
TEST_IN_DIFF=$(git diff --name-only "${BASE_COMMIT}..HEAD" -- tests/ || true)

if [ -z "$CHANGED_FILES" ]; then
    echo "错误: ${BASE_COMMIT}..HEAD 之间没有 ${PKG_PATH} 变更"
    exit 1
fi

echo ">>> ${PKG_PATH} 变更文件 ($(echo "$CHANGED_FILES" | wc -l) 个):"
echo "$CHANGED_FILES" | sed 's/^/  /'

if [ -n "$TEST_IN_DIFF" ]; then
    echo ""
    echo ">>> 已排除 tests/ 变更 ($(echo "$TEST_IN_DIFF" | wc -l) 个，不写入 patch):"
    echo "$TEST_IN_DIFF" | sed 's/^/  /'
fi

# 仅导出包目录，天然排除 tests/ 以及其它仓库根文件
git diff "${BASE_COMMIT}..HEAD" -- "$PKG_PATH" > "$OUT_FILE"

PATCH_FILES=$(grep -c '^diff --git' "$OUT_FILE" || true)
# 兜底：确认 patch 正文里没有 tests/ 路径
if grep -E '^(diff --git a/|\+\+\+ b/|--- a/)tests/' "$OUT_FILE" >/dev/null 2>&1; then
    echo "错误: patch 中意外包含 tests/ 路径"
    rm -f "$OUT_FILE"
    exit 1
fi

LINES=$(wc -l < "$OUT_FILE")
SIZE=$(stat -c%s "$OUT_FILE" 2>/dev/null || stat -f%z "$OUT_FILE")

echo ""
echo ">>> 生成完成"
echo "    文件:     $OUT_FILE"
echo "    大小:     $SIZE bytes, $LINES 行"
echo "    文件数:   $PATCH_FILES"
echo ""
echo ">>> 应用示例 (在 site-packages 对应包目录的上一级):"
echo "    patch -p1 < $OUT_FILE"
echo "    # 或配合你现有的 apply_site_packages_patches.sh"
