#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/release_bump.sh [--bump patch|minor|major] [--version x.y.z]

Examples:
  scripts/release_bump.sh --bump patch
  scripts/release_bump.sh --version 1.25.0
EOF
}

validate_semver() {
  local version="$1"
  [[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

semver_gt() {
  local left="$1"
  local right="$2"

  local l_major l_minor l_patch r_major r_minor r_patch
  IFS='.' read -r l_major l_minor l_patch <<< "${left}"
  IFS='.' read -r r_major r_minor r_patch <<< "${right}"

  if (( l_major != r_major )); then
    (( l_major > r_major ))
    return
  fi
  if (( l_minor != r_minor )); then
    (( l_minor > r_minor ))
    return
  fi
  (( l_patch > r_patch ))
}

bump_level="patch"
explicit_version=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bump)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --bump" >&2
        usage
        exit 1
      fi
      bump_level="$2"
      shift 2
      ;;
    --version)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --version" >&2
        usage
        exit 1
      fi
      explicit_version="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

current_version="$(perl -0ne 'if (/\[package\]\nname = "tokenizers-ffi"\nversion = "([^"]+)"/s) { print $1; exit }' crates/tokenizers/Cargo.toml)"
if [[ -z "${current_version}" ]]; then
  echo "Failed to read current version from crates/tokenizers/Cargo.toml" >&2
  exit 1
fi
if ! validate_semver "${current_version}"; then
  echo "Current version is not semver: ${current_version}" >&2
  exit 1
fi

latest_tag="$(git tag -l 'v*' --sort=-v:refname | head -n1 || true)"
latest_tag_version=""
if [[ -n "${latest_tag}" ]]; then
  latest_tag_version="${latest_tag#v}"
  if ! validate_semver "${latest_tag_version}"; then
    latest_tag_version=""
  fi
fi

base_version="${current_version}"
if [[ -n "${latest_tag_version}" ]] && semver_gt "${latest_tag_version}" "${base_version}"; then
  base_version="${latest_tag_version}"
fi

if [[ -n "${explicit_version}" ]]; then
  if ! validate_semver "${explicit_version}"; then
    echo "Invalid semver value for --version: ${explicit_version}" >&2
    exit 1
  fi
  if ! semver_gt "${explicit_version}" "${base_version}"; then
    echo "Explicit version must be greater than ${base_version}: ${explicit_version}" >&2
    exit 1
  fi
  next_version="${explicit_version}"
else
  IFS='.' read -r major minor patch <<< "${base_version}"
  case "${bump_level}" in
    patch)
      patch=$((patch + 1))
      ;;
    minor)
      minor=$((minor + 1))
      patch=0
      ;;
    major)
      major=$((major + 1))
      minor=0
      patch=0
      ;;
    *)
      echo "Invalid --bump value: ${bump_level} (expected patch|minor|major)" >&2
      exit 1
      ;;
  esac
  next_version="${major}.${minor}.${patch}"
fi

next_symbol="${next_version//./_}"

perl -0pi -e "s/(\\[package\\]\\nname = \"tokenizers-ffi\"\\nversion = \")[^\"]+(\"\\n)/\$1${next_version}\$2/s" crates/tokenizers/Cargo.toml
perl -0pi -e "s/(\\[\\[package\\]\\]\\nname = \"tokenizers-ffi\"\\nversion = \")[^\"]+(\"\\n)/\$1${next_version}\$2/s" Cargo.lock
perl -pi -e "s/tokenizers_version_\\d+_\\d+_\\d+/tokenizers_version_${next_symbol}/g" tokenizer.go crates/tokenizers/src/lib.rs
perl -pi -e 'if (/name = "tokenizers_rs"/) { $in = 1 } if ($in && /version = "/) { s/version = "[^"]+"/version = "'"${next_version}"'"/; $in = 0 }' crates/tokenizers/BUILD.bazel
perl -0pi -e "s/(module\\(\\n\\s*name = \"com_github_daulet_tokenizers\",\\n\\s*version = \")[^\"]+(\",\\n\\))/\$1${next_version}\$2/s" MODULE.bazel

if ! grep -Eq "^version = \"${next_version}\"$" crates/tokenizers/Cargo.toml; then
  echo "Failed to write crates/tokenizers/Cargo.toml version ${next_version}" >&2
  exit 1
fi
if ! grep -Eq "tokenizers_version_${next_symbol}" tokenizer.go; then
  echo "Failed to write tokenizer.go symbol tokenizers_version_${next_symbol}" >&2
  exit 1
fi
if ! grep -Eq "tokenizers_version_${next_symbol}" crates/tokenizers/src/lib.rs; then
  echo "Failed to write crates/tokenizers/src/lib.rs symbol tokenizers_version_${next_symbol}" >&2
  exit 1
fi
if ! grep -Eq "^    version = \"${next_version}\",$" MODULE.bazel; then
  echo "Failed to write MODULE.bazel module version ${next_version}" >&2
  exit 1
fi
if ! grep -Eq "^    version = \"${next_version}\",$" crates/tokenizers/BUILD.bazel; then
  echo "Failed to write crates/tokenizers/BUILD.bazel version ${next_version}" >&2
  exit 1
fi

echo "${next_version}"
